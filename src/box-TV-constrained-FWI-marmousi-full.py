import argparse
import csv
import json
import re
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from skimage.metrics import structural_similarity as ssim

import lib.signal_processing.diff_operator as diff_op
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelGradientCalculatorProps
from lib.signal_processing.misc import calc_psnr
from lib.signal_processing.norm import L12_norm
from lib.signal_processing.proximal_operator import proj_L12_norm_ball, prox_box_constraint
from lib.visualize import show_velocity_model


set_log_level("WARNING")

DEFAULT_MODEL_PATH = Path("datasets/marmousi/processed/marmousi_full_extent_full_extent_351x1701_10m.npz")


class FWIParams(NamedTuple):
    real_cell_size: Vec2D[int]
    cell_meter_size: Vec2D[float]
    damping_cell_thickness: int
    start_time: float
    unit_time: float
    simulation_times: int
    source_peek_time: float
    source_frequency: float
    n_shots: int
    n_receivers: int
    noise_sigma: float


class VelocityModelDataForOptimization(NamedTuple):
    true_data: npt.NDArray
    initial_data: npt.NDArray
    box_min_value: float
    box_max_value: float
    dz_m: float
    dx_m: float


def filename_value(value: Union[int, float, None]) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        value = f"{value:g}"
    return str(value).replace("-", "m").replace(".", "p")


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").lower()


def build_experiment_name(image_name: str, algorithm: str, alpha: float, noise_sigma: float, box_min_value: float, box_max_value: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return safe_filename(
        f"{timestamp}_{image_name}_{algorithm}_alpha-{filename_value(alpha)}_noise-{filename_value(noise_sigma)}_box-{filename_value(box_min_value)}-{filename_value(box_max_value)}"
    )


def remove_damping_cells(velocity_model: npt.NDArray, damping_cell_thickness: int):
    x = damping_cell_thickness
    return velocity_model[x:-x, x:-x]


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def load_marmousi_full_model(path: Path = DEFAULT_MODEL_PATH) -> VelocityModelDataForOptimization:
    if not path.exists():
        raise FileNotFoundError(
            f"Marmousi full-extent processed file was not found: {path}. "
            "Run: MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/prepare_marmousi_full_extent.py"
        )

    data = np.load(path, allow_pickle=True)
    true_velocity_model = data["vp_true"].astype(np.float32)
    initial_velocity_model = data["vp0"].astype(np.float32)
    if float(np.nanmax(true_velocity_model)) > 20.0:
        true_velocity_model = true_velocity_model / 1000.0
    if float(np.nanmax(initial_velocity_model)) > 20.0:
        initial_velocity_model = initial_velocity_model / 1000.0

    dz_m = float(data["dz"])
    dx_m = float(data["dx"])
    box_min_value = float(np.floor(float(true_velocity_model.min()) * 10) / 10)
    box_max_value = float(np.ceil(float(true_velocity_model.max()) * 10) / 10)

    print(f"Loaded Marmousi full-extent model from {path}")
    print(f"  shape: {true_velocity_model.shape}, spacing: dz={dz_m:g} m, dx={dx_m:g} m")
    print(f"  true velocity range: {float(true_velocity_model.min()):.4f} - {float(true_velocity_model.max()):.4f} km/s")
    print(f"  initial velocity range: {float(initial_velocity_model.min()):.4f} - {float(initial_velocity_model.max()):.4f} km/s")
    print(f"  box constraint: {box_min_value:.4f} - {box_max_value:.4f} km/s")
    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, box_min_value, box_max_value, dz_m, dx_m)


def marmousi_full_configuration(
    shape: tuple[int, int],
    dz_m: float,
    dx_m: float,
    n_shots: int,
    n_receivers: int,
    noise_sigma: float,
    simulation_times: int,
    source_frequency: float,
) -> FWIParams:
    nz, nx = shape
    return FWIParams(
        real_cell_size=Vec2D(nx, nz),
        cell_meter_size=Vec2D(dx_m, dz_m),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=simulation_times,
        source_peek_time=100,
        source_frequency=source_frequency,
        n_shots=n_shots,
        n_receivers=min(n_receivers, nx),
        noise_sigma=noise_sigma,
    )


def fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(
    params: FWIParams,
    true_velocity_model: npt.NDArray,
    initial_velocity_model: npt.NDArray,
    num_parallel_workers: int,
) -> FastParallelVelocityModelGradientCalculatorProps:
    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
    source_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_shots)], dtype=np.float32)
    receiver_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_receivers)], dtype=np.float32)
    return FastParallelVelocityModelGradientCalculatorProps(
        true_velocity_model,
        initial_velocity_model,
        shape,
        spacing,
        params.damping_cell_thickness,
        params.start_time,
        params.simulation_times,
        params.source_frequency,
        source_locations,
        receiver_locations,
        params.noise_sigma,
        num_parallel_workers,
    )


def save_experiment_results(
    output_dir: Path,
    config: dict,
    final_velocity_model_with_damping: npt.NDArray,
    final_velocity_model: npt.NDArray,
    dual_variable: npt.NDArray,
    true_velocity_model: npt.NDArray,
    initial_velocity_model: npt.NDArray,
    initial_velocity_model_with_damping: npt.NDArray,
    observed_seismic_data: npt.NDArray,
    source_locations: npt.NDArray,
    receiver_locations: npt.NDArray,
    objective_history: npt.NDArray,
    mse_history: npt.NDArray,
    psnr_history: npt.NDArray,
    ssim_history: npt.NDArray,
    tv_history: npt.NDArray,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir.joinpath(f"{output_dir.name}.npz")
    np.savez_compressed(
        result_path,
        final_velocity_model_with_damping=final_velocity_model_with_damping,
        final_velocity_model=final_velocity_model,
        dual_variable=dual_variable,
        true_velocity_model=true_velocity_model,
        initial_velocity_model=initial_velocity_model,
        initial_velocity_model_with_damping=initial_velocity_model_with_damping,
        observed_seismic_data=observed_seismic_data,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        objective_history=objective_history,
        mse_history=mse_history,
        psnr_history=psnr_history,
        ssim_history=ssim_history,
        tv_history=tv_history,
    )

    with output_dir.joinpath("metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective", "mse", "psnr", "ssim", "tv"])
        for i, values in enumerate(zip(objective_history, mse_history, psnr_history, ssim_history, tv_history), start=1):
            writer.writerow([i, *values])

    with output_dir.joinpath("config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved experiment results: {output_dir}")


def simulate_fwi(
    max_n_iters: int,
    n_shots: int,
    n_receivers: int,
    noise_sigma: float,
    algorithm: Union[Literal["pds"], Literal["gradient"]],
    gamma1: float,
    gamma2: Union[float, None],
    alpha: float,
    model_path: Path,
    simulation_times: int = 1000,
    source_frequency: float = 0.01,
    num_parallel_workers: int = 1,
    visualize_interval: Union[int, None] = None,
    result_root_path: Union[Path, None] = Path("results/marmousi_full"),
    image_name: str = "marmousi_full",
    random_seed: Union[int, None] = 0,
):
    if algorithm == "gradient":
        gamma2 = None
    if algorithm == "pds" and gamma2 is None:
        raise ValueError("gamma2 must be set when algorithm is pds")

    model_data = load_marmousi_full_model(model_path)
    true_velocity_model, initial_velocity_model, vmin, vmax, dz_m, dx_m = model_data
    if random_seed is not None:
        np.random.seed(random_seed)

    params = marmousi_full_configuration(true_velocity_model.shape, dz_m, dx_m, n_shots, n_receivers, noise_sigma, simulation_times, source_frequency)
    dsize = params.damping_cell_thickness

    show_velocity_model(true_velocity_model, vmax=vmax, vmin=vmin, title="marmousi full true velocity model", cmap="coolwarm")
    show_velocity_model(initial_velocity_model, vmax=vmax, vmin=vmin, title="marmousi full initial velocity model", cmap="coolwarm")
    total_variation_of_true_velocity_model = L12_norm(diff_op.D(true_velocity_model))
    print(f"TV of true velocity model: {total_variation_of_true_velocity_model}, alpha: {alpha}, ratio: {alpha / total_variation_of_true_velocity_model if total_variation_of_true_velocity_model else 0}")

    props = fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(params, true_velocity_model, initial_velocity_model, num_parallel_workers)
    grad_calculator = FastParallelVelocityModelGradientCalculator(props)
    initial_velocity_model_with_damping = grad_calculator.velocity_model.copy()
    observed_seismic_data = grad_calculator.true_observed_waveforms.copy()

    residual_norm_sum_values = []
    velocity_model_square_error_values = []
    psnr_values = []
    ssim_values = []
    total_variation_values = []

    v = grad_calculator.velocity_model.copy()
    y = diff_op.D(remove_damping_cells(v, dsize))
    th = -1
    start_time = time.perf_counter()
    elapsed = 0.0
    try:
        while True:
            th += 1
            residual_norm_sum, grad = grad_calculator.calc_grad(v)
            if not np.isfinite(residual_norm_sum):
                print(f"stopped: objective became non-finite at iteration {th + 1}: {residual_norm_sum}")
                break

            if algorithm == "gradient":
                v = v - gamma1 * grad
            elif algorithm == "pds":
                prev_v = v.copy()
                tmp = grad.copy()
                tmp[dsize:-dsize, dsize:-dsize] += diff_op.Dt(y)
                v = v - gamma1 * tmp
                v[dsize:-dsize, dsize:-dsize] = prox_box_constraint(remove_damping_cells(v, dsize), vmin, vmax)
                y = y + gamma2 * diff_op.D(2 * remove_damping_cells(v, dsize) - remove_damping_cells(prev_v, dsize))
                y = y - gamma2 * proj_L12_norm_ball(y / gamma2, alpha)

            if not np.all(np.isfinite(v)):
                print(f"stopped: velocity model became non-finite after iteration {th + 1}")
                break
            if float(np.min(v)) <= 0.0:
                print(f"stopped: velocity model became non-positive after iteration {th + 1}; min_velocity={float(np.min(v)):.6g} km/s")
                break

            v_core = remove_damping_cells(v, dsize)
            velocity_model_diff = v_core - true_velocity_model
            velocity_model_square_error = np.sum(velocity_model_diff * velocity_model_diff)
            psnr_value = calc_psnr(true_velocity_model, v_core, vmax)
            ssim_value = ssim(true_velocity_model, v_core, data_range=vmax - vmin)
            total_variation_value = L12_norm(diff_op.D(v_core))

            residual_norm_sum_values.append(residual_norm_sum)
            velocity_model_square_error_values.append(velocity_model_square_error)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            total_variation_values.append(total_variation_value)

            print(
                f"iters: {th + 1}, objective: {residual_norm_sum:.4f}, mse: {velocity_model_square_error:.4f}, "
                f"psnr: {psnr_value:.4f}, ssim: {ssim_value:.4f}, TV: {total_variation_value:.4f}"
            )

            if visualize_interval is not None and (th + 1) % visualize_interval == 0:
                show_velocity_model(v_core, title=f"marmousi full velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap="coolwarm")

            if th == max_n_iters - 1:
                break
    finally:
        elapsed = time.perf_counter() - start_time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        v_core = remove_damping_cells(v, dsize)
        show_velocity_model(v_core, title=f"marmousi full velocity model at final iteration {th + 1}", vmax=vmax, vmin=vmin, cmap="coolwarm")

        objective_history = np.asarray(residual_norm_sum_values)
        mse_history = np.asarray(velocity_model_square_error_values)
        psnr_history = np.asarray(psnr_values)
        ssim_history = np.asarray(ssim_values)
        tv_history = np.asarray(total_variation_values)

        if result_root_path is not None:
            experiment_name = build_experiment_name(image_name, algorithm, alpha, noise_sigma, vmin, vmax)
            output_dir = result_root_path.joinpath(experiment_name)
            config = {
                "image_name": image_name,
                "algorithm": algorithm,
                "max_n_iters": max_n_iters,
                "completed_iters": th + 1,
                "n_shots": n_shots,
                "n_receivers": params.n_receivers,
                "noise_sigma": noise_sigma,
                "gamma1": gamma1,
                "gamma2": gamma2,
                "alpha": alpha,
                "box_min_value": vmin,
                "box_max_value": vmax,
                "random_seed": random_seed,
                "elapsed": elapsed,
                "real_cell_size": {"x": params.real_cell_size.x, "y": params.real_cell_size.y},
                "cell_meter_size": {"x": params.cell_meter_size.x, "y": params.cell_meter_size.y},
                "damping_cell_thickness": params.damping_cell_thickness,
                "start_time": params.start_time,
                "unit_time": params.unit_time,
                "simulation_times": params.simulation_times,
                "source_peek_time": params.source_peek_time,
                "source_frequency": params.source_frequency,
                "model_path": str(model_path),
                "num_parallel_workers": num_parallel_workers,
                "velocity_unit": "km/s",
            }
            save_experiment_results(
                output_dir,
                config,
                v,
                v_core,
                y,
                true_velocity_model,
                initial_velocity_model,
                initial_velocity_model_with_damping,
                observed_seismic_data,
                grad_calculator.props.source_locations,
                grad_calculator.props.receiver_locations,
                objective_history,
                mse_history,
                psnr_history,
                ssim_history,
                tv_history,
            )

        print(f"elapsed: {elapsed}")
        del grad_calculator
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_marmousi_full_experiments(
    max_n_iters: int,
    n_shots: int,
    n_receivers: int,
    noise_sigma: float,
    gamma1: float,
    gamma2: float,
    model_path: Path,
    alpha_values: tuple[float, ...] | None = None,
    alpha_scales: tuple[float, ...] = (0.8,),
    simulation_times: int = 1000,
    source_frequency: float = 0.01,
    num_parallel_workers: int = 1,
    result_root_path: Path = Path("results/marmousi_full"),
    random_seed: Union[int, None] = 0,
):
    true_velocity_model, _, _, _, _, _ = load_marmousi_full_model(model_path)
    tv_true = L12_norm(diff_op.D(true_velocity_model))
    if alpha_values is None:
        alpha_values = tuple(scale * tv_true for scale in alpha_scales)

    print(f"Marmousi full TV: {tv_true}")
    print(f"Marmousi full alpha values: {alpha_values}")

    for alpha in alpha_values:
        if alpha == 0:
            simulate_fwi(
                max_n_iters,
                n_shots,
                n_receivers,
                noise_sigma,
                "gradient",
                gamma1,
                None,
                alpha,
                model_path=model_path,
                simulation_times=simulation_times,
                source_frequency=source_frequency,
                num_parallel_workers=num_parallel_workers,
                result_root_path=result_root_path,
                random_seed=random_seed,
            )
        else:
            simulate_fwi(
                max_n_iters,
                n_shots,
                n_receivers,
                noise_sigma,
                "pds",
                gamma1,
                gamma2,
                alpha,
                model_path=model_path,
                simulation_times=simulation_times,
                source_frequency=source_frequency,
                num_parallel_workers=num_parallel_workers,
                result_root_path=result_root_path,
                random_seed=random_seed,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FWI on the full-physical-extent Marmousi model.")
    parser.add_argument("--max-n-iters", type=int, default=10)
    parser.add_argument("--n-shots", type=int, default=10)
    parser.add_argument("--n-receivers", type=int, default=201)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--gamma1", type=float, default=1e-6)
    parser.add_argument("--gamma2", type=float, default=100.0)
    parser.add_argument("--alphas", default=None, help="Comma-separated alpha values. Use 0 for gradient baseline.")
    parser.add_argument("--alpha-scales", default="0.8", help="Comma-separated alpha scales relative to TV(true), used when --alphas is omitted.")
    parser.add_argument("--simulation-times", type=int, default=1000)
    parser.add_argument("--source-frequency", type=float, default=0.01)
    parser.add_argument("--num-parallel-workers", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--result-root-path", type=Path, default=Path("results/marmousi_full"))
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    alpha_values = parse_float_tuple(args.alphas) if args.alphas is not None else None
    alpha_scales = parse_float_tuple(args.alpha_scales)
    run_marmousi_full_experiments(
        max_n_iters=args.max_n_iters,
        n_shots=args.n_shots,
        n_receivers=args.n_receivers,
        noise_sigma=args.noise_sigma,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        model_path=args.model_path,
        alpha_values=alpha_values,
        alpha_scales=alpha_scales,
        simulation_times=args.simulation_times,
        source_frequency=args.source_frequency,
        num_parallel_workers=args.num_parallel_workers,
        result_root_path=args.result_root_path,
        random_seed=args.random_seed,
    )
