import csv
import json
import re
import signal
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from devito import set_log_level
from skimage.metrics import structural_similarity as ssim

import lib.signal_processing.diff_operator as diff_op
from lib.misc.historical_value import ValueHistoryList
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelGradientCalculatorProps
from lib.signal_processing.misc import calc_psnr
from lib.signal_processing.norm import L12_norm
from lib.signal_processing.proximal_operator import proj_L12_norm_ball, prox_box_constraint
from lib.visualize import show_velocity_model

# suppress to dump devito logs
set_log_level("WARNING")

"""
[NOTE] The number of parallel processes for gradient calculation.
       If you get an error because of insufficient computational resources, reduce this value.
"""
num_parallels = 1

BP2004_INITIAL_MODEL_PATH = Path("data/processed/bp2004/bp2004_initial_crop.npz")


class FWIParams(NamedTuple):
    real_cell_size: Vec2D[int]
    cell_meter_size: Vec2D[float]

    # damping
    damping_cell_thickness: int

    # time
    start_time: float
    unit_time: float
    simulation_times: int

    # input source
    source_peek_time: float
    source_frequency: float
    source_depth: float
    receiver_depth: float
    shot_margin: float

    # shorts
    n_shots: int
    n_receivers: int

    # noise
    noise_sigma: float


class VelocityModelDataForOptimization(NamedTuple):
    true_data: npt.NDArray
    initial_data: npt.NDArray
    box_min_value: float
    box_max_value: float


def bp2004_configuration(
    model_shape: tuple[int, int],
    dx_m: float,
    dz_m: float,
    n_shots: int,
    n_receivers: int,
    noise_sigma: float,
) -> FWIParams:
    nz, nx = model_shape
    return FWIParams(
        real_cell_size=Vec2D(nx, nz),
        cell_meter_size=Vec2D(dx_m, dz_m),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=999,
        source_peek_time=100,
        source_frequency=0.005,
        source_depth=25.0,
        receiver_depth=25.0,
        shot_margin=1000.0,
        n_shots=n_shots,
        n_receivers=n_receivers,
        noise_sigma=noise_sigma,
    )


def fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(
    params: FWIParams, true_velocity_model: npt.NDArray, initial_velocity_model: npt.NDArray
) -> FastParallelVelocityModelGradientCalculatorProps:
    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
    margin = min(params.shot_margin, max(0.0, width / 3.0))
    if params.n_shots <= 1:
        source_x = np.array([width / 2.0], dtype=np.float32)
    else:
        source_x = np.linspace(margin, width - margin, num=params.n_shots, dtype=np.float32)
    receiver_x = np.linspace(0, width, num=params.n_receivers, dtype=np.float32)
    source_locations = np.array([[params.source_depth, x] for x in source_x], dtype=np.float32)
    receiver_locations = np.array([[params.receiver_depth, x] for x in receiver_x], dtype=np.float32)
    props = FastParallelVelocityModelGradientCalculatorProps(
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
        num_parallels,
    )
    return props


def load_bp2004_model(path: Path = BP2004_INITIAL_MODEL_PATH) -> tuple[VelocityModelDataForOptimization, float, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"BP2004 preprocessed model not found: {path}. "
            "Run scripts/preprocess_bp2004.py and scripts/make_bp2004_initial_model.py first."
        )

    data = np.load(path, allow_pickle=True)
    true_velocity_model = (data["vp_true"].astype(np.float32) / 1000.0).copy()
    initial_velocity_model = (data["vp0"].astype(np.float32) / 1000.0).copy()
    dx_m = float(data["dx"])
    dz_m = float(data["dz"])

    vmin = 1.45
    vmax = 5.50
    print(f"Loaded BP2004 model from {path}")
    print(f"  shape: {true_velocity_model.shape}, spacing: dz={dz_m} m, dx={dx_m} m")
    print(f"  true velocity range: {float(np.min(true_velocity_model)):.4f} - {float(np.max(true_velocity_model)):.4f} km/s")
    print(f"  initial velocity range: {float(np.min(initial_velocity_model)):.4f} - {float(np.max(initial_velocity_model)):.4f} km/s")
    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, vmin, vmax), dx_m, dz_m


def remove_damping_cells(velocity_model: npt.NDArray, damping_cell_thickness: int):
    x = damping_cell_thickness
    return velocity_model[x:-x, x:-x]


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

    metrics_path = output_dir.joinpath("metrics.csv")
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective", "mse", "psnr", "ssim", "tv"])
        for i, values in enumerate(zip(objective_history, mse_history, psnr_history, ssim_history, tv_history), start=1):
            writer.writerow([i, *values])

    config_path = output_dir.joinpath("config.json")
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved experiment results: {output_dir}")


# 本来はalgorithmとgamma1, gamma2, alhpaなどのパラメータは紐づくはずだが、一旦これで
def simulate_fwi(
    max_n_iters: int,
    n_shots: int,
    noise_sigma: float,
    algorithm: Union[Literal["pds"], Literal["gradient"]],
    gamma1: float,
    gamma2: Union[float, None],
    alpha: float,
    visualize_interval: Union[int, None] = None,
    np_log_path: Union[Path, None] = None,
    result_root_path: Union[Path, None] = Path("results/bp2004"),
    image_name: str = "bp2004",
    random_seed: Union[int, None] = 0,
    n_receivers: int = 201,
    model_path: Path = BP2004_INITIAL_MODEL_PATH,
):
    if algorithm == "gradient":
        gamma2 = None
    if algorithm == "pds" and gamma2 is None:
        raise ValueError("gamma2 must be set when algorithm is pds")

    model_data, dx_m, dz_m = load_bp2004_model(model_path)
    true_velocity_model, initial_velocity_model, vmin, vmax = model_data
    params = bp2004_configuration(true_velocity_model.shape, dx_m, dz_m, n_shots, n_receivers, noise_sigma)

    # alias
    dsize = params.damping_cell_thickness

    if random_seed is not None:
        np.random.seed(random_seed)

    # simple visualize
    def simple_visualize():
        show_velocity_model(true_velocity_model, vmax=vmax, vmin=vmin, title="true velocity model", cmap="coolwarm")
        show_velocity_model(initial_velocity_model, vmax=vmax, vmin=vmin, title="initial velocity model", cmap="coolwarm")
        total_variation_of_true_velocity_model = L12_norm(diff_op.D(true_velocity_model))
        print(f"TV of true velocity model: {total_variation_of_true_velocity_model}, alpha: {alpha}, ratio: {alpha / total_variation_of_true_velocity_model}")

    simple_visualize()

    def create_grad_calculator():
        props = fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(params, true_velocity_model, initial_velocity_model)
        return FastParallelVelocityModelGradientCalculator(props)

    grad_calculator = create_grad_calculator()
    initial_velocity_model_with_damping = grad_calculator.velocity_model.copy()
    observed_seismic_data = grad_calculator.true_observed_waveforms.copy()

    residual_norm_sum_values = ValueHistoryList("objective", "less", [])
    velocity_model_square_error_values = ValueHistoryList("velocity model square error", "less", [])
    psnr_values = ValueHistoryList("psnr", "greater", [])
    ssim_values = ValueHistoryList("ssim", "greater", [])
    total_variation_values = ValueHistoryList("TV", None, [])

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
                print(
                    f"stopped: velocity model became non-positive after iteration {th + 1}; "
                    f"min_velocity={float(np.min(v)):.6g} km/s. Try a smaller gamma1."
                )
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
                f"iters: {th+1}, "
                f"{residual_norm_sum_values.prev_value_message(1)}, "
                f"{velocity_model_square_error_values.prev_value_message(3)}, "
                f"{psnr_values.prev_value_message(4)}, "
                f"{ssim_values.prev_value_message(4)}, "
                f"{total_variation_values.prev_value_message(4)}, "
            )

            if visualize_interval is not None and (th + 1) % visualize_interval == 0:
                show_velocity_model(v_core, title=f"Velocity model at iteration {th + 1}", vmax=vmax, vmin=vmin, cmap="coolwarm")

            if th == max_n_iters - 1:
                break

    finally:
        elapsed = time.perf_counter() - start_time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if np_log_path is not None:
            np.savez(
                np_log_path,
                v,
                y,
                velocity_model_square_error_values.values_as_np_array(),
                residual_norm_sum_values.values_as_np_array(),
                psnr_values.values_as_np_array(),
                ssim_values.values_as_np_array(),
            )

        v_core = remove_damping_cells(v, dsize)
        show_velocity_model(v_core, title=f"Velocity model at final iteration {th + 1}", vmax=vmax, vmin=vmin, cmap="coolwarm")

        objective_history = residual_norm_sum_values.values_as_np_array()
        mse_history = velocity_model_square_error_values.values_as_np_array()
        psnr_history = psnr_values.values_as_np_array()
        ssim_history = ssim_values.values_as_np_array()
        tv_history = total_variation_values.values_as_np_array()

        if result_root_path is not None:
            experiment_name = build_experiment_name(image_name, algorithm, alpha, noise_sigma, vmin, vmax)
            output_dir = result_root_path.joinpath(experiment_name)
            config = {
                "image_name": image_name,
                "algorithm": algorithm,
                "max_n_iters": max_n_iters,
                "completed_iters": th + 1,
                "n_shots": n_shots,
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
                "n_receivers": params.n_receivers,
                "source_depth": params.source_depth,
                "receiver_depth": params.receiver_depth,
                "shot_margin": params.shot_margin,
                "model_path": str(model_path),
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
        # release child process
        del grad_calculator

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_alpha_experiments(
    alphas: tuple[float, ...] = (0,),
    noise_sigmas: tuple[float, ...] = (0,),
    max_n_iters: int = 100,
    n_shots: int = 5,
    n_receivers: int = 201,
    gamma1: float = 1e-6,
    gamma2: float = 100,
    result_root_path: Path = Path("results/bp2004"),
    image_name: str = "bp2004",
    random_seed: Union[int, None] = 0,
    model_path: Path = BP2004_INITIAL_MODEL_PATH,
):
    for noise_sigma in noise_sigmas:
        for alpha in alphas:
            if alpha == 0:
                simulate_fwi(
                    max_n_iters,
                    n_shots,
                    noise_sigma,
                    "gradient",
                    gamma1,
                    None,
                    alpha,
                    result_root_path=result_root_path,
                    image_name=image_name,
                    random_seed=random_seed,
                    n_receivers=n_receivers,
                    model_path=model_path,
                )
            else:
                simulate_fwi(
                    max_n_iters,
                    n_shots,
                    noise_sigma,
                    "pds",
                    gamma1,
                    gamma2,
                    alpha,
                    result_root_path=result_root_path,
                    image_name=image_name,
                    random_seed=random_seed,
                    n_receivers=n_receivers,
                    model_path=model_path,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run box/TV constrained FWI code on the preprocessed BP2004 crop.")
    parser.add_argument("--max-n-iters", type=int, default=100)
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--n-receivers", type=int, default=201)
    parser.add_argument("--gamma1", type=float, default=1e-6)
    parser.add_argument("--gamma2", type=float, default=100.0)
    parser.add_argument("--alphas", default="0", help="Comma-separated alpha values. Use 0 for the plain gradient baseline.")
    parser.add_argument("--noise-sigmas", default="0", help="Comma-separated Gaussian noise sigma values.")
    parser.add_argument("--result-root-path", type=Path, default=Path("results/bp2004"))
    parser.add_argument("--model-path", type=Path, default=BP2004_INITIAL_MODEL_PATH)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())
    noise_sigmas = tuple(float(x) for x in args.noise_sigmas.split(",") if x.strip())
    run_alpha_experiments(
        alphas=alphas,
        noise_sigmas=noise_sigmas,
        max_n_iters=args.max_n_iters,
        n_shots=args.n_shots,
        n_receivers=args.n_receivers,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        result_root_path=args.result_root_path,
        random_seed=args.random_seed,
        model_path=args.model_path,
    )
