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
from lib.dataset import load_seismic_datasets__salt_model
from lib.dataset.load_overthrust_model import load_seismic_datasets__overthrust_model
from lib.misc import datasets_root_path
from lib.misc.historical_value import ValueHistoryList
from lib.model import Vec2D
from lib.seismic import FastParallelVelocityModelGradientCalculator, FastParallelVelocityModelGradientCalculatorProps
from lib.signal_processing.misc import calc_psnr, smoothing_with_gaussian_filter, zoom_and_crop
from lib.signal_processing.norm import L12_norm
from lib.signal_processing.proximal_operator import proj_L12_norm_ball, prox_box_constraint
from lib.visualize import show_velocity_model

# suppress to dump devito logs
set_log_level("WARNING")

"""
[NOTE] The number of parallel processes for gradient calculation.
       If you get an error because of insufficient computational resources, reduce this value.
"""
num_parallels = 20


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


def salt_model_test00_configuration(n_shots: int, noise_sigma: float) -> FWIParams:
    return FWIParams(
        real_cell_size=Vec2D(100, 50),
        cell_meter_size=Vec2D(10.0, 10.0),
        damping_cell_thickness=40,
        start_time=0,
        unit_time=1,
        simulation_times=1000,
        source_peek_time=100,
        source_frequency=0.01,
        n_shots=n_shots,
        n_receivers=101,
        noise_sigma=noise_sigma,
    )


def fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props(
    params: FWIParams, true_velocity_model: npt.NDArray, initial_velocity_model: npt.NDArray
) -> FastParallelVelocityModelGradientCalculatorProps:
    shape = (params.real_cell_size.y, params.real_cell_size.x)
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    width = ((params.real_cell_size - Vec2D(1, 1)) * params.cell_meter_size).x
    source_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_shots)])
    receiver_locations = np.array([[30, x] for x in np.linspace(0, width, num=params.n_receivers)])
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


def load_salt_model(real_cell_size: Vec2D[int], target_idx: int = 300):
    vmin, vmax = 1.5, 4.5

    seismic_data_path = datasets_root_path.joinpath("salt_and_overthrust_models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
    seismic_data = load_seismic_datasets__salt_model(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    assert vmin <= np.min(seismic_data) and np.max(seismic_data) <= vmax

    raw_true_velocity_model = seismic_data[target_idx]
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (real_cell_size.y, real_cell_size.x))
    initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[target_idx], 1, 80), (real_cell_size.y, real_cell_size.x))

    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, vmin, vmax)


def load_overthrust_model(real_cell_size: Vec2D[int], target_idx: int = 300):
    vmin, vmax = 2.1, 6.0

    seismic_data_path = datasets_root_path.joinpath("salt_and_overthrust_models/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites")
    seismic_data = load_seismic_datasets__overthrust_model(seismic_data_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    assert vmin <= np.min(seismic_data) and np.max(seismic_data) <= vmax

    raw_true_velocity_model = seismic_data[target_idx]
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (real_cell_size.y, real_cell_size.x))
    initial_velocity_model = zoom_and_crop(smoothing_with_gaussian_filter(seismic_data[target_idx], 1, 80), (real_cell_size.y, real_cell_size.x))

    return VelocityModelDataForOptimization(true_velocity_model, initial_velocity_model, vmin, vmax)


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
    result_root_path: Union[Path, None] = Path("results"),
    image_name: str = "salt",
    random_seed: Union[int, None] = 0,
):
    if algorithm == "gradient":
        gamma2 = None
    if algorithm == "pds" and gamma2 is None:
        raise ValueError("gamma2 must be set when algorithm is pds")

    # configures
    params = salt_model_test00_configuration(n_shots, noise_sigma)

    # alias
    dsize = params.damping_cell_thickness

    # load data
    true_velocity_model, initial_velocity_model, vmin, vmax = load_salt_model(params.real_cell_size)
    # true_velocity_model, initial_velocity_model, vmin, vmax = load_overthrust_model(params.real_cell_size)
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

    start_time = time.time()
    elapsed = 0.0
    try:
        while True:
            th += 1
            residual_norm_sum, grad = grad_calculator.calc_grad(v)

            if np.isnan(residual_norm_sum):
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
        elapsed = time.time() - start_time
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
    alphas: tuple[float, ...] = (0, 150, 350, 550),
    noise_sigmas: tuple[float, ...] = (0, 1),
    max_n_iters: int = 5000,
    n_shots: int = 20,
    gamma1: float = 1e-4,
    gamma2: float = 100,
    result_root_path: Path = Path("results"),
    image_name: str = "salt",
    random_seed: Union[int, None] = 0,
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
                )


if __name__ == "__main__":
    run_alpha_experiments()
