from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from devito import set_log_level

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.dataset import load_seismic_datasets__salt_model  # noqa: E402
from lib.misc import datasets_root_path  # noqa: E402
from lib.model import Vec2D  # noqa: E402
from lib.seismic import (  # noqa: E402
    FastParallelVelocityModelGradientCalculator,
    FastParallelVelocityModelGradientCalculatorProps,
)
from lib.signal_processing.misc import smoothing_with_gaussian_filter, zoom_and_crop  # noqa: E402

set_log_level("WARNING")


def filename_value(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def load_efficient_salt_model(args: argparse.Namespace):
    vmin, vmax = 1.5, 4.5
    salt_path = datasets_root_path.joinpath("salt_and_overthrust_models/3-D_Salt_Model/VEL_GRIDS/Saltf@@")
    seismic_data = load_seismic_datasets__salt_model(salt_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    if not (vmin <= float(np.min(seismic_data)) and float(np.max(seismic_data)) <= vmax):
        raise ValueError("Salt velocity range does not match expected [1.5, 4.5] km/s")

    real_cell_size = Vec2D(args.nx, args.ny)
    raw_true_velocity_model = seismic_data[args.target_idx]
    raw_initial_velocity_model = smoothing_with_gaussian_filter(raw_true_velocity_model, 1, args.smooth_sigma)
    true_velocity_model = zoom_and_crop(raw_true_velocity_model, (real_cell_size.y, real_cell_size.x)).astype(np.float32)
    initial_velocity_model = zoom_and_crop(raw_initial_velocity_model, (real_cell_size.y, real_cell_size.x)).astype(np.float32)
    if args.clip_velocity:
        true_velocity_model = np.clip(true_velocity_model, vmin, vmax).astype(np.float32)
        initial_velocity_model = np.clip(initial_velocity_model, vmin, vmax).astype(np.float32)
    return true_velocity_model, initial_velocity_model, vmin, vmax, salt_path


def build_props(args: argparse.Namespace, true_velocity_model: np.ndarray, initial_velocity_model: np.ndarray, noise_sigma: float):
    real_cell_size = Vec2D(args.nx, args.ny)
    cell_meter_size = Vec2D(args.dx, args.dy)
    shape = (real_cell_size.y, real_cell_size.x)
    spacing = (cell_meter_size.y, cell_meter_size.x)
    width = ((real_cell_size - Vec2D(1, 1)) * cell_meter_size).x
    source_locations = np.array([[args.source_depth_m, x] for x in np.linspace(0, width, num=args.n_shots)], dtype=np.float32)
    receiver_locations = np.array([[args.receiver_depth_m, x] for x in np.linspace(0, width, num=args.n_receivers)], dtype=np.float32)
    return FastParallelVelocityModelGradientCalculatorProps(
        true_velocity_model,
        initial_velocity_model,
        shape,
        spacing,
        args.damping_cell_thickness,
        args.start_time,
        args.end_time,
        args.source_frequency,
        source_locations,
        receiver_locations,
        noise_sigma,
        args.num_jobs,
    )


def save_dataset(
    output_dir: Path,
    args: argparse.Namespace,
    noise_sigma: float,
    true_velocity_model: np.ndarray,
    initial_velocity_model: np.ndarray,
    initial_velocity_model_with_damping: np.ndarray,
    clean_observed: np.ndarray,
    observed: np.ndarray,
    source_locations: np.ndarray,
    receiver_locations: np.ndarray,
    vmin: float,
    vmax: float,
    salt_path: Path,
):
    noise = observed - clean_observed
    config = {
        "source": "Efficient_and_Accurate_Full-Waveform_Inversion_with_Total_Variation_Constraint compatible Salt dataset",
        "salt_model_path": str(salt_path),
        "target_idx": args.target_idx,
        "smooth_sigma": args.smooth_sigma,
        "box_min_value": vmin,
        "box_max_value": vmax,
        "velocity_clip_applied": bool(args.clip_velocity),
        "true_velocity_min": float(true_velocity_model.min()),
        "true_velocity_max": float(true_velocity_model.max()),
        "initial_velocity_min": float(initial_velocity_model.min()),
        "initial_velocity_max": float(initial_velocity_model.max()),
        "velocity_axes": "yx",
        "source_receiver_coordinate_axes": "model_coordinates_depth_or_y_then_x",
        "noise_sigma": noise_sigma,
        "noise_is_added_to": "observed_seismic_data waveform samples",
        "random_seed": args.seed,
        "real_cell_size": {"x": args.nx, "y": args.ny},
        "cell_meter_size": {"x": args.dx, "y": args.dy},
        "damping_cell_thickness": args.damping_cell_thickness,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "source_frequency": args.source_frequency,
        "n_shots": args.n_shots,
        "n_receivers": args.n_receivers,
        "num_jobs": args.num_jobs,
        "observed_seismic_data_shape": list(observed.shape),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"salt_efficient_noise_sigma{filename_value(noise_sigma)}"
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        true_velocity_model=true_velocity_model.astype(np.float32),
        initial_velocity_model=initial_velocity_model.astype(np.float32),
        initial_velocity_model_with_damping=initial_velocity_model_with_damping.astype(np.float32),
        observed_seismic_data=observed.astype(np.float32),
        clean_observed_seismic_data=clean_observed.astype(np.float32),
        seismic_noise=noise.astype(np.float32),
        source_locations=source_locations.astype(np.float32),
        receiver_locations=receiver_locations.astype(np.float32),
        noise_sigma=np.float32(noise_sigma),
        box_min_value=np.float32(vmin),
        box_max_value=np.float32(vmax),
        velocity_min_value=np.float32(true_velocity_model.min()),
        velocity_max_value=np.float32(true_velocity_model.max()),
        velocity_axes="yx",
        source_receiver_coordinate_axes="model_coordinates_depth_or_y_then_x",
        config_json=json.dumps(config, indent=2),
    )
    (output_dir / f"{stem}.json").write_text(json.dumps(config, indent=2))
    print(f"saved {npz_path}")
    print(f"  true velocity: shape={true_velocity_model.shape} min={true_velocity_model.min():.4f} max={true_velocity_model.max():.4f} km/s")
    print(f"  initial velocity: shape={initial_velocity_model.shape} min={initial_velocity_model.min():.4f} max={initial_velocity_model.max():.4f} km/s")
    print(f"  observed seismic: shape={observed.shape} clean_std={clean_observed.std():.6g} noise_std={noise.std():.6g}")


def create(args: argparse.Namespace):
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = WORKSPACE / output_dir

    true_velocity_model, initial_velocity_model, vmin, vmax, salt_path = load_efficient_salt_model(args)

    clean_props = build_props(args, true_velocity_model, initial_velocity_model, noise_sigma=0.0)
    calculator = FastParallelVelocityModelGradientCalculator(clean_props)
    try:
        initial_velocity_model_with_damping = calculator.velocity_model.copy()
        clean_observed = calculator.true_observed_waveforms.copy()
    finally:
        del calculator

    for noise_sigma in args.noise_sigmas:
        np.random.seed(args.seed)
        if noise_sigma == 0:
            observed = clean_observed.copy()
        else:
            observed = clean_observed + np.random.normal(0, noise_sigma, clean_observed.shape).astype(np.float32)
        props = build_props(args, true_velocity_model, initial_velocity_model, noise_sigma=noise_sigma)
        save_dataset(
            output_dir,
            args,
            noise_sigma,
            true_velocity_model,
            initial_velocity_model,
            initial_velocity_model_with_damping,
            clean_observed,
            observed,
            props.source_locations,
            props.receiver_locations,
            vmin,
            vmax,
            salt_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/efficient_salt")
    parser.add_argument("--clip-velocity", action="store_true", help="Clip true/initial velocity to [1.5, 4.5]. Default leaves zoom/crop values unclipped.")
    parser.add_argument("--noise-sigmas", default="0,1")
    parser.add_argument("--target-idx", type=int, default=300)
    parser.add_argument("--smooth-sigma", type=float, default=80.0)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dy", type=float, default=10.0)
    parser.add_argument("--damping-cell-thickness", type=int, default=40)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=1000.0)
    parser.add_argument("--source-frequency", type=float, default=0.01)
    parser.add_argument("--n-shots", type=int, default=20)
    parser.add_argument("--n-receivers", type=int, default=101)
    parser.add_argument("--source-depth-m", type=float, default=30.0)
    parser.add_argument("--receiver-depth-m", type=float, default=30.0)
    parser.add_argument("--num-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.noise_sigmas = tuple(float(x) for x in args.noise_sigmas.split(",") if x.strip())
    return args


if __name__ == "__main__":
    create(parse_args())
