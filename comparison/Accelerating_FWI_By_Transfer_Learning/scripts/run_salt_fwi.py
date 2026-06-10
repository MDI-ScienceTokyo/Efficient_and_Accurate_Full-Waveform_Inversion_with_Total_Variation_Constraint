from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ZENODO = ROOT / "zenodo_source"
sys.path.insert(0, str(ZENODO))

import AdjointMethod  # noqa: E402
import FiniteDifferencePyTorchConv as FiniteDifference  # noqa: E402
import Utilities  # noqa: E402


def filename_value(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def infer_box_label(
    model: dict[str, np.ndarray | float | str],
    dataset_path: Path | None,
    requested_label: str,
) -> str:
    if requested_label != "auto":
        return requested_label
    box_min = float(model.get("box_min_value", model.get("vmin_km_s", 0.0)))
    box_max = float(model.get("box_max_value", model.get("vmax_km_s", 0.0)))
    vel_min = float(model.get("velocity_min_value", model.get("vmin_km_s", box_min)))
    vel_max = float(model.get("velocity_max_value", model.get("vmax_km_s", box_max)))
    if abs(vel_min - box_min) < 1.0e-4 and abs(vel_max - box_max) < 1.0e-4:
        return f"{filename_value(box_min)}-{filename_value(box_max)}"
    if dataset_path is not None and "unclipped" in dataset_path.parent.name.lower():
        return "unclipped"
    return f"{filename_value(vel_min)}-{filename_value(vel_max)}"


def load_model(path: Path) -> dict[str, np.ndarray | float]:
    data = np.load(path)
    model = {
        "gamma": data["gamma"].astype(np.float32),
        "initial_gamma": data["initial_gamma"].astype(np.float32),
        "velocity": data["velocity"].astype(np.float32),
        "initial_velocity": data["initial_velocity"].astype(np.float32),
        "vmin_km_s": float(data["vmin_km_s"]),
        "vmax_km_s": float(data["vmax_km_s"]),
    }
    return model


def load_efficient_dataset(path: Path, gamma_floor: float) -> dict[str, np.ndarray | float | str]:
    data = np.load(path)
    velocity = data["true_velocity_model"].astype(np.float32)
    initial_velocity = data["initial_velocity_model"].astype(np.float32)
    if "velocity_min_value" in data.files:
        vmin = float(data["velocity_min_value"])
    else:
        vmin = float(data["box_min_value"])
    if "velocity_max_value" in data.files:
        vmax = float(data["velocity_max_value"])
    else:
        vmax = float(data["box_max_value"])
    gamma = np.clip(vmin / velocity.T, gamma_floor, 1.0).astype(np.float32)
    initial_gamma = np.clip(vmin / initial_velocity.T, gamma_floor, 1.0).astype(np.float32)
    model: dict[str, np.ndarray | float | str] = {
        "gamma": gamma,
        "initial_gamma": initial_gamma,
        "velocity": velocity,
        "initial_velocity": initial_velocity,
        "vmin_km_s": vmin,
        "vmax_km_s": vmax,
        "box_min_value": float(data["box_min_value"]),
        "box_max_value": float(data["box_max_value"]),
        "velocity_min_value": vmin,
        "velocity_max_value": vmax,
        "dataset_path": str(path),
    }
    for key in (
        "observed_seismic_data",
        "clean_observed_seismic_data",
        "seismic_noise",
        "source_locations",
        "receiver_locations",
    ):
        if key in data.files:
            model[f"efficient_{key}"] = data[key].astype(np.float32)
    return model


def gamma_to_velocity_km_s(gamma: np.ndarray, vmin_km_s: float, vmax_km_s: float) -> np.ndarray:
    gamma_safe = np.maximum(gamma.astype(np.float32), np.finfo(np.float32).eps)
    velocity = vmin_km_s / gamma_safe
    return np.clip(velocity, vmin_km_s, vmax_km_s).astype(np.float32)


def gamma_xy_to_velocity_yx(gamma_xy: np.ndarray, vmin_km_s: float, vmax_km_s: float) -> np.ndarray:
    return gamma_to_velocity_km_s(gamma_xy, vmin_km_s, vmax_km_s).T.astype(np.float32)


def resize_bilinear(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    src_h, src_w = image.shape
    if image.shape == shape:
        return image.astype(np.float32)
    y = np.linspace(0, src_h - 1, target_h)
    x = np.linspace(0, src_w - 1, target_w)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = (1.0 - wx) * image[y0[:, None], x0] + wx * image[y0[:, None], x1]
    bottom = (1.0 - wx) * image[y1[:, None], x0] + wx * image[y1[:, None], x1]
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def make_source(args: argparse.Namespace, source_x: list[int], source_y: list[int]) -> torch.Tensor:
    return Utilities.getSource(
        args.frequency,
        args.cycles,
        args.amplitude,
        source_x,
        source_y,
        args.nx,
        args.ny,
        args.dx,
        args.dy,
        args.lx,
        args.ly,
        args.time_steps,
        args.dt,
    )


def resolve_waveform_noise_std(
    measurement: torch.Tensor,
    sigma: float,
    mode: str,
    velocity_min: float,
    velocity_max: float,
    reference_noise_ratio: float | None = None,
) -> float:
    if sigma <= 0:
        return 0.0
    if mode == "absolute":
        return float(sigma)
    signal_std = float(measurement.std().detach().cpu())
    if mode == "relative_std":
        return float(sigma) * signal_std
    if mode == "efficient_reference":
        if reference_noise_ratio is None:
            raise ValueError("efficient_reference noise mode requires a reference noise ratio")
        return float(reference_noise_ratio) * signal_std
    if mode == "velocity_relative":
        # Efficient and Accurate FWI uses velocity in km/s with box [1.5, 4.5].
        # Interpret sigma in that velocity scale, then carry only the relative
        # fraction of the model dynamic range to this solver's waveform scale.
        velocity_range = float(velocity_max - velocity_min)
        if velocity_range <= 0:
            raise ValueError("velocity_max must be larger than velocity_min")
        return float(sigma) / velocity_range * signal_std
    raise ValueError(f"unknown noise mode: {mode}")


def add_noise(measurement: torch.Tensor, sigma: float, seed: int, std: float) -> torch.Tensor:
    if sigma <= 0 or std <= 0:
        return measurement
    generator = torch.Generator(device=measurement.device).manual_seed(seed)
    return measurement + torch.normal(
        mean=0.0,
        std=std,
        size=measurement.shape,
        generator=generator,
        device=measurement.device,
    )



def locations_from_efficient_dataset(
    model: dict[str, np.ndarray | float | str],
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[int], list[int]]:
    src = model.get("efficient_source_locations")
    rec = model.get("efficient_receiver_locations")
    if not isinstance(src, np.ndarray) or not isinstance(rec, np.ndarray) or src.size == 0 or rec.size == 0:
        raise ValueError("Efficient reference geometry requested but source/receiver locations are missing")

    def to_indices(coords: np.ndarray) -> tuple[list[int], list[int]]:
        y_idx = np.rint(coords[:, 0] / args.dy).astype(int) + 1
        x_idx = np.rint(coords[:, 1] / args.dx).astype(int) + 1
        x_idx = np.clip(x_idx, 1, args.nx + 1)
        y_idx = np.clip(y_idx, 1, args.ny + 1)
        return x_idx.tolist(), y_idx.tolist()

    source_x, source_y = to_indices(src)
    sensor_x, sensor_y = to_indices(rec)
    return source_x, source_y, sensor_x, sensor_y

def run_case(args: argparse.Namespace, noise_sigma: float, dataset_path: Path | None = None) -> Path:
    device = torch.device("cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if dataset_path is not None:
        model = load_efficient_dataset(dataset_path, args.gamma_floor)
        args.ny = int(model["velocity"].shape[0] - 1)
        args.nx = int(model["velocity"].shape[1] - 1)
        args.lx = args.nx * args.dx
        args.ly = args.ny * args.dy
        print(f"loaded efficient dataset: {dataset_path}")
    else:
        model = load_model(Path(args.model))
    solver_shape = (args.nx + 1, args.ny + 1)
    gamma_np = resize_bilinear(model["gamma"], solver_shape)
    initial_np = resize_bilinear(model["initial_gamma"], solver_shape)
    public_shape = (args.ny + 1, args.nx + 1)
    velocity_true_np = resize_bilinear(model["velocity"], public_shape)
    velocity_initial_np = resize_bilinear(model["initial_velocity"], public_shape)
    vmin_km_s = float(model["vmin_km_s"])
    vmax_km_s = float(model["vmax_km_s"])

    if dataset_path is not None and args.geometry_mode == "efficient_reference":
        source_x, source_y, sensor_x, sensor_y = locations_from_efficient_dataset(model, args)
    else:
        source_x, source_y = Utilities.getSourceLocations(
            args.nx, args.ny, args.source_spacing, args.sources
        )
        sensor_x, sensor_y = Utilities.getSensorLocations(
            args.nx, args.ny, args.sensor_spacing, source_x
        )
    args.sources = len(source_x)
    reference_noise_ratio = None
    if dataset_path is not None and noise_sigma > 0:
        efficient_clean = model.get("efficient_clean_observed_seismic_data")
        efficient_noise = model.get("efficient_seismic_noise")
        if isinstance(efficient_clean, np.ndarray) and isinstance(efficient_noise, np.ndarray) and efficient_clean.size > 0:
            clean_ref_std = float(efficient_clean.std())
            if clean_ref_std > 0:
                reference_noise_ratio = float(efficient_noise.std()) / clean_ref_std
    print(
        f"device={device} geometry={args.geometry_mode} sources={len(source_x)} sensors={len(sensor_x)} "
        f"noise_sigma={noise_sigma} reference_noise_ratio={reference_noise_ratio}"
    )

    solver = FiniteDifference.FiniteDifference(args.dt, args.dx, args.dy, args.c, args.rho, device=device)
    f = make_source(args, source_x, source_y).to(device)
    u0 = torch.zeros((args.sources, 1, args.nx + 3, args.ny + 3), dtype=torch.float32, device=device)
    u1 = torch.zeros_like(u0)

    gamma_true = torch.ones((1, 1, args.nx + 3, args.ny + 3), dtype=torch.float32, device=device)
    gamma_true[0, 0, 1:-1, 1:-1] = torch.from_numpy(gamma_np).to(device)

    started = time.perf_counter()
    measurement_clean = solver.forwardNSteps(
        u0.clone(), u1.clone(), gamma_true, f, args.nx, args.ny, args.time_steps, args.sources, device
    )[:, sensor_x, sensor_y, 1:]
    waveform_noise_std = resolve_waveform_noise_std(
        measurement_clean,
        noise_sigma,
        args.noise_mode,
        args.velocity_min,
        args.velocity_max,
        reference_noise_ratio,
    )
    measurement = add_noise(measurement_clean, noise_sigma, args.seed, waveform_noise_std)
    clean_std = float(measurement_clean.std().detach().cpu())
    actual_noise_std = float((measurement - measurement_clean).std().detach().cpu())
    print(
        f"generated measurements in {time.perf_counter() - started:.2f}s "
        f"clean_std={clean_std:.6g} waveform_noise_std={waveform_noise_std:.6g} "
        f"actual_noise_std={actual_noise_std:.6g}"
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if dataset_path is not None:
        box_label = infer_box_label(model, dataset_path, args.box_label)
    else:
        box_label = args.box_label if args.box_label != "auto" else "none-none"
    run_dir = (
        ROOT
        / "results"
        / f"{timestamp}_salt_{args.method_label}_alpha-{args.alpha_label}_noise-{filename_value(noise_sigma)}_box-{box_label}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "observed_data.npz",
        measurement=measurement.detach().cpu().numpy(),
        source_x=np.array(source_x),
        source_y=np.array(source_y),
        sensor_x=np.array(sensor_x),
        sensor_y=np.array(sensor_y),
        noise_sigma=noise_sigma,
        noise_mode=args.noise_mode,
        waveform_noise_std=waveform_noise_std,
        clean_measurement_std=clean_std,
        actual_noise_std=actual_noise_std,
        reference_noise_ratio=np.float32(reference_noise_ratio if reference_noise_ratio is not None else np.nan),
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        geometry_mode=args.geometry_mode,
        efficient_dataset_path=str(dataset_path) if dataset_path is not None else "",
        efficient_observed_seismic_data=model.get("efficient_observed_seismic_data", np.array([], dtype=np.float32)),
        efficient_clean_observed_seismic_data=model.get("efficient_clean_observed_seismic_data", np.array([], dtype=np.float32)),
        efficient_seismic_noise=model.get("efficient_seismic_noise", np.array([], dtype=np.float32)),
        efficient_source_locations=model.get("efficient_source_locations", np.array([], dtype=np.float32)),
        efficient_receiver_locations=model.get("efficient_receiver_locations", np.array([], dtype=np.float32)),
    )

    gamma_pred = torch.nn.Parameter(torch.ones((1, 1, args.nx + 3, args.ny + 3), device=device))
    gamma_pred.data[0, 0, 1:-1, 1:-1] = torch.from_numpy(initial_np).to(device)
    optimizer = torch.optim.Adam((gamma_pred,), lr=args.lr)

    history = []
    gamma_history = []
    velocity_history = []
    for epoch in range(args.epochs):
        cost, gradient = AdjointMethod.getAdjointGradient(
            solver,
            u0.clone(),
            u1.clone(),
            args.c,
            args.rho,
            gamma_pred.detach(),
            f,
            args.nx,
            args.dx,
            args.ny,
            args.dy,
            args.time_steps,
            args.dt,
            args.sources,
            measurement,
            sensor_x,
            sensor_y,
            device,
        )
        gamma_pred.grad = torch.zeros_like(gamma_pred)
        gamma_pred.grad[0, 0, 1:-1, 1:-1] = gradient * args.cost_scaling
        gamma_pred.grad[0, 0, sensor_x, sensor_y] = 0
        gamma_pred.grad[0, 0, source_x, source_y] = 0
        torch.nn.utils.clip_grad_norm_((gamma_pred,), args.clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gamma_pred.data.clamp_(args.gamma_floor, 1.0)

        gamma_mse = 0.5 * torch.mean((gamma_true - gamma_pred) ** 2).detach().cpu().item()
        gamma_epoch = gamma_pred.detach().cpu().numpy()[0, 0]
        velocity_epoch = gamma_xy_to_velocity_yx(gamma_epoch[1:-1, 1:-1], vmin_km_s, vmax_km_s)
        velocity_mse = 0.5 * float(np.mean((velocity_true_np - velocity_epoch) ** 2))
        history.append((epoch + 1, float(cost), gamma_mse, velocity_mse))
        gamma_history.append(gamma_epoch)
        velocity_history.append(velocity_epoch)
        print(
            f"epoch={epoch + 1}/{args.epochs} cost={float(cost):.6e} "
            f"gamma_mse={gamma_mse:.6e} velocity_mse={velocity_mse:.6e}"
        )

    with (run_dir / "metrics.csv").open("w", newline="") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["epoch", "cost", "mse", "velocity_mse"])
        writer.writerows(history)
    gamma_final = gamma_pred.detach().cpu().numpy()[0, 0]
    velocity_final = gamma_xy_to_velocity_yx(gamma_final[1:-1, 1:-1], vmin_km_s, vmax_km_s)
    np.savez_compressed(
        run_dir / "fwi_result.npz",
        velocity_true=velocity_true_np.astype(np.float32),
        velocity_initial=velocity_initial_np.astype(np.float32),
        velocity_final=velocity_final.astype(np.float32),
        velocity_history=np.array(velocity_history, dtype=np.float32),
        final_velocity_model=velocity_final.astype(np.float32),
        true_velocity_model=velocity_true_np.astype(np.float32),
        initial_velocity_model=velocity_initial_np.astype(np.float32),
        gamma_true=gamma_true.detach().cpu().numpy()[0, 0],
        gamma_initial=initial_np,
        gamma_final=gamma_final,
        gamma_history=np.array(gamma_history, dtype=np.float32),
        vmin_km_s=vmin_km_s,
        vmax_km_s=vmax_km_s,
        velocity_axes="yx",
        gamma_axes="xy",
        geometry_mode=args.geometry_mode,
        efficient_dataset_path=str(dataset_path) if dataset_path is not None else "",
    )
    print(f"saved {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "data" / "processed" / "salt_model.npz"))
    parser.add_argument("--noise-sigmas", default="0,1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--nx", type=int, default=127)
    parser.add_argument("--ny", type=int, default=63)
    parser.add_argument("--lx", type=float, default=1270.0)
    parser.add_argument("--ly", type=float, default=630.0)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dy", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--time-steps", type=int, default=300)
    parser.add_argument("--c", type=float, default=1500.0)
    parser.add_argument("--rho", type=float, default=2700.0)
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--amplitude", type=float, default=1.0e9)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--source-spacing", type=int, default=18)
    parser.add_argument("--sensor-spacing", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--clip-grad", type=float, default=1.0e-5)
    parser.add_argument("--cost-scaling", type=float, default=1.0e-2)
    parser.add_argument(
        "--noise-mode",
        choices=("velocity_relative", "relative_std", "absolute", "efficient_reference"),
        default="velocity_relative",
        help=(
            "How --noise-sigmas is mapped to waveform noise. "
            "velocity_relative treats sigma in the [velocity_min, velocity_max] km/s scale; "
            "relative_std uses sigma * clean waveform std; efficient_reference uses the "
            "noise/clean std ratio stored in the Efficient dataset."
        ),
    )
    parser.add_argument("--velocity-min", type=float, default=1.5)
    parser.add_argument("--velocity-max", type=float, default=4.5)
    parser.add_argument("--gamma-floor", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--method-label", default="tlfwi", help="Method label used in result folder names.")
    parser.add_argument("--alpha-label", default="none", help="Alpha label used in result folder names.")
    parser.add_argument("--box-label", default="auto", help="Box label used in result folder names. Use auto, 1p5-4p5, unclipped, etc.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--efficient-dataset-dir",
        default=None,
        help="Directory containing salt_efficient_noise_sigma*.npz. When set, true/initial/noise metadata are loaded from these datasets.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("zenodo", "efficient_reference"),
        default="zenodo",
        help="Use Zenodo's centered grid geometry or source/receiver locations stored in the Efficient dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for item in args.noise_sigmas.split(","):
        if item.strip():
            noise_sigma = float(item)
            dataset_path = None
            if args.efficient_dataset_dir is not None:
                dataset_dir = Path(args.efficient_dataset_dir)
                if not dataset_dir.is_absolute():
                    dataset_dir = ROOT / dataset_dir
                dataset_path = dataset_dir / f"salt_efficient_noise_sigma{filename_value(noise_sigma)}.npz"
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Efficient dataset not found: {dataset_path}")
            run_case(args, noise_sigma, dataset_path)
