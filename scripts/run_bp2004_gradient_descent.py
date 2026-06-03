from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from bp2004_utils import ensure_dirs, load_config, npz_meta, read_npz_meta, save_velocity_plot
from run_bp2004_gradient import objective_and_gradient


def plot_history(values: np.ndarray, path: Path, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, values.size + 1), values, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BP2004 plain gradient descent without TV or box constraint.")
    parser.add_argument("--config", default="configs/bp2004_gradient.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--observed", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--gamma1", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--progress-interval", type=int, default=1, help="Write progress files every N iterations.")
    args = parser.parse_args()

    config = load_config(args.config)
    _, processed_dir, base_output_dir = ensure_dirs(config)
    output_dir = Path(args.output_dir) if args.output_dir else base_output_dir / "gradient_descent_no_tv_no_box"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model) if args.model else processed_dir / "bp2004_initial_crop.npz"
    observed_path = Path(args.observed) if args.observed else base_output_dir / "observed_data.npz"
    model_data = np.load(model_path, allow_pickle=True)
    obs_data = np.load(observed_path, allow_pickle=True)
    observed = {key: obs_data[key] for key in obs_data.files}

    vp_true = model_data["vp_true"].astype(np.float32)
    vp = model_data["vp0"].astype(np.float32).copy()
    dx = float(model_data["dx"])
    dz = float(model_data["dz"])
    iterations = int(args.iterations if args.iterations is not None else config.get("gradient_descent", {}).get("iterations", 100))
    gamma1 = float(args.gamma1 if args.gamma1 is not None else config.get("gradient_descent", {}).get("gamma1", 1.0e-2))

    objective_history: list[float] = []
    grad_norm_history: list[float] = []
    model_mse_history: list[float] = []
    vmin_history: list[float] = []
    vmax_history: list[float] = []
    progress_csv_path = output_dir / "progress.csv"
    status_path = output_dir / "status.json"
    progress_interval = max(1, int(args.progress_interval))

    completed = 0
    diverged = False
    stop_reason = "completed"
    start_time = time.perf_counter()
    with progress_csv_path.open("w", newline="") as progress_f:
        progress_writer = csv.writer(progress_f)
        progress_writer.writerow(["iteration", "objective", "grad_norm", "model_mse", "vp_min", "vp_max", "gamma1", "elapsed_seconds"])
        progress_f.flush()

        for iteration in range(1, iterations + 1):
            objective, gradient = objective_and_gradient(vp, dx, dz, observed, config["modeling"])
            grad_norm = float(np.linalg.norm(gradient.ravel()))
            mse = float(np.mean((vp.astype(np.float64) - vp_true.astype(np.float64)) ** 2))
            vmin = float(np.min(vp))
            vmax = float(np.max(vp))
            elapsed_seconds = time.perf_counter() - start_time

            objective_history.append(float(objective))
            grad_norm_history.append(grad_norm)
            model_mse_history.append(mse)
            vmin_history.append(vmin)
            vmax_history.append(vmax)
            completed = iteration

            finite = np.isfinite(objective) and np.isfinite(grad_norm) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(vp))
            line = (
                f"iter={iteration:03d}, objective={objective:.6e}, grad_norm={grad_norm:.6e}, "
                f"mse={mse:.6e}, vp_min={vmin:.2f}, vp_max={vmax:.2f}, gamma1={gamma1:.3e}, elapsed={elapsed_seconds:.1f}s"
            )
            print(line, flush=True)

            if iteration % progress_interval == 0 or iteration == 1:
                progress_writer.writerow([iteration, objective, grad_norm, mse, vmin, vmax, gamma1, elapsed_seconds])
                progress_f.flush()
                status = {
                    "state": "running",
                    "iteration": iteration,
                    "iterations_requested": iterations,
                    "objective": float(objective),
                    "grad_norm": grad_norm,
                    "model_mse": mse,
                    "vp_min": vmin,
                    "vp_max": vmax,
                    "gamma1": gamma1,
                    "elapsed_seconds": elapsed_seconds,
                    "diverged": False,
                    "stop_reason": None,
                }
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

            if not finite:
                diverged = True
                stop_reason = "non-finite objective, gradient, or velocity"
                break
            if iteration > 1 and objective_history[-1] > 1.0e6 * max(objective_history[0], 1.0):
                diverged = True
                stop_reason = "objective exceeded 1e6 times the initial objective"
                break

            vp = (vp - gamma1 * gradient).astype(np.float32)
            if not np.all(np.isfinite(vp)):
                diverged = True
                stop_reason = "update produced non-finite velocity"
                break

    objective_arr = np.asarray(objective_history, dtype=np.float64)
    grad_norm_arr = np.asarray(grad_norm_history, dtype=np.float64)
    mse_arr = np.asarray(model_mse_history, dtype=np.float64)
    vmin_arr = np.asarray(vmin_history, dtype=np.float64)
    vmax_arr = np.asarray(vmax_history, dtype=np.float64)

    meta = read_npz_meta(model_data["meta"])
    elapsed_seconds = time.perf_counter() - start_time
    meta.update(
        {
            "algorithm": "gradient",
            "tv_constraint": False,
            "box_constraint": False,
            "iterations_requested": iterations,
            "iterations_completed": completed,
            "gamma1": gamma1,
            "diverged": diverged,
            "stop_reason": stop_reason,
            "parameterization": "velocity",
            "elapsed_seconds": elapsed_seconds,
        }
    )
    np.savez(
        output_dir / "gradient_descent_result.npz",
        vp_final=vp,
        vp_initial=model_data["vp0"],
        vp_true=vp_true,
        objective_history=objective_arr,
        grad_norm_history=grad_norm_arr,
        model_mse_history=mse_arr,
        vmin_history=vmin_arr,
        vmax_history=vmax_arr,
        dx=dx,
        dz=dz,
        meta=npz_meta(meta),
    )

    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective", "grad_norm", "model_mse", "vp_min", "vp_max"])
        for i, row in enumerate(zip(objective_arr, grad_norm_arr, mse_arr, vmin_arr, vmax_arr), start=1):
            writer.writerow([i, *row])

    final_status = {
        "state": "finished",
        "iteration": completed,
        "iterations_requested": iterations,
        "objective": float(objective_arr[-1]) if objective_arr.size else None,
        "grad_norm": float(grad_norm_arr[-1]) if grad_norm_arr.size else None,
        "model_mse": float(mse_arr[-1]) if mse_arr.size else None,
        "vp_min": float(vmin_arr[-1]) if vmin_arr.size else None,
        "vp_max": float(vmax_arr[-1]) if vmax_arr.size else None,
        "gamma1": gamma1,
        "elapsed_seconds": elapsed_seconds,
        "diverged": diverged,
        "stop_reason": stop_reason,
    }
    status_path.write_text(json.dumps(final_status, indent=2) + "\n", encoding="utf-8")

    save_velocity_plot(vp, output_dir / "vp_final.png", "BP2004 gradient descent final velocity", dx, dz)
    save_velocity_plot(vp - vp_true, output_dir / "vp_final_minus_true.png", "BP2004 final - true velocity", dx, dz, cmap="seismic", label="Velocity difference [m/s]")
    plot_history(objective_arr, output_dir / "objective_history.png", "Objective", "BP2004 objective history")
    plot_history(grad_norm_arr, output_dir / "grad_norm_history.png", "Gradient L2 norm", "BP2004 gradient norm history")
    plot_history(mse_arr, output_dir / "model_mse_history.png", "MSE [(m/s)^2]", "BP2004 model MSE history")

    print(f"Gradient descent finished: completed={completed}, diverged={diverged}, stop_reason={stop_reason}", flush=True)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
