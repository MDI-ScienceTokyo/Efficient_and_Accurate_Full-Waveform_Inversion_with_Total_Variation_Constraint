from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bp2004_utils import ensure_dirs, load_config, save_shot_gather_plot, save_velocity_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BP2004 preprocessing, observed-data, and gradient outputs.")
    parser.add_argument("--config", default="configs/bp2004_gradient.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    _, processed_dir, output_dir = ensure_dirs(config)
    model_path = processed_dir / "bp2004_initial_crop.npz"
    observed_path = output_dir / "observed_data.npz"
    gradient_path = output_dir / "gradient_result.npz"

    model = np.load(model_path, allow_pickle=True)
    dx = float(model["dx"])
    dz = float(model["dz"])
    vp_true = model["vp_true"]
    vp0 = model["vp0"]

    save_velocity_plot(vp_true, output_dir / "vp_true.png", "BP2004 true velocity", dx, dz)
    save_velocity_plot(vp0, output_dir / "vp_initial.png", "BP2004 initial velocity", dx, dz)
    save_velocity_plot(vp_true - vp0, output_dir / "vp_difference.png", "BP2004 true - initial velocity", dx, dz, cmap="seismic", label="Velocity difference [m/s]")

    if gradient_path.exists():
        gradient = np.load(gradient_path, allow_pickle=True)["gradient"]
        save_velocity_plot(gradient, output_dir / "gradient.png", "BP2004 FWI gradient", dx, dz, cmap="seismic", label="Gradient")

    if observed_path.exists():
        observed = np.load(observed_path, allow_pickle=True)
        save_shot_gather_plot(observed["d_obs"][0], output_dir / "shot_gather_example.png", float(observed["dt"]))

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
