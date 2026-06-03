from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

from bp2004_utils import ensure_dirs, load_config, npz_meta, read_npz_meta, save_velocity_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a smoothed BP2004 initial velocity model.")
    parser.add_argument("--config", default="configs/bp2004_gradient.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    _, processed_dir, output_dir = ensure_dirs(config)
    input_path = Path(args.input) if args.input else processed_dir / "bp2004_exact_crop.npz"
    output_path = Path(args.output) if args.output else processed_dir / "bp2004_initial_crop.npz"

    data = np.load(input_path, allow_pickle=True)
    vp = data["vp"].astype(np.float32)
    dx = float(data["dx"])
    dz = float(data["dz"])
    init_cfg = config["initial_model"]
    sigma = (float(init_cfg["smoothing_sigma_z_grid"]), float(init_cfg["smoothing_sigma_x_grid"]))
    vp0 = gaussian_filter(vp, sigma=sigma).astype(np.float32)
    vp0 = np.clip(vp0, float(init_cfg["min_velocity"]), float(init_cfg["max_velocity"])).astype(np.float32)

    meta = read_npz_meta(data["meta"])
    meta.update({"initial_model": dict(init_cfg), "parameterization": "velocity"})
    np.savez(output_path, vp0=vp0, vp_true=vp, dx=dx, dz=dz, x=data["x"], z=data["z"], meta=npz_meta(meta))

    print(f"Initial model smoothing: sigma_z={sigma[0]:g}, sigma_x={sigma[1]:g}")
    print(f"Initial velocity range: min={float(np.min(vp0)):.2f} m/s, max={float(np.max(vp0)):.2f} m/s")
    save_velocity_plot(vp, output_dir / "vp_true.png", "BP2004 true velocity", dx, dz)
    save_velocity_plot(vp0, output_dir / "vp_initial.png", "BP2004 initial velocity", dx, dz)
    save_velocity_plot(vp - vp0, output_dir / "vp_difference.png", "BP2004 true - initial velocity", dx, dz, cmap="seismic", label="Velocity difference [m/s]")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.hist((vp - vp0).ravel(), bins=100)
    plt.title("BP2004 true - initial velocity histogram")
    plt.xlabel("Velocity difference [m/s]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "vp_difference_histogram.png", dpi=180)
    plt.close()
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
