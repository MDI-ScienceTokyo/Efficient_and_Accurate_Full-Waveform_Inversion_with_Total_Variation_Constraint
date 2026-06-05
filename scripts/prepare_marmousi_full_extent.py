from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from prepare_marmousi import download_if_needed, extract_if_needed, find_vp_file, read_segy_as_2d_array


PROCESSED_DIR = Path("datasets/marmousi/processed")


def resample_to_spacing(vp: np.ndarray, raw_dz_m: float, raw_dx_m: float, target_dz_m: float, target_dx_m: float) -> np.ndarray:
    raw_nz, raw_nx = vp.shape
    raw_depth_m = (raw_nz - 1) * raw_dz_m
    raw_width_m = (raw_nx - 1) * raw_dx_m
    target_nz = int(round(raw_depth_m / target_dz_m)) + 1
    target_nx = int(round(raw_width_m / target_dx_m)) + 1
    scale_z = target_nz / raw_nz
    scale_x = target_nx / raw_nx
    return zoom(vp, (scale_z, scale_x), order=1).astype(np.float32)


def save_preview(arr: np.ndarray, path: Path, title: str, dx_m: float, dz_m: float) -> None:
    nz, nx = arr.shape
    extent = [0, (nx - 1) * dx_m / 1000.0, (nz - 1) * dz_m / 1000.0, 0]
    plt.figure(figsize=(12, 3.5))
    plt.imshow(arr, aspect="auto", extent=extent, cmap="coolwarm")
    plt.colorbar(label="Velocity [km/s]")
    plt.title(title)
    plt.xlabel("x [km]")
    plt.ylabel("depth [km]")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a full-physical-extent Marmousi VP model for FWI.")
    parser.add_argument("--target-dz-m", type=float, default=10.0)
    parser.add_argument("--target-dx-m", type=float, default=10.0)
    parser.add_argument("--raw-dz-m", type=float, default=1.25)
    parser.add_argument("--raw-dx-m", type=float, default=1.25)
    parser.add_argument(
        "--initial-smoothing-radius-m",
        type=float,
        default=280.0,
        help="Gaussian smoothing radius for the initial model, in meters. 280 m matches the previous 8 px smoothing on the 35 m downsampled model.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    download_if_needed()
    extract_if_needed()
    vp_file = find_vp_file()
    vp_raw = read_segy_as_2d_array(vp_file)
    vp_true = resample_to_spacing(vp_raw, args.raw_dz_m, args.raw_dx_m, args.target_dz_m, args.target_dx_m)

    sigma = (args.initial_smoothing_radius_m / args.target_dz_m, args.initial_smoothing_radius_m / args.target_dx_m)
    vp_init = gaussian_filter(vp_true, sigma=sigma).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"full_extent_{vp_true.shape[0]}x{vp_true.shape[1]}_{args.target_dz_m:g}m"
    output_path = args.output_dir / f"marmousi_full_extent_{suffix}.npz"
    true_png = args.output_dir / f"marmousi_full_extent_true_{vp_true.shape[0]}x{vp_true.shape[1]}.png"
    init_png = args.output_dir / f"marmousi_full_extent_init_{vp_true.shape[0]}x{vp_true.shape[1]}.png"

    raw_depth_m = (vp_raw.shape[0] - 1) * args.raw_dz_m
    raw_width_m = (vp_raw.shape[1] - 1) * args.raw_dx_m
    np.savez_compressed(
        output_path,
        vp_true=vp_true,
        vp0=vp_init,
        dz=np.float32(args.target_dz_m),
        dx=np.float32(args.target_dx_m),
        raw_shape=np.asarray(vp_raw.shape, dtype=np.int32),
        raw_spacing_m=np.asarray([args.raw_dz_m, args.raw_dx_m], dtype=np.float32),
        raw_extent_m=np.asarray([raw_depth_m, raw_width_m], dtype=np.float32),
        initial_smoothing_radius_m=np.float32(args.initial_smoothing_radius_m),
        source=str(vp_file),
        velocity_unit="km/s",
        orientation="vp[z, x]",
    )

    save_preview(vp_true, true_png, f"Marmousi VP true full extent, {vp_true.shape}", args.target_dx_m, args.target_dz_m)
    save_preview(vp_init, init_png, f"Marmousi VP initial full extent, {vp_true.shape}", args.target_dx_m, args.target_dz_m)

    print(f"[info] raw shape={vp_raw.shape}, raw extent={raw_depth_m:g} m x {raw_width_m:g} m")
    print(f"[info] target shape={vp_true.shape}, spacing={args.target_dz_m:g} m x {args.target_dx_m:g} m")
    print(f"[info] true velocity range={float(vp_true.min()):.4f} - {float(vp_true.max()):.4f} km/s")
    print(f"[info] initial velocity range={float(vp_init.min()):.4f} - {float(vp_init.max()):.4f} km/s")
    print("[saved]")
    print(f"  - {output_path}")
    print(f"  - {true_png}")
    print(f"  - {init_png}")


if __name__ == "__main__":
    main()
