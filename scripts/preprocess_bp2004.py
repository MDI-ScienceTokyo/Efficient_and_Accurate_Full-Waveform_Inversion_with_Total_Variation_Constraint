from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bp2004_utils import ensure_dirs, load_config, npz_meta, save_velocity_plot


def read_bp2004_velocity(path: Path, nz: int, nx: int) -> np.ndarray:
    print(f"[read] {path}")
    try:
        import segyio

        with segyio.open(str(path), ignore_geometry=True) as segy:
            traces = np.asarray([trace for trace in segy.trace], dtype=np.float32)
    except ImportError:
        try:
            from obspy.io.segy.segy import _read_segy
        except ImportError as exc:
            raise ImportError("Install segyio (`pip install segyio`) or obspy to read BP2004 SEG-Y files.") from exc
        print("[warn] segyio is not installed; falling back to obspy.")
        segy = _read_segy(str(path))
        traces = np.asarray([trace.data for trace in segy.traces], dtype=np.float32)

    print(f"[info] raw trace-major shape={traces.shape}")
    if traces.shape == (nx, nz):
        vp = traces.T
    elif traces.shape == (nz, nx):
        vp = traces
    elif traces.size == nz * nx:
        vp = traces.reshape(nx, nz).T
    else:
        raise ValueError(f"Cannot infer BP2004 shape from {traces.shape}; expected {nz} x {nx}.")

    vp = np.nan_to_num(vp, nan=float(np.nanmedian(vp))).astype(np.float32)
    if np.nanmax(vp) < 20.0:
        print("[info] values look like km/s; converting to m/s")
        vp *= 1000.0
    return vp


def crop_and_resample(vp: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp = config["bp2004"]
    crop = config["crop"]
    resample = config["resample"]
    dz_full = float(bp["dz_full_m"])
    dx_full = float(bp["dx_full_m"])
    x_full = np.arange(vp.shape[1], dtype=np.float32) * dx_full
    z_full = np.arange(vp.shape[0], dtype=np.float32) * dz_full

    x_min, x_max = float(crop["x_min_m"]), float(crop["x_max_m"])
    z_min, z_max = float(crop["z_min_m"]), float(crop["z_max_m"])
    x_mask = (x_full >= x_min) & (x_full <= x_max)
    z_mask = (z_full >= z_min) & (z_full <= z_max)
    cropped = vp[np.ix_(z_mask, x_mask)]
    x_crop = x_full[x_mask]
    z_crop = z_full[z_mask]

    dx = float(resample["dx_m"])
    dz = float(resample["dz_m"])
    x = np.arange(x_min, x_max + 0.5 * dx, dx, dtype=np.float32)
    z = np.arange(z_min, z_max + 0.5 * dz, dz, dtype=np.float32)
    interp = RegularGridInterpolator((z_crop, x_crop), cropped, bounds_error=False, fill_value=None)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    processed = interp(np.column_stack([zz.ravel(), xx.ravel()])).reshape(z.size, x.size).astype(np.float32)
    return processed, x, z


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the BP2004 exact velocity model for a small gradient test.")
    parser.add_argument("--config", default="configs/bp2004_gradient.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir, processed_dir, output_dir = ensure_dirs(config)
    input_path = Path(args.input) if args.input else raw_dir / "vel_z6.25m_x12.5m_exact.segy"
    output_path = Path(args.output) if args.output else processed_dir / "bp2004_exact_crop.npz"

    bp = config["bp2004"]
    vp_full = read_bp2004_velocity(input_path, int(bp["nz_full"]), int(bp["nx_full"]))
    print("Loaded BP2004 velocity model")
    print(f"Original shape: {vp_full.shape[0]} x {vp_full.shape[1]}")
    print(f"Original spacing: dz={bp['dz_full_m']} m, dx={bp['dx_full_m']} m")
    print(f"Original velocity range: min={float(np.min(vp_full)):.2f} m/s, max={float(np.max(vp_full)):.2f} m/s")

    vp, x, z = crop_and_resample(vp_full, config)
    dx = float(config["resample"]["dx_m"])
    dz = float(config["resample"]["dz_m"])
    crop = config["crop"]
    print(f"Cropped physical range: x=[{crop['x_min_m']}, {crop['x_max_m']}] m, z=[{crop['z_min_m']}, {crop['z_max_m']}] m")
    print(f"Processed shape: {vp.shape}, spacing: dz={dz} m, dx={dx} m")
    print(f"Processed velocity range: min={float(np.min(vp)):.2f} m/s, max={float(np.max(vp)):.2f} m/s")

    meta = {
        "source": "SEG/BP 2004 velocity estimation benchmark exact velocity model",
        "original_shape": tuple(vp_full.shape),
        "original_spacing_m": (float(bp["dz_full_m"]), float(bp["dx_full_m"])),
        "velocity_unit": "m/s",
        "orientation": "vp[z, x]",
        "crop": dict(crop),
        "resample": dict(config["resample"]),
    }
    np.savez(output_path, vp=vp, dx=dx, dz=dz, x0=float(x[0]), z0=float(z[0]), x=x, z=z, meta=npz_meta(meta))
    save_velocity_plot(vp, output_dir / "vp_true.png", "BP2004 true velocity", dx, dz)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
