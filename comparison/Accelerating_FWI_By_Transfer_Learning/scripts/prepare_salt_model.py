from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT.parents[1]
    / "datasets"
    / "salt_and_overthrust_models"
    / "3-D_Salt_Model"
    / "VEL_GRIDS"
    / "Saltf@@"
)


def resize_bilinear(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    src_h, src_w = image.shape
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


def box_smooth(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32)
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (total / float(size * size)).astype(np.float32)


def load_seg_eage_salt(path: Path) -> np.ndarray:
    nz, ny, nx = 210, 676, 676
    vel = np.fromfile(path, dtype=np.dtype("float32").newbyteorder(">"))
    vel = vel.reshape(nx, ny, nz, order="F")
    vel = np.flip(vel, 2)
    return np.transpose(vel, (2, 1, 0)).astype(np.float32)


def prepare(args: argparse.Namespace) -> Path:
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"Salt model not found: {source}\n"
            "Run the TV-constraint project salt download task first, or pass --source."
        )

    raw = load_seg_eage_salt(source)
    # Match the TV-constraint project convention: use a y-slice and crop/resize it.
    slice_2d = raw.transpose((1, 0, 2))[args.slice_index].astype(np.float32) / 1000.0
    # Public velocity arrays are saved as (y, x), matching image/seismic convention:
    # rows are vertical/depth samples and columns are horizontal samples.
    velocity = resize_bilinear(slice_2d, (args.ny, args.nx))
    initial_velocity = resize_bilinear(box_smooth(slice_2d, args.smooth_radius), (args.ny, args.nx))

    # The Zenodo solver internally uses arrays indexed as (x, y). Keep gamma in
    # that internal orientation, but leave velocity outputs in (y, x).
    vmin = float(np.min(velocity))
    gamma = np.clip(vmin / velocity.T, args.gamma_floor, 1.0).astype(np.float32)
    initial_gamma = np.clip(vmin / initial_velocity.T, args.gamma_floor, 1.0).astype(np.float32)

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        velocity=velocity,
        initial_velocity=initial_velocity,
        gamma=gamma,
        initial_gamma=initial_gamma,
        vmin_km_s=vmin,
        vmax_km_s=float(np.max(velocity)),
        source=str(source),
        slice_index=args.slice_index,
        velocity_axes="yx",
        gamma_axes="xy",
    )
    print(f"saved {out}")
    print(f"velocity shape (y, x)={velocity.shape} min={velocity.min():.4f} max={velocity.max():.4f} km/s")
    print(f"gamma shape (x, y)={gamma.shape} min={gamma.min():.4f} max={gamma.max():.4f}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default="data/processed/salt_model.npz")
    parser.add_argument("--slice-index", type=int, default=300)
    parser.add_argument("--nx", type=int, default=127)
    parser.add_argument("--ny", type=int, default=63)
    parser.add_argument("--smooth-radius", type=int, default=12)
    parser.add_argument("--gamma-floor", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
