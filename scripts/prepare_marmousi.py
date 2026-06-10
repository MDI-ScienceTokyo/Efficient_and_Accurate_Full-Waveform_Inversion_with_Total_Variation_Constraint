from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy
from scipy.ndimage import gaussian_filter, zoom


MARM_URL = "https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz"

RAW_DIR = Path("datasets/marmousi/raw")
PROCESSED_DIR = Path("datasets/marmousi/processed")
TARBALL = RAW_DIR / "elastic-marmousi-model.tar.gz"
EXTRACTED_DIR = RAW_DIR / "elastic-marmousi-model"

# Use the full Marmousi extent and downsample while preserving aspect ratio.
TARGET_NZ = 101
ORIGINAL_DZ_M = 1.25
ORIGINAL_DX_M = 1.25

# Initial model smoothing in downsampled pixel units.
INIT_SMOOTH_SIGMA = 8.0


def download_if_needed() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARBALL.exists():
        print(f"[skip] tarball already exists: {TARBALL}")
        return

    print(f"[download] {MARM_URL}")
    urllib.request.urlretrieve(MARM_URL, TARBALL)
    print(f"[done] saved to {TARBALL}")


def extract_if_needed() -> None:
    if EXTRACTED_DIR.exists():
        print(f"[skip] already extracted: {EXTRACTED_DIR}")
        return

    print(f"[extract] {TARBALL}")
    with tarfile.open(TARBALL, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    print(f"[done] extracted to {RAW_DIR}")


def find_vp_file() -> Path:
    candidates: list[Path] = []
    for pattern in ("*P-WAVE*", "*P_wave*", "*p-wave*", "*VP*", "*vp*", "*Vp*"):
        candidates.extend(EXTRACTED_DIR.rglob(pattern))

    candidates = sorted({p for p in candidates if p.is_file()})

    print("[info] VP candidates:")
    for p in candidates:
        print(f"  - {p}")

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        parent = str(path.parent).lower()
        rank = 100
        if "model" in parent:
            rank -= 20
        if "p-wave" in name or "p_wave" in name:
            rank -= 20
        if "velocity" in name:
            rank -= 10
        if name.endswith((".segy", ".sgy", ".su", ".segy.tar.gz", ".sgy.tar.gz", ".su.tar.gz")):
            rank -= 5
        return rank, str(path)

    segy_candidates = [
        p
        for p in candidates
        if p.suffix.lower() in {".segy", ".sgy", ".su", ""}
        or p.name.lower().endswith((".segy.tar.gz", ".sgy.tar.gz", ".su.tar.gz"))
    ]
    if segy_candidates:
        return sorted(segy_candidates, key=score)[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No VP file found under {EXTRACTED_DIR}. Please inspect the extracted files manually.")


def extract_nested_segy_if_needed(path: Path) -> Path:
    if not path.name.lower().endswith((".segy.tar.gz", ".sgy.tar.gz", ".su.tar.gz")):
        return path

    output_dir = path.with_suffix("").with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in output_dir.iterdir() if p.is_file()]
    if existing:
        print(f"[skip] nested SEG-Y already extracted: {output_dir}")
    else:
        print(f"[extract] nested SEG-Y archive: {path}")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(output_dir)

    segy_files = sorted(
        p
        for p in output_dir.rglob("*")
        if p.is_file() and (p.suffix.lower() in {".segy", ".sgy", ".su"} or p.name.lower() in {"vp", "vp.segy"})
    )
    if not segy_files:
        raise FileNotFoundError(f"No SEG-Y file found after extracting {path} to {output_dir}")
    return segy_files[0]


def read_segy_as_2d_array(path: Path) -> np.ndarray:
    path = extract_nested_segy_if_needed(path)
    print(f"[read] {path}")
    segy = _read_segy(str(path))

    raw_vp = np.array([tr.data for tr in segy.traces], dtype=np.float32)
    print(f"[info] raw VP trace-major shape={raw_vp.shape}")

    # Interpret each trace as a vertical column: output shape is (nz, nx).
    vp = raw_vp.T
    print("[info] interpreted orientation: axis 0 = depth, axis 1 = horizontal")

    # Convert m/s to km/s if needed.
    if np.nanmax(vp) > 20:
        vp = vp / 1000.0

    median_value = float(np.nanmedian(vp))
    vp = np.nan_to_num(vp, nan=median_value, posinf=median_value, neginf=median_value).astype(np.float32)

    print(f"[info] original oriented shape={vp.shape}, min={vp.min():.4f}, max={vp.max():.4f}")
    return vp


def resize_full_model_preserving_aspect(vp: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    original_nz, original_nx = vp.shape
    target_nx = int(round(original_nx / original_nz * TARGET_NZ))

    scale_z = TARGET_NZ / original_nz
    scale_x = target_nx / original_nx

    vp_resized = zoom(vp, (scale_z, scale_x), order=1).astype(np.float32)
    target_nz, target_nx = vp_resized.shape
    effective_dz_m = ORIGINAL_DZ_M * original_nz / target_nz
    effective_dx_m = ORIGINAL_DX_M * original_nx / target_nx
    metadata = {
        "original_nz": original_nz,
        "original_nx": original_nx,
        "target_nz": target_nz,
        "target_nx": target_nx,
        "original_dz_m": ORIGINAL_DZ_M,
        "original_dx_m": ORIGINAL_DX_M,
        "effective_dz_m": effective_dz_m,
        "effective_dx_m": effective_dx_m,
        "velocity_unit": "km/s",
        "distance_unit": "m",
        "time_unit": "ms",
        "orientation": "axis0_depth_axis1_horizontal",
    }

    print(f"[info] target shape={vp_resized.shape}, min={vp_resized.min():.4f}, max={vp_resized.max():.4f}")
    print("[info] Marmousi metadata:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")
    return vp_resized, metadata


def save_preview(arr: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(12, 3))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="Velocity [km/s]")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Depth")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_outputs(vp_true: np.ndarray, metadata: dict[str, object]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    vp_init = gaussian_filter(vp_true, sigma=INIT_SMOOTH_SIGMA).astype(np.float32)

    nz, nx = vp_true.shape
    suffix = f"{nz}x{nx}"

    true_npy = PROCESSED_DIR / f"marmousi_vp_true_full_{suffix}.npy"
    init_npy = PROCESSED_DIR / f"marmousi_vp_init_full_{suffix}.npy"
    metadata_npz = PROCESSED_DIR / f"marmousi_metadata_full_{suffix}.npz"
    true_png = PROCESSED_DIR / f"marmousi_vp_true_full_{suffix}.png"
    init_png = PROCESSED_DIR / f"marmousi_vp_init_full_{suffix}.png"

    np.save(true_npy, vp_true)
    np.save(init_npy, vp_init)
    np.savez(metadata_npz, **metadata)

    save_preview(vp_true, true_png, f"Marmousi VP true full extent, {suffix}")
    save_preview(vp_init, init_png, f"Marmousi VP initial full extent, {suffix}")

    print("[saved]")
    print(f"  - {true_npy}")
    print(f"  - {init_npy}")
    print(f"  - {metadata_npz}")
    print(f"  - {true_png}")
    print(f"  - {init_png}")


def main() -> None:
    download_if_needed()
    extract_if_needed()
    vp_file = find_vp_file()
    vp = read_segy_as_2d_array(vp_file)
    vp_true, metadata = resize_full_model_preserving_aspect(vp)
    save_outputs(vp_true, metadata)


if __name__ == "__main__":
    main()
