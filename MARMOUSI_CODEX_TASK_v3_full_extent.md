# Codex task: Add a Marmousi experiment without cropping the model

This document is intended to be passed to Codex inside VS Code.

Important policy:

- Do **not crop** the Marmousi model by default.
- Use the full lateral and depth extent of the downloaded Marmousi P-wave velocity model.
- Downsample only if needed for computational feasibility.
- Keep the Marmousi aspect ratio as much as possible.
- The existing Salt/Overthrust experiment must remain unchanged.

Repository:

```text
https://github.com/MDI-ScienceTokyo/Efficient_and_Accurate_Full-Waveform_Inversion_with_Total_Variation_Constraint
```

Current status:

- The original proposed method script already runs successfully.
- The environment is WSL2 + VS Code.
- The repository uses Poetry.
- We now want to add a Marmousi-specific experiment.

---

## 1. Goal

Add a new Marmousi experiment to the repository.

Expected new files:

```text
scripts/prepare_marmousi.py
src/box-TV-constrained-FWI-marmousi.py
```

Expected directories:

```text
datasets/marmousi/raw/
datasets/marmousi/processed/
results/marmousi/
```

The Marmousi experiment should:

1. Download the Marmousi model.
2. Extract the P-wave velocity model.
3. Use the full model extent by default.
4. Downsample the full model to a computationally feasible grid while preserving aspect ratio.
5. Generate a smoothed initial model.
6. Run the proposed TV-constrained FWI method on the Marmousi model.
7. Save results separately from Salt/Overthrust.

---

## 2. Marmousi data source

Use the AGL Elastic Marmousi model:

```text
https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz
```

Download location:

```text
datasets/marmousi/raw/elastic-marmousi-model.tar.gz
```

Extraction location:

```text
datasets/marmousi/raw/elastic-marmousi-model/
```

We only need the P-wave velocity model, usually a SEG-Y file containing `VP` or `vp` in the filename.

---

## 3. Grid-size policy: no cropping by default

Do not crop the Marmousi model unless explicitly requested later.

Instead:

1. Read the full VP model.
2. Preserve the full model extent.
3. Downsample to a feasible grid size.

Use a target vertical size and automatically determine the horizontal size from the original aspect ratio.

Recommended first setting:

```python
TARGET_NZ = 101
```

Then:

```python
TARGET_NX = round(original_nx / original_nz * TARGET_NZ)
```

This preserves the original aspect ratio.

If this becomes too wide/heavy, use:

```python
TARGET_NZ = 81
```

If computation is affordable, try:

```python
TARGET_NZ = 151
```

Do not hard-code `50 x 100`.

The preprocessing script should print:

```text
original shape
target shape
velocity min/max
```

so that we can verify the scale.

---

## 4. Add dependencies if needed

Check `pyproject.toml`. If the following packages are missing, add them:

```bash
poetry add obspy scipy matplotlib
```

`obspy` reads SEG-Y.  
`scipy` handles resizing and Gaussian smoothing.  
`matplotlib` saves preview images.

---

## 5. Implement `scripts/prepare_marmousi.py`

Create this script.

Expected command:

```bash
poetry run python scripts/prepare_marmousi.py
```

The script should:

1. Create necessary directories.
2. Download the Marmousi tarball if it does not already exist.
3. Extract it if it is not already extracted.
4. Search recursively for a VP SEG-Y file.
5. Read VP into a 2D NumPy array.
6. Convert from m/s to km/s if needed.
7. Use the full model without cropping.
8. Downsample while preserving aspect ratio.
9. Generate an initial model by Gaussian smoothing.
10. Save `.npy` arrays and preview `.png` figures.

Suggested implementation:

```python
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy
from scipy.ndimage import gaussian_filter, zoom


MARM_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/"
    "elastic-marmousi/elastic-marmousi-model.tar.gz"
)

RAW_DIR = Path("datasets/marmousi/raw")
PROCESSED_DIR = Path("datasets/marmousi/processed")
TARBALL = RAW_DIR / "elastic-marmousi-model.tar.gz"
EXTRACTED_DIR = RAW_DIR / "elastic-marmousi-model"

# Use full Marmousi extent and downsample while preserving aspect ratio.
TARGET_NZ = 101

# Initial model smoothing. This may need adjustment after visual inspection.
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
    candidates = []
    for pattern in ("*VP*", "*vp*", "*Vp*"):
        candidates.extend(EXTRACTED_DIR.rglob(pattern))

    candidates = [p for p in candidates if p.is_file()]

    print("[info] VP candidates:")
    for p in candidates:
        print(f"  - {p}")

    segy_candidates = [
        p for p in candidates
        if p.suffix.lower() in {".segy", ".sgy", ".su", ""}
    ]

    if segy_candidates:
        return segy_candidates[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No VP file found under {EXTRACTED_DIR}. "
        "Please inspect the extracted files manually."
    )


def read_segy_as_2d_array(path: Path) -> np.ndarray:
    print(f"[read] {path}")
    segy = _read_segy(str(path))

    # Interpret each trace as a vertical column.
    vp = np.array([tr.data for tr in segy.traces], dtype=np.float32).T

    # Convert m/s to km/s if needed.
    if np.nanmax(vp) > 20:
        vp = vp / 1000.0

    median_value = float(np.nanmedian(vp))
    vp = np.nan_to_num(vp, nan=median_value, posinf=median_value, neginf=median_value)

    print(f"[info] original shape={vp.shape}, min={vp.min():.4f}, max={vp.max():.4f}")
    return vp


def resize_full_model_preserving_aspect(vp: np.ndarray) -> np.ndarray:
    original_nz, original_nx = vp.shape
    target_nx = int(round(original_nx / original_nz * TARGET_NZ))

    scale_z = TARGET_NZ / original_nz
    scale_x = target_nx / original_nx

    vp_resized = zoom(vp, (scale_z, scale_x), order=1).astype(np.float32)

    print(
        f"[info] target shape={vp_resized.shape}, "
        f"min={vp_resized.min():.4f}, max={vp_resized.max():.4f}"
    )

    return vp_resized


def save_preview(arr: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(12, 3))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="Velocity [km/s]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_outputs(vp_true: np.ndarray) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Smooth in pixel units after downsampling.
    vp_init = gaussian_filter(vp_true, sigma=INIT_SMOOTH_SIGMA).astype(np.float32)

    nz, nx = vp_true.shape
    suffix = f"{nz}x{nx}"

    true_npy = PROCESSED_DIR / f"marmousi_vp_true_full_{suffix}.npy"
    init_npy = PROCESSED_DIR / f"marmousi_vp_init_full_{suffix}.npy"
    true_png = PROCESSED_DIR / f"marmousi_vp_true_full_{suffix}.png"
    init_png = PROCESSED_DIR / f"marmousi_vp_init_full_{suffix}.png"

    np.save(true_npy, vp_true)
    np.save(init_npy, vp_init)

    save_preview(vp_true, true_png, f"Marmousi VP true full extent, {suffix}")
    save_preview(vp_init, init_png, f"Marmousi VP initial full extent, {suffix}")

    print("[saved]")
    print(f"  - {true_npy}")
    print(f"  - {init_npy}")
    print(f"  - {true_png}")
    print(f"  - {init_png}")


def main() -> None:
    download_if_needed()
    extract_if_needed()
    vp_file = find_vp_file()
    vp = read_segy_as_2d_array(vp_file)
    vp_true = resize_full_model_preserving_aspect(vp)
    save_outputs(vp_true)


if __name__ == "__main__":
    main()
```

---

## 6. Verify preprocessing

Run:

```bash
poetry run python scripts/prepare_marmousi.py
```

Then:

```bash
ls -lh datasets/marmousi/processed/
```

Expected file names will contain the actual aspect-ratio-preserved shape, for example:

```text
marmousi_vp_true_full_101xXXX.npy
marmousi_vp_init_full_101xXXX.npy
marmousi_vp_true_full_101xXXX.png
marmousi_vp_init_full_101xXXX.png
```

where `XXX` depends on the original model aspect ratio.

Please inspect the PNGs. The true image should preserve the full wide Marmousi structure. If it looks upside down or transposed, fix the SEG-Y orientation.

---

## 7. Add Marmousi FWI script

Copy the original script:

```bash
cp src/box-TV-constrained-FWI.py src/box-TV-constrained-FWI-marmousi.py
```

Modify the copied script to load the processed Marmousi `.npy` files.

Because the exact horizontal size is automatically determined, do not hard-code the file name if possible. Instead, find the generated files:

```python
from pathlib import Path
import numpy as np

processed_dir = Path("datasets/marmousi/processed")
true_candidates = sorted(processed_dir.glob("marmousi_vp_true_full_*.npy"))
init_candidates = sorted(processed_dir.glob("marmousi_vp_init_full_*.npy"))

if not true_candidates or not init_candidates:
    raise FileNotFoundError(
        "Marmousi processed files were not found. "
        "Run: poetry run python scripts/prepare_marmousi.py"
    )

m_true = np.load(true_candidates[-1])
m_init = np.load(init_candidates[-1])

print("Marmousi true:", m_true.shape, m_true.min(), m_true.max())
print("Marmousi init:", m_init.shape, m_init.min(), m_init.max())
```

Use the actual variable names in the original script. For example, if the original script uses `vp`, `m0`, `ground_truth`, or `initial_model`, adapt accordingly.

Do not overwrite the original Salt/Overthrust code.

---

## 8. Update geometry for the full-extent Marmousi model

Because the Marmousi grid will be wider than the original `50 x 100` experiment, check all geometry settings in the copied script.

Search:

```bash
grep -n "shape\|spacing\|origin\|nbl\|source\|receiver\|rec\|src\|nt\|tn\|dt" src/box-TV-constrained-FWI-marmousi.py
```

Required changes:

- The model/grid shape should be derived from `m_true.shape`.
- Source locations should be distributed along the surface.
- Receiver locations should be distributed along the surface.
- Avoid hard-coded `50`, `100`, or old source/receiver positions.
- Keep the physical spacing reasonable.
- Use fewer sources and iterations for the first smoke test.

Recommended smoke-test acquisition:

```text
number of sources: 3 to 5
number of receivers: 101 or fewer
number of iterations: 10 to 50
num_parallels: 1
alpha values: one value only
```

After confirming that it runs, increase these settings.

---

## 9. Bounds and velocity scaling

After loading Marmousi, print:

```python
print("Marmousi true:", m_true.shape, m_true.min(), m_true.max())
print("Marmousi init:", m_init.shape, m_init.min(), m_init.max())
```

If velocities are around `1500-5000`, they are still in m/s and must be divided by `1000`.

Use bounds based on the model:

```python
lower_bound = float(np.floor(m_true.min() * 10) / 10)
upper_bound = float(np.ceil(m_true.max() * 10) / 10)
```

If the original code requires fixed bounds, use something like:

```python
lower_bound = 1.0
upper_bound = 5.0
```

or values consistent with `m_true.min()` and `m_true.max()`.

---

## 10. TV radius alpha for Marmousi

Do not assume the Salt/Overthrust alpha range is appropriate for the full-extent Marmousi model.

For the first run, compute the TV value of the Marmousi ground truth and choose alpha values around that scale.

If no helper exists, add this simple function for choosing alpha only:

```python
def isotropic_tv_2d(x: np.ndarray) -> float:
    dz = np.diff(x, axis=0, append=x[-1:, :])
    dx = np.diff(x, axis=1, append=x[:, -1:])
    return float(np.sum(np.sqrt(dx**2 + dz**2)))
```

Then use:

```python
alpha_gt = isotropic_tv_2d(m_true)
alpha_values = [
    0.6 * alpha_gt,
    0.8 * alpha_gt,
    1.0 * alpha_gt,
    1.2 * alpha_gt,
]
```

For the smoke test, use only one alpha, for example:

```python
alpha_values = [0.8 * alpha_gt]
```

The actual optimization should continue using the repository's existing TV/projection implementation.

---

## 11. Run the Marmousi experiment

After implementing the Marmousi script:

```bash
poetry run python src/box-TV-constrained-FWI-marmousi.py
```

If computation is too heavy, reduce:

- `TARGET_NZ` in `scripts/prepare_marmousi.py`
- number of sources
- number of receivers
- number of iterations
- number of alpha values
- `num_parallels`

---

## 12. Output policy

Save all Marmousi outputs under:

```text
results/marmousi/
```

or another Marmousi-specific directory.

The output should include:

- true velocity model preview
- initial velocity model preview
- reconstructed velocity models
- RMSE / SSIM curves if the original script computes them
- parameter settings used for the run

Do not mix Marmousi outputs with Salt/Overthrust outputs.

---

## 13. Acceptance criteria

Implementation is acceptable when:

1. Original Salt/Overthrust experiment still runs.
2. Marmousi preprocessing runs by:

```bash
poetry run python scripts/prepare_marmousi.py
```

3. Processed Marmousi files are created using the full model extent.
4. The processed shape is not hard-coded to `50 x 100`.
5. Marmousi preview PNGs look like a wide full Marmousi model.
6. Marmousi FWI runs by:

```bash
poetry run python src/box-TV-constrained-FWI-marmousi.py
```

7. Results are saved separately under `results/marmousi/`.
8. No hard-coded Salt/Overthrust geometry remains in the Marmousi script.

---

## 14. Suggested paper wording

Use wording like:

```text
We additionally evaluated the proposed method on the Marmousi P-wave velocity model. 
For computational feasibility, the full model extent was downsampled while preserving its aspect ratio, and the initial model was generated by Gaussian smoothing.
```

Do not claim that the full-resolution Marmousi model or original Marmousi acquisition geometry was used unless that is implemented exactly.

---

## 15. Suggested commit message

```text
Add full-extent Marmousi TV-constrained FWI experiment
```
