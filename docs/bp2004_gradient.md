# BP2004 FWI Workflow

This workflow downloads the SEG/BP 2004 exact velocity model, crops it to a smaller 2D model, creates a smoothed initial model, and runs BP2004 FWI with the repository's proposed box-TV code path.

Install the project dependencies first. SEG-Y reading uses `segyio` when available and falls back to `obspy`.

```bash
poetry install
poetry run pip install pyyaml segyio
```

Prepare the BP2004 model:

```bash
python scripts/download_bp2004.py --decompress
python scripts/preprocess_bp2004.py --config configs/bp2004_gradient.yaml
python scripts/make_bp2004_initial_model.py --config configs/bp2004_gradient.yaml
```

Expected prepared data:

```text
data/processed/bp2004/bp2004_exact_crop.npz
data/processed/bp2004/bp2004_initial_crop.npz
```

Run gradient descent without TV by setting `--alphas 0`. The BP2004 driver still applies the configured velocity box bounds, matching the existing box-constrained update path.

```bash
MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 0 \
  --noise-sigmas 0,0.01 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --result-root-path results/bp2004
```

Run the proposed TV + box constrained version by using a positive `--alphas` value:

```bash
MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 500 \
  --noise-sigmas 0,0.01 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --gamma2 100 \
  --result-root-path results/bp2004
```

The saved velocity arrays in `data/processed/bp2004` are in m/s. The BP2004 FWI driver converts them to km/s internally, matching the existing FWI code in this repository.
