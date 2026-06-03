# BP2004 Gradient Smoke Test

This workflow downloads the SEG/BP 2004 exact velocity model, crops it to a smaller 2D model, creates a smoothed initial model, generates synthetic observed data, and computes one FWI objective/gradient at the initial model.

Install the project dependencies first. SEG-Y reading uses `segyio` when available and falls back to `obspy`.

```bash
poetry install
poetry run pip install pyyaml segyio
```

Run the workflow:

```bash
python scripts/download_bp2004.py --decompress
python scripts/preprocess_bp2004.py --config configs/bp2004_gradient.yaml
python scripts/make_bp2004_initial_model.py --config configs/bp2004_gradient.yaml
python scripts/run_bp2004_forward.py --config configs/bp2004_gradient.yaml
python scripts/run_bp2004_gradient.py --config configs/bp2004_gradient.yaml
python scripts/plot_bp2004_results.py --config configs/bp2004_gradient.yaml
```

To generate noisy synthetic observed data, pass `--noise-sigma` and `--seed`:

```bash
python scripts/run_bp2004_forward.py \
  --config configs/bp2004_gradient.yaml \
  --noise-sigma 0.01 \
  --seed 0 \
  --output outputs/bp2004_gradient/observed_data_noise_sigma0p01.npz
```

Expected outputs:

```text
data/processed/bp2004/bp2004_exact_crop.npz
data/processed/bp2004/bp2004_initial_crop.npz
outputs/bp2004_gradient/observed_data.npz
outputs/bp2004_gradient/gradient_result.npz
outputs/bp2004_gradient/vp_true.png
outputs/bp2004_gradient/vp_initial.png
outputs/bp2004_gradient/vp_difference.png
outputs/bp2004_gradient/gradient.png
outputs/bp2004_gradient/shot_gather_example.png
```

The saved velocity arrays are in m/s. Devito modeling converts them internally to km/s, matching the existing FWI code in this repository.
