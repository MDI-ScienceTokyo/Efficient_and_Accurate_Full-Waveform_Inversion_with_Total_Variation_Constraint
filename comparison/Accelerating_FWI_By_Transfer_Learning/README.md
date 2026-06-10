# Accelerating FWI By Transfer Learning: Salt FWI workspace

This folder contains the Zenodo code from <https://zenodo.org/records/13150916> and a small Salt-model adapter.

This workspace lives inside the comparison area of the parent project:

`/projects/Efficient_and_Accurate_Full-Waveform_Inversion_with_Total_Variation_Constraint/comparison/Accelerating_FWI_By_Transfer_Learning`

## Layout

- `zenodo_source/`: downloaded Zenodo scripts and pretrained model.
- `scripts/prepare_salt_model.py`: loads the SEG/EAGE Salt model used by the TV-constraint project and converts it to the Zenodo solver's `gamma` field.
- `scripts/run_salt_fwi.py`: generates observed data and runs conventional FWI for one or more noise levels.
- `data/`: generated datasets and preprocessed models. This directory is ignored by git.
- `results/`: generated observed data and FWI results. This directory is ignored by git.

## Environment

```bash
cd /projects/Efficient_and_Accurate_Full-Waveform_Inversion_with_Total_Variation_Constraint/comparison/Accelerating_FWI_By_Transfer_Learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs `numpy`, `pandas`, `matplotlib`, `tables`, `scikit-image`, and CUDA 12.4 compatible `torch==2.6.0+cu124`. If CUDA is unavailable, `scripts/run_salt_fwi.py --cpu` can be used for small smoke tests.

## Prepare Salt Data

```bash
python scripts/prepare_salt_model.py
```

Prepared velocity arrays are saved as `(y, x)` for plotting and evaluation. The internal `gamma` arrays are saved as `(x, y)` because the Zenodo finite-difference solver indexes its grid that way.

The default source is the already downloaded TV-constraint Salt file:

`../../datasets/salt_and_overthrust_models/3-D_Salt_Model/VEL_GRIDS/Saltf@@`

## Run FWI For noise sigma 0 and 1

```bash
python scripts/run_salt_fwi.py --noise-sigmas 0,1
```

By default, `--noise-mode velocity_relative` interprets `noise_sigma` in the Salt velocity scale `[1.5, 4.5]` km/s and maps it to waveform noise as `sigma / (4.5 - 1.5) * clean_waveform_std`. Use `--noise-mode absolute` only when you want to add the raw sigma value directly to the waveform samples.

To force CPU execution, add `--cpu`.

For a quick smoke test:

```bash
python scripts/prepare_salt_model.py --nx 31 --ny 15 --output data/processed/salt_smoke.npz
python scripts/run_salt_fwi.py --model data/processed/salt_smoke.npz --nx 31 --ny 15 --lx 310 --ly 150 --time-steps 30 --epochs 1 --sources 2 --source-spacing 6 --sensor-spacing 3 --noise-sigmas 0,1 --cpu
```

## Run on Efficient-style Salt datasets

Clipped dataset, matching the previous `[1.5, 4.5]` box-constrained experiment:

```bash
../../.venv/bin/python scripts/create_efficient_salt_datasets.py --output-dir data/efficient_salt --noise-sigmas 0,1 --clip-velocity
python scripts/run_salt_fwi.py --efficient-dataset-dir data/efficient_salt --geometry-mode efficient_reference --noise-mode efficient_reference --noise-sigmas 0,1 --epochs 100 --time-steps 736 --lr 0.03 --cost-scaling 0.1 --clip-grad 1e-5
```

Unclipped dataset, preserving the raw `zoom_and_crop` range after Salt preprocessing:

```bash
../../.venv/bin/python scripts/create_efficient_salt_datasets.py --output-dir data/efficient_salt_unclipped --noise-sigmas 0,1
python scripts/run_salt_fwi.py --efficient-dataset-dir data/efficient_salt_unclipped --geometry-mode efficient_reference --noise-mode efficient_reference --noise-sigmas 0,1 --epochs 100 --time-steps 791 --lr 0.03 --cost-scaling 0.1 --clip-grad 1e-5
```

This mode loads `true_velocity_model`, `initial_velocity_model`, source/receiver geometry, and reference Devito signals from `salt_efficient_noise_sigma*.npz`. The Efficient/Devito `observed_seismic_data` is stored in the output as a reference, while the loss uses a Zenodo-solver-compatible measurement regenerated from the loaded true model with matching Efficient-style geometry.

Result folders are named as:

```text
YYYYMMDD_HHMMSS_salt_tlfwi_alpha-none_noise-{noise}_box-{box}
```

For example, clipped Salt runs use `box-1p5-4p5`, while unclipped Salt runs use `box-unclipped`.


## Git policy

The repository tracks the TL-FWI code, Zenodo source files, requirements, README, and lightweight markdown notes. Generated artifacts are intentionally not tracked:

- `.venv/`
- `data/`
- `results/`
- `reports/assets/`
- Python cache directories

Regenerate datasets and results with the commands above after cloning.
