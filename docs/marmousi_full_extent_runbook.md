# Marmousi full-extent FWI runbook

## What This Uses

This setup does not use the old `101 x 490` Marmousi model that was physically shrunk by the driver.

It uses the Marmousi full physical extent:

```text
raw Marmousi VP:      2801 x 13601, 1.25 m grid, about 3.5 km x 17 km
10 m processed model:  351 x 1701, 10 m grid, about 3.5 km x 17 km
20 m processed model:  176 x 851,  20 m grid, about 3.5 km x 17 km
```

The raw `1.25 m` model is not used directly for FWI because it has about 38 million cells before damping. That is not practical for this Devito workflow on a normal workstation. The new preprocessing keeps the full physical extent and resamples it to a workable grid.

The `10 m` grid is still heavy for `simulation_times=1000`: Devito may warn that wavefield memory will exceed physical memory. The recommended practical starting point is the `20 m` full-extent model.

## Prepare Data

Recommended `20 m` grid:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache \
.venv/bin/python scripts/prepare_marmousi_full_extent.py \
  --target-dz-m 20 \
  --target-dx-m 20
```

Heavier `10 m` grid:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache \
.venv/bin/python scripts/prepare_marmousi_full_extent.py \
  --target-dz-m 10 \
  --target-dx-m 10
```

The `20 m` command writes:

```text
datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz
```

## Smoke Test

Gradient path:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR OMP_NUM_THREADS=1 \
.venv/bin/python src/box-TV-constrained-FWI-marmousi-full.py \
  --model-path datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz \
  --max-n-iters 1 \
  --n-shots 1 \
  --n-receivers 21 \
  --alphas 0 \
  --gamma1 1e-10 \
  --simulation-times 500 \
  --num-parallel-workers 1 \
  --result-root-path results/marmousi_full_20m_smoke
```

TV + box path:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR OMP_NUM_THREADS=1 \
.venv/bin/python src/box-TV-constrained-FWI-marmousi-full.py \
  --model-path datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz \
  --max-n-iters 1 \
  --n-shots 1 \
  --n-receivers 21 \
  --alpha-scales 0.8 \
  --gamma1 1e-10 \
  --simulation-times 500 \
  --num-parallel-workers 1 \
  --result-root-path results/marmousi_full_20m_smoke
```

## Run With 10 Shots / 201 Receivers

Gradient baseline:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR OMP_NUM_THREADS=1 \
.venv/bin/python src/box-TV-constrained-FWI-marmousi-full.py \
  --model-path datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz \
  --max-n-iters 5000 \
  --n-shots 10 \
  --n-receivers 201 \
  --alphas 0 \
  --noise-sigma 0 \
  --gamma1 1e-10 \
  --simulation-times 1000 \
  --num-parallel-workers 1 \
  --result-root-path results/marmousi_full_20m
```

TV + box constraint with alpha = 0.8 * TV(true):

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR OMP_NUM_THREADS=1 \
.venv/bin/python src/box-TV-constrained-FWI-marmousi-full.py \
  --model-path datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz \
  --max-n-iters 5000 \
  --n-shots 10 \
  --n-receivers 201 \
  --alpha-scales 0.8 \
  --noise-sigma 0 \
  --gamma1 1e-10 \
  --gamma2 100 \
  --simulation-times 1000 \
  --num-parallel-workers 1 \
  --result-root-path results/marmousi_full_20m
```

Multiple TV settings:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR OMP_NUM_THREADS=1 \
.venv/bin/python src/box-TV-constrained-FWI-marmousi-full.py \
  --model-path datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz \
  --max-n-iters 5000 \
  --n-shots 10 \
  --n-receivers 201 \
  --alpha-scales 0.5,0.8,1.0 \
  --noise-sigma 0 \
  --gamma1 1e-10 \
  --gamma2 100 \
  --simulation-times 1000 \
  --num-parallel-workers 1 \
  --result-root-path results/marmousi_full_20m
```

## Notes

- `--alphas 0` runs the gradient baseline.
- `--alpha-scales 0.8` runs PDS / TV + box with `alpha = 0.8 * TV(true)`.
- `--num-parallel-workers 1` is conservative. Increase it only if memory is comfortable.
- `n_shots=10` is synthetic acquisition geometry, not a fixed property of Marmousi.
- `n_receivers=201` places receivers across the full 17 km width, about every 85 m.
- `gamma1=1e-6` is too large for the full-extent Marmousi gradient run; it can make velocity non-positive after a few iterations.
- In a `20 m`, `n_shots=10`, `n_receivers=201`, `simulation_times=1000` test, even `gamma1=1e-8` became non-positive at iteration 3. Start with `1e-10` or smaller, or add box projection / line search to the gradient baseline.
- Before launching `5000` iterations, run `--max-n-iters 10` or `50` first and check that objective, velocity range, and runtime look sane.
