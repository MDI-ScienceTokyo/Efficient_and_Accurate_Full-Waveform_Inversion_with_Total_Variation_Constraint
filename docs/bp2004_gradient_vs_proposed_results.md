# BP2004: Gradient Baseline vs Proposed TV+Box Method

Date: 2026-06-03

## Summary

BP2004 cropped model に対して、既存の `src/box-TV-constrained-FWI-BP2004.py` で実行した gradient baseline と提案法を比較した。

ここでの提案法は、コピー元の `box-TV-constrained-FWI.py` と同じ PDS 更新であり、TV constraint と box constraint を含む。

```python
tmp = grad.copy()
tmp[dsize:-dsize, dsize:-dsize] += diff_op.Dt(y)

v = v - gamma1 * tmp
v[dsize:-dsize, dsize:-dsize] = prox_box_constraint(
    remove_damping_cells(v, dsize), vmin, vmax
)

y = y + gamma2 * diff_op.D(
    2 * remove_damping_cells(v, dsize) - remove_damping_cells(prev_v, dsize)
)
y = y - gamma2 * proj_L12_norm_ball(y / gamma2, alpha)
```

Important caveat:

現時点で保存されている結果は、gradient が 5000 iteration、提案法 PDS が 10 iteration の smoke result である。したがって、以下は完全に公平な 5000 iteration 比較ではなく、暫定比較である。公平比較には PDS も同じ iteration 数で回す必要がある。

## Compared Runs

| method | result directory | iterations | noise | gamma1 | gamma2 | alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gradient | `results/bp2004/20260603_170432_bp2004_gradient_alpha-0_noise-0_box-1p45-5p5` | 5000 | 0 | `1.0e-5` | none | 0 |
| proposed PDS | `results/bp2004_pds_smoke/20260603_182555_bp2004_pds_alpha-500_noise-0_box-1p45-5p5` | 10 | 0 | `1.0e-6` | 100 | 500 |

Model and geometry:

| item | value |
| --- | ---: |
| model | BP2004 cropped exact model |
| model shape | `241 x 801` |
| spacing | `dz = 25 m`, `dx = 25 m` |
| velocity unit in solver | km/s |
| shots | 5 |
| receivers | 201 |
| source frequency | `0.005` |
| source/receiver depth | `25 m` |
| box constraint | `[1.45, 5.50] km/s` |

## Velocity Images

True, initial, gradient final, and proposed-method final velocity models are shown below.

![Velocity comparison](../outputs/bp2004_proposal_report_figures/velocity_true_initial_gradient_proposed.png)

## Error Maps

Each panel shows final velocity minus true velocity.

![Error maps](../outputs/bp2004_proposal_report_figures/error_maps_gradient_proposed.png)

## Error per Iteration

The following figure shows the data error/objective,

```text
0.5 * ||F(v) - d_obs||_2^2
```

per iteration.

![Objective error per iteration](../outputs/bp2004_proposal_report_figures/objective_error_per_iter.png)

The following figure shows model squared error per iteration.

![Model error per iteration](../outputs/bp2004_proposal_report_figures/model_error_per_iter.png)

For the first 10 iterations, the two methods can be compared on the same horizontal scale.

![First 10 metrics comparison](../outputs/bp2004_proposal_report_figures/first10_metrics_comparison.png)

## RMSE per Iteration

RMSE is computed against the true cropped BP2004 velocity model.

![RMSE per iteration](../outputs/bp2004_proposal_report_figures/rmse_per_iter.png)

## SSIM per Iteration

SSIM is computed against the true cropped BP2004 velocity model.

![SSIM per iteration](../outputs/bp2004_proposal_report_figures/ssim_per_iter.png)

## Numeric Summary

| metric | gradient | proposed PDS |
| --- | ---: | ---: |
| iterations | 5000 | 10 |
| objective first | `2.534119e+04` | `2.534119e+04` |
| objective last | `7.883757e+00` | `1.181588e+04` |
| model squared error first | `1.927716e+04` | `1.927738e+04` |
| model squared error last | `1.928194e+04` | `1.927722e+04` |
| RMSE first | `0.316007 km/s` | `0.316009 km/s` |
| RMSE last | `0.316046 km/s` | `0.316008 km/s` |
| SSIM first | `0.927646` | `0.927656` |
| SSIM last | `0.926004` | `0.927650` |
| TV first | `2361.8306` | `2358.4141` |
| TV last | `2490.7976` | `2360.9941` |
| final velocity min | `1.1951 km/s` | `1.4728 km/s` |
| final velocity max | `4.7797 km/s` | `4.7797 km/s` |

## Discussion

### Gradient Baseline

Gradient baseline drives the data objective down very strongly, from approximately `2.53e4` to `7.88`. However, the model error and SSIM do not improve. In fact, RMSE increases slightly and SSIM decreases.

This is consistent with the earlier BP2004 observation: minimizing the data misfit with a small number of shots does not necessarily move the velocity model closer to the true model. The inversion can reduce waveform residuals while producing a model that is less faithful in image metrics.

Another important point is that gradient baseline has no box constraint. Its final velocity minimum is `1.1951 km/s`, below the intended BP2004 physical lower bound and below the configured box lower bound of `1.45 km/s`.

### Proposed PDS with TV+Box

The proposed method result currently available is only 10 iterations, so it should be treated as a smoke test rather than a final convergence comparison.

Within those 10 iterations:

- objective decreases from `2.53e4` to `1.18e4`;
- RMSE slightly decreases;
- SSIM stays nearly constant;
- velocity stays inside the box constraint, with final minimum `1.4728 km/s`;
- TV remains close to the initial value.

This is qualitatively different from the gradient baseline: the proposed method is more conservative and keeps the velocity in the physically constrained range.

### Current Limitation

The current comparison is not iteration-matched:

- gradient: 5000 iterations
- proposed PDS: 10 iterations

Therefore, the main conclusion at this stage is not that one method is definitively better, but that:

1. the BP2004-specific proposed-method code runs;
2. the gradient baseline can overfit the data objective while harming image metrics;
3. the TV+box method enforces the physical velocity range and behaves stably in the smoke test.

For a publication-quality comparison, run PDS for the same iteration count and compare the histories again.

## Recommended Full PDS Run

To run the proposed method for 5000 iterations:

```bash
MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 500 \
  --noise-sigmas 0 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --gamma2 100 \
  --result-root-path results/bp2004_pds
```

To sweep alpha values:

```bash
MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 500,1500,3000 \
  --noise-sigmas 0 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --gamma2 100 \
  --result-root-path results/bp2004_pds_alpha_sweep
```

## Figure Files

- `outputs/bp2004_proposal_report_figures/velocity_true_initial_gradient_proposed.png`
- `outputs/bp2004_proposal_report_figures/error_maps_gradient_proposed.png`
- `outputs/bp2004_proposal_report_figures/objective_error_per_iter.png`
- `outputs/bp2004_proposal_report_figures/model_error_per_iter.png`
- `outputs/bp2004_proposal_report_figures/rmse_per_iter.png`
- `outputs/bp2004_proposal_report_figures/ssim_per_iter.png`
- `outputs/bp2004_proposal_report_figures/first10_metrics_comparison.png`
