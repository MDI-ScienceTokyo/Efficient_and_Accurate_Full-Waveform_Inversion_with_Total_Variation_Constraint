# BP2004 Gradient Descent 5000 Iteration Results

Date: 2026-06-03

## Summary

BP2004 cropped model に対して、TV constraint なし、box constraint なしの plain gradient descent を 5000 回実行した。

更新式は次の通り。

```python
vp = vp - gamma1 * gradient
```

反復中の clipping は行っていない。つまり `np.clip`、box projection、TV projection は使っていない。

結論:

- noise なし、noise sigma 0.01 の両方で 5000 回完走した。
- 両方とも `diverged = False`、`stop_reason = completed`。
- objective は 5000 回まで単調非増加だった。
- gradient norm も大きく低下した。
- ただし box constraint なしのため、最終速度の最小値は約 `702 m/s` まで下がった。これは物理的にはかなり低く、長期反復では box constraint なしの副作用として注意が必要。
- model MSE はわずかに増加したため、data misfit は下がっているが真値速度モデルに近づいているとは限らない。

## Conditions

| item | value |
| --- | ---: |
| model | SEG/BP 2004 exact velocity cropped model |
| model shape | `241 x 801` |
| grid spacing | `dz = 25 m`, `dx = 25 m` |
| initial model | true model with Gaussian smoothing |
| algorithm | plain gradient descent |
| TV constraint | no |
| box constraint | no |
| clipping during iteration | no |
| iterations | 5000 |
| `gamma1` | `1.0e-2` |
| shots | 5 |
| receivers | 201 |
| observed data samples | `(5, 314, 201)` |

## Runs

| label | observed data | output directory |
| --- | --- | --- |
| noise0 | `outputs/bp2004_gradient/observed_data.npz` | `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000` |
| noise_sigma0p01 | `outputs/bp2004_gradient/observed_data_noise_sigma0p01.npz` | `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000` |

## Final Status

| metric | noise0 | noise sigma 0.01 |
| --- | ---: | ---: |
| completed iterations | 5000 | 5000 |
| diverged | false | false |
| stop reason | completed | completed |
| objective first | `2.2179546828e+06` | `2.2179953668e+06` |
| objective last | `1.7751392525e+06` | `1.7751809202e+06` |
| objective decrease | `-4.4281543027e+05` | `-4.4281444656e+05` |
| objective decrease ratio | `-19.9650 %` | `-19.9646 %` |
| objective minimum iteration | 5000 | 5000 |
| objective monotone non-increasing | true | true |
| number of objective increases | 0 | 0 |
| grad norm first | `1.6837614441e+02` | `1.6837544250e+02` |
| grad norm last | `2.2135976791e+01` | `2.2135036469e+01` |
| model MSE first | `9.9861725973e+04` | `9.9861725973e+04` |
| model MSE last | `9.9888998269e+04` | `9.9888997692e+04` |
| model MSE change ratio | `+0.02731 %` | `+0.02731 %` |
| velocity min first | `1489.1875 m/s` | `1489.1875 m/s` |
| velocity min last | `701.8007 m/s` | `701.8079 m/s` |
| velocity max first | `4779.6572 m/s` | `4779.6572 m/s` |
| velocity max last | `4779.6572 m/s` | `4779.6572 m/s` |
| final RMSE vs true | `316.0522 m/s` | `316.0522 m/s` |
| final relative error | `0.103595` | `0.103595` |
| final SSIM vs true | `0.913567` | `0.913568` |
| elapsed time | `8890.1 s` | `8894.4 s` |

## Objective Samples

| iteration | noise0 objective | noise sigma 0.01 objective |
| ---: | ---: | ---: |
| 1 | `2.2179546828e+06` | `2.2179953668e+06` |
| 10 | `2.2131371451e+06` | `2.2131761700e+06` |
| 100 | `2.1733545777e+06` | `2.1733932117e+06` |
| 500 | `2.0681592623e+06` | `2.0681982161e+06` |
| 1000 | `1.9862681258e+06` | `1.9863061663e+06` |
| 2000 | `1.8882620338e+06` | `1.8883016734e+06` |
| 3000 | `1.8385755082e+06` | `1.8386154176e+06` |
| 4000 | `1.7981614446e+06` | `1.7982029394e+06` |
| 5000 | `1.7751392525e+06` | `1.7751809202e+06` |

## Result Figures

### True, Initial, and Final Velocity Models

The color scale is shared across panels. Because no box constraint was used, the final models include values lower than the original BP2004 water velocity.

![Velocity comparison](../outputs/bp2004_gradient/report_figures/velocity_true_initial_final_comparison.png)

### Final Error Maps

Each panel shows `final velocity - true velocity`.

![Final error maps](../outputs/bp2004_gradient/report_figures/final_error_maps.png)

### RMSE per Iteration

RMSE is computed against the true cropped BP2004 velocity model.

![RMSE per iteration](../outputs/bp2004_gradient/report_figures/rmse_per_iteration.png)

### Relative Model Error per Iteration

Relative error is `RMSE / RMS(true velocity)`.

![Relative error per iteration](../outputs/bp2004_gradient/report_figures/relative_error_per_iteration.png)

### Data Misfit Error per Iteration

This is the FWI objective value, `0.5 * ||F(v) - d_obs||_2^2`.

![Objective error per iteration](../outputs/bp2004_gradient/report_figures/objective_error_per_iteration.png)

### SSIM Summary

SSIM is computed against the true cropped BP2004 velocity model. The current result files save the initial and final velocity models plus scalar histories, but do not save every intermediate velocity model. Therefore this figure compares the initial model and the final models, not SSIM at every iteration.

![SSIM summary](../outputs/bp2004_gradient/report_figures/ssim_summary.png)

| model | SSIM vs true |
| --- | ---: |
| initial | `0.913951` |
| final noise0 | `0.913567` |
| final noise sigma 0.01 | `0.913568` |

## Observations

### No Divergence

両条件とも objective は 5000 iteration まで単調に低下した。NaN/Inf による停止や objective 爆増による停止はなかった。

### Noise Effect Is Small

今回の `noise_sigma = 0.01` では、noise なしとほぼ同じ収束挙動になった。objective の絶対値は noise ありの方が少しだけ大きいが、差は小さい。

### Velocity Minimum Becomes Unphysical

box constraint なし、clip なしのため、速度の最小値が初期の `1489 m/s` から約 `702 m/s` まで低下した。これは acoustic velocity model としてはかなり低い。

この実験では数値的には発散していないが、物理的妥当性を保つには box constraint または別の step-size / regularization の検討が必要。

### Data Misfit and Model MSE Do Not Agree

objective は約 20% 低下した一方で、model MSE は約 0.027% 増加した。合成データに対する misfit は下がっているが、真値速度モデルへの距離は改善していない。

これは FWI の非線形性、少数 shot、低周波設定、box/TV なしの更新によるものと考えられる。

RMSE と SSIM も同じ傾向を示している。初期 RMSE は `316.0091 m/s`、最終 RMSE は両条件とも約 `316.0522 m/s` で、わずかに悪化した。SSIM も `0.913951` から約 `0.913567` にわずかに低下した。

## Output Files

### Noise 0

- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/gradient_descent_result.npz`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/metrics.csv`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/progress.csv`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/status.json`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/objective_history.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/grad_norm_history.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/model_mse_history.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/vp_final.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/vp_final_minus_true.png`

### Noise Sigma 0.01

- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/gradient_descent_result.npz`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/metrics.csv`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/progress.csv`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/status.json`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/objective_history.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/grad_norm_history.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/model_mse_history.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/vp_final.png`
- `outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/vp_final_minus_true.png`

### Report Figures

- `outputs/bp2004_gradient/report_figures/velocity_true_initial_final_comparison.png`
- `outputs/bp2004_gradient/report_figures/final_error_maps.png`
- `outputs/bp2004_gradient/report_figures/rmse_per_iteration.png`
- `outputs/bp2004_gradient/report_figures/relative_error_per_iteration.png`
- `outputs/bp2004_gradient/report_figures/objective_error_per_iteration.png`
- `outputs/bp2004_gradient/report_figures/ssim_summary.png`

## Reproduction Commands

Noise 0:

```bash
export MPLCONFIGDIR=.matplotlib-cache

DEVITO_LOGGING=ERROR .venv/bin/python -u scripts/run_bp2004_gradient_descent.py \
  --config configs/bp2004_gradient.yaml \
  --observed outputs/bp2004_gradient/observed_data.npz \
  --iterations 5000 \
  --gamma1 1e-2 \
  --output-dir outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000 \
  --progress-interval 1
```

Noise sigma 0.01:

```bash
export MPLCONFIGDIR=.matplotlib-cache

.venv/bin/python scripts/run_bp2004_forward.py \
  --config configs/bp2004_gradient.yaml \
  --noise-sigma 0.01 \
  --seed 0 \
  --output outputs/bp2004_gradient/observed_data_noise_sigma0p01.npz

DEVITO_LOGGING=ERROR .venv/bin/python -u scripts/run_bp2004_gradient_descent.py \
  --config configs/bp2004_gradient.yaml \
  --observed outputs/bp2004_gradient/observed_data_noise_sigma0p01.npz \
  --iterations 5000 \
  --gamma1 1e-2 \
  --output-dir outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000 \
  --progress-interval 1
```
