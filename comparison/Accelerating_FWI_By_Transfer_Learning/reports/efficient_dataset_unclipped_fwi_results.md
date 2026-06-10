# TL-FWI on Unclipped Efficient-style Salt dataset

## 概要

前回の `data/efficient_salt` では `zoom_and_crop` 後の velocity を `[1.5, 4.5] km/s` にclipしていた。今回はそのclipを外し、`data/efficient_salt_unclipped` を作成して同じFWI条件で実験した。

clipなしの true velocity 範囲は `1.4168-4.8336 km/s`。initial velocity 範囲は `2.1310-2.5815 km/s`。Devito側の observed seismic data は shape `(20, 791, 101)` になったため、Zenodo FWIも `--time-steps 791` で実行した。

```bash
python scripts/create_efficient_salt_datasets.py --output-dir data/efficient_salt_unclipped --noise-sigmas 0,1
python scripts/run_salt_fwi.py --efficient-dataset-dir data/efficient_salt_unclipped --geometry-mode efficient_reference --noise-mode efficient_reference --noise-sigmas 0,1 --epochs 100 --time-steps 791 --lr 0.03 --cost-scaling 0.1 --clip-grad 1e-5
```

## Summary

`velocity objective` は `0.5 * mean((v_true - v_pred)^2)`。SSIMはtrue velocityの実レンジを `data_range` として計算した。

| noise sigma | epochs | time steps | true range (km/s) | initial objective | final objective | best objective | best epoch | final cost | noise/clean std |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 791 | 1.4168-4.8336 | 0.220905 | 0.015303 | 0.015303 | 100 | 0.000116284 | 0.000 |
| 1 | 100 | 791 | 1.4168-4.8336 | 0.220905 | 0.033103 | 0.027281 | 50 | 0.115744 | 0.480 |

## RMSE and SSIM

| noise sigma | model | epoch | velocity objective | RMSE (km/s) | SSIM |
|---:|:---|---:|---:|---:|---:|
| 0 | initial | 0 | 0.220905 | 0.664688 | 0.469645 |
| 0 | best | 100 | 0.015303 | 0.174947 | 0.744266 |
| 0 | final | 100 | 0.015303 | 0.174947 | 0.744266 |
| 1 | initial | 0 | 0.220905 | 0.664688 | 0.469645 |
| 1 | best | 50 | 0.027281 | 0.233587 | 0.573724 |
| 1 | final | 100 | 0.033103 | 0.257307 | 0.478732 |

## 保存先

実験で使ったdataset、FWI結果、図は以下に保存している。

| 種類 | noise sigma | folder | file |
|:---|---:|:---|:---|
| dataset | 0 | `data/efficient_salt_unclipped/` | `salt_efficient_noise_sigma0.npz`, `salt_efficient_noise_sigma0.json` |
| dataset | 1 | `data/efficient_salt_unclipped/` | `salt_efficient_noise_sigma1.npz`, `salt_efficient_noise_sigma1.json` |
| FWI result | 0 | `results/20260610_150307_salt_tlfwi_alpha-none_noise-0_box-unclipped/` | `fwi_result.npz`, `metrics.csv`, `observed_data.npz` |
| FWI result | 1 | `results/20260610_150430_salt_tlfwi_alpha-none_noise-1_box-unclipped/` | `fwi_result.npz`, `metrics.csv`, `observed_data.npz` |
| model figure | 0 | `reports/assets/` | `efficient_unclipped_noise_sigma0_models.png` |
| model figure | 1 | `reports/assets/` | `efficient_unclipped_noise_sigma1_models.png` |
| metrics figure | 0,1 | `reports/assets/` | `efficient_unclipped_metrics.png` |
| waveform figure | 0,1 | `reports/assets/` | `efficient_unclipped_waveforms.png` |

`fwi_result.npz` には `velocity_true`, `velocity_initial`, `velocity_final`, `velocity_history`, `gamma_true`, `gamma_initial`, `gamma_final`, `gamma_history` を保存している。`metrics.csv` には epochごとの `cost`, `gamma mse`, `velocity objective` を保存している。

## Velocity Models

### noise sigma = 0

![sigma 0 models](assets/efficient_unclipped_noise_sigma0_models.png)

### noise sigma = 1

![sigma 1 models](assets/efficient_unclipped_noise_sigma1_models.png)

## Metrics

![metrics](assets/efficient_unclipped_metrics.png)

## Seismic Signals

![waveforms](assets/efficient_unclipped_waveforms.png)

## 考察

clipを外すとtrue velocityに `1.5 km/s` 未満と `4.5 km/s` 超の値が入り、評価対象が少し難しくなる。noiseなしでは100 epochまで改善し、final objectiveは `0.015303`。clipありの `0.013266` よりやや悪いが、同じオーダーで収束している。

noiseありでは epoch `50` の best objective `0.027281` が最良で、その後100 epoch finalでは `0.033103` まで悪化した。clipありと同様に、noiseありでは早期停止が重要になる。
