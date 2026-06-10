# TL-FWI on Efficient-style Salt dataset

## 概要

提案法側の Salt dataset 作成コードを参考に、`true_velocity_model`、`initial_velocity_model`、source/receiver 配置、Devito の seismic signal を含む `noise_sigma=0` と `noise_sigma=1` の dataset を作成した。その dataset の true / initial velocity を Zenodo solver に読み込ませ、TL-FWI (Transfer Learning FWI) として整理した実験名でFWIを実行した。

今回の改善版では `--geometry-mode efficient_reference` を使い、source/receiver 配置と時間ステップ数を Efficient dataset 側に合わせた。`noise_sigma=1` は Efficient dataset 内の `seismic_noise.std / clean_observed.std = 0.480` を Zenodo waveform に写像している。

```bash
python scripts/run_salt_fwi.py --efficient-dataset-dir data/efficient_salt --geometry-mode efficient_reference --noise-mode efficient_reference --noise-sigmas 0,1 --epochs 100 --time-steps 736 --lr 0.03 --cost-scaling 0.1 --clip-grad 1e-5
```

## Summary

`velocity objective` は実行コードに合わせて `0.5 * mean((v_true - v_pred)^2)` としている。誤差マップのカラーバーは実速度差 `km/s`。

| noise sigma | epochs | initial velocity objective | final velocity objective | best velocity objective | best epoch | final cost | noise/clean std |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 0.219690 | 0.013266 | 0.013266 | 100 | 2.13043e-05 | 0.000 |
| 1 | 100 | 0.219690 | 0.033476 | 0.024753 | 41 | 0.0970935 | 0.480 |

- `noise_sigma=0`: final objective は `0.013266`。initial `0.219690` から `94.0%` 改善した。
- `noise_sigma=1`: best objective は epoch `41` の `0.024753` で、そこから 100 epoch まで進めると `0.033476` へ悪化した。
- best epoch 同士で見ると、noiseありは noiseなしより `86.6%` 悪い。100 epoch final 同士では `152.3%` 悪い。

## RMSE and SSIM

`RMSE` と `SSIM` は true velocity に対して計算した。SSIM は `skimage.metrics.structural_similarity` を使い、速度の表示範囲に合わせて `data_range=3.0` とした。

| noise sigma | model | epoch | velocity objective | RMSE (km/s) | SSIM |
|---:|:---|---:|---:|---:|---:|
| 0 | initial | 0 | 0.219690 | 0.662857 | 0.433598 |
| 0 | best | 100 | 0.013266 | 0.162887 | 0.740589 |
| 0 | final | 100 | 0.013266 | 0.162887 | 0.740589 |
| 1 | initial | 0 | 0.219690 | 0.662857 | 0.433598 |
| 1 | best | 41 | 0.024753 | 0.222500 | 0.561009 |
| 1 | final | 100 | 0.033476 | 0.258753 | 0.441344 |

## 保存先

実験で使ったdataset、FWI結果、図は以下に保存している。

| 種類 | noise sigma | folder | file |
|:---|---:|:---|:---|
| dataset | 0 | `data/efficient_salt/` | `salt_efficient_noise_sigma0.npz`, `salt_efficient_noise_sigma0.json` |
| dataset | 1 | `data/efficient_salt/` | `salt_efficient_noise_sigma1.npz`, `salt_efficient_noise_sigma1.json` |
| FWI result | 0 | `results/20260610_131242_salt_tlfwi_alpha-none_noise-0_box-1p5-4p5/` | `fwi_result.npz`, `metrics.csv`, `observed_data.npz` |
| FWI result | 1 | `results/20260610_131306_salt_tlfwi_alpha-none_noise-1_box-1p5-4p5/` | `fwi_result.npz`, `metrics.csv`, `observed_data.npz` |
| model figure | 0 | `reports/assets/` | `efficient_reference_noise_sigma0_models.png` |
| model figure | 1 | `reports/assets/` | `efficient_reference_noise_sigma1_models.png` |
| metrics figure | 0,1 | `reports/assets/` | `efficient_reference_metrics.png` |
| waveform figure | 0,1 | `reports/assets/` | `efficient_reference_waveforms.png` |

`fwi_result.npz` には `velocity_true`, `velocity_initial`, `velocity_final`, `velocity_history`, `gamma_true`, `gamma_initial`, `gamma_final`, `gamma_history` を保存している。`metrics.csv` には epochごとの `cost`, `gamma mse`, `velocity objective` を保存している。

## Velocity Models

### noise sigma = 0

![sigma 0 models](assets/efficient_reference_noise_sigma0_models.png)

### noise sigma = 1

![sigma 1 models](assets/efficient_reference_noise_sigma1_models.png)

## Metrics

![metrics](assets/efficient_reference_metrics.png)

## Seismic Signals

Efficient/Devito dataset に保存されている shot gather の例。`noise_sigma=1` は clean waveform 標準偏差の約48%の Gaussian noise を含む。

![waveforms](assets/efficient_reference_waveforms.png)

## 考察

初期モデルは提案法の `load_salt_model(Vec2D(100, 50))` で作られる initial model と一致している。以前の Zenodo デフォルト配置では、観測条件が dataset 側とずれていたため、noiseあり/なしの差がかなり小さく見えていた。

source/receiver 配置と `time_steps=736` を dataset 側に合わせると、noiseなしは100 epochまで安定して改善し、noiseありは約40 epochで最良になったあと、waveform loss を下げようとして velocity objective が悪化した。これは「ノイズに影響される」挙動として自然で、noiseありでは早期停止または正則化が必要になる。

現状の `noise_sigma=0` は提案法と同程度を狙える水準まで改善している一方、`noise_sigma=1` は最良 epoch を使っても明確に劣る。比較実験としては、noiseなしは 100 epoch、noiseありは validation/true MSE を見ない実験設定なら waveform loss の停滞を基準に早期停止する、という扱いが妥当。
