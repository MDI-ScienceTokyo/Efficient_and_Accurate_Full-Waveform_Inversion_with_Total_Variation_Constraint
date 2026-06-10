# Salt FWI Results: velocity-scale outputs

## 概要

x/y の向きを修正し、保存・図示用の速度モデルを `velocity[y, x]` に統一した。モデル比較図は上段に true / initial / final、下段に error map を並べ、各パネルのアスペクト比を実配列の `(y, x)` に合わせている。

- velocity range: `[1.5, 4.5]` km/s
- velocity arrays: `(y, x)`; rows are y/depth samples, columns are x samples
- gamma arrays: `(x, y)` internal solver variables
- noise mode: `velocity_relative`
- `sigma=1`: clean waveform std の約 1/3 の Gaussian noise
- optimizer: Adam, `lr=0.01`, `cost_scaling=0.01`, `clip_grad=1e-5`
- epochs: 5

## Summary Table

| noise sigma | epochs | velocity shape | initial cost | final cost | initial velocity MSE | final velocity MSE | best velocity MSE | max velocity update | clean waveform std | actual noise std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | (64, 128) = (y, x) | 7.7389e-05 | 1.99714e-05 | 0.0214663 | 0.019961 | 0.019961 @ 5 | 0.435301 | 0.00786249 | 0 |
| 1 | 5 | (64, 128) = (y, x) | 0.0025321 | 0.00246429 | 0.0214663 | 0.0200613 | 0.0200613 @ 5 | 0.456389 | 0.00786249 | 0.00261237 |

## Velocity Model Comparison

### noise sigma = 0

![noise sigma 0 velocity models](assets/noise_sigma0_velocity_models.png)

### noise sigma = 1

![noise sigma 1 velocity models](assets/noise_sigma1_velocity_models.png)

## Metrics

![metrics history](assets/metrics_history.png)

## Observed Data

![observed shot gathers](assets/observed_shot_gathers.png)

## 考察

上段の3枚は同じ速度スケール `[1.5, 4.5]` km/s で表示している。下段は `Final - Initial` と `Final - True` を独立した差分スケールで表示しているため、更新量と残差構造を見やすい。

今回の 5 epoch 実行では `noise_sigma=0` と `noise_sigma=1` のどちらも velocity MSE が低下している。`gamma_*` は solver 内部の都合で `(x, y)` のまま残しているため、可視化・比較・論文用の評価には `velocity_*` を使う。

## Saved Arrays

`results/salt_noise_sigma*/fwi_result.npz` に以下を保存している。

- `velocity_true`: true model in km/s, axes `(y, x)`
- `velocity_initial`: initial model in km/s, axes `(y, x)`
- `velocity_final`: final model in km/s, axes `(y, x)`
- `velocity_history`: per-epoch velocity models in km/s, axes `(epoch, y, x)`
- `gamma_true`, `gamma_initial`, `gamma_final`, `gamma_history`: solver internal variables, axes `(x, y)` with ghost cells where applicable
- `vmin_km_s`, `vmax_km_s`: conversion range

## Reproduction

```bash
cd /projects/Efficient_and_Accurate_Full-Waveform_Inversion_with_Total_Variation_Constraint/external/Accelerating_FWI_By_Transfer_Learning
source .venv/bin/activate
python scripts/prepare_salt_model.py
python scripts/run_salt_fwi.py --noise-sigmas 0,1 --epochs 5
```
