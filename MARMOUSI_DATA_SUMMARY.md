# Marmousiデータ概要

このメモは、Marmousi実験で使う以下のデータをMarkdown形式でまとめたものです。

- true velocity model
- initial model
- observed seismic data

## 前処理済みモデル

MarmousiのP波速度モデルは、`scripts/prepare_marmousi.py` により作成されます。

```bash
MPLCONFIGDIR=.matplotlib-cache poetry run python scripts/prepare_marmousi.py
```

この前処理では、Marmousiの全体extentをcropせずに読み込み、アスペクト比を維持したまま `TARGET_NZ=101` にdownsampleしています。

| 項目 | 値 |
| --- | --- |
| 元データ | AGL Elastic Marmousi P-wave velocity model |
| 読み込みファイル | `MODEL_P-WAVE_VELOCITY_1.25m.segy` |
| 前処理後shape | `(101, 490)` |
| 単位 | km/s |
| crop | なし |
| downsample | あり、アスペクト比維持 |

## True Velocity Model

真の速度モデルです。FWIの評価基準として使います。

| 項目 | 値 |
| --- | --- |
| 保存先 | `datasets/marmousi/processed/marmousi_vp_true_full_101x490.npy` |
| preview | `datasets/marmousi/processed/marmousi_vp_true_full_101x490.png` |
| shape | `(101, 490)` |
| dtype | `float32` |
| min | `1.0280 km/s` |
| max | `4.7000 km/s` |
| mean | `2.6719 km/s` |

![Marmousi true velocity model](datasets/marmousi/processed/marmousi_vp_true_full_101x490.png)

## Initial Model

初期解です。前処理後のtrue velocity modelにGaussian smoothingをかけて作成しています。

| 項目 | 値 |
| --- | --- |
| 保存先 | `datasets/marmousi/processed/marmousi_vp_init_full_101x490.npy` |
| preview | `datasets/marmousi/processed/marmousi_vp_init_full_101x490.png` |
| shape | `(101, 490)` |
| dtype | `float32` |
| smoothing | Gaussian smoothing, `sigma=8.0` |
| min | `1.5111 km/s` |
| max | `4.1592 km/s` |
| mean | `2.6719 km/s` |

![Marmousi initial model](datasets/marmousi/processed/marmousi_vp_init_full_101x490.png)

## Observed Seismic Data

観測Seismic dataは、Marmousi FWI実行時にtrue velocity modelから合成されます。前処理段階では作成されず、`src/box-TV-constrained-FWI-marmousi.py` の実行結果 `.npz` 内に保存されます。

直近の確認実行では、以下に保存されています。

```text
results/marmousi/20260602_214643_marmousi_pds_alpha-6449p11_noise-0_box-1-4p7/
  20260602_214643_marmousi_pds_alpha-6449p11_noise-0_box-1-4p7.npz
```

| 項目 | 値 |
| --- | --- |
| `.npz` key | `observed_seismic_data` |
| shape | `(3, 769, 101)` |
| shapeの意味 | `(n_shots, time_length, n_receivers)` |
| dtype | `float32` |
| n_shots | `3` |
| n_receivers | `101` |
| noise_sigma | `0.0` |
| min | `-22.9288` |
| max | `44.0182` |
| mean | `0.0012` |
| std | `1.0045` |

## 観測条件

直近のMarmousi実行では、以下の条件で観測Seismic dataを生成しています。

| 項目 | 値 |
| --- | --- |
| algorithm | `pds` |
| alpha | `6449.105078125001` |
| box constraint | `[1.0, 4.7] km/s` |
| source_frequency | `0.01` |
| simulation_times | `1000` |
| start_time | `0` |
| cell_meter_size | `(10.0, 10.0)` |
| real_cell_size | `(x=490, y=101)` |
| damping_cell_thickness | `40` |
| random_seed | `0` |

## Pythonでの読み込み例

```python
from pathlib import Path
import numpy as np

true_v = np.load("datasets/marmousi/processed/marmousi_vp_true_full_101x490.npy")
init_v = np.load("datasets/marmousi/processed/marmousi_vp_init_full_101x490.npy")

result_dir = Path("results/marmousi/20260602_214643_marmousi_pds_alpha-6449p11_noise-0_box-1-4p7")
result = np.load(result_dir / f"{result_dir.name}.npz")

observed = result["observed_seismic_data"]
source_locations = result["source_locations"]
receiver_locations = result["receiver_locations"]

print(true_v.shape)
print(init_v.shape)
print(observed.shape)
print(source_locations.shape)
print(receiver_locations.shape)
```

## 注意

- `datasets/` と `results/` はGit管理外です。
- clone直後はMarmousiデータも実験結果も存在しません。
- 別PCでは、まず `scripts/prepare_marmousi.py` を実行して前処理済みモデルを作成してください。
- observed seismic dataはFWI実行時に生成されるため、必要な場合は `src/box-TV-constrained-FWI-marmousi.py` を実行してください。
