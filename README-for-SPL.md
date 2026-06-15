# SPL実験用メモ

このファイルは、`Expt/SPL` ブランチで追加した比較実験用コードの使い方をまとめたものです。

## 目的

`src/box-TV-constrained-FWI.py` は、Saltモデルに対してFWIを実行し、TV制約なし/ありの結果を比較するための実験スクリプトです。

現在のデフォルト設定では、ノイズなし/ありの2条件それぞれについて、次の4条件を順番に実行します。

| alpha | algorithm | 内容 |
| ---: | --- | --- |
| 0 | `gradient` | TV制約なしのベースライン |
| 150 | `pds` | TV制約あり |
| 350 | `pds` | TV制約あり |
| 550 | `pds` | TV制約あり |

`alpha=0` のときはTV制約を使わず、単純な勾配法で更新します。`alpha=150, 350, 550` のときは primal-dual splitting によりTV制約付きの更新を行います。

ノイズ条件は `noise_sigma = 0, 1` です。コード上の `noise_sigma` はガウスノイズの標準偏差なので、`noise_sigma=1` は variance=1 の Gaussian noise に対応します。

## セットアップ

このリポジトリはPython 3.12とPoetryを使います。clone後、以下を実行してください。

```bash
poetry install
```

`poetry.toml` により、仮想環境はリポジトリ直下の `.venv/` に作成され、Poetryのキャッシュは `.poetry-cache/` に置かれます。これらはGit管理外です。

## データセット

実験にはSEG/EAGE Salt and Overthrust Modelsを使います。

データセットはサイズが大きいため、Gitには含めていません。clone後に次のコマンドでダウンロード・展開してください。

```bash
poetry run inv download-salt-and-overthrust-models
```

このコマンドは `tasks.py` に定義されています。実行すると、以下のように `datasets/` 配下へ保存されます。

```text
datasets/
  salt_and_overthrust_models.tar.gz
  salt_and_overthrust_models/
    3-D_Salt_Model/
    3-D_Overthrust_Model_Disk1/
    ...
```

`datasets/` は `.gitignore` されています。別PCで実行するときは、再度このコマンドを実行する必要があります。

現在の実験コードではSaltモデルを使います。読み込み箇所は `load_salt_model()` です。

```python
true_velocity_model, initial_velocity_model, vmin, vmax = load_salt_model(params.real_cell_size)
```

Overthrustモデル用の読み込み関数もありますが、現在はコメントアウトされています。

## 実行方法

通常のSPL比較実験は次のコマンドで実行します。

```bash
MPLCONFIGDIR=.matplotlib-cache poetry run python src/box-TV-constrained-FWI.py
```

`MPLCONFIGDIR=.matplotlib-cache` は、GUIのない環境やホームディレクトリに書き込めない環境でMatplotlibの警告を避けるための指定です。

このコマンドを実行すると、`run_alpha_experiments()` が呼ばれ、`noise_sigma = 0, 1` と `alpha = 0, 150, 350, 550` の組み合わせで、合計8実験が順番に走ります。

## BP2004 gradient smoke test

SEG/BP 2004 velocity benchmark data をダウンロードし、切り出し、合成観測データ生成、FWI gradient 1回計算まで確認する手順は [docs/bp2004_gradient.md](docs/bp2004_gradient.md) にまとめています。

## TL-FWI comparison workspace

Transfer Learning FWI比較用コードは次の場所に置いています。

```text
comparison/Accelerating_FWI_By_Transfer_Learning/
```

このフォルダは、Zenodoの `Accelerating_FWI_By_Transfer_Learning` コードをベースに、SEG/EAGE Saltモデルを読み込むためのadapter、Efficient and Accurate FWI形式のSalt dataset生成、TL-FWI実行、レポート作成用の補助コードを追加したものです。手法名としては結果フォルダ・README上で `TL-FWI` と表記します。

### TL-FWI環境の復元

clone後、親リポジトリ側の依存関係とSaltデータを準備してから、TL-FWI側の仮想環境を作成します。

```bash
poetry install
poetry run inv download-salt-and-overthrust-models

cd comparison/Accelerating_FWI_By_Transfer_Learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPUが使える環境では `requirements.txt` のCUDA 12.4対応PyTorchを使います。GPUがない、または短い確認だけ行う場合は `scripts/run_salt_fwi.py --cpu` を付けてください。

### TL-FWI用Salt dataset生成

Efficient and Accurate FWIのSalt前処理に合わせたdatasetはGit管理外です。再現時は次のコマンドで再生成します。

```bash
# [1.5, 4.5] km/s にclipする版
../../.venv/bin/python scripts/create_efficient_salt_datasets.py \
  --output-dir data/efficient_salt \
  --noise-sigmas 0,1 \
  --clip-velocity

# clipしない版
../../.venv/bin/python scripts/create_efficient_salt_datasets.py \
  --output-dir data/efficient_salt_unclipped \
  --noise-sigmas 0,1
```

生成される `data/efficient_salt*/salt_efficient_noise_sigma*.npz` には、`true_velocity_model`, `initial_velocity_model`, source/receiver geometry, Efficient/Devito由来の `observed_seismic_data` などが入ります。

### TL-FWI実行コマンド

論文側のpretraining epochとは別に、`scripts/run_salt_fwi.py --epochs` はFWIそのものの反復回数です。今回の35/100 iteration比較は次の設定で実行しました。

```bash
cd comparison/Accelerating_FWI_By_Transfer_Learning
source .venv/bin/activate

python scripts/run_salt_fwi.py \
  --efficient-dataset-dir data/efficient_salt_unclipped \
  --geometry-mode efficient_reference \
  --noise-mode efficient_reference \
  --noise-sigmas 0,1 \
  --epochs 35 \
  --time-steps 791 \
  --lr 0.03 \
  --cost-scaling 0.1 \
  --clip-grad 1e-5

python scripts/run_salt_fwi.py \
  --efficient-dataset-dir data/efficient_salt_unclipped \
  --geometry-mode efficient_reference \
  --noise-mode efficient_reference \
  --noise-sigmas 0,1 \
  --epochs 100 \
  --time-steps 791 \
  --lr 0.03 \
  --cost-scaling 0.1 \
  --clip-grad 1e-5
```

clipped datasetを使う場合は `--efficient-dataset-dir data/efficient_salt` とし、既存のclipped datasetでは `--time-steps 736` を使います。

### TL-FWIのGit管理方針

次回同じ環境を復元するためにGitへ入れる対象は、コード、`requirements.txt`、README、軽量なMarkdown/CSVレポート、レポート本文から参照する小さなPNG図です。次の生成物はGit管理外にします。

- `comparison/Accelerating_FWI_By_Transfer_Learning/.venv/`
- `comparison/Accelerating_FWI_By_Transfer_Learning/data/`
- `comparison/Accelerating_FWI_By_Transfer_Learning/results/`
- `comparison/Accelerating_FWI_By_Transfer_Learning/reports/assets/` のうち、汎用生成画像。ただし `tlfwi_unclipped_noise*_models_35_vs_100.png` と `tlfwi_unclipped_noise*_rmse_ssim_35_vs_100.png` はレポート用に管理対象
- `comparison/Accelerating_FWI_By_Transfer_Learning/reports/*.pdf`
- `.poetry-tool/`, `.poetry-cache/`, `.matplotlib-cache/`
- `results_env_check/`, `results_poetry_check/`

TL-FWI論文PDF本体はGitに入れず、必要ならローカルに再配置します。抽出済みメモ `comparison/Accelerating_FWI_By_Transfer_Learning/reports/tlfwi_paper_extracted.md` は軽量なので、パラメータ決定の記録として管理対象にできます。

## 出力される結果

実験結果は `results/` 配下に保存されます。各実験ごとに1つのディレクトリが作られます。

ディレクトリ名には、実験日時、画像名、アルゴリズム、alpha、noise、box constraintの下限/上限が入ります。

例:

```text
results/
  20260602_184500_salt_gradient_alpha-0_noise-0_box-1p5-4p5/
  20260602_185910_salt_pds_alpha-150_noise-0_box-1p5-4p5/
  20260602_190220_salt_pds_alpha-350_noise-0_box-1p5-4p5/
  20260602_190540_salt_pds_alpha-550_noise-0_box-1p5-4p5/
  20260602_190710_salt_gradient_alpha-0_noise-1_box-1p5-4p5/
  20260602_191120_salt_pds_alpha-150_noise-1_box-1p5-4p5/
  20260602_192430_salt_pds_alpha-350_noise-1_box-1p5-4p5/
  20260602_193740_salt_pds_alpha-550_noise-1_box-1p5-4p5/
```

各ディレクトリには、次の3種類のファイルが保存されます。

```text
<experiment_name>.npz
metrics.csv
config.json
```

### `.npz`

Pythonで解析するためのメインの保存ファイルです。`numpy.load()` で読み込めます。

```python
import numpy as np

data = np.load("results/<experiment_name>/<experiment_name>.npz")
print(data.files)
```

保存される主な配列は以下です。

| キー | 内容 |
| --- | --- |
| `final_velocity_model` | damping領域を除いた最終速度モデル |
| `final_velocity_model_with_damping` | damping領域を含む最終速度モデル |
| `true_velocity_model` | 真の速度モデル |
| `initial_velocity_model` | 初期解 |
| `initial_velocity_model_with_damping` | damping領域を含む初期解 |
| `observed_seismic_data` | 観測Seismic data |
| `source_locations` | 震源位置 |
| `receiver_locations` | 受振器位置 |
| `objective_history` | 目的関数値の履歴 |
| `mse_history` | MSEの履歴 |
| `psnr_history` | PSNRの履歴 |
| `ssim_history` | SSIMの履歴 |
| `tv_history` | TV値の履歴 |
| `dual_variable` | PDSで使う双対変数 `y` |

`observed_seismic_data` のshapeは、おおよそ次の形式です。

```text
(n_shots, time_length, n_receivers)
```

### `metrics.csv`

反復ごとの指標をCSVで保存します。表計算や簡単なプロットに使いやすい形式です。

列は以下です。

```text
iteration, objective, mse, psnr, ssim, tv
```

### `config.json`

実験条件を保存します。主な項目は以下です。

```text
image_name
algorithm
max_n_iters
completed_iters
n_shots
noise_sigma
gamma1
gamma2
alpha
box_min_value
box_max_value
random_seed
elapsed
real_cell_size
cell_meter_size
damping_cell_thickness
simulation_times
source_frequency
n_receivers
```

`random_seed` はデフォルトで `0` です。これにより、TVなし/TVありの各実験で同じノイズ実現を使いやすくしています。

## 図の保存

GUIのない環境では `plt.show()` が使えないため、図は `outputs/figures/` にPNGとして保存されます。

例:

```text
outputs/figures/
  001_true_velocity_model.png
  002_initial_velocity_model.png
  003_velocity_model_at_final_iteration_5000.png
```

`outputs/` はGit管理外です。

## パラメータ調整

主な調整箇所は `src/box-TV-constrained-FWI.py` です。

### 並列数

```python
num_parallels = 20
```

計算資源が足りない場合は、この値を下げてください。メモリ不足やプロセス数が多すぎる問題が出る場合は、まず `4` や `8` などに下げるのがよいです。

### alpha sweep

`run_alpha_experiments()` の `alphas` を変更します。

```python
def run_alpha_experiments(
    alphas: tuple[float, ...] = (0, 150, 350, 550),
    noise_sigmas: tuple[float, ...] = (0, 1),
    ...
):
```

`alpha=0` はTVなしの `gradient` として実行されます。`alpha>0` はTVありの `pds` として実行されます。

### 反復回数

```python
max_n_iters: int = 5000
```

短い動作確認をしたい場合は、例えば `10` や `100` に下げてください。

### shot数

```python
n_shots: int = 20
```

shot数を減らすと計算は軽くなりますが、実験条件が変わります。

### ノイズ

```python
noise_sigmas: tuple[float, ...] = (0, 1)
```

観測Seismic dataに加えるガウスノイズの標準偏差です。`0` はノイズなし、`1` は variance=1 の Gaussian noise です。

### ステップサイズ

```python
gamma1: float = 1e-4
gamma2: float = 100
```

`gamma1` は速度モデル更新側のステップサイズです。`gamma2` はPDSの双対変数更新側で使われます。`gradient` 実行時には `gamma2` は使われません。

### モデルサイズ・観測条件

`salt_model_test00_configuration()` で設定しています。

```python
real_cell_size=Vec2D(100, 50)
cell_meter_size=Vec2D(10.0, 10.0)
damping_cell_thickness=40
simulation_times=1000
n_receivers=101
```

サイズや時間ステップを大きくすると計算負荷が増えます。

### box constraint

Saltモデルでは `load_salt_model()` 内で固定されています。

```python
vmin, vmax = 1.5, 4.5
```

これがbox constraintの下限・上限として使われ、結果ディレクトリ名にも入ります。

Overthrustモデルを使う場合は `load_overthrust_model()` 側の値を使う想定です。

```python
vmin, vmax = 2.1, 6.0
```

## 短い動作確認

フル実験は重いので、動作確認だけならPythonから小さい設定で呼び出すと便利です。

```bash
MPLCONFIGDIR=.matplotlib-cache poetry run python -c "import runpy; ns=runpy.run_path('src/box-TV-constrained-FWI.py'); ns['num_parallels']=1; ns['run_alpha_experiments'](alphas=(0, 150), max_n_iters=1, n_shots=1, noise_sigma=0.0, result_root_path=ns['Path']('results_test'), image_name='salt_test')"
```

この実行では `results_test/` に小さな結果が保存されます。`results_test/` はGit管理外です。

## 解析例

保存された `.npz` と `metrics.csv` はPythonで直接読み込めます。

```python
from pathlib import Path
import numpy as np
import pandas as pd

result_dir = Path("results/20260602_190710_salt_gradient_alpha-0_noise-1_box-1p5-4p5")
data = np.load(result_dir / f"{result_dir.name}.npz")
metrics = pd.read_csv(result_dir / "metrics.csv")

final_v = data["final_velocity_model"]
observed = data["observed_seismic_data"]

print(final_v.shape)
print(observed.shape)
print(metrics.tail())
```

`pandas` は現在の `pyproject.toml` には含まれていません。必要であれば追加してください。`metrics.csv` は標準ライブラリの `csv` や `numpy.loadtxt` でも読み込めます。

## 注意点

- `datasets/`, `outputs/`, `results/`, `results_test/` はGit管理外です。
- 別PCでcloneした場合、`datasets/` は再ダウンロードが必要です。
- フル実験は重いです。READMEの環境例ではCPUが13th Gen Intel Core i9-13900K、メモリ32GBです。
- リソース不足の場合は、まず `num_parallels` と `n_shots` を下げてください。
- `alpha=0` はTV制約なし比較のため `gradient` で実行されます。`pds` の `alpha=0` ではありません。
