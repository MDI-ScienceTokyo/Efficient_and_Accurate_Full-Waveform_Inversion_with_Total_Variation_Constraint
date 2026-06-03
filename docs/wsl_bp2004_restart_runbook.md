# WSL再起動後にBP2004を動かす手順

このメモは、VS CodeでWSLに接続し直したあと、BP2004のgradient実験を再開するための手順です。

## 1. WSLを止めて再起動する

Windows側のPowerShellで実行します。

```powershell
wsl --shutdown
```

その後、VS Codeからもう一度WSLに接続します。

```text
VS Code -> Remote Explorer -> WSL -> 対象ディストリビューションへ接続
```

これでWSL内で走っていたPython/Devitoのバックグラウンド計算も基本的に止まります。

## 2. リポジトリへ移動する

VS CodeのWSLターミナルで実行します。

```bash
cd /projects/Efficient_and_Accurate_Full-Waveform_Inversion_with_Total_Variation_Constraint
```

## 3. 環境確認

まず、既存の仮想環境 `.venv` が使えるか確認します。

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib, obspy, devito; print('venv deps ok')"
```

`venv deps ok` と出れば、Poetryを入れ直す必要はありません。このリポジトリでは `.venv/bin/python` を直接使えばBP2004実験を動かせます。

Matplotlibがホームディレクトリにキャッシュを書けない警告を出すことがあります。実行時は以下を付けると避けられます。

```bash
export MPLCONFIGDIR=.matplotlib-cache
```

## 4. Poetryは入れ直すべきか

通常は不要です。

このワークスペースにはすでに `.venv/` があり、BP2004に必要な依存も入っています。したがって、再起動後は `.venv/bin/python` で実行すれば十分です。

Poetryが必要になるのは、次のような場合だけです。

- `.venv/` が消えた
- `.venv/` の依存が壊れた
- `pyproject.toml` / `poetry.lock` から環境を作り直したい

その場合は、Poetryをインストールしてから環境を作り直します。

```bash
python3 -m pip install --user poetry
~/.local/bin/poetry install
```

もしPoetryを使わずに復旧するなら、最低限は次のように入れます。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy matplotlib obspy devito scikit-image tqdm requests pillow invoke joblib colorama pytest
```

`segyio` は入っていなくても、現在のBP2004前処理は `obspy` fallback で動きます。

## 5. BP2004データがあるか確認

```bash
ls -lh data/raw/bp2004
ls -lh data/processed/bp2004
ls -lh outputs/bp2004_gradient
```

すでに以下があれば、ダウンロードと前処理はやり直さなくてよいです。

```text
data/raw/bp2004/vel_z6.25m_x12.5m_exact.segy
data/processed/bp2004/bp2004_exact_crop.npz
data/processed/bp2004/bp2004_initial_crop.npz
outputs/bp2004_gradient/observed_data.npz
```

足りない場合だけ、順番に作り直します。

```bash
.venv/bin/python scripts/download_bp2004.py --decompress
.venv/bin/python scripts/preprocess_bp2004.py --config configs/bp2004_gradient.yaml
.venv/bin/python scripts/make_bp2004_initial_model.py --config configs/bp2004_gradient.yaml
.venv/bin/python scripts/run_bp2004_forward.py --config configs/bp2004_gradient.yaml
```

## 6. ノイズあり観測データを作る

ノイズなし観測データは通常 `outputs/bp2004_gradient/observed_data.npz` です。

ノイズありを作る場合:

```bash
export MPLCONFIGDIR=.matplotlib-cache

.venv/bin/python scripts/run_bp2004_forward.py \
  --config configs/bp2004_gradient.yaml \
  --noise-sigma 0.01 \
  --seed 0 \
  --output outputs/bp2004_gradient/observed_data_noise_sigma0p01.npz
```

## 7. 5000回実験を起動する

### ノイズなし

```bash
export MPLCONFIGDIR=.matplotlib-cache

DEVITO_LOGGING=ERROR .venv/bin/python -u scripts/run_bp2004_gradient_descent.py \
  --config configs/bp2004_gradient.yaml \
  --observed outputs/bp2004_gradient/observed_data.npz \
  --iterations 5000 \
  --gamma1 1e-2 \
  --output-dir outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000 \
  --progress-interval 1 \
  > outputs/bp2004_gradient/gradient_descent_noise0_iter5000.log 2>&1 &
```

### ノイズあり

```bash
export MPLCONFIGDIR=.matplotlib-cache

DEVITO_LOGGING=ERROR .venv/bin/python -u scripts/run_bp2004_gradient_descent.py \
  --config configs/bp2004_gradient.yaml \
  --observed outputs/bp2004_gradient/observed_data_noise_sigma0p01.npz \
  --iterations 5000 \
  --gamma1 1e-2 \
  --output-dir outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000 \
  --progress-interval 1 \
  > outputs/bp2004_gradient/gradient_descent_noise_sigma0p01_iter5000.log 2>&1 &
```

コマンドの最後の `&` は「バックグラウンドで実行する」という意味です。

実行直後に次のような表示が出ることがあります。

```text
[4] 177674
[5] 177675
```

これはエラーではありません。`[4]` や `[5]` はジョブ番号、`177674` などはプロセスIDです。

## 8. 実行中の進捗を見る

最新状態だけ見る:

```bash
cat outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/status.json
cat outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/status.json
```

CSVを追いかける:

```bash
tail -f outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/progress.csv
```

ログを追いかける:

```bash
tail -f outputs/bp2004_gradient/gradient_descent_noise0_iter5000.log
```

10秒ごとに最新状態を見る:

```bash
while true; do
  clear
  echo "noise=0"
  cat outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/status.json
  echo
  echo "noise_sigma=0.01"
  cat outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/status.json
  sleep 10
done
```

## 9. 実行中プロセスを確認・停止する

確認:

```bash
ps -f -u $(whoami) | grep run_bp2004_gradient_descent
```

停止:

```bash
pkill -f run_bp2004_gradient_descent.py
```

より慎重に止める場合は、`ps` でPIDを確認してから対象PIDだけ止めます。

```bash
kill <PID>
```

それでも止まらない場合だけ:

```bash
kill -9 <PID>
```

## 10. 完了後の出力

各 `--output-dir` に以下が保存されます。

```text
gradient_descent_result.npz
metrics.csv
progress.csv
status.json
objective_history.png
grad_norm_history.png
model_mse_history.png
vp_final.png
vp_final_minus_true.png
```

最終結果を簡単に見る:

```bash
cat outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise0_iter5000/status.json
cat outputs/bp2004_gradient/gradient_descent_no_tv_no_box_noise_sigma0p01_iter5000/status.json
```
