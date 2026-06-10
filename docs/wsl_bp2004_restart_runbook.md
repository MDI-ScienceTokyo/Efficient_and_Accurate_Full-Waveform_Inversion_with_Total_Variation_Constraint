# WSL再起動後にBP2004を動かす手順

このメモは、VS CodeでWSLに接続し直したあと、BP2004実験を再開するための手順です。

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
```

すでに以下があれば、ダウンロードと前処理はやり直さなくてよいです。

```text
data/raw/bp2004/vel_z6.25m_x12.5m_exact.segy
data/processed/bp2004/bp2004_exact_crop.npz
data/processed/bp2004/bp2004_initial_crop.npz
```

足りない場合だけ、順番に作り直します。

```bash
.venv/bin/python scripts/download_bp2004.py --decompress
.venv/bin/python scripts/preprocess_bp2004.py --config configs/bp2004_gradient.yaml
.venv/bin/python scripts/make_bp2004_initial_model.py --config configs/bp2004_gradient.yaml
```

## 6. BP2004 gradientを回す

`--alphas 0` にするとTV制約なしのgradient版として動きます。現在のBP2004ドライバは既存の提案法コードの更新経路を使うため、速度のbox制約は有効です。

```bash
export MPLCONFIGDIR=.matplotlib-cache

DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 0 \
  --noise-sigmas 0,0.01 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --result-root-path results/bp2004
```

## 7. BP2004 TV + box constraintを回す

`--alphas` を正の値にするとTV近接ステップを含む提案法として動きます。

```bash
export MPLCONFIGDIR=.matplotlib-cache

DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 500 \
  --noise-sigmas 0,0.01 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --gamma2 100000 \
  --result-root-path results/bp2004
```

BP2004では `gamma1=1e-6` に対して `gamma2=100` だとTV制約の効きがかなり遅くなります。`gamma2=100000` 程度に上げるとTV制約が効きやすくなります。安定条件はコード側で `gamma1 * gamma2 * 8 < 1` として確認されます。

## 8. 実行中の進捗を見る

このドライバは各iterationで目的関数、モデル誤差、PSNR、SSIM、TVを標準出力に表示します。ログを残したい場合は `tee` を使います。

```bash
DEVITO_LOGGING=ERROR .venv/bin/python src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 0 \
  --noise-sigmas 0 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --result-root-path results/bp2004 2>&1 | tee results/bp2004_gradient_noise0.log
```

バックグラウンド実行する場合は `-u` と `tail -f` を使うと進捗を確認しやすいです。

```bash
DEVITO_LOGGING=ERROR .venv/bin/python -u src/box-TV-constrained-FWI-BP2004.py \
  --max-n-iters 5000 \
  --alphas 0 \
  --noise-sigmas 0 \
  --n-shots 5 \
  --n-receivers 201 \
  --gamma1 1e-6 \
  --result-root-path results/bp2004 \
  > results/bp2004_gradient_noise0.log 2>&1 &

tail -f results/bp2004_gradient_noise0.log
```

## 9. 実行中プロセスを確認・停止する

確認:

```bash
ps -f -u $(whoami) | grep box-TV-constrained-FWI-BP2004.py
```

停止:

```bash
pkill -f box-TV-constrained-FWI-BP2004.py
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

各実験は `--result-root-path` 以下にタイムスタンプ付きディレクトリとして保存されます。ディレクトリ名にはBP2004、alpha、noise、box範囲が入ります。
