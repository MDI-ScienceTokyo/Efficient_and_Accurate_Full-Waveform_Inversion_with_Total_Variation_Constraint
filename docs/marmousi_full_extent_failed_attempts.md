# Marmousi full-extent FWI 試行メモ

## 結論

Marmousi を「縮小版ではなく full physical extent」で回すために、raw Marmousi から full extent を保った 10 m / 20 m grid の前処理データと、新しい実行ドライバを作った。

ただし、現状の gradient baseline は安定には回っていない。

- 10 m grid は Devito の wavefield memory が重すぎて swap 警告が出る。
- 20 m grid に落としても、`n_shots=10`, `n_receivers=201`, `simulation_times=1000` では gradient 更新後に速度が非正になり、3 iteration で停止する。
- `alpha=0` の gradient baseline は box projection を通らないため、速度が負になった時点で破綻する。

したがって、今回の状態では **full-extent Marmousi の gradient baseline は成功していない**。

## 追加したもの

### 前処理

新規ファイル:

```text
scripts/prepare_marmousi_full_extent.py
```

raw Marmousi VP:

```text
shape: 2801 x 13601
spacing: 1.25 m x 1.25 m
physical extent: about 3.5 km x 17 km
```

これを full physical extent を保ったまま resample した。

生成済みデータ:

```text
datasets/marmousi/processed/marmousi_full_extent_full_extent_351x1701_10m.npz
datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz
```

### 実行ドライバ

新規ファイル:

```text
src/box-TV-constrained-FWI-marmousi-full.py
```

主な特徴:

- `.npz` の `vp_true`, `vp0`, `dx`, `dz` を読む。
- `--n-shots`, `--n-receivers`, `--model-path`, `--alphas`, `--alpha-scales` を CLI から指定できる。
- `--alphas 0` で gradient baseline。
- `--alpha-scales 0.8` などで TV + box の PDS 実験。

## 試行結果

| 条件 | grid | shots / receivers | simulation times | gamma1 | max iters | completed | 結果 |
|---|---:|---:|---:|---:|---:|---:|---|
| smoke gradient | 10 m, 351 x 1701 | 1 / 21 | 50 | `1e-6` | 1 | 1 | 成功 |
| smoke TV + box | 10 m, 351 x 1701 | 1 / 21 | 50 | `1e-6` | 1 | 1 | 成功 |
| gradient 本番寄り | 10 m, 351 x 1701 | 10 / 201 | 1000 | `1e-6` | 5000 | 3 | swap 警告 + 負速度で停止 |
| smoke gradient | 20 m, 176 x 851 | 2 / 51 | 500 | `1e-8` | 1 | 1 | 成功 |
| gradient test | 20 m, 176 x 851 | 10 / 201 | 1000 | `1e-8` | 10 | 3 | 負速度で停止 |
| gradient test | 20 m, 176 x 851 | 10 / 201 | 1000 | `1e-6` | 10 | 3 | 大きく負速度で停止 |

## 代表ログ

### 10 m full extent, gradient, gamma1=1e-6

実行条件:

```text
model: datasets/marmousi/processed/marmousi_full_extent_full_extent_351x1701_10m.npz
shape: 351 x 1701
n_shots: 10
n_receivers: 201
simulation_times: 1000
gamma1: 1e-6
```

ログ上の問題:

```text
Trying to allocate more memory for symbol u than available on physical device, this will start swapping
stopped: velocity model became non-positive after iteration 3; min_velocity=-3.96068 km/s
```

保存結果:

```text
results/marmousi_full/20260605_165941_marmousi_full_gradient_alpha-0_noise-0_box-1-4p7
```

### 20 m full extent, gradient, gamma1=1e-6

実行条件:

```text
model: datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz
shape: 176 x 851
n_shots: 10
n_receivers: 201
simulation_times: 1000
gamma1: 1e-6
```

ログ上の問題:

```text
stopped: velocity model became non-positive after iteration 3; min_velocity=-347.145 km/s
```

保存結果:

```text
results/marmousi_full_20m_test/20260605_173507_marmousi_full_gradient_alpha-0_noise-0_box-1-4p7
```

### 20 m full extent, gradient, gamma1=1e-8

`gamma1` を 100 分の 1 にしても、本番寄り条件では 3 iteration で停止した。

保存結果:

```text
results/marmousi_full_20m_test/20260605_173425_marmousi_full_gradient_alpha-0_noise-0_box-1-4p7
```

最終速度範囲:

```text
min: -0.0953 km/s
max: 4.1639 km/s
```

## うまく回らなかった理由

### 1. 10 m full extent はメモリが重い

10 m 版は `351 x 1701` cells あり、damping 付きではさらに大きくなる。Devito は forward / adjoint の wavefield を時間方向にも保持するため、`simulation_times=1000` と `n_shots=10` では wavefield memory が大きい。

実際に Devito から swap 警告が出た。

```text
Trying to allocate more memory for symbol u than available on physical device
```

### 2. Gradient baseline は box constraint を使わない

`alpha=0` の場合、コードは `algorithm="gradient"` になり、更新は単純に:

```python
v = v - gamma1 * grad
```

だけになる。

TV + box の PDS 経路では:

```python
prox_box_constraint(...)
```

が入るが、gradient baseline では入らない。したがって、更新が大きいと速度が `0 km/s` 以下になり、その時点で停止する。

### 3. Marmousi full extent では gradient scale が Salt/BP2004 と違う

Marmousi full extent は横幅が約 17 km あり、構造も細かい。`n_shots=10`, `simulation_times=1000` にすると目的関数と gradient のスケールが大きくなり、Salt や小さい smoke test で動いた `gamma1` がそのまま使えない。

20 m 版でも `gamma1=1e-6` は明らかに大きすぎた。`gamma1=1e-8` でもまだ 3 iteration で負速度になったため、単純な固定 step gradient baseline はかなり繊細。

## 現時点の判断

Marmousi full extent で `n_shots=10`, `n_receivers=201` を使う方向性自体は妥当。ただし、現在の gradient baseline 実装をそのまま使うと安定に長時間回らない。

特に、`alpha=0` を gradient baseline として比較したい場合、以下のどれかが必要になる。

- `gamma1` をさらに小さくする。例: `1e-9`, `1e-10`, `1e-11`
- gradient baseline にも box projection を入れる。
- line search / backtracking を入れて、非正速度になる更新を避ける。
- velocity の下限を割ったら step を縮めて retry する。
- simulation time / shot 数を段階的に増やす continuation にする。

## 次に試すなら

まずは原因切り分けとして、20 m 版でかなり小さい step から試す。

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache DEVITO_LOGGING=ERROR OMP_NUM_THREADS=1 \
.venv/bin/python src/box-TV-constrained-FWI-marmousi-full.py \
  --model-path datasets/marmousi/processed/marmousi_full_extent_full_extent_176x851_20m.npz \
  --max-n-iters 10 \
  --n-shots 10 \
  --n-receivers 201 \
  --alphas 0 \
  --noise-sigma 0 \
  --gamma1 1e-10 \
  --simulation-times 1000 \
  --num-parallel-workers 1 \
  --result-root-path results/marmousi_full_20m_gamma_test
```

ただし、単に `gamma1` を下げるだけでは収束が非常に遅くなる可能性が高い。実験としてきれいに比較したいなら、gradient baseline 側にも box projection か line search を入れるのが現実的。

## まとめ

今回やったことは、Marmousi を縮小版ではなく full physical extent に近い形で動かすための準備だった。

前処理と実行コードは作成でき、smoke test も通った。しかし、本番に近い `n_shots=10`, `n_receivers=201`, `simulation_times=1000` では、gradient baseline が 3 iteration で破綻した。

そのため、現時点では **Marmousi full extent の gradient baseline は「環境・データ準備はできたが、数値的に安定して回せていない」** という結論。
