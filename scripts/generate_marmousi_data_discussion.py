from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lib.signal_processing.diff_operator as diff_op
from lib.signal_processing.norm import L12_norm


DOC_PATH = Path("docs/marmousi_data_discussion.md")
ASSET_DIR = Path("docs/assets")
PROCESSED_DIR = Path("datasets/marmousi/processed")


def latest(pattern: str) -> Path:
    candidates = sorted(PROCESSED_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No Marmousi processed files matched {pattern}. Run scripts/prepare_marmousi.py first.")
    return candidates[-1]


def load_marmousi() -> tuple[np.ndarray, np.ndarray, Path, Path]:
    true_path = latest("marmousi_vp_true_full_*.npy")
    init_path = latest("marmousi_vp_init_full_*.npy")
    true_model = np.load(true_path).astype(np.float32)
    initial_model = np.load(init_path).astype(np.float32)
    if float(np.nanmax(true_model)) > 20.0:
        true_model = true_model / 1000.0
    if float(np.nanmax(initial_model)) > 20.0:
        initial_model = initial_model / 1000.0
    return true_model, initial_model, true_path, init_path


def survey_geometry(shape: tuple[int, int], n_shots: int = 3, n_receivers: int = 101) -> dict[str, np.ndarray | float | int]:
    nz, nx = shape
    dz = 10.0
    dx = 10.0
    width_m = (nx - 1) * dx
    source_x = np.linspace(0.0, width_m, num=n_shots, dtype=np.float32)
    receiver_x = np.linspace(0.0, width_m, num=min(n_receivers, nx), dtype=np.float32)
    return {
        "dz": dz,
        "dx": dx,
        "width_m": width_m,
        "depth_m": (nz - 1) * dz,
        "source_depth_m": 30.0,
        "receiver_depth_m": 30.0,
        "source_x": source_x,
        "receiver_x": receiver_x,
        "n_shots": n_shots,
        "n_receivers": receiver_x.size,
    }


def save_figures(true_model: np.ndarray, initial_model: np.ndarray, geom: dict[str, np.ndarray | float | int]) -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    vmin = float(np.floor(float(true_model.min()) * 10) / 10)
    vmax = float(np.ceil(float(true_model.max()) * 10) / 10)
    width_km = float(geom["width_m"]) / 1000.0
    depth_km = float(geom["depth_m"]) / 1000.0
    extent = [0, width_km, depth_km, 0]

    assets: dict[str, Path] = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    im = axes[0].imshow(true_model, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
    axes[0].set_title("Marmousi: true velocity")
    axes[0].set_xlabel("x [km]")
    axes[0].set_ylabel("depth [km]")
    fig.colorbar(im, ax=axes[0], shrink=0.88, label="Velocity [km/s]")

    im = axes[1].imshow(true_model, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
    rx = np.asarray(geom["receiver_x"], dtype=np.float32) / 1000.0
    sx = np.asarray(geom["source_x"], dtype=np.float32) / 1000.0
    rz = np.full_like(rx, float(geom["receiver_depth_m"]) / 1000.0)
    sz = np.full_like(sx, float(geom["source_depth_m"]) / 1000.0)
    axes[1].scatter(rx, rz, s=18, c="#111111", marker="v", label="Receivers")
    axes[1].scatter(sx, sz, s=90, c="#ffd400", edgecolors="#111111", marker="*", label="Sources")
    axes[1].set_title("Marmousi: true + acquisition geometry")
    axes[1].set_xlabel("x [km]")
    axes[1].set_ylabel("depth [km]")
    axes[1].legend(loc="lower left", fontsize=8, framealpha=0.88)
    fig.colorbar(im, ax=axes[1], shrink=0.88, label="Velocity [km/s]")
    path = ASSET_DIR / "marmousi_true_and_geometry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    assets["geometry"] = path

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for ax, model, title in zip(axes, (true_model, initial_model), ("True", "Initial")):
        im = ax.imshow(model, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
        ax.set_title(f"Marmousi: {title}")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("depth [km]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label="Velocity [km/s]")
    path = ASSET_DIR / "marmousi_true_initial.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    assets["initial"] = path

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    diff = initial_model - true_model
    lim = float(np.max(np.abs(diff)))
    im = ax.imshow(diff, cmap="seismic", vmin=-lim, vmax=lim, extent=extent, aspect="auto")
    ax.set_title("Marmousi: initial - true")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("depth [km]")
    fig.colorbar(im, ax=ax, shrink=0.88, label="Velocity difference [km/s]")
    path = ASSET_DIR / "marmousi_initial_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    assets["initial_error"] = path
    return assets


def rel(path: Path) -> str:
    return path.relative_to(DOC_PATH.parent).as_posix()


def write_markdown(
    true_model: np.ndarray,
    initial_model: np.ndarray,
    true_path: Path,
    init_path: Path,
    geom: dict[str, np.ndarray | float | int],
    assets: dict[str, Path],
) -> None:
    nz, nx = true_model.shape
    raw_shape = (2801, 13601)
    raw_spacing_m = 1.25
    raw_depth_m = (raw_shape[0] - 1) * raw_spacing_m
    raw_width_m = (raw_shape[1] - 1) * raw_spacing_m
    preserving_aspect_spacing_z = raw_depth_m / (nz - 1)
    preserving_aspect_spacing_x = raw_width_m / (nx - 1)
    driver_dz = float(geom["dz"])
    driver_dx = float(geom["dx"])
    width_m = float(geom["width_m"])
    depth_m = float(geom["depth_m"])
    source_x = np.asarray(geom["source_x"], dtype=np.float32)
    receiver_x = np.asarray(geom["receiver_x"], dtype=np.float32)
    rmse_init = float(np.sqrt(np.mean((initial_model - true_model) ** 2)))
    tv_true = float(L12_norm(diff_op.D(true_model)))
    tv_init = float(L12_norm(diff_op.D(initial_model)))
    vmin_box = float(np.floor(float(true_model.min()) * 10) / 10)
    vmax_box = float(np.ceil(float(true_model.max()) * 10) / 10)

    lines: list[str] = []
    lines.append("# Marmousi データ考察")
    lines.append("")
    lines.append("## 概要")
    lines.append("")
    lines.append("`src/box-TV-constrained-FWI-marmousi.py` で使う Marmousi データについて、Salt/BP2004 と同じ観点で整理する。ここでの Marmousi は `scripts/prepare_marmousi.py` により、公開 elastic Marmousi model の P-wave velocity を full extent のまま縦 101 cell に縮小したもの。")
    lines.append("")
    lines.append(f"![Marmousi true and acquisition geometry]({rel(assets['geometry'])})")
    lines.append("")
    lines.append(f"![Marmousi true and initial]({rel(assets['initial'])})")
    lines.append("")
    lines.append(f"![Marmousi initial error]({rel(assets['initial_error'])})")
    lines.append("")
    lines.append("## 基本データ")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|---|---:|")
    lines.append(f"| raw VP shape [z, x] | `{raw_shape}` |")
    lines.append(f"| raw grid spacing | `1.25 m` |")
    lines.append(f"| raw physical size [depth, width] | `{raw_depth_m:g} m x {raw_width_m:g} m` |")
    lines.append(f"| processed true shape [z, x] | `{true_model.shape}` |")
    lines.append(f"| aspect-preserving equivalent spacing [z, x] | `{preserving_aspect_spacing_z:.3f} m x {preserving_aspect_spacing_x:.3f} m` |")
    lines.append(f"| FWI driver spacing [z, x] | `{driver_dz:g} m x {driver_dx:g} m` |")
    lines.append(f"| FWI driver physical size [depth, width] | `{depth_m:g} m x {width_m:g} m` |")
    lines.append(f"| processed cells | `{true_model.size:,}` |")
    lines.append(f"| true velocity range | `{true_model.min():.3f} - {true_model.max():.3f} km/s` |")
    lines.append(f"| initial velocity range | `{initial_model.min():.3f} - {initial_model.max():.3f} km/s` |")
    lines.append(f"| box constraint in driver | `{vmin_box:.1f} - {vmax_box:.1f} km/s` |")
    lines.append(f"| initial RMSE | `{rmse_init:.6f} km/s` |")
    lines.append(f"| true TV | `{tv_true:.3f}` |")
    lines.append(f"| initial TV | `{tv_init:.3f}` |")
    lines.append(f"| true file | `{true_path.as_posix()}` |")
    lines.append(f"| initial file | `{init_path.as_posix()}` |")
    lines.append("")
    lines.append("## Acquisition / FWI 設定")
    lines.append("")
    lines.append("| 項目 | 値 |")
    lines.append("|---|---:|")
    lines.append(f"| default shots | `{int(geom['n_shots'])}` |")
    lines.append(f"| default receivers | `{int(geom['n_receivers'])}` |")
    lines.append(f"| source x range | `{float(source_x[0]):g} - {float(source_x[-1]):g} m` |")
    lines.append(f"| receiver x range | `{float(receiver_x[0]):g} - {float(receiver_x[-1]):g} m` |")
    lines.append(f"| source spacing | `{float(np.mean(np.diff(source_x))):.3f} m` |")
    lines.append(f"| receiver spacing | `{float(np.mean(np.diff(receiver_x))):.3f} m` |")
    lines.append(f"| source depth | `{float(geom['source_depth_m']):g} m` |")
    lines.append(f"| receiver depth | `{float(geom['receiver_depth_m']):g} m` |")
    lines.append("| damping cell thickness | `40` |")
    lines.append("| simulation time steps | `1000` |")
    lines.append("| source frequency | `0.01` |")
    lines.append("| source peak time | `100` |")
    lines.append("| num_parallels | `4` |")
    lines.append("")
    lines.append("## Salt / BP2004 との位置づけ")
    lines.append("")
    lines.append("| 項目 | Salt | Marmousi | BP2004 |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| 実験用 shape [z, x] | `50 x 100` | `101 x 490` | `241 x 801` |")
    lines.append("| 実験セル数 | `5,000` | `49,490` | `193,041` |")
    lines.append("| driver spacing [m] | `10 x 10` | `10 x 10` | `25 x 25` |")
    lines.append("| driver size [depth x width] | `0.49 km x 0.99 km` | `1.00 km x 4.89 km` | `6.00 km x 20.00 km` |")
    lines.append("| default shots | `20` | `3` | `5` |")
    lines.append("| default receivers | `101` | `101` | `201` |")
    lines.append("| source/receiver depth | `30 m` | `30 m` | `25 m` |")
    lines.append("| velocity range [km/s] | `about 1.4-4.8 after zoom` | `1.028-4.700` | `1.486-4.790` |")
    lines.append("")
    lines.append("## 考察")
    lines.append("")
    lines.append("- Marmousi は Salt より約 9.9 倍セル数が多く、BP2004 よりは約 1/3.9 のセル数。Salt よりは本格的だが、BP2004 より軽い中間的なベンチマークとして使いやすい。")
    lines.append("- Marmousi は横長で薄いモデルなので、横方向の反射・層構造の連続性が支配的。Salt のような単一の大きな salt body というより、細かい層構造・褶曲・断層的な不連続をどれだけ保てるかを見るデータ。")
    lines.append("- default は 3 shots / 101 receivers なので、BP2004 よりさらに shot 数が少ない。FWI の照明はかなり限られ、深部や端部の再構成は source 配置に強く依存する。")
    lines.append("- `prepare_marmousi.py` は raw の物理スケールを保って縦 101 cell に縮小しているが、FWI driver は spacing を `10 m x 10 m` と固定している。raw Marmousi の 1.25 m grid から full extent を縮小したと考えると等価 spacing は約 35 m なので、現在の driver 上の物理サイズは元データより約 3.5 倍小さく扱われている。この点は波動伝播条件、source frequency、走時解釈に効くため、物理スケールを厳密に議論する場合は spacing を見直すべき。")
    lines.append("- initial model は Gaussian smoothing sigma=8 pixel で作られており、層境界の細かい構造はかなり消える。一方で全体の低速/高速トレンドは残るため、TV 制約の効果を見るには妥当な初期値。")
    lines.append("- box constraint は true の min/max を 0.1 km/s 刻みに丸めた `1.0-4.7 km/s`。Salt より低速域が広く、浅部低速層の再構成が評価に入りやすい。")
    lines.append("- TV 制約では、Marmousi の細かい層状構造を過度に平滑化しない alpha 選びが重要。Salt で有効な強めの TV は、Marmousi では層構造を潰す可能性がある。")
    lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    true_model, initial_model, true_path, init_path = load_marmousi()
    geom = survey_geometry(true_model.shape)
    assets = save_figures(true_model, initial_model, geom)
    write_markdown(true_model, initial_model, true_path, init_path, geom, assets)
    print(f"Wrote {DOC_PATH}")
    for path in assets.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
