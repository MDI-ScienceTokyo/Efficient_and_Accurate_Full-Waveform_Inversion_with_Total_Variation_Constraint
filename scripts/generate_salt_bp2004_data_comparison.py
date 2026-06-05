from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.dataset import load_seismic_datasets__salt_model
from lib.misc import datasets_root_path
from lib.signal_processing.misc import smoothing_with_gaussian_filter, zoom_and_crop


DOC_PATH = Path("docs/salt_bp2004_data_comparison.md")
ASSET_DIR = Path("docs/assets")


@dataclass
class Survey:
    name: str
    true_model: np.ndarray
    initial_model: np.ndarray
    raw_shape: tuple[int, ...]
    raw_spacing_m: tuple[float, ...]
    raw_extent_m: tuple[float, ...]
    experiment_shape: tuple[int, int]
    experiment_spacing_m: tuple[float, float]
    velocity_min_kms: float
    velocity_max_kms: float
    n_shots: int
    n_receivers: int
    source_depth_m: float
    receiver_depth_m: float
    source_x_m: np.ndarray
    receiver_x_m: np.ndarray
    damping_cell_thickness: int
    simulation_times: int
    source_frequency: float
    source_peek_time: float
    notes: str

    @property
    def width_m(self) -> float:
        return (self.experiment_shape[1] - 1) * self.experiment_spacing_m[1]

    @property
    def depth_m(self) -> float:
        return (self.experiment_shape[0] - 1) * self.experiment_spacing_m[0]

    @property
    def source_spacing_m(self) -> float | None:
        if self.source_x_m.size <= 1:
            return None
        return float(np.mean(np.diff(self.source_x_m)))

    @property
    def receiver_spacing_m(self) -> float | None:
        if self.receiver_x_m.size <= 1:
            return None
        return float(np.mean(np.diff(self.receiver_x_m)))


def load_salt_survey() -> Survey:
    raw_path = datasets_root_path / "salt_and_overthrust_models/3-D_Salt_Model/VEL_GRIDS/Saltf@@"
    raw = load_seismic_datasets__salt_model(raw_path).transpose((1, 0, 2)).astype(np.float32) / 1000.0
    target_idx = 300
    true_model = zoom_and_crop(raw[target_idx], (50, 100))
    initial_model = zoom_and_crop(smoothing_with_gaussian_filter(raw[target_idx], 1, 80), (50, 100))
    width = (100 - 1) * 10.0
    source_x = np.linspace(0.0, width, num=20, dtype=np.float32)
    receiver_x = np.linspace(0.0, width, num=101, dtype=np.float32)
    return Survey(
        name="Salt",
        true_model=true_model,
        initial_model=initial_model,
        raw_shape=(210, 676, 676),
        raw_spacing_m=(20.0, 20.0, 20.0),
        raw_extent_m=((210 - 1) * 20.0, (676 - 1) * 20.0, (676 - 1) * 20.0),
        experiment_shape=true_model.shape,
        experiment_spacing_m=(10.0, 10.0),
        velocity_min_kms=float(np.min(raw)),
        velocity_max_kms=float(np.max(raw)),
        n_shots=20,
        n_receivers=101,
        source_depth_m=30.0,
        receiver_depth_m=30.0,
        source_x_m=source_x,
        receiver_x_m=receiver_x,
        damping_cell_thickness=40,
        simulation_times=1000,
        source_frequency=0.01,
        source_peek_time=100.0,
        notes="SEG/EAGE 3-D Salt model の y-index 300 断面を取り、100 x 50 cells に zoom/crop して使用。",
    )


def load_bp2004_survey() -> Survey:
    data = np.load("data/processed/bp2004/bp2004_initial_crop.npz", allow_pickle=True)
    true_model = data["vp_true"].astype(np.float32) / 1000.0
    initial_model = data["vp0"].astype(np.float32) / 1000.0
    dz = float(data["dz"])
    dx = float(data["dx"])
    width = (true_model.shape[1] - 1) * dx
    margin = min(1000.0, max(0.0, width / 3.0))
    source_x = np.linspace(margin, width - margin, num=5, dtype=np.float32)
    receiver_x = np.linspace(0.0, width, num=201, dtype=np.float32)
    meta = data["meta"].item()
    return Survey(
        name="BP2004",
        true_model=true_model,
        initial_model=initial_model,
        raw_shape=tuple(meta["original_shape"]),
        raw_spacing_m=tuple(meta["original_spacing_m"]),
        raw_extent_m=((1911 - 1) * 6.25, (5395 - 1) * 12.5),
        experiment_shape=true_model.shape,
        experiment_spacing_m=(dz, dx),
        velocity_min_kms=float(np.min(true_model)),
        velocity_max_kms=float(np.max(true_model)),
        n_shots=5,
        n_receivers=201,
        source_depth_m=25.0,
        receiver_depth_m=25.0,
        source_x_m=source_x,
        receiver_x_m=receiver_x,
        damping_cell_thickness=40,
        simulation_times=999,
        source_frequency=0.005,
        source_peek_time=100.0,
        notes="BP2004 exact velocity model を x=25-45 km, z=0-6 km で crop し、25 m grid に resample して使用。",
    )


def cell_coords(x_m: np.ndarray, z_m: float, survey: Survey) -> tuple[np.ndarray, np.ndarray]:
    x = x_m / survey.experiment_spacing_m[1]
    z = np.full_like(x, z_m / survey.experiment_spacing_m[0], dtype=np.float32)
    return x, z


def plot_true(ax: plt.Axes, survey: Survey, show_title: str) -> None:
    im = ax.imshow(
        survey.true_model,
        cmap="coolwarm",
        vmin=survey.velocity_min_kms,
        vmax=survey.velocity_max_kms,
        extent=[0, survey.width_m / 1000.0, survey.depth_m / 1000.0, 0],
        aspect="auto",
    )
    ax.set_title(show_title)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("depth [km]")
    return im


def add_geometry(ax: plt.Axes, survey: Survey) -> None:
    rx, rz = cell_coords(survey.receiver_x_m, survey.receiver_depth_m, survey)
    sx, sz = cell_coords(survey.source_x_m, survey.source_depth_m, survey)
    ax.scatter(rx * survey.experiment_spacing_m[1] / 1000.0, rz * survey.experiment_spacing_m[0] / 1000.0, s=16, c="#111111", marker="v", label="Receivers")
    ax.scatter(sx * survey.experiment_spacing_m[1] / 1000.0, sz * survey.experiment_spacing_m[0] / 1000.0, s=70, c="#ffd400", edgecolors="#111111", marker="*", label="Sources")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.88)


def save_figures(salt: Survey, bp2004: Survey) -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, survey in zip(axes, (salt, bp2004)):
        im = plot_true(row[0], survey, f"{survey.name}: true velocity")
        fig.colorbar(im, ax=row[0], shrink=0.84, label="Velocity [km/s]")
        im = plot_true(row[1], survey, f"{survey.name}: true + acquisition geometry")
        add_geometry(row[1], survey)
        fig.colorbar(im, ax=row[1], shrink=0.84, label="Velocity [km/s]")
    path = ASSET_DIR / "salt_bp2004_true_and_geometry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    assets["combined"] = path

    for survey in (salt, bp2004):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        im = plot_true(axes[0], survey, f"{survey.name}: true velocity")
        fig.colorbar(im, ax=axes[0], shrink=0.84, label="Velocity [km/s]")
        im = plot_true(axes[1], survey, f"{survey.name}: true + geometry")
        add_geometry(axes[1], survey)
        fig.colorbar(im, ax=axes[1], shrink=0.84, label="Velocity [km/s]")
        path = ASSET_DIR / f"{survey.name.lower()}_true_and_geometry.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        assets[survey.name.lower()] = path

    return assets


def fmt_tuple(values: tuple[float, ...]) -> str:
    return " x ".join(f"{v:g}" for v in values)


def fmt_opt(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def rel(path: Path) -> str:
    return path.relative_to(DOC_PATH.parent).as_posix()


def write_markdown(salt: Survey, bp2004: Survey, assets: dict[str, Path]) -> None:
    lines: list[str] = []
    lines.append("# Salt と BP2004 データ比較")
    lines.append("")
    lines.append("## 概要")
    lines.append("")
    lines.append("このメモは、現在の FWI 実験コードで使っている Salt model と BP2004 model の違いをまとめたもの。ここでの Salt は `src/box-TV-constrained-FWI.py` の標準設定、BP2004 は `src/box-TV-constrained-FWI-BP2004.py` と直近の BP2004 実験設定を基準にしている。")
    lines.append("")
    lines.append(f"![Salt and BP2004 true models with acquisition geometry]({rel(assets['combined'])})")
    lines.append("")
    lines.append("## True 画像と観測配置")
    lines.append("")
    lines.append(f"![Salt true and geometry]({rel(assets['salt'])})")
    lines.append("")
    lines.append(f"![BP2004 true and geometry]({rel(assets['bp2004'])})")
    lines.append("")
    lines.append("## 基本データ")
    lines.append("")
    lines.append("| 項目 | Salt | BP2004 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 元データ shape | `{salt.raw_shape}` | `{bp2004.raw_shape}` |")
    lines.append(f"| 元データ grid spacing [m] | `{fmt_tuple(salt.raw_spacing_m)}` | `{fmt_tuple(bp2004.raw_spacing_m)}` |")
    lines.append(f"| 元データ物理サイズ [m] | `{fmt_tuple(salt.raw_extent_m)}` | `{fmt_tuple(bp2004.raw_extent_m)}` |")
    lines.append(f"| 実験用 true shape [z, x] | `{salt.experiment_shape}` | `{bp2004.experiment_shape}` |")
    lines.append(f"| 実験用 grid spacing [z, x] [m] | `{fmt_tuple(salt.experiment_spacing_m)}` | `{fmt_tuple(bp2004.experiment_spacing_m)}` |")
    lines.append(f"| 実験用物理サイズ [depth, width] [m] | `{salt.depth_m:g} x {salt.width_m:g}` | `{bp2004.depth_m:g} x {bp2004.width_m:g}` |")
    lines.append(f"| 実験セル数 | `{salt.true_model.size:,}` | `{bp2004.true_model.size:,}` |")
    lines.append(f"| true velocity range [km/s] | `{np.min(salt.true_model):.3f} - {np.max(salt.true_model):.3f}` | `{np.min(bp2004.true_model):.3f} - {np.max(bp2004.true_model):.3f}` |")
    lines.append(f"| initial velocity range [km/s] | `{np.min(salt.initial_model):.3f} - {np.max(salt.initial_model):.3f}` | `{np.min(bp2004.initial_model):.3f} - {np.max(bp2004.initial_model):.3f}` |")
    lines.append(f"| box constraint [km/s] | `1.5 - 4.5` | `1.45 - 5.5` |")
    lines.append(f"| initial model | y-index 300 断面を Gaussian smoothing, then zoom/crop | crop 後 true を Gaussian smoothing, sigma z=20, x=40 |")
    lines.append("")
    lines.append("## Acquisition / FWI 設定")
    lines.append("")
    lines.append("| 項目 | Salt | BP2004 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| shots | `{salt.n_shots}` | `{bp2004.n_shots}` |")
    lines.append(f"| receivers | `{salt.n_receivers}` | `{bp2004.n_receivers}` |")
    lines.append(f"| source x range [m] | `{float(salt.source_x_m[0]):g} - {float(salt.source_x_m[-1]):g}` | `{float(bp2004.source_x_m[0]):g} - {float(bp2004.source_x_m[-1]):g}` |")
    lines.append(f"| receiver x range [m] | `{float(salt.receiver_x_m[0]):g} - {float(salt.receiver_x_m[-1]):g}` | `{float(bp2004.receiver_x_m[0]):g} - {float(bp2004.receiver_x_m[-1]):g}` |")
    lines.append(f"| source spacing [m] | `{fmt_opt(salt.source_spacing_m)}` | `{fmt_opt(bp2004.source_spacing_m)}` |")
    lines.append(f"| receiver spacing [m] | `{fmt_opt(salt.receiver_spacing_m)}` | `{fmt_opt(bp2004.receiver_spacing_m)}` |")
    lines.append(f"| source depth [m] | `{salt.source_depth_m:g}` | `{bp2004.source_depth_m:g}` |")
    lines.append(f"| receiver depth [m] | `{salt.receiver_depth_m:g}` | `{bp2004.receiver_depth_m:g}` |")
    lines.append(f"| damping cell thickness | `{salt.damping_cell_thickness}` | `{bp2004.damping_cell_thickness}` |")
    lines.append(f"| simulation time steps | `{salt.simulation_times}` | `{bp2004.simulation_times}` |")
    lines.append(f"| source frequency | `{salt.source_frequency:g}` | `{bp2004.source_frequency:g}` |")
    lines.append(f"| source peak time | `{salt.source_peek_time:g}` | `{bp2004.source_peek_time:g}` |")
    lines.append("")
    lines.append("## 特徴と違い")
    lines.append("")
    lines.append("- Salt は元データが 3-D cube で、コードでは y-index 300 の 2-D 断面をかなり小さい `100 x 50` grid に切り出して使う。計算は軽く、アルゴリズム検証やパラメータ探索に向いている。")
    lines.append("- Salt の元データは `1.5-4.482 km/s` の範囲だが、実験用断面は `scipy.ndimage.zoom` による補間後の crop なので、true velocity が box constraint の外へ少しオーバーシュートしている。これは Salt 実験の RMSE や box constraint 評価で注意が必要。")
    lines.append("- BP2004 は 2-D velocity benchmark の一部を crop/resample しており、実験 grid は `801 x 241`。Salt 実験よりセル数が約 38.6 倍大きく、1 iteration あたりの計算負荷もかなり重い。")
    lines.append("- Salt は横幅が約 0.99 km、BP2004 は約 20 km。BP2004 は横方向にかなり広いので、同じ receiver 数でも receiver spacing は Salt より大きい。")
    lines.append("- Salt は 20 shots / 101 receivers、BP2004 は 5 shots / 201 receivers。BP2004 は shot 数を抑えつつ、receiver を広域に配置している。")
    lines.append("- Salt の source/receiver は端から端まで配置される。BP2004 の source は端から 1 km の margin を置き、receiver は crop 全幅に配置される。")
    lines.append("- Salt の速度範囲は典型的な salt body を含む `1.5-4.5 km/s` 付近。BP2004 は最大速度が今回の crop では約 `4.79 km/s` だが、box constraint は余裕を見て `5.5 km/s` まで許している。")
    lines.append("- BP2004 は深さ 6 km・横幅 20 km のモデルなので、Salt よりも大域的な構造、照明不足、shot 配置の影響を受けやすい。Salt の小型実験で良い gamma や alpha が、そのまま BP2004 に移るとは限らない。")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("- Salt model: `datasets/salt_and_overthrust_models/3-D_Salt_Model/VEL_GRIDS/Saltf@@`")
    lines.append("- BP2004 raw: `data/raw/bp2004/vel_z6.25m_x12.5m_exact.segy`")
    lines.append("- BP2004 processed: `data/processed/bp2004/bp2004_initial_crop.npz`")
    lines.append("- Salt driver: `src/box-TV-constrained-FWI.py`")
    lines.append("- BP2004 driver: `src/box-TV-constrained-FWI-BP2004.py`")
    lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    salt = load_salt_survey()
    bp2004 = load_bp2004_survey()
    assets = save_figures(salt, bp2004)
    write_markdown(salt, bp2004, assets)
    print(f"Wrote {DOC_PATH}")
    for path in assets.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
