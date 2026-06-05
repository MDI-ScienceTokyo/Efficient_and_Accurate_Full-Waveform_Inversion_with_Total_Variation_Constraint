from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

import lib.signal_processing.diff_operator as diff_op
from lib.signal_processing.norm import L12_norm


RESULT_ROOT = Path("results/bp2004")
DOC_PATH = Path("docs/bp2004_tv_box_results.md")
ASSET_DIR = Path("docs/assets")


@dataclass
class Run:
    alpha: float
    name: str
    directory: Path
    config: dict
    npz: np.lib.npyio.NpzFile
    iteration: np.ndarray
    objective: np.ndarray
    rmse: np.ndarray
    psnr: np.ndarray
    ssim: np.ndarray
    tv: np.ndarray


def load_metrics(path: Path, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["iteration"]),
                    float(row["objective"]),
                    float(row["mse"]),
                    float(row["psnr"]),
                    float(row["ssim"]),
                    float(row["tv"]),
                ]
            )
    data = np.asarray(rows, dtype=np.float64)
    iteration = data[:, 0]
    objective = data[:, 1]
    rmse = np.sqrt(data[:, 2] / n_cells)
    psnr = data[:, 3]
    ssim_history = data[:, 4]
    tv = data[:, 5]
    return iteration, objective, rmse, psnr, ssim_history, tv


def load_runs() -> list[Run]:
    runs: list[Run] = []
    for config_path in sorted(RESULT_ROOT.glob("*/config.json")):
        with config_path.open() as f:
            config = json.load(f)
        npz_path = next(config_path.parent.glob("*.npz"))
        npz = np.load(npz_path)
        true_model = npz["true_velocity_model"]
        n_cells = true_model.size
        iteration, objective, rmse, psnr, ssim_history, tv = load_metrics(config_path.parent / "metrics.csv", n_cells)
        runs.append(
            Run(
                alpha=float(config["alpha"]),
                name=config_path.parent.name,
                directory=config_path.parent,
                config=config,
                npz=npz,
                iteration=iteration,
                objective=objective,
                rmse=rmse,
                psnr=psnr,
                ssim=ssim_history,
                tv=tv,
            )
        )
    return sorted(runs, key=lambda r: r.alpha)


def save_model_comparison(runs: list[Run]) -> Path:
    true_model = runs[0].npz["true_velocity_model"]
    models = [true_model]
    titles = ["True"]
    for run in runs:
        models.append(run.npz["final_velocity_model"])
        titles.append(f"alpha={run.alpha:g}")

    vmin = min(float(np.min(model)) for model in models)
    vmax = max(float(np.max(model)) for model in models)
    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 4.0), constrained_layout=True)
    for ax, model, title in zip(axes, models, titles):
        im = ax.imshow(model, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x cell")
        ax.set_ylabel("z cell")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="Velocity [km/s]")
    output_path = ASSET_DIR / "bp2004_tv_box_velocity_models.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_error_maps(runs: list[Run]) -> Path:
    true_model = runs[0].npz["true_velocity_model"]
    errors = [run.npz["final_velocity_model"] - true_model for run in runs]
    lim = max(float(np.max(np.abs(err))) for err in errors)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.1 * len(runs), 4.0), constrained_layout=True)
    for ax, err, run in zip(axes, errors, runs):
        im = ax.imshow(err, cmap="seismic", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"alpha={run.alpha:g}")
        ax.set_xlabel("x cell")
        ax.set_ylabel("z cell")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="Final - true [km/s]")
    output_path = ASSET_DIR / "bp2004_tv_box_error_maps.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_metric_plot(runs: list[Run], attr: str, ylabel: str, filename: str, yscale: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for run in runs:
        values = getattr(run, attr)
        ax.plot(run.iteration, values, label=f"alpha={run.alpha:g}", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()
    output_path = ASSET_DIR / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def rel(path: Path) -> str:
    return path.relative_to(DOC_PATH.parent).as_posix()


def fmt_seconds(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h"


def write_markdown(runs: list[Run], assets: dict[str, Path]) -> None:
    true_model = runs[0].npz["true_velocity_model"]
    initial_model = runs[0].npz["initial_velocity_model"]
    n_cells = true_model.size
    data_range = float(runs[0].config["box_max_value"] - runs[0].config["box_min_value"])
    initial_rmse = float(np.sqrt(np.sum((initial_model - true_model) ** 2) / n_cells))
    initial_ssim = float(ssim(true_model, initial_model, data_range=data_range))
    true_tv = float(L12_norm(diff_op.D(true_model)))

    lines: list[str] = []
    lines.append("# BP2004 TV + box constrained FWI results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("BP2004 crop に対して TV + box constraint の PDS 版を `alpha=500,1500,3000` で比較した。全 run は 5000 iteration 完了しており、noise は 0、shot は 5、receiver は 201。速度単位は km/s。")
    lines.append("")
    lines.append(f"- Model shape: `{true_model.shape[0]} x {true_model.shape[1]}` cells")
    lines.append(f"- Grid spacing: `{runs[0].config['cell_meter_size']['y']} m x {runs[0].config['cell_meter_size']['x']} m`")
    lines.append(f"- Box constraint: `{runs[0].config['box_min_value']} <= v <= {runs[0].config['box_max_value']}` km/s")
    lines.append(f"- Initial RMSE: `{initial_rmse:.6f}` km/s")
    lines.append(f"- Initial SSIM: `{initial_ssim:.6f}`")
    lines.append(f"- True-model TV reference: `{true_tv:.3f}`")
    lines.append(f"- Alpha / true TV: `{', '.join(f'{run.alpha / true_tv:.3f}' for run in runs)}` for alpha `{', '.join(f'{run.alpha:g}' for run in runs)}`")
    lines.append("")
    lines.append("## Velocity Models")
    lines.append("")
    lines.append(f"![True and final velocity models]({rel(assets['models'])})")
    lines.append("")
    lines.append("## Final Error Maps")
    lines.append("")
    lines.append(f"![Final error maps]({rel(assets['error_maps'])})")
    lines.append("")
    lines.append("## Metrics Per Iteration")
    lines.append("")
    lines.append(f"![RMSE per iteration]({rel(assets['rmse'])})")
    lines.append("")
    lines.append(f"![SSIM per iteration]({rel(assets['ssim'])})")
    lines.append("")
    lines.append(f"![Objective per iteration]({rel(assets['objective'])})")
    lines.append("")
    lines.append(f"![TV per iteration]({rel(assets['tv'])})")
    lines.append("")
    lines.append("## Final Metrics")
    lines.append("")
    lines.append("| alpha | final RMSE [km/s] | best RMSE [km/s] | final SSIM | best SSIM | final objective | final TV | elapsed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        best_rmse = float(np.min(run.rmse))
        best_ssim = float(np.max(run.ssim))
        lines.append(
            "| "
            f"{run.alpha:g} | "
            f"{run.rmse[-1]:.6f} | "
            f"{best_rmse:.6f} | "
            f"{run.ssim[-1]:.6f} | "
            f"{best_ssim:.6f} | "
            f"{run.objective[-1]:.6g} | "
            f"{run.tv[-1]:.3f} | "
            f"{fmt_seconds(float(run.config['elapsed']))} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `metrics.csv` の `mse` は全セルの二乗誤差和なので、このレポートでは `sqrt(mse / number_of_cells)` として RMSE に変換した。")
    lines.append("- `error per iter` は波形残差由来の目的関数 `objective_history` として描画した。")
    lines.append("- `alpha=3000` は objective と RMSE が最も低く、最終速度場も true に最も近い。一方で SSIM は `alpha=1500` がわずかに高い。差は小さいため、構造類似度だけを見ると中程度の TV 制約も候補になる。")
    lines.append("- `alpha=500` は TV が最も小さく、平滑化が強すぎる。目的関数と RMSE が iteration とともに悪化しており、この設定では制約半径が狭すぎてデータ適合とモデル再構成の両方を阻害している可能性が高い。")
    lines.append("- `alpha=1500` は `alpha=500` より安定しているが、後半で objective が増加している。今回の `gamma1=1e-6`, `gamma2=100` では、alpha によっては長時間反復で単調改善しない。")
    lines.append("- `alpha=3000` は true model の TV reference に近い制約半径で、TV が過度に圧縮されず、objective も低い水準に収束している。今回の 3 条件では最有力。")
    lines.append("")
    lines.append("## Run Directories")
    lines.append("")
    for run in runs:
        lines.append(f"- alpha={run.alpha:g}: `{run.directory.as_posix()}`")
    lines.append("")

    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise RuntimeError(f"No runs found under {RESULT_ROOT}")

    assets = {
        "models": save_model_comparison(runs),
        "error_maps": save_error_maps(runs),
        "rmse": save_metric_plot(runs, "rmse", "RMSE [km/s]", "bp2004_tv_box_rmse_per_iter.png"),
        "ssim": save_metric_plot(runs, "ssim", "SSIM", "bp2004_tv_box_ssim_per_iter.png"),
        "objective": save_metric_plot(runs, "objective", "Objective / waveform error", "bp2004_tv_box_objective_per_iter.png", yscale="log"),
        "tv": save_metric_plot(runs, "tv", "TV", "bp2004_tv_box_tv_per_iter.png"),
    }
    write_markdown(runs, assets)
    print(f"Wrote {DOC_PATH}")
    for path in assets.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
