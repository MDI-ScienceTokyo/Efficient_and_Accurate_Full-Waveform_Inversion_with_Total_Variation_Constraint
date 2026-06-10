from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as calc_ssim

import lib.signal_processing.diff_operator as diff_op
from lib.signal_processing.norm import L12_norm


RESULT_ROOT = Path("results/bp2004")
DOC_PATH = Path("docs/bp2004_tv_box_alpha_sweep_gamma2_100000_results.md")
ASSET_DIR = Path("docs/assets")
TARGET_ALPHAS = (500.0, 1000.0, 1500.0)
TARGET_NOISE = 0.0
TARGET_N_SHOTS = 10
TARGET_GAMMA2 = 100000.0


@dataclass
class Run:
    alpha: float
    directory: Path
    config: dict
    npz_path: Path
    data: np.lib.npyio.NpzFile
    iteration: np.ndarray
    objective: np.ndarray
    rmse: np.ndarray
    ssim: np.ndarray
    tv: np.ndarray


def read_metrics(path: Path, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    float(row["iteration"]),
                    float(row["objective"]),
                    (float(row["mse"]) / n_cells) ** 0.5,
                    float(row["ssim"]),
                    float(row["tv"]),
                )
            )
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def matches(config: dict) -> bool:
    return (
        config.get("algorithm") == "pds"
        and float(config.get("noise_sigma")) == TARGET_NOISE
        and int(config.get("n_shots")) == TARGET_N_SHOTS
        and abs(float(config.get("gamma2")) - TARGET_GAMMA2) < 1e-9
        and float(config.get("alpha")) in TARGET_ALPHAS
    )


def load_runs() -> list[Run]:
    by_alpha: dict[float, Run] = {}
    for config_path in RESULT_ROOT.glob("*/config.json"):
        with config_path.open() as f:
            config = json.load(f)
        if not matches(config):
            continue
        npz_path = next(config_path.parent.glob("*.npz"))
        data = np.load(npz_path)
        n_cells = data["true_velocity_model"].size
        iteration, objective, rmse, ssim, tv = read_metrics(config_path.parent / "metrics.csv", n_cells)
        run = Run(
            alpha=float(config["alpha"]),
            directory=config_path.parent,
            config=config,
            npz_path=npz_path,
            data=data,
            iteration=iteration,
            objective=objective,
            rmse=rmse,
            ssim=ssim,
            tv=tv,
        )
        previous = by_alpha.get(run.alpha)
        if previous is None or run.directory.name > previous.directory.name:
            by_alpha[run.alpha] = run

    missing = [alpha for alpha in TARGET_ALPHAS if alpha not in by_alpha]
    if missing:
        raise RuntimeError(f"Missing target alpha runs: {missing}")
    return [by_alpha[alpha] for alpha in TARGET_ALPHAS]


def relative(path: Path) -> str:
    return path.relative_to(DOC_PATH.parent).as_posix()


def save_velocity_models(runs: list[Run]) -> Path:
    true_model = runs[0].data["true_velocity_model"]
    initial_model = runs[0].data["initial_velocity_model"]
    models = [true_model, initial_model] + [run.data["final_velocity_model"] for run in runs]
    titles = ["True", "Initial"] + [f"alpha={run.alpha:g}" for run in runs]
    finite_models = [model[np.isfinite(model)] for model in models]
    vmin = min(float(model.min()) for model in finite_models if model.size)
    vmax = max(float(model.max()) for model in finite_models if model.size)

    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.0), constrained_layout=True)
    for ax, model, title in zip(axes, models, titles):
        im = ax.imshow(np.ma.masked_invalid(model), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x cell")
        ax.set_ylabel("z cell")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Velocity [km/s]")
    output = ASSET_DIR / "bp2004_gamma2_100000_velocity_models.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_seismic_signal(run: Run) -> Path:
    observed = run.data["observed_seismic_data"]
    shot_indices = [0, observed.shape[0] // 2, observed.shape[0] - 1]
    clip = np.percentile(np.abs(observed), 99.0)
    if not np.isfinite(clip) or clip <= 0:
        clip = float(np.max(np.abs(observed))) if observed.size else 1.0

    fig, axes = plt.subplots(1, len(shot_indices), figsize=(4.6 * len(shot_indices), 4.2), constrained_layout=True)
    for ax, shot_idx in zip(axes, shot_indices):
        gather = observed[shot_idx]
        im = ax.imshow(gather, cmap="gray", aspect="auto", vmin=-clip, vmax=clip)
        ax.set_title(f"Observed shot {shot_idx + 1}")
        ax.set_xlabel("Receiver index")
        ax.set_ylabel("Time index")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Amplitude")
    output = ASSET_DIR / "bp2004_gamma2_100000_observed_seismic_signal.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_metric_plot(runs: list[Run], attr: str, ylabel: str, filename: str, yscale: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for run in runs:
        ax.plot(run.iteration, getattr(run, attr), label=f"alpha={run.alpha:g}", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()
    output = ASSET_DIR / filename
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_markdown(runs: list[Run], assets: dict[str, Path]) -> None:
    true_model = runs[0].data["true_velocity_model"]
    initial_model = runs[0].data["initial_velocity_model"]
    n_cells = true_model.size
    data_range = float(runs[0].config["box_max_value"] - runs[0].config["box_min_value"])
    initial_rmse = float(np.sqrt(np.sum((initial_model - true_model) ** 2) / n_cells))
    initial_ssim = float(calc_ssim(true_model, initial_model, data_range=data_range))
    initial_tv = float(L12_norm(diff_op.D(initial_model)))
    true_tv = float(L12_norm(diff_op.D(true_model)))

    lines: list[str] = []
    lines.append("# BP2004 TV + Box Constraint Alpha Sweep Results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "BP2004 crop に対して TV + box constraint の PDS 版を "
        "`alpha=500,1000,1500`、`noise_sigma=0`、`n_shots=10`、`n_receivers=201`、"
        "`gamma1=1e-6`、`gamma2=100000` で比較した。速度単位は km/s。"
    )
    lines.append("")
    lines.append(f"- Model shape: `{true_model.shape[0]} x {true_model.shape[1]}` cells")
    lines.append(f"- Initial RMSE: `{initial_rmse:.6f}` km/s")
    lines.append(f"- Initial SSIM: `{initial_ssim:.6f}`")
    lines.append(f"- True TV: `{true_tv:.3f}`")
    lines.append(f"- Initial TV: `{initial_tv:.3f}`")
    lines.append("")
    lines.append("## Velocity Models")
    lines.append("")
    lines.append(f"![True, initial, and final models]({relative(assets['models'])})")
    lines.append("")
    lines.append("## Seismic Signal")
    lines.append("")
    lines.append("観測波形 `observed_seismic_data` のshot gather例。今回のnoiseなし条件ではalphaによらず同じ観測データを使うため、代表としてalpha=500 runから描画した。")
    lines.append("")
    lines.append(f"![Observed seismic shot gathers]({relative(assets['seismic'])})")
    lines.append("")
    lines.append("## RMSE Per Iteration")
    lines.append("")
    lines.append(f"![RMSE per iteration]({relative(assets['rmse'])})")
    lines.append("")
    lines.append("## SSIM Per Iteration")
    lines.append("")
    lines.append(f"![SSIM per iteration]({relative(assets['ssim'])})")
    lines.append("")
    lines.append("## Additional Curves")
    lines.append("")
    lines.append(f"![Objective per iteration]({relative(assets['objective'])})")
    lines.append("")
    lines.append(f"![TV per iteration]({relative(assets['tv'])})")
    lines.append("")
    lines.append("## Final Metrics")
    lines.append("")
    lines.append("| alpha | completed iters | finite final model | final RMSE [km/s] | best RMSE [km/s] | final SSIM | best SSIM | final TV | final objective | elapsed [h] |")
    lines.append("|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        finite_final = bool(np.all(np.isfinite(run.data["final_velocity_model"])))
        lines.append(
            "| "
            f"{run.alpha:g} | "
            f"{int(run.config['completed_iters'])} | "
            f"{'yes' if finite_final else 'no'} | "
            f"{run.rmse[-1]:.6f} | "
            f"{float(np.min(run.rmse)):.6f} | "
            f"{run.ssim[-1]:.6f} | "
            f"{float(np.max(run.ssim)):.6f} | "
            f"{run.tv[-1]:.3f} | "
            f"{run.objective[-1]:.6g} | "
            f"{float(run.config['elapsed']) / 3600:.3f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `metrics.csv` の `mse` は全セルの二乗誤差和なので、`sqrt(mse / number_of_cells)` としてRMSEへ変換した。")
    lines.append("- `alpha=500` と `alpha=1000` は5000 iterationには到達しておらず、それぞれ保存済みconfigの `completed_iters` は179, 187だった。`gamma2=100000` でTV制約が強く効き、低alpha条件では早期停止した可能性が高い。")
    lines.append("- `alpha=1000` の保存済みfinal modelには非finite値が含まれていた。速度モデル図では該当値をマスクして表示しているため、このrunは失敗扱いで解釈する。")
    lines.append("- `alpha=1500` は5000 iterationまで完走しており、この3条件の中では実験として最も安定して比較できる。")
    lines.append("- `gamma2=100` の過去runではTV制約の効きが遅すぎたが、`gamma2=100000` ではPDSのTV側更新が明確に強くなる。")
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
    assets = {
        "models": save_velocity_models(runs),
        "seismic": save_seismic_signal(runs[0]),
        "rmse": save_metric_plot(runs, "rmse", "RMSE [km/s]", "bp2004_gamma2_100000_rmse_per_iter.png"),
        "ssim": save_metric_plot(runs, "ssim", "SSIM", "bp2004_gamma2_100000_ssim_per_iter.png"),
        "objective": save_metric_plot(runs, "objective", "Objective", "bp2004_gamma2_100000_objective_per_iter.png", yscale="log"),
        "tv": save_metric_plot(runs, "tv", "TV", "bp2004_gamma2_100000_tv_per_iter.png"),
    }
    write_markdown(runs, assets)
    print(f"Wrote {DOC_PATH}")
    for path in assets.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
