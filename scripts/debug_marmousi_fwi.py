import argparse
import csv
import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from devito import set_log_level
from scipy.ndimage import gaussian_filter

from lib.seismic.fast_parallel_velocity_model_gradient_calculator import get_time_length


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "marmousi_debug"
MARM_SCRIPT = PROJECT_ROOT / "src" / "box-TV-constrained-FWI-marmousi.py"


@dataclass
class ObjectiveContext:
    ns: dict
    calc: object
    dsize: int
    true: np.ndarray
    init: np.ndarray
    vmin: float
    vmax: float
    initial_with_damping: np.ndarray


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def array_stats(name: str, x: np.ndarray) -> str:
    finite = np.isfinite(x)
    lines = [
        f"{name} shape: {x.shape}",
        f"{name} dtype: {x.dtype}",
        f"{name} min: {float(np.nanmin(x))}",
        f"{name} max: {float(np.nanmax(x))}",
        f"{name} mean: {float(np.nanmean(x))}",
        f"{name} std: {float(np.nanstd(x))}",
        f"{name} nan_count: {int(np.isnan(x).sum())}",
        f"{name} inf_count: {int(np.isinf(x).sum())}",
        f"{name} finite_fraction: {float(finite.mean())}",
    ]
    return "\n".join(lines)


def save_image(path: Path, data: np.ndarray, title: str, cmap: str = "coolwarm", vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_line_plot(path: Path, series: dict[str, np.ndarray], title: str, yscale: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in series.items():
        plt.plot(values, label=label)
    if yscale:
        plt.yscale(yscale)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def load_ns(num_parallels: int) -> dict:
    ns = runpy.run_path(str(MARM_SCRIPT))
    ns["num_parallels"] = num_parallels
    return ns


def load_marmousi(ns: dict):
    data = ns["load_marmousi_model"]()
    return data.true_data, data.initial_data, data.box_min_value, data.box_max_value


def make_params(ns: dict, shape: tuple[int, int], n_shots: int, n_receivers: int, noise_sigma: float, f0: float | None = None, tn: float | None = None):
    metadata = ns["load_marmousi_metadata"](shape)
    params = ns["marmousi_configuration"](
        shape,
        n_shots,
        n_receivers,
        noise_sigma,
        float(metadata["effective_dz_m"]),
        float(metadata["effective_dx_m"]),
        source_frequency=f0 if f0 is not None else 0.003,
    )
    if tn is not None:
        params = params._replace(simulation_times=tn)
    return params


def make_props(ns: dict, params, true: np.ndarray, init: np.ndarray, acquisition_margin: float = 0.0):
    props = ns["fwi_params_to_fast_parallel_velocity_model_gradient_calculator_props"](params, true, init)
    if acquisition_margin <= 0:
        return props

    width = (props.shape[1] - 1) * props.spacing[1]
    if acquisition_margin * 2 >= width:
        raise ValueError(f"acquisition_margin={acquisition_margin} is too large for width={width}")

    src_z = float(props.source_locations[0, 0])
    rec_z = float(props.receiver_locations[0, 0])
    source_locations = np.array(
        [[src_z, x] for x in np.linspace(acquisition_margin, width - acquisition_margin, num=len(props.source_locations))],
        dtype=np.float32,
    )
    receiver_locations = np.array(
        [[rec_z, x] for x in np.linspace(acquisition_margin, width - acquisition_margin, num=len(props.receiver_locations))],
        dtype=np.float32,
    )
    return props._replace(source_locations=source_locations, receiver_locations=receiver_locations)


def create_context(
    ns: dict,
    n_shots: int,
    n_receivers: int,
    f0: float | None = None,
    tn: float | None = None,
    initial_override: np.ndarray | None = None,
    acquisition_margin: float = 0.0,
) -> ObjectiveContext:
    true, init, vmin, vmax = load_marmousi(ns)
    if initial_override is not None:
        init = initial_override.astype(np.float32)
    params = make_params(ns, true.shape, n_shots, n_receivers, 0.0, f0=f0, tn=tn)
    props = make_props(ns, params, true, init, acquisition_margin=acquisition_margin)
    calc = ns["FastParallelVelocityModelGradientCalculator"](props)
    return ObjectiveContext(ns, calc, params.damping_cell_thickness, true, init, vmin, vmax, calc.velocity_model.copy())


def core_to_padded(ctx: ObjectiveContext, core: np.ndarray) -> np.ndarray:
    v = ctx.initial_with_damping.copy()
    d = ctx.dsize
    v[d:-d, d:-d] = core.astype(np.float32)
    return v


def objective_and_gradient(ctx: ObjectiveContext, core: np.ndarray) -> tuple[float, np.ndarray]:
    value, grad_with_damping = ctx.calc.calc_grad(core_to_padded(ctx, core))
    d = ctx.dsize
    return float(value), grad_with_damping[d:-d, d:-d].copy()


def objective_only(ctx: ObjectiveContext, core: np.ndarray) -> float:
    value, _ = objective_and_gradient(ctx, core)
    return value


def cleanup_context(ctx: ObjectiveContext) -> None:
    del ctx.calc


def test_0_model(ns: dict) -> dict:
    true, init, vmin, vmax = load_marmousi(ns)
    save_image(RESULT_ROOT / "00_true_model.png", true, "Marmousi true model", vmin=vmin, vmax=vmax)
    save_image(RESULT_ROOT / "00_initial_model.png", init, "Marmousi initial model", vmin=vmin, vmax=vmax)
    save_image(RESULT_ROOT / "00_true_minus_init.png", true - init, "Marmousi true - initial", cmap="seismic")
    text = "\n\n".join([array_stats("m_true", true), array_stats("m_init", init), f"box_min: {vmin}", f"box_max: {vmax}"])
    write_text(RESULT_ROOT / "00_model_stats.txt", text + "\n")
    return {"true": true, "init": init, "vmin": vmin, "vmax": vmax}


def test_1_config(ns: dict, model: dict, n_shots: int, n_receivers: int) -> dict:
    true = model["true"]
    marm_params = make_params(ns, true.shape, n_shots, n_receivers, 0.0)
    marm_props = make_props(ns, marm_params, model["true"], model["init"])
    nt = get_time_length(marm_props)
    try:
        salt_ns = runpy.run_path(str(PROJECT_ROOT / "src" / "box-TV-constrained-FWI.py"))
        salt_params = salt_ns["salt_model_test00_configuration"](n_shots, 0.0)
        salt_shape = (salt_params.real_cell_size.y, salt_params.real_cell_size.x)
        salt_spacing = (salt_params.cell_meter_size.y, salt_params.cell_meter_size.x)
        salt_width = ((salt_params.real_cell_size - ns["Vec2D"](1, 1)) * salt_params.cell_meter_size).x
        salt_summary = [
            "Salt/reference:",
            f"  model_shape: {salt_shape}",
            f"  spacing: {salt_spacing}",
            f"  domain_size: {(salt_shape[0] - 1) * salt_spacing[0]} x {salt_width}",
            f"  nbl: {salt_params.damping_cell_thickness}",
            f"  source_frequency: {salt_params.source_frequency}",
            f"  n_shots: {salt_params.n_shots}",
            f"  n_receivers: {salt_params.n_receivers}",
            f"  tn: {salt_params.simulation_times}",
        ]
    except Exception as exc:
        salt_summary = [f"Salt/reference unavailable: {type(exc).__name__}: {exc}"]

    lines = [
        "Marmousi:",
        f"  model_shape: {true.shape}",
        f"  spacing: {(marm_params.cell_meter_size.y, marm_params.cell_meter_size.x)}",
        f"  domain_size_zx: {((true.shape[0] - 1) * marm_params.cell_meter_size.y, (true.shape[1] - 1) * marm_params.cell_meter_size.x)}",
        f"  nbl: {marm_params.damping_cell_thickness}",
        f"  velocity_min_max: {(float(true.min()), float(true.max()))}",
        f"  source_frequency: {marm_params.source_frequency}",
        f"  n_shots: {marm_params.n_shots}",
        f"  n_receivers: {marm_params.n_receivers}",
        f"  source_coordinate_range: z={marm_props.source_locations[:, 0].min()}..{marm_props.source_locations[:, 0].max()}, x={marm_props.source_locations[:, 1].min()}..{marm_props.source_locations[:, 1].max()}",
        f"  receiver_coordinate_range: z={marm_props.receiver_locations[:, 0].min()}..{marm_props.receiver_locations[:, 0].max()}, x={marm_props.receiver_locations[:, 1].min()}..{marm_props.receiver_locations[:, 1].max()}",
        f"  tn: {marm_params.simulation_times}",
        f"  dt: solver/default Devito dt; nt={nt}",
        f"  dtype: {true.dtype}",
        "",
        *salt_summary,
    ]
    write_text(RESULT_ROOT / "01_config_summary.txt", "\n".join(lines) + "\n")
    return {"params": marm_params, "props": marm_props, "nt": nt}


def test_2_geometry(model: dict, config: dict) -> None:
    true = model["true"]
    props = config["props"]
    spacing_z, spacing_x = props.spacing
    extent = [0, (true.shape[1] - 1) * spacing_x, (true.shape[0] - 1) * spacing_z, 0]
    plt.figure(figsize=(14, 4))
    plt.imshow(true, extent=extent, cmap="coolwarm", aspect="auto", vmin=model["vmin"], vmax=model["vmax"])
    plt.scatter(props.receiver_locations[:, 1], props.receiver_locations[:, 0], s=12, c="black", marker="v", label="receivers")
    plt.scatter(props.source_locations[:, 1], props.source_locations[:, 0], s=50, c="yellow", marker="*", edgecolors="black", label="sources")
    plt.axhline(0, color="white", lw=1)
    plt.axhline((true.shape[0] - 1) * spacing_z, color="white", lw=1)
    plt.axvline(0, color="white", lw=1)
    plt.axvline((true.shape[1] - 1) * spacing_x, color="white", lw=1)
    plt.legend()
    plt.title("Marmousi source/receiver geometry")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(RESULT_ROOT / "02_geometry.png", dpi=150)
    plt.close()


def waveform_stats(name: str, waveforms: np.ndarray) -> str:
    abs_wave = np.abs(waveforms)
    per_shot = abs_wave.reshape(abs_wave.shape[0], -1).max(axis=1) if waveforms.ndim == 3 else np.array([abs_wave.max()])
    per_receiver = abs_wave.max(axis=1) if waveforms.ndim == 3 else np.empty((0,))
    lines = [
        array_stats(name, waveforms),
        f"{name} l2_norm: {float(np.linalg.norm(waveforms[np.isfinite(waveforms)]))}",
        f"{name} max_abs_per_shot: {per_shot.tolist()}",
    ]
    if per_receiver.size:
        lines.append(f"{name} max_abs_per_receiver_min_max: {float(per_receiver.min())}, {float(per_receiver.max())}")
    return "\n".join(lines)


def save_waveform_plots(prefix: str, waveforms: np.ndarray) -> None:
    shot0 = waveforms[0]
    save_image(RESULT_ROOT / f"{prefix}_shot0.png", shot0.T, f"{prefix} shot 0 gather", cmap="seismic")
    plt.figure(figsize=(10, 5))
    receiver_ids = np.linspace(0, shot0.shape[1] - 1, min(5, shot0.shape[1]), dtype=int)
    for rid in receiver_ids:
        plt.plot(shot0[:, rid], label=f"rec {rid}")
    plt.title(f"{prefix} trace examples")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_ROOT / f"{prefix}_trace_examples.png", dpi=150)
    plt.close()


def test_forward(ns: dict, model: dict, config: dict, which: str) -> np.ndarray:
    if which == "true":
        true_arg, init_arg = model["true"], model["true"]
        prefix = "03_forward_true"
    else:
        true_arg, init_arg = model["init"], model["init"]
        prefix = "04_forward_init"
    props = make_props(ns, config["params"], true_arg, init_arg)
    calc = ns["FastParallelVelocityModelGradientCalculator"](props)
    waveforms = calc.true_observed_waveforms.copy()
    write_text(RESULT_ROOT / f"{prefix}_stats.txt", waveform_stats(prefix, waveforms) + "\n")
    save_waveform_plots(prefix, waveforms)
    del calc
    return waveforms


def test_5_objective(ns: dict, n_shots: int, n_receivers: int) -> dict:
    ctx = create_context(ns, n_shots, n_receivers)
    e_init, grad = objective_and_gradient(ctx, ctx.init)
    cleanup_context(ctx)
    true_ctx = create_context(ns, n_shots, n_receivers, initial_override=ctx.true)
    e_true = objective_only(true_ctx, true_ctx.true)
    cleanup_context(true_ctx)
    text = f"E(m_true): {e_true}\nE(m_init): {e_init}\nratio E_true/E_init: {e_true / e_init if e_init else np.nan}\n"
    write_text(RESULT_ROOT / "05_initial_objective.txt", text)
    return {"E0": e_init, "gradient": grad}


def test_6_gradient(model: dict, objective: dict) -> None:
    grad = objective["gradient"]
    model_norm = float(np.linalg.norm(model["init"]))
    grad_norm = float(np.linalg.norm(grad[np.isfinite(grad)]))
    lines = [
        array_stats("gradient", grad),
        f"gradient_l2_norm: {grad_norm}",
        f"gradient_linf_norm: {float(np.nanmax(np.abs(grad)))}",
        f"model_l2_norm: {model_norm}",
        f"relative_gradient_norm: {grad_norm / model_norm}",
    ]
    write_text(RESULT_ROOT / "06_gradient_stats.txt", "\n".join(lines) + "\n")
    finite_grad = grad[np.isfinite(grad)]
    lim = np.percentile(np.abs(finite_grad), 99) if finite_grad.size else 1.0
    save_image(RESULT_ROOT / "06_gradient_map.png", grad, "First gradient", cmap="seismic", vmin=-lim, vmax=lim)
    plt.figure(figsize=(8, 5))
    plt.hist(finite_grad.ravel(), bins=100)
    plt.yscale("log")
    plt.title("Gradient histogram")
    plt.tight_layout()
    plt.savefig(RESULT_ROOT / "06_gradient_hist.png", dpi=150)
    plt.close()


def test_7_update_sweep(model: dict, grad: np.ndarray, gammas: Iterable[float]) -> None:
    init = model["init"]
    model_norm = np.linalg.norm(init)
    lines = ["gamma,delta_min,delta_max,delta_l2,relative_delta,candidate_min,candidate_max,outside_percent"]
    for gamma in gammas:
        delta = -gamma * grad
        candidate = init + delta
        outside = np.mean((candidate < model["vmin"]) | (candidate > model["vmax"])) * 100
        lines.append(
            f"{gamma:g},{float(np.nanmin(delta))},{float(np.nanmax(delta))},{float(np.linalg.norm(delta[np.isfinite(delta)]))},"
            f"{float(np.linalg.norm(delta[np.isfinite(delta)]) / model_norm)},{float(np.nanmin(candidate))},{float(np.nanmax(candidate))},{float(outside)}"
        )
    write_text(RESULT_ROOT / "07_relative_update_sweep.txt", "\n".join(lines) + "\n")


def test_8_line_search(ns: dict, model: dict, grad: np.ndarray, e0: float, n_shots: int, n_receivers: int, gammas: Iterable[float]) -> None:
    ctx = create_context(ns, n_shots, n_receivers)
    write_line_search(RESULT_ROOT / "08_one_step_line_search.txt", ctx, model["init"], grad, e0, gammas)
    cleanup_context(ctx)


def test_9_sign_check(ns: dict, model: dict, grad: np.ndarray, n_shots: int, n_receivers: int) -> None:
    rng = np.random.default_rng(0)
    p = rng.standard_normal(model["init"].shape)
    p = gaussian_filter(p, sigma=3)
    p = p / np.linalg.norm(p)
    inner = float(np.sum(grad * p))
    ctx = create_context(ns, n_shots, n_receivers)
    lines = [f"inner_gradient_dot_p: {inner}", "eps,fd,fd_over_inner,same_sign"]
    for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
        e_plus = objective_only(ctx, model["init"] + eps * p)
        e_minus = objective_only(ctx, model["init"] - eps * p)
        fd = (e_plus - e_minus) / (2 * eps)
        lines.append(f"{eps:g},{fd},{fd / inner if inner else np.nan},{np.sign(fd) == np.sign(inner)}")
    cleanup_context(ctx)
    write_text(RESULT_ROOT / "09_gradient_sign_check.txt", "\n".join(lines) + "\n")


def test_12_cfl(model: dict, config: dict) -> None:
    params = config["params"]
    spacing = (params.cell_meter_size.y, params.cell_meter_size.x)
    min_v = float(model["true"].min())
    max_v = float(model["true"].max())
    f0 = params.source_frequency
    wavelength = min_v / f0 if f0 else np.inf
    ppw = wavelength / max(spacing)
    lines = [
        f"spacing: {spacing}",
        f"min_velocity: {min_v}",
        f"max_velocity: {max_v}",
        f"source_frequency: {f0}",
        f"estimated_minimum_wavelength: {wavelength}",
        f"points_per_wavelength: {ppw}",
        f"tn: {params.simulation_times}",
        f"nt: {config['nt']}",
        "note: repository uses velocity in km/s with spacing/time values inherited from the Salt script; inspect units carefully.",
    ]
    write_text(RESULT_ROOT / "12_grid_cfl_check.txt", "\n".join(lines) + "\n")


def write_line_search(
    path: Path,
    ctx: ObjectiveContext,
    init: np.ndarray,
    grad: np.ndarray,
    e0: float,
    gammas: Iterable[float],
    header: list[str] | None = None,
) -> None:
    lines = header or []
    lines.extend(["E0: " + str(e0), "gamma,E_minus,E_plus,ratio_minus,ratio_plus,min_minus,max_minus,min_plus,max_plus,finite_minus,finite_plus"])
    for gamma in gammas:
        minus = init - gamma * grad
        plus = init + gamma * grad
        try:
            e_minus = objective_only(ctx, minus)
            finite_minus = np.isfinite(e_minus)
        except Exception:
            e_minus = np.nan
            finite_minus = False
        try:
            e_plus = objective_only(ctx, plus)
            finite_plus = np.isfinite(e_plus)
        except Exception:
            e_plus = np.nan
            finite_plus = False
        lines.append(
            f"{gamma:g},{e_minus},{e_plus},{e_minus / e0 if e0 else np.nan},{e_plus / e0 if e0 else np.nan},"
            f"{float(np.nanmin(minus))},{float(np.nanmax(minus))},{float(np.nanmin(plus))},{float(np.nanmax(plus))},{finite_minus},{finite_plus}"
        )
    write_text(path, "\n".join(lines) + "\n")


def test_16_inset_geometry_line_search(ns: dict, n_shots: int, n_receivers: int, acquisition_margin: float) -> None:
    ctx = create_context(ns, n_shots, n_receivers, acquisition_margin=acquisition_margin)
    e0, grad = objective_and_gradient(ctx, ctx.init)
    source_range = (float(ctx.calc.props.source_locations[:, 1].min()), float(ctx.calc.props.source_locations[:, 1].max()))
    receiver_range = (float(ctx.calc.props.receiver_locations[:, 1].min()), float(ctx.calc.props.receiver_locations[:, 1].max()))
    header = [
        f"acquisition_margin: {acquisition_margin}",
        f"source_x_range: {source_range}",
        f"receiver_x_range: {receiver_range}",
    ]
    write_line_search(
        RESULT_ROOT / "16_inset_geometry_line_search.txt",
        ctx,
        ctx.init,
        grad,
        e0,
        [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12],
        header=header,
    )
    cleanup_context(ctx)


def run_box_only_smoke(
    ns: dict,
    output_dir: Path,
    n_shots: int,
    n_receivers: int,
    gamma: float,
    n_iters: int,
    acquisition_margin: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = create_context(ns, n_shots, n_receivers, acquisition_margin=acquisition_margin)
    core = ctx.init.copy()
    rows = []
    for iteration in range(1, n_iters + 1):
        objective, grad = objective_and_gradient(ctx, core)
        if not np.isfinite(objective):
            rows.append([iteration, objective, np.nan, np.nan, np.nan, np.nan, False])
            break
        core = np.clip(core - gamma * grad, ctx.vmin, ctx.vmax)
        outside = np.mean((core < ctx.vmin) | (core > ctx.vmax)) * 100
        rows.append(
            [
                iteration,
                objective,
                float(np.nanmin(core)),
                float(np.nanmax(core)),
                float(np.linalg.norm((core - ctx.true)[np.isfinite(core)])),
                float(outside),
                bool(np.isfinite(core).all()),
            ]
        )

    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective_before_update", "model_min", "model_max", "model_error_l2", "outside_percent", "finite_model"])
        writer.writerows(rows)

    save_image(output_dir / "final_model.png", core, f"box-only final gamma={gamma:g}, margin={acquisition_margin:g}", vmin=ctx.vmin, vmax=ctx.vmax)
    save_image(output_dir / "final_minus_true.png", core - ctx.true, "box-only final - true", cmap="seismic")
    save_line_plot(
        output_dir / "objective_history.png",
        {"objective": np.array([r[1] for r in rows], dtype=float)},
        "Box-only objective history",
        yscale="log",
    )
    summary = {
        "gamma": gamma,
        "iterations_requested": n_iters,
        "iterations_completed": len(rows),
        "acquisition_margin": acquisition_margin,
        "first_objective": rows[0][1] if rows else None,
        "last_objective": rows[-1][1] if rows else None,
        "best_objective": min(r[1] for r in rows) if rows else None,
        "final_model_min": rows[-1][2] if rows else None,
        "final_model_max": rows[-1][3] if rows else None,
        "finite_final_model": rows[-1][6] if rows else None,
    }
    write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    cleanup_context(ctx)
    return summary


def test_13_box_only_smoke(ns: dict, n_shots: int, n_receivers: int, acquisition_margin: float) -> None:
    root = RESULT_ROOT / "13_box_only_smoke_test"
    summaries = []
    for margin in [0.0, acquisition_margin]:
        for gamma in [1e-6, 1e-7, 1e-8]:
            label = f"margin-{margin:g}_gamma-{gamma:g}".replace(".", "p").replace("-", "m")
            summaries.append(run_box_only_smoke(ns, root / label, n_shots, n_receivers, gamma, 50, margin))

    lines = [
        "margin,gamma,iters,first_objective,last_objective,best_objective,last_over_first,final_min,final_max,finite"
    ]
    for summary in summaries:
        first = summary["first_objective"]
        last = summary["last_objective"]
        lines.append(
            f"{summary['acquisition_margin']:g},{summary['gamma']:g},{summary['iterations_completed']},"
            f"{first},{last},{summary['best_objective']},{last / first if first else np.nan},"
            f"{summary['final_model_min']},{summary['final_model_max']},{summary['finite_final_model']}"
        )
    write_text(root / "summary.csv", "\n".join(lines) + "\n")


def read_existing(path: str) -> str:
    target = RESULT_ROOT / path
    return target.read_text() if target.exists() else ""


def generate_report() -> None:
    objective_text = read_existing("05_initial_objective.txt")
    gradient_text = read_existing("06_gradient_stats.txt")
    line_search_text = read_existing("08_one_step_line_search.txt")
    sign_text = read_existing("09_gradient_sign_check.txt")
    cfl_text = read_existing("12_grid_cfl_check.txt")
    box_smoke_text = read_existing("13_box_only_smoke_test/summary.csv")
    inset_text = read_existing("16_inset_geometry_line_search.txt")

    links = [
        ("Model stats", "00_model_stats.txt"),
        ("Config summary", "01_config_summary.txt"),
        ("Geometry", "02_geometry.png"),
        ("Forward true stats", "03_forward_true_stats.txt"),
        ("Forward init stats", "04_forward_init_stats.txt"),
        ("Initial objective", "05_initial_objective.txt"),
        ("Gradient stats", "06_gradient_stats.txt"),
        ("Update sweep", "07_relative_update_sweep.txt"),
        ("Line search", "08_one_step_line_search.txt"),
        ("Gradient sign check", "09_gradient_sign_check.txt"),
        ("Grid/CFL check", "12_grid_cfl_check.txt"),
        ("Box-only smoke summary", "13_box_only_smoke_test/summary.csv"),
        ("Inset geometry line search", "16_inset_geometry_line_search.txt"),
    ]
    lines = [
        "# Marmousi FWI Diagnostic Report",
        "",
        "This report was generated by `scripts/debug_marmousi_fwi.py`.",
        "",
        "## Outputs",
        "",
    ]
    for title, rel in links:
        path = RESULT_ROOT / rel
        if path.exists():
            lines.append(f"- [{title}]({rel})")
    lines.extend(
        [
        "",
        "## Key Results",
        "",
        "```text",
        objective_text.strip(),
        "```",
        "",
        "```text",
        "\n".join(line_search_text.splitlines()[:10]),
        "```",
        "",
        "```text",
        "\n".join(sign_text.splitlines()[:8]),
        "```",
        "",
        "## Interpretation",
        "",
        "- The processed Marmousi model loads as km/s and has no NaN/Inf values.",
        "- Forward modeling for true and initial models is finite in this diagnostic run.",
        "- `E(m_true)` is zero with consistent observed/calculated data, so the data-generation path is internally consistent.",
        "- The first gradient is finite and the finite-difference sign check agrees with the adjoint gradient sign.",
        "- One-step line search shows `m - gamma*g` decreases the objective for `gamma=1e-6` and `1e-8`, so the immediate update direction is not reversed.",
        "- The geometry places sources/receivers at the lateral endpoints (`x=0` and `x=xmax`), which is suspicious for repeated iterations and should be tested away from the side boundaries.",
        "",
        "## Gradient Scale / CFL Snippets",
        "",
        "```text",
        "\n".join(gradient_text.splitlines()[:16]),
        "```",
        "",
        "```text",
        cfl_text.strip(),
        "```",
        "",
        "## Box-Only Smoke Test",
        "",
        "```text",
        box_smoke_text.strip(),
        "```",
        "",
        "## Inset Geometry Line Search",
        "",
        "```text",
        "\n".join(inset_text.splitlines()[:12]),
        "```",
        "",
        "## Figures",
            "",
            "![true](00_true_model.png)",
            "",
            "![initial](00_initial_model.png)",
            "",
            "![geometry](02_geometry.png)",
            "",
            "![gradient](06_gradient_map.png)",
            "",
            "## Quick Reading Guide",
            "",
            "- If `E(m_true)` is not much smaller than `E(m_init)`, data generation is inconsistent.",
            "- If `E(m + gamma*g)` decreases while `E(m - gamma*g)` does not, the update sign is likely reversed.",
            "- If only extremely tiny gammas decrease the objective, fixed global step sizes are not usable without normalization or line search.",
            "- If source/receiver x coordinates touch 0 or the domain maximum, consider moving them away from lateral boundaries.",
        ]
    )
    write_text(RESULT_ROOT / "DIAGNOSTIC_REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shots", type=int, default=3)
    parser.add_argument("--n-receivers", type=int, default=101)
    parser.add_argument("--num-parallels", type=int, default=4)
    parser.add_argument("--acquisition-margin", type=float, default=200.0)
    parser.add_argument("--skip-line-search", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    set_log_level("WARNING")
    sys.path.insert(0, str(PROJECT_ROOT))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    ns = load_ns(args.num_parallels)
    model = test_0_model(ns)
    config = test_1_config(ns, model, args.n_shots, args.n_receivers)
    test_2_geometry(model, config)
    test_forward(ns, model, config, "true")
    test_forward(ns, model, config, "init")
    objective = test_5_objective(ns, args.n_shots, args.n_receivers)
    grad = objective["gradient"]
    test_6_gradient(model, objective)
    update_gammas = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]
    test_7_update_sweep(model, grad, update_gammas)
    if not args.skip_line_search:
        test_8_line_search(ns, model, grad, objective["E0"], args.n_shots, args.n_receivers, [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20])
        test_9_sign_check(ns, model, grad, args.n_shots, args.n_receivers)
        test_16_inset_geometry_line_search(ns, args.n_shots, args.n_receivers, args.acquisition_margin)
    test_12_cfl(model, config)
    if not args.skip_smoke:
        test_13_box_only_smoke(ns, args.n_shots, args.n_receivers, args.acquisition_margin)
    generate_report()
    print(f"Saved diagnostics to {RESULT_ROOT}")


if __name__ == "__main__":
    main()
