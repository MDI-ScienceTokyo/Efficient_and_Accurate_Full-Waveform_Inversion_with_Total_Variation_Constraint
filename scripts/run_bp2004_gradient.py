from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from devito import Eq, Function, Inc, Operator, TimeFunction, solve

from bp2004_utils import ensure_dirs, load_config, npz_meta, read_npz_meta, save_velocity_plot
from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


def make_model(vp_m_s: np.ndarray, dx: float, dz: float, nbl: int) -> SeismicModel:
    vp_km_s = (vp_m_s / 1000.0).astype(np.float32)
    return SeismicModel(space_order=2, vp=vp_km_s, origin=(0.0, 0.0), shape=vp_km_s.shape, dtype=np.float32, spacing=(dz, dx), nbl=nbl, bcs="damp", fs=False)


def create_gradient_operator(model: SeismicModel, geometry: AcquisitionGeometry, solver: AcousticWaveSolver) -> Operator:
    grad = Function(name="grad", grid=model.grid)
    u = TimeFunction(name="u", grid=model.grid, save=geometry.nt, time_order=2, space_order=solver.space_order)
    v = TimeFunction(name="v", grid=model.grid, save=None, time_order=2, space_order=solver.space_order)
    eqns = [Eq(v.backward, solve(model.m * v.dt2 - v.laplace + model.damp * v.dt.T, v.backward))]
    rec_term = geometry.rec.inject(field=v.backward, expr=geometry.rec * model.grid.stepping_dim.spacing**2 / model.m)
    gradient_update = Inc(grad, -u.dt2 * v * model.m**1.5)
    return Operator(eqns + rec_term + [gradient_update], subs=model.spacing_map, name="BP2004Gradient")


def objective_and_gradient(vp0_m_s: np.ndarray, dx: float, dz: float, observed: dict, modeling: dict) -> tuple[float, np.ndarray]:
    nbl = int(modeling["absorbing_boundary_cells"])
    source_positions = observed["source_positions"].astype(np.float32)
    receiver_positions = observed["receiver_positions"].astype(np.float32)
    d_obs = observed["d_obs"].astype(np.float32)
    dt_s = float(observed["dt"])
    nt = int(observed["nt"])
    end_time_ms = (nt - 1) * dt_s * 1000.0
    f0_khz = float(modeling["source_frequency_hz"]) / 1000.0

    model = make_model(vp0_m_s, dx, dz, nbl)
    geometry = AcquisitionGeometry(model, receiver_positions, source_positions[:1], 0.0, end_time_ms, f0=f0_khz, src_type="Ricker")
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    grad_op = create_gradient_operator(model, geometry, solver)

    grad = Function(name="grad", grid=model.grid)
    objective = 0.0
    for i, src_pos in enumerate(source_positions):
        geometry.src_positions[0, :] = src_pos
        d_syn = Receiver(name=f"d_syn_{i}", grid=model.grid, time_range=geometry.time_axis, coordinates=receiver_positions)
        residual = Receiver(name=f"residual_{i}", grid=model.grid, time_range=geometry.time_axis, coordinates=receiver_positions)
        _, wavefield, _ = solver.forward(vp=model.vp, save=True, rec=d_syn)
        residual.data[:] = d_syn.data[:] - d_obs[i, : geometry.nt, :]
        residual64 = residual.data.astype(np.float64, copy=False)
        objective += 0.5 * float(np.sum(residual64**2))
        grad_op.apply(rec=residual, grad=grad, u=wavefield, dt=solver.dt, vp=model.vp)
        print(f"[gradient] shot {i + 1}/{source_positions.shape[0]} partial_objective={objective:.6e}")

    grad_m_s = (-grad.data[model.nbl : -model.nbl, model.nbl : -model.nbl] / 1000.0).astype(np.float32)
    return objective, grad_m_s


def check_directional_derivative(vp0: np.ndarray, grad: np.ndarray, dx: float, dz: float, observed: dict, modeling: dict, eps: float) -> None:
    rng = np.random.default_rng(0)
    direction = rng.normal(size=vp0.shape).astype(np.float32)
    direction /= np.linalg.norm(direction.ravel()) + 1e-12
    j_plus, _ = objective_and_gradient(vp0 + eps * direction, dx, dz, observed, modeling)
    j_minus, _ = objective_and_gradient(vp0 - eps * direction, dx, dz, observed, modeling)
    fd = (j_plus - j_minus) / (2.0 * eps)
    adj = float(np.sum(grad.astype(np.float64) * direction.astype(np.float64)))
    rel = abs(fd - adj) / max(abs(fd), abs(adj), 1e-12)
    print(f"Finite-difference check: fd={fd:.6e}, <g,p>={adj:.6e}, relative_error={rel:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute one BP2004 FWI objective and gradient at the initial model.")
    parser.add_argument("--config", default="configs/bp2004_gradient.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--observed", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--check-gradient", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    _, processed_dir, output_dir = ensure_dirs(config)
    model_path = Path(args.model) if args.model else processed_dir / "bp2004_initial_crop.npz"
    observed_path = Path(args.observed) if args.observed else output_dir / "observed_data.npz"
    output_path = Path(args.output) if args.output else output_dir / "gradient_result.npz"

    model_data = np.load(model_path, allow_pickle=True)
    obs_data = np.load(observed_path, allow_pickle=True)
    observed = {key: obs_data[key] for key in obs_data.files}
    vp0 = model_data["vp0"].astype(np.float32)
    vp_true = model_data["vp_true"].astype(np.float32)
    dx = float(model_data["dx"])
    dz = float(model_data["dz"])

    objective, gradient = objective_and_gradient(vp0, dx, dz, observed, config["modeling"])
    if not np.isfinite(objective):
        raise FloatingPointError(f"Objective is not finite: {objective}")
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Gradient contains NaN or Inf values.")

    grad_norm = float(np.linalg.norm(gradient.ravel()))
    print(f"FWI objective at initial model: {objective:.6e}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Gradient min/max/norm: {float(np.min(gradient)):.6e} / {float(np.max(gradient)):.6e} / {grad_norm:.6e}")

    meta = read_npz_meta(model_data["meta"])
    meta.update({"gradient": dict(config["gradient"]), "parameterization": "velocity", "gradient_unit": "objective derivative per m/s"})
    np.savez(output_path, objective=objective, gradient=gradient, vp0=vp0, vp_true=vp_true, dx=dx, dz=dz, meta=npz_meta(meta))
    save_velocity_plot(gradient, output_dir / "gradient.png", "BP2004 FWI gradient", dx, dz, cmap="seismic", label="Gradient")

    import matplotlib.pyplot as plt

    extent = [0, (vp0.shape[1] - 1) * dx / 1000.0, (vp0.shape[0] - 1) * dz / 1000.0, 0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, arr, title, cmap in zip(axes, [vp_true, vp0, gradient], ["True velocity", "Initial velocity", "Gradient"], ["viridis", "viridis", "seismic"]):
        im = ax.imshow(arr, aspect="auto", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Depth [km]")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(output_dir / "vp_true_initial_gradient.png", dpi=180)
    plt.close(fig)

    if args.check_gradient or bool(config["gradient"].get("check_gradient", False)):
        check_directional_derivative(vp0, gradient, dx, dz, observed, config["modeling"], float(config["gradient"]["finite_difference_epsilon"]))

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
