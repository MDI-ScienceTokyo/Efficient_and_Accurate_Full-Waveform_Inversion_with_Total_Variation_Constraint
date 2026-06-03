from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bp2004_utils import build_geometry, ensure_dirs, load_config, npz_meta, read_npz_meta, save_shot_gather_plot
from lib.seismic.devito_example import AcquisitionGeometry, Receiver, SeismicModel
from lib.seismic.devito_example.acoustic import AcousticWaveSolver


def make_model(vp_m_s: np.ndarray, dx: float, dz: float, nbl: int) -> SeismicModel:
    vp_km_s = (vp_m_s / 1000.0).astype(np.float32)
    return SeismicModel(space_order=2, vp=vp_km_s, origin=(0.0, 0.0), shape=vp_km_s.shape, dtype=np.float32, spacing=(dz, dx), nbl=nbl, bcs="damp", fs=False)


def generate_observed_data(
    vp_true_m_s: np.ndarray,
    dx: float,
    dz: float,
    modeling: dict,
    noise_sigma: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    source_positions, receiver_positions = build_geometry(vp_true_m_s, dx, dz, modeling)
    nt_requested = int(modeling["nt"])
    dt_s = float(modeling["dt"])
    end_time_ms = (nt_requested - 1) * dt_s * 1000.0
    f0_khz = float(modeling["source_frequency_hz"]) / 1000.0
    nbl = int(modeling["absorbing_boundary_cells"])

    model = make_model(vp_true_m_s, dx, dz, nbl)
    geometry = AcquisitionGeometry(model, receiver_positions, source_positions[:1], 0.0, end_time_ms, f0=f0_khz, src_type="Ricker")
    solver = AcousticWaveSolver(model, geometry, space_order=4)

    d_obs = np.zeros((source_positions.shape[0], geometry.nt, receiver_positions.shape[0]), dtype=np.float32)
    for i, src_pos in enumerate(source_positions):
        geometry.src_positions[0, :] = src_pos
        rec = Receiver(name=f"d_obs_{i}", grid=model.grid, time_range=geometry.time_axis, coordinates=receiver_positions)
        solver.forward(vp=model.vp, rec=rec)
        d_obs[i] = rec.data[:]
        print(f"[forward] shot {i + 1}/{source_positions.shape[0]} source_depth={src_pos[0]:.1f} m source_x={src_pos[1]:.1f} m")

    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_sigma, size=d_obs.shape).astype(np.float32)
        d_obs = (d_obs + noise).astype(np.float32)
        print(f"[noise] added Gaussian noise: sigma={noise_sigma:g}, seed={seed}")

    return d_obs, source_positions, receiver_positions, geometry.src.data.copy(), float(geometry.time_axis.step) / 1000.0, int(geometry.nt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic observed data on the BP2004 true model.")
    parser.add_argument("--config", default="configs/bp2004_gradient.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--noise-sigma", type=float, default=None, help="Gaussian noise standard deviation added to d_obs. Default comes from config.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Gaussian noise.")
    args = parser.parse_args()

    config = load_config(args.config)
    _, processed_dir, output_dir = ensure_dirs(config)
    input_path = Path(args.input) if args.input else processed_dir / "bp2004_initial_crop.npz"
    output_path = Path(args.output) if args.output else output_dir / "observed_data.npz"

    data = np.load(input_path, allow_pickle=True)
    vp_true = data["vp_true"].astype(np.float32)
    dx = float(data["dx"])
    dz = float(data["dz"])
    noise_sigma = float(args.noise_sigma if args.noise_sigma is not None else config["modeling"].get("noise_sigma", 0.0))
    seed = args.seed if args.seed is not None else config["modeling"].get("noise_seed", None)
    d_obs, source_positions, receiver_positions, wavelet, dt_actual, nt_actual = generate_observed_data(vp_true, dx, dz, config["modeling"], noise_sigma=noise_sigma, seed=seed)

    meta = read_npz_meta(data["meta"])
    meta.update(
        {
            "modeling": dict(config["modeling"]),
            "calculation_velocity_unit": "km/s",
            "stored_velocity_unit": "m/s",
            "time_unit": "s",
            "coordinate_order": "(depth_m, x_m)",
            "noise_sigma": noise_sigma,
            "noise_seed": seed,
        }
    )
    np.savez(
        output_path,
        d_obs=d_obs,
        source_positions=source_positions,
        receiver_positions=receiver_positions,
        wavelet=wavelet,
        dt=dt_actual,
        nt=nt_actual,
        meta=npz_meta(meta),
    )
    save_shot_gather_plot(d_obs[0], output_dir / "shot_gather_example.png", dt_actual)
    print(f"Generated observed data: n_shots={d_obs.shape[0]}, nt={d_obs.shape[1]}, n_receivers={d_obs.shape[2]}, noise_sigma={noise_sigma:g}")
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
