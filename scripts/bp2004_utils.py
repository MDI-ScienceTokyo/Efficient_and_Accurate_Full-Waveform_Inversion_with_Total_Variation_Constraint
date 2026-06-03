from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "raw_dir": "data/raw/bp2004",
        "processed_dir": "data/processed/bp2004",
        "output_dir": "outputs/bp2004_gradient",
    },
    "bp2004": {
        "nz_full": 1911,
        "nx_full": 5395,
        "dz_full_m": 6.25,
        "dx_full_m": 12.5,
    },
    "crop": {
        "x_min_m": 25000.0,
        "x_max_m": 45000.0,
        "z_min_m": 0.0,
        "z_max_m": 6000.0,
    },
    "resample": {
        "dx_m": 25.0,
        "dz_m": 25.0,
    },
    "initial_model": {
        "smoothing_sigma_z_grid": 20,
        "smoothing_sigma_x_grid": 40,
        "min_velocity": 1450.0,
        "max_velocity": 5500.0,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        print(f"[warn] config not found, using defaults: {config_path}")
        return config

    try:
        import yaml
    except ImportError:
        loaded = load_simple_yaml(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, loaded)


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported config line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value.strip())
    return root


def npz_meta(meta: dict[str, Any]) -> np.ndarray:
    return np.array(meta, dtype=object)


def read_npz_meta(value: np.ndarray | Any) -> dict[str, Any]:
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return item
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, output_dir


def save_velocity_plot(arr: np.ndarray, path: Path, title: str, dx: float, dz: float, cmap: str = "viridis", label: str = "Velocity [m/s]") -> None:
    import matplotlib.pyplot as plt

    nz, nx = arr.shape
    extent = [0.0, (nx - 1) * dx / 1000.0, (nz - 1) * dz / 1000.0, 0.0]
    plt.figure(figsize=(10, 4))
    plt.imshow(arr, cmap=cmap, aspect="auto", extent=extent)
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel("X [km]")
    plt.ylabel("Depth [km]")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
