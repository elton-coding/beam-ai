"""Scene sets, geometry hypotheses and parallel helpers shared by experiments."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from direct_network.core.paths import ARTIFACTS
from direct_network.core.physics import Scene, blockage_ratio
from nfbeam.channel import perturbed


PACKAGE = Path(__file__).resolve().parent
RESULTS = PACKAGE / "results"
FIGURES = PACKAGE / "figures"

GRID = dict(n_grid=8192, obstacle_step_m=0.0015)

# Geometry hypotheses used by robust design: obstacle centre shift and width scale.
HYPOTHESIS_SHIFTS_M = (-0.003, 0.0, 0.003)
HYPOTHESIS_WIDTH_SCALES = (0.95, 1.0, 1.05)


def load_scenes(split: str | None = "test") -> list[tuple[int, Scene]]:
    records = json.loads((ARTIFACTS / "general_scenes.json").read_text(encoding="utf-8"))
    out = []
    for r in records:
        if split is not None and r["split"] != split:
            continue
        scene = Scene(**{k: v for k, v in r.items() if k not in ("scene_id", "split")})
        out.append((int(r["scene_id"]), replace(scene, **GRID)))
    return out


def fixed_scene() -> Scene:
    return replace(Scene(), **GRID)


def hypotheses(scene: Scene) -> list[Scene]:
    return [perturbed(scene, dx_m=dx, width_scale=ws)
            for dx in HYPOTHESIS_SHIFTS_M for ws in HYPOTHESIS_WIDTH_SCALES]


def random_mismatch(scene: Scene, rng: np.random.Generator, strong: bool = False) -> Scene:
    """Truth geometry differing from the nominal one; never on the hypothesis grid."""
    if strong:
        return perturbed(scene, dx_m=rng.uniform(-0.006, 0.006), width_scale=rng.uniform(0.9, 1.1),
                         dz_m=rng.uniform(-0.005, 0.005), depth_scale=rng.uniform(0.85, 1.15))
    return perturbed(scene, dx_m=rng.uniform(-0.004, 0.004), width_scale=rng.uniform(0.94, 1.06))


def deep(scene: Scene, x: float) -> bool:
    return blockage_ratio(scene, float(x)) >= scene.activation_bl


def run_parallel(fn: Callable, items: Iterable, workers: int | None = None) -> list:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    items = list(items)
    workers = workers or min(len(items), 12, max(1, (os.cpu_count() or 2) - 2))
    if workers <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def scene_bootstrap(values_by_scene: np.ndarray, seed: int = 20261118, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values_by_scene, dtype=float)
    draws = rng.integers(0, len(v), (n, len(v)))
    return np.quantile(v[draws].mean(axis=1), [0.025, 0.975]).tolist()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
