"""Generate 2,400 offline Airy labels across 80 independent geometries.

Scene IDs, not receiver positions, define train/validation/test partitions.
The label optimizer is offline only; the deployed predictor has no search.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.generate_scene_labels import optimize_label
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio
from direct_network.core.paths import ROOT


OUT = ROOT / "artifacts"
N_SCENES = 80
POSITIONS_PER_SCENE = 30
SEED = 20261104


def build_scenes() -> list[dict]:
    rng = np.random.default_rng(SEED)
    records = []
    for scene_id in range(N_SCENES):
        scene = replace(
            Scene(),
            obstacle_z_m=float(rng.uniform(.18, .22)),
            obstacle_x_m=float(rng.uniform(-.012, .012)),
            obstacle_depth_m=float(rng.uniform(.020, .040)),
            obstacle_width_m=float(rng.uniform(.075, .105)),
            receiver_z_m=float(rng.uniform(.34, .40)),
            n_grid=8192,
            obstacle_step_m=.0015,
        )
        # Rare geometries have no position above the activation threshold.
        # Resample only that scene, keeping every saved scene ID stable.
        check_grid = np.linspace(-.11, .11, 881)
        if sum(blockage_ratio(scene, float(x)) >= scene.activation_bl for x in check_grid) < POSITIONS_PER_SCENE:
            local_rng = np.random.default_rng(SEED + 74000 + scene_id)
            for _ in range(100):
                candidate = replace(scene,
                                    obstacle_width_m=float(local_rng.uniform(.085, .105)),
                                    obstacle_x_m=float(local_rng.uniform(-.006, .006)),
                                    obstacle_depth_m=float(local_rng.uniform(.025, .040)))
                if sum(blockage_ratio(candidate, float(x)) >= candidate.activation_bl
                       for x in check_grid) >= POSITIONS_PER_SCENE:
                    scene = candidate
                    break
            else:
                raise RuntimeError(f"Cannot construct active scene {scene_id}")
        split = "train" if scene_id < 48 else "validation" if scene_id < 64 else "test"
        records.append({"scene_id": scene_id, "split": split, **asdict(scene)})
    return records


def positions(scene: Scene, scene_id: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1000 + scene_id)
    grid = np.linspace(-.11, .11, 881)
    active = grid[np.array([blockage_ratio(scene, float(x)) >= scene.activation_bl for x in grid])]
    if len(active) < POSITIONS_PER_SCENE:
        raise ValueError(f"Scene {scene_id} has too little deep-shadow range")
    if scene_id < 38:
        # Preserve the coordinates of the first checkpointed scenes. Later
        # scenes use the active grid because some active sets are disconnected.
        low, high = float(active.min()), float(active.max())
        core = rng.uniform(low + .001, high - .001, POSITIONS_PER_SCENE - 6)
        boundary = np.r_[rng.uniform(low, low + .005, 3),
                         rng.uniform(high - .005, high, 3)]
        values = np.sort(np.r_[core, boundary])
        if any(blockage_ratio(scene, float(x)) < scene.activation_bl for x in values):
            raise AssertionError("Checkpointed sample changed")
        return values
    # The active set can be disconnected due to diffraction ripples; sample
    # actual active grid points instead of the interval between its extremes.
    core = rng.choice(active, POSITIONS_PER_SCENE - 6, replace=False)
    remaining = active[~np.isin(active, core)]
    boundary = np.r_[rng.choice(remaining[:min(12, len(remaining) // 2)], 3, replace=False),
                     rng.choice(remaining[-min(12, len(remaining) // 2):], 3, replace=False)]
    values = np.sort(np.r_[core, boundary])
    return values


def main() -> None:
    OUT.mkdir(exist_ok=True)
    scenes = build_scenes()
    scenes_path = OUT / "general_scenes.json"
    scenes_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    target = OUT / "general_labels.csv"
    existing = pd.read_csv(target) if target.exists() else pd.DataFrame()
    completed = set(existing.scene_id.unique()) if len(existing) else set()
    if len(existing) and any(existing.groupby("scene_id").size() != POSITIONS_PER_SCENE):
        raise ValueError("Interrupted scene found; remove its rows before resuming")
    for record in scenes:
        scene_id = record["scene_id"]
        if scene_id in completed:
            continue
        scene = Scene(**{key: value for key, value in record.items() if key not in ("scene_id", "split")})
        engine = AngularSpectrum(scene)
        rows = []
        for x in positions(scene, scene_id):
            rows.append({"scene_id": scene_id, "split": record["split"],
                         **optimize_label(scene, engine, float(x))})
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        existing.to_csv(target, index=False, float_format="%.12g")
        print(f"offline labels: {scene_id + 1}/{N_SCENES} scenes, {len(existing)} rows", flush=True)
    if len(existing) != N_SCENES * POSITIONS_PER_SCENE:
        raise AssertionError("General label dataset is incomplete")
    print(target)


if __name__ == "__main__":
    main()
