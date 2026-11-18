"""High-resolution held-out-geometry evaluation of the general Airy network."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.core.general_model import GeneralSceneNetwork
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio, features, parameter_bounds
from direct_network.core.paths import ROOT


OUT = ROOT / "results" / "stage3"


def direct_parameters(scene: Scene, model: GeneralSceneNetwork, x: float) -> np.ndarray:
    bl = blockage_ratio(scene, x)
    lo, hi, f0, theta0 = parameter_bounds(scene, x)
    if bl < scene.activation_bl:
        return np.array([0.0, f0, theta0])
    b, delta_f, delta_theta = model.predict(features(scene, x, bl))
    return np.clip([b, f0 + delta_f, theta0 + delta_theta], lo, hi)


def summarize(frame: pd.DataFrame) -> dict:
    if not len(frame):
        return {}
    ratio = frame.power_network / frame.power_focus
    scene_mean = frame.assign(gain_pct=100 * (ratio - 1)).groupby("scene_id").gain_pct.mean()
    rng = np.random.default_rng(20261104)
    samples = rng.choice(scene_mean.to_numpy(), size=(10000, len(scene_mean)), replace=True)
    return {"positions": len(frame), "scenes": frame.scene_id.nunique(),
            "network_vs_focus_mean_pointwise_pct": float(100 * np.mean(ratio - 1)),
            "network_vs_focus_mean_db": float(np.mean(10 * np.log10(ratio))),
            "network_mean_normalized_power": float(frame.power_network.mean()),
            "strict_unit_power_upper_bound_headroom_pct": float(100 *
                (1 / frame.power_network.mean() - 1)),
            "network_better_fraction": float(np.mean(ratio > 1)),
            "network_gain_p05_pct": float(np.quantile(100 * (ratio - 1), .05)),
            "scene_mean_bootstrap_95pct_ci_pct": [float(v) for v in np.quantile(samples.mean(axis=1), [.025, .975])]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "artifacts" / "general_labels.csv")
    scene_records = json.loads((ROOT / "artifacts" / "general_scenes.json").read_text(encoding="utf-8"))
    model = GeneralSceneNetwork()
    rows = []
    for record in scene_records:
        if record["split"] != "test":
            continue
        scene_id = record["scene_id"]
        scene = Scene(**{k: v for k, v in record.items() if k not in ("scene_id", "split")})
        fine = replace(scene, n_grid=16384, obstacle_step_m=.00075)
        engine = AngularSpectrum(fine)
        labels = data[data.scene_id == scene_id]
        positions = [("labeled", float(row.rx_x_m), row) for row in labels.itertuples(index=False)]
        grid = np.linspace(-.11, .11, 881)
        active = grid[np.array([blockage_ratio(fine, float(x)) >= fine.activation_bl for x in grid])]
        extra_grid = active[np.rint(np.linspace(0, len(active) - 1, 21)).astype(int)]
        rng = np.random.default_rng(410000 + scene_id)
        for x in extra_grid:
            candidate = float(x + rng.uniform(-.00012, .00012))
            if blockage_ratio(fine, candidate) < fine.activation_bl:
                candidate = float(x)
            positions.append(("new_position", candidate, None))
        for group, x, label in positions:
            _, _, f0, theta0 = parameter_bounds(fine, x)
            focus = engine.received_power(x, 0.0, f0, theta0)
            direct = engine.received_power(x, *direct_parameters(fine, model, x))
            teacher = (engine.received_power(x, label.B_per_m, label.F_m, label.theta_deg)
                       if label is not None else np.nan)
            rows.append({"scene_id": scene_id, "group": group, "rx_x_m": x,
                         "obstacle_x_m": fine.obstacle_x_m, "obstacle_width_m": fine.obstacle_width_m,
                         "obstacle_z_m": fine.obstacle_z_m, "receiver_z_m": fine.receiver_z_m,
                         "blockage_ratio": blockage_ratio(fine, x), "power_focus": focus,
                         "power_network": direct, "power_offline_teacher": teacher})
        print(f"evaluated held-out scene {scene_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "general_scene_frames.csv", index=False, float_format="%.12g")
    result = {"scene_disjoint_test_scenes": 16, "physics_grid": 16384,
              "physics_obstacle_step_m": .00075,
              "labeled_positions": summarize(frame[frame.group == "labeled"]),
              "new_positions": summarize(frame[frame.group == "new_position"])}
    labeled = frame[frame.group == "labeled"]
    result["offline_teacher_vs_network"] = {
        "mean_pointwise_pct": float(100 * np.mean(labeled.power_offline_teacher / labeled.power_network - 1)),
        "mean_power_ratio_pct": float(100 * (labeled.power_offline_teacher.mean() /
                                               labeled.power_network.mean() - 1)),
        "teacher_better_fraction": float(np.mean(labeled.power_offline_teacher > labeled.power_network)),
        "note": "Teacher is a feasible offline coarse-grid search reference, not a certified global optimum."}
    (OUT / "general_scene_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
