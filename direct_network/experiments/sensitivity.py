"""Re-score fixed one-shot beams on a finer propagation grid."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.core.physics import AngularSpectrum, Scene, gain_db, parameter_bounds
from direct_network.core.paths import ROOT




def main() -> None:
    scene = replace(Scene(), n_grid=32768, obstacle_step_m=0.000375)
    engine = AngularSpectrum(scene)
    static = pd.read_csv(ROOT / "results" / "static_results.csv")
    static = static[static.trajectory == "deep_shadow"].copy()
    differences = []
    for row in static.itertuples():
        x = float(row.receiver_x_m)
        _, _, f0, theta0 = parameter_bounds(scene, x)
        focus = engine.received_power(x, 0, f0, theta0)
        direct = engine.received_power(x, row.B_per_m, row.F_m, row.theta_deg)
        differences.append(gain_db(direct, focus) - row.gain_db)
    delayed = pd.read_csv(ROOT / "results" / "delayed_results.csv")
    gain_base = delayed.gain_pred_vs_stale_db.to_numpy()
    gain_fine = []
    for row in delayed.itertuples():
        actual = float(row.actual_x_m)
        pred = engine.received_power(actual, row.B_predicted, row.F_predicted, row.theta_predicted)
        stale = engine.received_power(actual, row.B_stale, row.F_stale, row.theta_stale)
        gain_fine.append(gain_db(pred, stale))
    gain_fine = np.asarray(gain_fine)
    result = {
        "fine_grid_points": scene.n_grid,
        "fine_obstacle_step_mm": scene.obstacle_step_m * 1000,
        "fixed_parameters_only": True,
        "deep_shadow_mean_gain_change_db": float(np.mean(differences)),
        "deep_shadow_max_abs_gain_change_db": float(np.max(np.abs(differences))),
        "delayed_mean_gain_base_db": float(np.mean(gain_base)),
        "delayed_mean_gain_fine_db": float(np.mean(gain_fine)),
        "delayed_mean_gain_change_db": float(np.mean(gain_fine - gain_base)),
        "delayed_max_abs_frame_gain_change_db": float(np.max(np.abs(gain_fine - gain_base))),
    }
    (ROOT / "results" / "sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
