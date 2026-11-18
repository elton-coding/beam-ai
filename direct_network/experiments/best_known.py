"""Offline Airy reference and a rigorous passive-power ceiling.

The normalized transmitter emits unit power. Angular-spectrum propagation and
absorbing masks are contractions, so power in the receiver aperture cannot
exceed one. This is a rigorous but relaxed bound for the Airy phase-only family.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.generate_scene_labels import optimize_label
from direct_network.core.physics import AngularSpectrum, Scene, gain_db
from direct_network.core.paths import ROOT




def describe(gaps: np.ndarray) -> dict:
    return {"mean_db": float(np.mean(gaps)), "min_db": float(np.min(gaps)),
            "max_db": float(np.max(gaps))}


def main() -> None:
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=0.00075)
    engine = AngularSpectrum(scene)
    transmitted_power = float(np.sum(np.abs(engine.source(0, 0.36, 0)) ** 2) * engine.dx)
    if abs(transmitted_power - 1.0) > 1e-12:
        raise AssertionError("The source is not normalized to unit power")
    static = pd.read_csv(ROOT / "results" / "static_results.csv")
    static = static[static.trajectory == "deep_shadow"].copy()
    delayed = pd.read_csv(ROOT / "results" / "delayed_results.csv").copy()
    positions = sorted(set(np.round(np.r_[static.receiver_x_m, delayed.actual_x_m], 9)))
    offline = {}
    for i, x in enumerate(positions, 1):
        offline[x] = optimize_label(scene, engine, float(x))
        if i % 10 == 0 or i == len(positions):
            print(f"offline Airy reference {i}/{len(positions)}", flush=True)

    static_rows = []
    for row in static.itertuples():
        x = float(row.receiver_x_m)
        label = offline[round(x, 9)]
        reference = max(float(row.power_direct), float(label["power_best"]))
        static_rows.append({"receiver_x_m": x, "power_focus": row.power_focus,
                            "power_direct": row.power_direct,
                            "power_best_known_airy": reference,
                            "power_theoretical_ceiling": transmitted_power,
                            "B_best_per_m": label["B_per_m"], "F_best_m": label["F_m"],
                            "theta_best_deg": label["theta_deg"],
                            "best_known_headroom_db": gain_db(reference, row.power_direct),
                            "theoretical_headroom_db": gain_db(transmitted_power, row.power_direct),
                            "best_known_gain_vs_focus_db": gain_db(reference, row.power_focus),
                            "theoretical_gain_vs_focus_db": gain_db(transmitted_power, row.power_focus)})
    a = pd.DataFrame(static_rows)
    a.to_csv(ROOT / "results" / "bounds_static.csv", index=False, float_format="%.12g")

    delayed_rows = []
    for row in delayed.itertuples():
        x = float(row.actual_x_m)
        label = offline[round(x, 9)]
        predicted = float(row.power_predicted_direct)
        reference = max(predicted, float(row.power_oracle_direct), float(label["power_best"]))
        delayed_rows.append({"time_index": row.time_index, "actual_x_m": x,
                             "power_predicted_direct": predicted,
                             "power_focus_at_true_x": row.power_oracle_focus,
                             "power_best_known_airy": reference,
                             "power_theoretical_ceiling": transmitted_power,
                             "best_known_headroom_db": gain_db(reference, predicted),
                             "theoretical_headroom_db": gain_db(transmitted_power, predicted),
                             "best_known_gain_vs_focus_db": gain_db(reference, row.power_oracle_focus),
                             "theoretical_gain_vs_focus_db": gain_db(transmitted_power, row.power_oracle_focus)})
    b = pd.DataFrame(delayed_rows)
    b.to_csv(ROOT / "results" / "bounds_delayed.csv", index=False, float_format="%.12g")
    summary = {
        "interpretation": {
            "best_known_airy": "Offline multi-start local search plus evaluated direct beams; feasible reference, not a certified global optimum.",
            "theoretical_ceiling": "Unit transmitted power; rigorous for this passive normalized simulator, but allows arbitrary complex wavefronts and is loose for Airy phase-only beams.",
        },
        "transmitted_power": transmitted_power,
        "unique_reference_positions": len(positions),
        "offline_physics_evaluations": int(sum(offline[x]["physics_evaluations"] for x in positions)),
        "static_deep_shadow": {
            "positions": len(a),
            "direct_mean_gain_vs_focus_db": float(np.mean(10 * np.log10(a.power_direct / a.power_focus))),
            "best_known_mean_gain_vs_focus_db": float(a.best_known_gain_vs_focus_db.mean()),
            "theoretical_mean_gain_vs_focus_db": float(a.theoretical_gain_vs_focus_db.mean()),
            "gap_to_best_known": describe(a.best_known_headroom_db.to_numpy()),
            "gap_to_theoretical_ceiling": describe(a.theoretical_headroom_db.to_numpy()),
            "positions_with_better_known_airy": int(np.count_nonzero(a.best_known_headroom_db > 1e-6)),
        },
        "delayed_predicted_beam_at_true_x": {
            "frames": len(b),
            "predicted_mean_gain_vs_true_focus_db": float(np.mean(10 * np.log10(b.power_predicted_direct / b.power_focus_at_true_x))),
            "best_known_mean_gain_vs_true_focus_db": float(b.best_known_gain_vs_focus_db.mean()),
            "theoretical_mean_gain_vs_true_focus_db": float(b.theoretical_gain_vs_focus_db.mean()),
            "gap_to_best_known": describe(b.best_known_headroom_db.to_numpy()),
            "gap_to_theoretical_ceiling": describe(b.theoretical_headroom_db.to_numpy()),
            "frames_with_better_known_airy": int(np.count_nonzero(b.best_known_headroom_db > 1e-6)),
        },
    }
    (ROOT / "results" / "bounds_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
