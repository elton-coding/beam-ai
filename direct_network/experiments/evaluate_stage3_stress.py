"""Reproducible sensing stress tests for the fixed-scene causal pipeline.

These tests use independent random streams. No target position enters beam
selection; exact position is used only to synthesize observations and score it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.evaluate_sensing import crossing_trajectory
from direct_network.experiments.evaluate_stage2 import CommunicationLookup, FRAME_S
from direct_network.core.physics import Scene, blockage_ratio
from direct_network.core.sensing import CausalPositionTracker
from direct_network.core.sensing_stage2 import CONDITIONS, PilotLibrary, PositionInstrument
from direct_network.core.paths import ROOT


OUT = ROOT / "results" / "stage3"
TAU = .0001
RUNS = 20


def make_cases() -> list[dict]:
    cases = []
    for mode, n in (("passive", 9), ("passive", 21), ("active", 9), ("active", 21)):
        cases.append({"case": f"baseline_{mode}_{n}", "mode": mode, "beams": n})
    cases += [
        {"case": "adaptive_passive_9", "mode": "passive", "beams": 21, "selected_beams": 9},
        {"case": "adaptive_active_9", "mode": "active", "beams": 21, "selected_beams": 9},
        {"case": "no_background_passive_21", "mode": "passive", "beams": 21, "background": 0},
        {"case": "stale_background_passive_21", "mode": "passive", "beams": 21, "stale": True},
        {"case": "mismatch_passive_21", "mode": "passive", "beams": 21, "mismatch": True},
        {"case": "mismatch_active_9", "mode": "active", "beams": 9, "mismatch": True},
    ]
    for db in (-12, -6, 0, 6):
        cases.append({"case": f"active_9_power_{db:+d}db", "mode": "active", "beams": 9,
                      "pilot_power_db": db})
    return cases


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=.00075)
    library = PilotLibrary(scene)
    # Independent forward geometry and discretization; estimator keeps nominal templates.
    truth_scene = replace(scene, obstacle_x_m=scene.obstacle_x_m + .003,
                          obstacle_width_m=scene.obstacle_width_m * 1.05,
                          obstacle_depth_m=scene.obstacle_depth_m * 1.1,
                          n_grid=8192, obstacle_step_m=.001)
    truth_library = PilotLibrary(truth_scene)
    trajectory = crossing_trajectory(motion="variable")
    lookup = CommunicationLookup(scene, library, {"variable": trajectory})
    data_noise = lookup.reference_power / 10
    condition = CONDITIONS[1]
    rows = []
    null_rows = []
    for case_id, case in enumerate(make_cases()):
        mode = case["mode"]
        n_used = case.get("selected_beams", case["beams"])
        instrument = PositionInstrument(
            library, mode, condition, case["beams"], 198000 + case_id,
            background_snapshots=case.get("background", 64),
            truth_library=truth_library if case.get("mismatch", False) else None)
        if case.get("background") == 0:
            # No blank-scene reference is available: use analytic noise-only
            # threshold. Residual obstacle echo then drives actual false alarms.
            instrument.threshold = float(-instrument.noise_power * np.log(
                1 - .99 ** (1 / n_used)))
            instrument.threshold_cache = {tuple(range(case["beams"])): instrument.threshold}
        null_rng = np.random.default_rng(199000 + case_id)
        for phase in ("initial", "drifted"):
            if phase == "drifted" and case.get("stale"):
                instrument.clutter *= 1.25 * np.exp(.35j)
            null = instrument._null_peaks(null_rng, 10000)
            null_rows.append({"case": case["case"], "phase": phase,
                              "false_alarm_rate": float(np.mean(null > instrument.threshold)),
                              "threshold": instrument.threshold})
            if phase == "drifted" and case.get("stale"):
                instrument.clutter /= 1.25 * np.exp(.35j)
        for run in range(RUNS):
            rng = np.random.default_rng(210000 + case_id * 1000 + run)
            rho = 10 ** (rng.uniform(-6, 0) / 10) if mode == "passive" else 10 ** (
                case.get("pilot_power_db", 0) / 10)
            tracker = CausalPositionTracker(dt_s=FRAME_S)
            instrument.reset()
            last_observed = None
            for t, actual in enumerate(trajectory):
                if case.get("stale") and t == 50:
                    instrument.clutter *= 1.25 * np.exp(.35j)
                if "selected_beams" in case and tracker.initialized:
                    pointing = library.pointing_x[instrument.indices]
                    selected = np.sort(np.argsort(np.abs(pointing - tracker.state[0]))[:n_used])
                elif "selected_beams" in case:
                    selected = np.rint(np.linspace(0, case["beams"] - 1, n_used)).astype(int)
                else:
                    selected = None
                measured, peak = instrument.observe(float(actual), rng, rho,
                                                    selected_local_indices=selected)
                estimate, uncertainty = tracker.step(measured)
                if measured is not None:
                    last_observed = measured
                hold = float(last_observed if last_observed is not None else estimate)
                power_network, power_focus = lookup.at(estimate, t)
                power_hold, _ = lookup.at(hold, t)
                fraction = ((FRAME_S - n_used * TAU) /
                            (FRAME_S + (64 if mode == "passive" and case.get("background", 64) else 0)
                             * n_used * TAU / len(trajectory)))
                rows.append({"case": case["case"], "mode": mode, "run": run, "frame": t,
                             "n_pilots": n_used, "actual_x_m": actual,
                             "deep_shadow": blockage_ratio(scene, float(actual)) >= scene.activation_bl,
                             "detected": measured is not None, "measured_x_m": measured,
                             "estimated_x_m": estimate, "estimated_std_m": uncertainty,
                             "hold_x_m": hold, "peak": peak,
                             "threshold": instrument.last_threshold,
                             "power_network": power_network, "power_focus": power_focus,
                             "power_hold": power_hold,
                             "power_true_network": lookup.true_network[t],
                             "power_true_focus": lookup.true_focus[t],
                             "effective_rate_bpshz": fraction * np.log2(1 + power_network / data_noise),
                             "effective_focus_bpshz": fraction * np.log2(1 + power_focus / data_noise),
                             "effective_hold_bpshz": fraction * np.log2(1 + power_hold / data_noise),
                             "data_fraction": fraction})
            if case.get("stale"):
                instrument.clutter /= 1.25 * np.exp(.35j)
        print(f"completed {case['case']}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "stress_frames.csv", index=False, float_format="%.12g")
    pd.DataFrame(null_rows).to_csv(OUT / "null_false_alarm.csv", index=False)
    summary = []
    for (case, region), group in frame.groupby(["case", "deep_shadow"]):
        miss = (~group.detected).to_numpy()
        streaks = []
        for _, run_group in group.groupby("run"):
            current = longest = 0
            for detected in run_group.detected:
                current = 0 if detected else current + 1
                longest = max(longest, current)
            streaks.append(longest)
        errors = 1000 * np.abs(group.estimated_x_m - group.actual_x_m)
        rate = group.effective_rate_bpshz
        summary.append({"case": case, "region": "deep" if region else "other",
                        "frames": len(group), "runs": group.run.nunique(),
                        "detection_pct": float(100 * group.detected.mean()),
                        "tracked_mae_mm": float(errors.mean()),
                        "tracked_p95_mm": float(errors.quantile(.95)),
                        "longest_miss_streak_mean_frames": float(np.mean(streaks)),
                        "rate_mean_bpshz": float(rate.mean()),
                        "rate_p05_bpshz": float(rate.quantile(.05)),
                        "outage_rate_below_1bpshz_pct": float(100 * (rate < 1).mean()),
                        "network_vs_estimated_focus_rate_pct": float(100 * (rate.mean() / group.effective_focus_bpshz.mean() - 1)),
                        "network_vs_hold_rate_pct": float(100 * (rate.mean() / group.effective_hold_bpshz.mean() - 1)),
                        "oracle_position_network_rate_bpshz": float(
                            (group.data_fraction * np.log2(1 + group.power_true_network / data_noise)).mean())})
    result = {"simulation_only": True, "runs_per_case": RUNS, "frames_per_run": len(trajectory),
              "data_noise_reference": data_noise, "pilot_duration_s": TAU,
              "mismatch_truth_scene": {"obstacle_x_m": truth_scene.obstacle_x_m,
                                       "obstacle_width_m": truth_scene.obstacle_width_m,
                                       "obstacle_depth_m": truth_scene.obstacle_depth_m,
                                       "n_grid": truth_scene.n_grid},
              "summary": summary, "null_false_alarm": null_rows}
    (OUT / "stress_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {len(frame)} frames", flush=True)


if __name__ == "__main__":
    main()
