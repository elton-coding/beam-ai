"""Derive reliability, ROC, resource and ablation tables from saved frames."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.core.physics import Scene
from direct_network.core.sensing_stage2 import CONDITIONS, PilotLibrary, PositionInstrument
from direct_network.core.paths import ROOT


OUT = ROOT / "results" / "stage3"


def main() -> None:
    old = pd.read_csv(ROOT / "results" / "stage2" / "frames.csv")
    stress = pd.read_csv(OUT / "stress_frames.csv")
    ref = json.loads((ROOT / "results" / "stage2" / "summary.json").read_text(encoding="utf-8"))
    noise = ref["data_noise_power_at_10db_reference"]
    base = old[(old.condition == "reflective") & (old.motion == "variable")].copy()
    base["fraction"] = np.where(base["mode"] == "passive",
                                (1 - base.n_pilots * .0001 / .02) /
                                (1 + 64 * base.n_pilots * .0001 / (101 * .02)),
                                1 - base.n_pilots * .0001 / .02)
    for col in ("power_network", "power_estimated_focus", "power_hold", "power_true_network"):
        base["rate_" + col] = base.fraction * np.log2(1 + base[col] / noise)
    reliability = []
    for (mode, beams, deep), group in base.groupby(["mode", "n_pilots", "deep_shadow"]):
        error = 1000 * np.abs(group.estimated_x_m - group.actual_x_m)
        rate = group.rate_power_network
        max_streak = []
        for _, run in group.groupby("run"):
            streak = longest = 0
            for detected in run.detected:
                streak = 0 if detected else streak + 1
                longest = max(longest, streak)
            max_streak.append(longest)
        reliability.append({"mode": mode, "n_pilots": int(beams),
                            "region": "deep" if deep else "other", "frames": len(group),
                            "detection_pct": float(100 * group.detected.mean()),
                            "tracked_mae_mm": float(error.mean()),
                            "tracked_p95_mm": float(error.quantile(.95)),
                            "mean_max_miss_streak_frames": float(np.mean(max_streak)),
                            "effective_rate_mean_bpshz": float(rate.mean()),
                            "effective_rate_p05_bpshz": float(rate.quantile(.05)),
                            "outage_below_1bpshz_pct": float(100 * (rate < 1).mean()),
                            "network_vs_estimated_focus_pct": float(100 * (
                                rate.mean() / group.rate_power_estimated_focus.mean() - 1)),
                            "network_vs_hold_pct": float(100 * (
                                rate.mean() / group.rate_power_hold.mean() - 1)),
                            "true_position_reference_delta_pct": float(100 * (
                                group.rate_power_true_network.mean() / rate.mean() - 1))})
    pd.DataFrame(reliability).to_csv(OUT / "reliability_ablation.csv", index=False)

    # Threshold sweep uses independent, no-target Monte Carlo draws. Thus the
    # x-axis is empirical false-alarm probability, not a nominal target.
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=.00075)
    library = PilotLibrary(scene)
    roc = []
    for mode, beams in (("passive", 21), ("active", 9)):
        instrument = PositionInstrument(library, mode, CONDITIONS[1], beams, 9912)
        null = instrument._null_peaks(np.random.default_rng(9913), 50000)
        target = stress[(stress.case == f"baseline_{mode}_{beams}") & stress.deep_shadow].peak.to_numpy()
        for nominal_pfa in (.001, .003, .01, .03, .1):
            threshold = float(np.quantile(null, 1 - nominal_pfa))
            roc.append({"mode": mode, "beams": beams, "deep_target_frames": len(target),
                        "no_target_draws": len(null), "threshold": threshold,
                        "empirical_false_alarm_pct": float(100 * np.mean(null > threshold)),
                        "deep_detection_pct": float(100 * np.mean(target > threshold))})
    pd.DataFrame(roc).to_csv(OUT / "roc_deep.csv", index=False)

    # Only a controlled normalized energy proxy: active return scales with
    # pilot power while noise remains fixed, and beam time stays 0.1 ms.
    power = []
    for db in (-12, -6, 0, 6):
        subset = stress[stress.case == f"active_9_power_{db:+d}db"]
        for deep, group in subset.groupby("deep_shadow"):
            power.append({"pilot_power_db": db, "normalized_pilot_energy": float(10 ** (db / 10)),
                          "region": "deep" if deep else "other", "frames": len(group),
                          "detection_pct": float(100 * group.detected.mean()),
                          "rate_mean_bpshz": float(group.effective_rate_bpshz.mean())})
    pd.DataFrame(power).to_csv(OUT / "active_power_sweep.csv", index=False)

    summary = {"source_stage2_frames": len(old), "source_stress_frames": len(stress),
               "reliability": reliability, "roc": roc, "normalized_pilot_power": power,
               "interpretation": "All rates are AWGN spectral-efficiency surrogates; no hardware measurements."}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved analysis tables", flush=True)


if __name__ == "__main__":
    main()
