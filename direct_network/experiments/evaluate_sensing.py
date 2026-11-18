"""Evaluate reflection-based position acquisition before one-pass Airy prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.evaluate import beam_parameters
from direct_network.core.model import SceneNetwork
from direct_network.core.physics import Scene, blockage_ratio, parameter_bounds
from direct_network.core.sensing import CausalPositionTracker, ReflectionScanner, SensingConfig
from direct_network.core.paths import ROOT




def crossing_trajectory(n_frames: int = 101, motion: str = "variable") -> np.ndarray:
    """Smooth changing-speed traverse; all positions stay in the scan window."""
    u = np.linspace(0, 1, n_frames)
    linear = -0.095 + 0.190 * u
    if motion == "variable":
        return linear + 0.007 * np.sin(6 * np.pi * u)
    if motion == "steady":
        return linear
    if motion == "alternate":
        return linear + 0.005 * np.sin(4 * np.pi * u + .7) - 0.005 * np.sin(.7)
    raise ValueError(f"Unknown motion: {motion}")


def pct(a: pd.Series, b: pd.Series) -> float:
    return float(100 * np.mean(a.to_numpy() / b.to_numpy() - 1))


def subgroup(frame: pd.DataFrame) -> dict:
    missed = frame.loc[~frame.detected]
    found = frame.loc[frame.detected]
    runs = frame.groupby("run", sort=False)
    max_miss_run = 0
    for _, one in runs:
        streak = 0
        for detected in one.detected:
            streak = 0 if detected else streak + 1
            max_miss_run = max(max_miss_run, streak)
    return {
        "samples": len(frame),
        "detection_rate": float(frame.detected.mean()),
        "max_consecutive_missed_frames": max_miss_run,
        "measured_position_mae_mm_when_detected": float(1000 * (found.measured_x_m - found.actual_x_m).abs().mean()) if len(found) else None,
        "tracker_position_mae_mm": float(1000 * (frame.estimated_x_m - frame.actual_x_m).abs().mean()),
        "tracker_position_mae_mm_when_missed": float(1000 * (missed.estimated_x_m - missed.actual_x_m).abs().mean()) if len(missed) else None,
        "hold_position_mae_mm": float(1000 * (frame.hold_x_m - frame.actual_x_m).abs().mean()),
        "tracker_vs_true_focus_power_change_pct": pct(frame.power_tracker, frame.power_true_focus),
        "tracker_vs_estimated_focus_power_change_pct": pct(frame.power_tracker, frame.power_estimated_focus),
        "hold_vs_true_focus_power_change_pct": pct(frame.power_hold, frame.power_true_focus),
        "oracle_position_vs_true_focus_power_change_pct": pct(frame.power_oracle, frame.power_true_focus),
        "tracker_vs_hold_power_change_pct": pct(frame.power_tracker, frame.power_hold),
        "oracle_position_vs_tracker_power_change_pct": pct(frame.power_oracle, frame.power_tracker),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--noise-power", type=float, default=SensingConfig.complex_noise_power)
    parser.add_argument("--motion", choices=("variable", "steady", "alternate"), default="variable")
    parser.add_argument("--output-stem", default="sensing")
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=0.00075)
    if args.noise_power <= 0 or not args.output_stem.replace("_", "").isalnum():
        raise ValueError("noise-power must be positive and output-stem alphanumeric/underscore")
    config = replace(SensingConfig(), complex_noise_power=args.noise_power)
    scanner = ReflectionScanner(scene, config)
    engine = scanner.engine
    model = SceneNetwork()
    trajectory = crossing_trajectory(motion=args.motion)
    dt_s = .02
    baseline = []
    for actual in trajectory:
        actual = float(actual)
        _, _, f0, theta0 = parameter_bounds(scene, actual)
        oracle_params, _, _ = beam_parameters(scene, model, actual)
        baseline.append((engine.received_power(actual, 0.0, f0, theta0),
                         engine.received_power(actual, *oracle_params)))
    rows = []
    for run in range(args.runs):
        rng = np.random.default_rng(config.seed + run)
        tracker = CausalPositionTracker(dt_s=dt_s)
        last_observed = None
        for t, actual in enumerate(trajectory):
            actual = float(actual)
            observation = scanner.observe(actual, rng)
            estimated, uncertainty = tracker.step(observation.measured_x_m)
            if observation.measured_x_m is not None:
                last_observed = observation.measured_x_m
            hold = float(last_observed if last_observed is not None else estimated)
            tracker_params, tracker_mode, _ = beam_parameters(scene, model, estimated)
            _, _, estimated_f0, estimated_theta0 = parameter_bounds(scene, estimated)
            hold_params, _, _ = beam_parameters(scene, model, hold)
            tracker_power = engine.received_power(actual, *tracker_params)
            estimated_focus_power = (tracker_power if tracker_mode == "focus" else
                                     engine.received_power(actual, 0.0, estimated_f0, estimated_theta0))
            rows.append({
                "run": run, "frame": t, "time_s": t * dt_s,
                "actual_x_m": actual, "blockage_ratio": blockage_ratio(scene, actual),
                "deep_shadow": blockage_ratio(scene, actual) >= scene.activation_bl,
                "detected": observation.detected, "measured_x_m": observation.measured_x_m,
                "estimated_x_m": estimated, "estimated_std_m": uncertainty,
                "hold_x_m": hold, "selected_tracker": tracker_mode,
                "peak_echo_power": observation.peak_echo_power,
                "detection_threshold_power": observation.threshold_power,
                "power_true_focus": baseline[t][0], "power_oracle": baseline[t][1],
                "power_tracker": tracker_power,
                "power_estimated_focus": estimated_focus_power,
                "power_hold": engine.received_power(actual, *hold_params),
            })
        print(f"completed sensing run {run + 1}/{args.runs}", flush=True)
    table = pd.DataFrame(rows)
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    table.to_csv(result_dir / f"{args.output_stem}_frames.csv", index=False, float_format="%.12g")
    summary = {
        "method": "Focused pilot sweep -> noisy two-way reflection proxy -> threshold detection -> causal Kalman position estimate -> unchanged one-pass Airy network",
        "scene": asdict(scene),
        "sensing": asdict(config),
        "scan_pilots_per_frame": config.scan_beams,
        "frame_interval_s": dt_s,
        "motion": args.motion,
        "tracker": {"measurement_std_m": .003, "acceleration_std_m_s2": 5.0,
                    "velocity_retention_without_detection": .9},
        "runs": args.runs,
        "trajectory_frames": len(trajectory),
        "all": subgroup(table),
        "deep_shadow": subgroup(table.loc[table.deep_shadow]),
        "other_positions": subgroup(table.loc[~table.deep_shadow]),
        "limitations": [
            "Two-way power is a reciprocal square-law proxy, not a full radar waveform/channel simulation.",
            "Fixed reflectivity and white receiver noise are assumed; clutter, multipath, hardware switching and scan time are not modeled.",
            "True positions generate synthetic echoes and score the beam; they are never passed to the tracker or online network.",
        ],
    }
    (result_dir / f"{args.output_stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
