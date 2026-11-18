"""Evaluate passive and cooperative position acquisition with pilot overhead.

This is an offline Monte Carlo experiment. Position estimation and one-pass
beam selection are causal; true x is used only to synthesize observations and
to score the selected beam. No online physical parameter search is performed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.evaluate import beam_parameters
from direct_network.experiments.evaluate_sensing import crossing_trajectory
from direct_network.core.model import SceneNetwork
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio, parameter_bounds
from direct_network.core.sensing import CausalPositionTracker
from direct_network.core.sensing_stage2 import ABLATIONS, CONDITIONS, PilotLibrary, PositionInstrument
from direct_network.core.paths import ROOT


FRAME_S = 0.020
PILOT_DURATIONS_S = (0.00005, 0.0001, 0.0002)
REFERENCE_SNR_DB = (0, 10, 20)


class CommunicationLookup:
    """Score a 1 mm quantized design location at each exact actual location."""

    def __init__(self, scene: Scene, library: PilotLibrary, trajectories: dict[str, np.ndarray]):
        self.scene = scene
        self.engine: AngularSpectrum = library.engine
        self.design_x = library.grid_x
        self.actual_x = np.concatenate(tuple(trajectories.values()))
        self.motion_offsets = {}
        offset = 0
        for motion, values in trajectories.items():
            self.motion_offsets[motion] = offset
            offset += len(values)
        self.network = np.empty((len(self.design_x), len(self.actual_x)))
        self.focus = np.empty_like(self.network)
        self.true_focus = np.empty(len(self.actual_x))
        self.true_network = np.empty(len(self.actual_x))
        model = SceneNetwork()
        left_index = np.searchsorted(self.engine.x, self.actual_x - scene.rx_width_m / 2, side="left")
        right_index = np.searchsorted(self.engine.x, self.actual_x + scene.rx_width_m / 2, side="right")

        def powers(params: tuple[float, float, float] | np.ndarray) -> np.ndarray:
            field = self.engine.received_field(self.engine.source(*params))
            cumulative = np.r_[0.0, np.cumsum(np.abs(field) ** 2) * self.engine.dx]
            return cumulative[right_index] - cumulative[left_index]

        for i, x in enumerate(self.design_x):
            _, _, focal, theta = parameter_bounds(scene, float(x))
            self.focus[i] = powers((0.0, focal, theta))
            params, selection, _ = beam_parameters(scene, model, float(x))
            self.network[i] = self.focus[i] if selection == "focus" else powers(params)
        for i, x in enumerate(self.actual_x):
            _, _, focal, theta = parameter_bounds(scene, float(x))
            # Exact true-position references, with no design-grid quantization.
            focus_field = self.engine.received_field(self.engine.source(0.0, focal, theta))
            mask = np.abs(self.engine.x - x) <= scene.rx_width_m / 2
            self.true_focus[i] = float(np.sum(np.abs(focus_field[mask]) ** 2) * self.engine.dx)
            params, selection, _ = beam_parameters(scene, model, float(x))
            if selection == "focus":
                self.true_network[i] = self.true_focus[i]
            else:
                field = self.engine.received_field(self.engine.source(*params))
                self.true_network[i] = float(np.sum(np.abs(field[mask]) ** 2) * self.engine.dx)
        # Reference at an unshadowed path endpoint; this calibrates the
        # dimensionless receiver-noise sweep, not a measured hardware SNR.
        self.reference_power = float(self.true_focus[len(trajectories["variable"]) - 1])

    def at(self, design_x: float, actual_index: int) -> tuple[float, float]:
        i = int(np.clip(np.rint((design_x - self.design_x[0]) / 0.001), 0, len(self.design_x) - 1))
        return float(self.network[i, actual_index]), float(self.focus[i, actual_index])


def summarize(group: pd.DataFrame, noise_data: float) -> dict:
    found = group[group.detected]
    missed = group[~group.detected]
    out = {
        "samples": int(len(group)),
        "detection_rate": float(group.detected.mean()),
        "measured_mae_mm_when_detected": float(1000 * (found.measured_x_m - found.actual_x_m).abs().mean()) if len(found) else None,
        "tracked_mae_mm": float(1000 * (group.estimated_x_m - group.actual_x_m).abs().mean()),
        "tracked_mae_mm_when_missed": float(1000 * (missed.estimated_x_m - missed.actual_x_m).abs().mean()) if len(missed) else None,
        "hold_mae_mm": float(1000 * (group.hold_x_m - group.actual_x_m).abs().mean()),
        "network_vs_estimated_focus_power_pct": float(100 * np.mean(group.power_network / group.power_estimated_focus - 1)),
        "network_vs_hold_power_pct": float(100 * np.mean(group.power_network / group.power_hold - 1)),
        "network_vs_true_focus_power_pct": float(100 * np.mean(group.power_network / group.power_true_focus - 1)),
    }
    for duration in PILOT_DURATIONS_S:
        online_fraction = 1 - group.n_pilots.iloc[0] * duration / FRAME_S
        calibration_scans = 64 if group["mode"].iloc[0] == "passive" else 0
        # A session contains 101 communication frames. Passive background
        # calibration takes time before the first frame and carries no data.
        fraction = ((FRAME_S - group.n_pilots.iloc[0] * duration)
                    / (FRAME_S + calibration_scans * group.n_pilots.iloc[0] * duration / 101))
        if online_fraction <= 0:
            raise ValueError("Pilot budget exceeds frame")
        key = f"{duration*1000:.2f}ms"
        rate_network = fraction * np.log2(1 + group.power_network.to_numpy() / noise_data)
        rate_focus = fraction * np.log2(1 + group.power_estimated_focus.to_numpy() / noise_data)
        rate_hold = fraction * np.log2(1 + group.power_hold.to_numpy() / noise_data)
        out[f"rate_{key}"] = {
            "online_data_fraction": float(online_fraction),
            "session_data_fraction_including_calibration": float(fraction),
            "network_mean_bpshz": float(rate_network.mean()),
            "estimated_focus_mean_bpshz": float(rate_focus.mean()),
            "hold_mean_bpshz": float(rate_hold.mean()),
            "network_p05_bpshz": float(np.quantile(rate_network, .05)),
            "network_vs_estimated_focus_pct": float(100 * (rate_network.mean() / rate_focus.mean() - 1)),
        }
    out["rate_0.10ms_snr_sweep"] = {
        f"{snr_db}db_network_mean_bpshz": float(
            out["rate_0.10ms"]["session_data_fraction_including_calibration"]
            * np.log2(1 + group.power_network.to_numpy() / (noise_data * 10 ** ((10 - snr_db) / 10))).mean())
        for snr_db in REFERENCE_SNR_DB
    }
    return out


def run_experiment(n_runs: int, robust_runs: int) -> None:
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=.00075)
    library = PilotLibrary(scene)
    trajectories = {name: crossing_trajectory(motion=name) for name in ("variable", "steady", "alternate")}
    lookup = CommunicationLookup(scene, library, trajectories)
    data_noise = lookup.reference_power / 10  # 10 dB at the reference focus location.
    rows = []
    instrument_meta = []
    for condition_index, condition in enumerate(CONDITIONS + ABLATIONS):
        for mode_index, mode in enumerate(("passive", "active")):
            if condition in ABLATIONS and mode != "passive":
                continue
            for n_pilots in (5, 9, 21):
                if condition in ABLATIONS and n_pilots != 21:
                    continue
                instrument = PositionInstrument(library, mode, condition, n_pilots,
                                                calibration_seed=20261104 + 1000 * condition_index
                                                + 100 * mode_index + n_pilots)
                instrument_meta.append({"condition": condition.name, "mode": mode,
                                        "n_pilots": n_pilots, "threshold": instrument.threshold,
                                        "null_false_alarm_rate": instrument.no_target_false_alarm})
                motions = ("variable", "steady", "alternate") if condition.name == "reflective" and n_pilots == 21 else ("variable",)
                for motion in motions:
                    repeats = n_runs if motion == "variable" else robust_runs
                    for run in range(repeats):
                        seed = 20261104 + condition_index * 10000 + run
                        # Fixed target strength for a run, shared by modes and
                        # pilot budgets; observation noise remains independent.
                        strength_rng = np.random.default_rng(seed)
                        rho_db = float(strength_rng.uniform(condition.passive_reflectivity_db_low,
                                                             condition.passive_reflectivity_db_high))
                        rho = 10 ** (rho_db / 10) if mode == "passive" else 1.0
                        rng = np.random.default_rng(seed + 100000 * mode_index + 1000 * n_pilots
                                                    + (0 if motion == "variable" else 200000 if motion == "steady" else 300000))
                        tracker = CausalPositionTracker(dt_s=FRAME_S)
                        instrument.reset()
                        last_observed = None
                        for t, actual in enumerate(trajectories[motion]):
                            actual = float(actual)
                            measured, peak = instrument.observe(actual, rng, rho)
                            estimate, uncertainty = tracker.step(measured)
                            if measured is not None:
                                last_observed = measured
                            hold = float(last_observed if last_observed is not None else estimate)
                            actual_index = lookup.motion_offsets[motion] + t
                            power_network, power_focus = lookup.at(estimate, actual_index)
                            power_hold, _ = lookup.at(hold, actual_index)
                            rows.append({
                                "condition": condition.name, "mode": mode, "n_pilots": n_pilots,
                                "motion": motion, "run": run, "frame": t, "time_s": t * FRAME_S,
                                "actual_x_m": actual, "deep_shadow": blockage_ratio(scene, actual) >= scene.activation_bl,
                                "target_reflectivity": rho if mode == "passive" else np.nan,
                                "detected": measured is not None, "measured_x_m": measured,
                                "estimated_x_m": estimate, "estimated_std_m": uncertainty,
                                "hold_x_m": hold, "peak_echo_power": peak,
                                "detection_threshold_power": instrument.threshold,
                                "power_network": power_network, "power_estimated_focus": power_focus,
                                "power_hold": power_hold,
                                "power_true_focus": lookup.true_focus[actual_index],
                                "power_true_network": lookup.true_network[actual_index],
                            })
                    print(f"finished {condition.name} {mode} {n_pilots} pilots {motion}", flush=True)

    frame = pd.DataFrame(rows)
    output_dir = ROOT / "results" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "frames.csv", index=False, float_format="%.12g")
    result_rows = []
    for keys, group in frame.groupby(["condition", "mode", "n_pilots", "motion", "deep_shadow"], sort=True):
        condition, mode, n_pilots, motion, deep_shadow = keys
        result_rows.append({"condition": condition, "mode": mode, "n_pilots": int(n_pilots),
                            "motion": motion, "region": "deep" if deep_shadow else "other",
                            **summarize(group, data_noise)})
    for keys, group in frame.groupby(["condition", "mode", "n_pilots", "motion"], sort=True):
        condition, mode, n_pilots, motion = keys
        result_rows.append({"condition": condition, "mode": mode, "n_pilots": int(n_pilots),
                            "motion": motion, "region": "all", **summarize(group, data_noise)})
    paired_ci = []
    base = frame[(frame.condition == "reflective") & (frame.motion == "variable")].copy()
    base["rate"] = np.log2(1 + base.power_network / data_noise)
    base["fraction"] = np.where(
        base["mode"] == "passive",
        (FRAME_S - base.n_pilots * .0001) / (FRAME_S + 64 * base.n_pilots * .0001 / 101),
        1 - base.n_pilots * .0001 / FRAME_S,
    )
    base["rate"] *= base["fraction"]
    rng_ci = np.random.default_rng(20261105)
    for region, subset in (("all", base), ("deep", base[base.deep_shadow])):
        rates = subset.groupby(["run", "mode", "n_pilots"]).rate.mean().unstack(["mode", "n_pilots"])
        bootstrap_ids = rng_ci.integers(0, len(rates), (10000, len(rates)))
        for label, left, right in (
            ("active_9_minus_passive_21", ("active", 9), ("passive", 21)),
            ("active_9_minus_active_21", ("active", 9), ("active", 21)),
            ("passive_21_minus_passive_9", ("passive", 21), ("passive", 9)),
        ):
            difference = (rates[left] - rates[right]).to_numpy()
            means = difference[bootstrap_ids].mean(axis=1)
            paired_ci.append({"region": region, "comparison": label,
                              "mean_difference_bpshz": float(difference.mean()),
                              "bootstrap_95pct_ci_bpshz": [float(v) for v in np.quantile(means, [.025, .975])]})
    summary = {
        "experiment": "passive-return versus cooperative-uplink pilot observation; unchanged one-pass Airy beam network",
        "scene": asdict(scene),
        "conditions": [asdict(c) for c in CONDITIONS + ABLATIONS],
        "trajectories": {k: len(v) for k, v in trajectories.items()},
        "main_runs_per_configuration": n_runs,
        "robust_runs_per_motion": robust_runs,
        "pilot_durations_s": PILOT_DURATIONS_S,
        "frame_s": FRAME_S,
        "data_reference_power": lookup.reference_power,
        "data_noise_power_at_10db_reference": data_noise,
        "data_snr_reference_db_sweep": REFERENCE_SNR_DB,
        "position_design_grid_mm": 1.0,
        "null_calibration_samples": 4000,
        "null_validation_samples": 10000,
        "instrument": instrument_meta,
        "rows": result_rows,
        "paired_run_bootstrap": paired_ci,
        "limitations": [
            "Target return and obstacle echo are coherent parametric proxies, not a full Maxwell radar simulation.",
            "Passive static clutter is estimated from 64 no-target calibration sweeps; their time is amortized over the 101-frame session in reported effective rate.",
            "Active uplink is modeled by a reciprocal one-way power response; timing, synchronization and hardware are idealized.",
            "The receiver z coordinate and beam-network geometry remain fixed; only transverse x is estimated.",
            "Data rate is an AWGN spectral-efficiency surrogate, not measured throughput or bit error rate.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(frame)} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--robust-runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1 or args.robust_runs < 1:
        raise ValueError("run counts must be positive")
    run_experiment(args.runs, args.robust_runs)


if __name__ == "__main__":
    main()
