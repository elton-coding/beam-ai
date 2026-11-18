"""Compare dedicated sweeps and cooperative reuse of downlink data-frame pilots."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.evaluate_general_pipeline import FRAME_S, TAU
from direct_network.experiments.evaluate_general_scene import direct_parameters
from direct_network.core.general_model import GeneralSceneNetwork
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio, parameter_bounds
from direct_network.core.sensing import CausalPositionTracker
from direct_network.core.sensing_stage2 import CONDITIONS, complex_normal
from direct_network.core.paths import ROOT


OUT = ROOT / "results" / "pilot_reuse"
GRID_X = np.linspace(-.11, .11, 221)
SCAN_X = np.linspace(-.10, .10, 9)
SUBARRAYS = 4
CONTROL_S = TAU
RUNS = 5
METHODS = ("dedicated", "periodic", "shared", "shared_recovery")
CONDITIONS_TO_TEST = ("matched", "geometry_offset", "high_pilot_noise")
NOISE = CONDITIONS[1].noise_power
RECOVERY_AFTER = 2
PERIODIC_INTERVAL = 3


def design_index(x: float) -> int:
    return int(np.clip(np.rint((x - GRID_X[0]) / .001), 0, len(GRID_X) - 1))


def find_longest_streak(mask: np.ndarray) -> int:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), initial=0))


def trajectory(run: int) -> np.ndarray:
    t = np.linspace(0, 1, 101)
    if run < 3:
        t = t + .09 * np.sin(2 * np.pi * t + run) - .09 * np.sin(run)
    else:
        # One brief reversal during an otherwise complete crossing.
        t = t + .11 * np.sin(4 * np.pi * t)
    return (-1 if run % 2 else 1) * (.19 * t - .095)


def aperture_power(engine: AngularSpectrum, params: tuple, x: np.ndarray,
                   n_subarrays: int) -> np.ndarray:
    field = engine.received_field(engine.source(*params))
    integral = np.r_[0., np.cumsum(np.abs(field) ** 2) * engine.dx]
    edges = engine.x[0] - engine.dx / 2 + np.arange(len(integral)) * engine.dx
    offsets = np.linspace(-engine.scene.rx_width_m / 2, engine.scene.rx_width_m / 2,
                          n_subarrays + 1)
    cumulative = np.interp(x[:, None] + offsets, edges, integral)
    return np.diff(cumulative, axis=1).T


def propagation_tables(nominal: Scene, truth: Scene, beams: list[tuple]) -> np.ndarray:
    """Use nominal commands on both geometries, including when scoring payload."""
    scan_beams = []
    for x in SCAN_X:
        _, _, focal, angle = parameter_bounds(nominal, x)
        scan_beams.append((0., focal, angle))
    engine = AngularSpectrum(truth)
    return np.stack([aperture_power(engine, params, GRID_X, SUBARRAYS)
                     for params in beams + scan_beams])


def channel_amplitude(power: np.ndarray, x: float) -> np.ndarray:
    left = int(np.clip(np.floor((x - GRID_X[0]) / .001), 0, len(GRID_X) - 2))
    weight = np.clip((x - GRID_X[left]) / .001, 0., 1.)
    return np.sqrt((1 - weight) * power[..., left] + weight * power[..., left + 1])


def observe(power: np.ndarray, x: float, gain: float, phase: float,
            noise: np.ndarray, noise_power: float) -> np.ndarray:
    main = channel_amplitude(power, x)
    extra = channel_amplitude(power, x + .020)
    # A coherent secondary component in the same spirit as the previous proxy.
    received = np.sqrt(gain) * (main + .15 * extra * np.exp(1j * phase))
    return np.abs(received + np.sqrt(noise_power) * noise).ravel()


def match(amplitude: np.ndarray, template: np.ndarray, noise_power: float,
          center: float | None = None, window: float = .110) -> tuple[float | None, bool]:
    # Under noise only, each squared amplitude is exponential with mean noise_power.
    threshold = -noise_power * np.log1p(-.99 ** (1 / amplitude.size))
    if np.max(amplitude ** 2) <= threshold:
        return None, False
    permitted = np.ones(len(GRID_X), dtype=bool) if center is None else abs(GRID_X - center) <= window
    candidates = GRID_X[permitted]
    signatures = template[:, permitted]
    norm = np.maximum(np.sum(signatures**2, axis=0), 1e-20)
    data = np.sqrt(np.maximum(amplitude**2 - noise_power, 0))
    scale = np.clip(data @ signatures / norm, 0., np.inf)
    error = np.sum((data[:, None] - scale * signatures) ** 2, axis=0)
    best = int(error.argmin())
    if center is not None:
        # Accept an update only when the power shape rules out distant positions.
        # This is a fixed gate using a high-SNR residual scale, not a calibrated CI.
        plausible = error <= error[best] + 2 * noise_power
        span = np.ptp(candidates[plausible])
        if span > .016 or span > .65 * np.ptp(candidates):
            return None, True
    return float(candidates[best]), True


def simulate(scene_id: int, condition: str, truth: Scene, nominal: np.ndarray,
             actual: np.ndarray, noise_power: float, reference_power: float) -> list[dict]:
    rows = []
    scan = slice(len(GRID_X), None)
    scan_template = np.sqrt(nominal[scan]).reshape(-1, len(GRID_X))
    reuse_template = np.sqrt(nominal[:len(GRID_X)])
    for run in range(RUNS):
        rng = np.random.default_rng(181100 + 100 * scene_id + run)
        path = trajectory(run)
        deep = [blockage_ratio(truth, x) >= truth.activation_bl for x in path]
        pilot_noise = complex_normal(rng, (len(path), SUBARRAYS))
        scan_noise = complex_normal(rng, (len(path), len(SCAN_X), SUBARRAYS))
        gains = 10 ** (rng.normal(0, .75, len(path)) / 10)
        phases = rng.uniform(-np.pi, np.pi, len(path))
        data_noise = reference_power / 10
        for method in METHODS:
            tracker = CausalPositionTracker(dt_s=FRAME_S, acceleration_std_m_s2=.5)
            misses = 0
            for frame, x in enumerate(path):
                gain, phase = gains[frame], phases[frame]
                scanned = (method == "dedicated" or frame == 0 or
                           (method == "periodic" and frame % PERIODIC_INTERVAL == 0) or
                           (method == "shared_recovery" and misses >= RECOVERY_AFTER))
                prior = float(np.clip(tracker.state[0] + FRAME_S * tracker.state[1],
                                      GRID_X[0], GRID_X[-1]))
                if scanned:
                    amplitude = observe(actual[scan], x, gain, phase, scan_noise[frame], noise_power)
                    measurement, detected = match(amplitude, scan_template, noise_power)
                    estimate, sigma = tracker.step(measurement)
                    beam = design_index(estimate)
                else:
                    beam = design_index(prior)
                    measurement, detected = None, None
                    if method != "periodic":
                        amplitude = observe(actual[beam], x, gain, phase, pilot_noise[frame], noise_power)
                        window = np.clip(3 * np.sqrt(tracker.cov[0, 0]) + .010, .025, .110)
                        center = prior if tracker.initialized else 0.
                        if not tracker.initialized:
                            window = .110
                        measurement, detected = match(amplitude, reuse_template[beam], noise_power, center, window)
                    estimate, sigma = tracker.step(measurement)
                misses = 0 if measurement is not None else misses + 1
                payload_fraction = (FRAME_S - TAU - CONTROL_S - scanned * len(SCAN_X) * TAU) / FRAME_S
                main = channel_amplitude(actual[beam], x)
                extra = channel_amplitude(actual[beam], x + .020)
                power = float(gain * np.sum(abs(main + .15 * extra * np.exp(1j * phase)) ** 2))
                rows.append({
                    "condition": condition, "scene": scene_id, "run": run, "frame": frame,
                    "method": method, "deep": deep[frame], "actual_x_m": x,
                    "beam_x_m": GRID_X[beam], "estimated_x_m": estimate, "sigma_m": sigma,
                    "measurement_x_m": measurement, "detected": detected is True,
                    "attempted": detected is not None, "scanned": scanned,
                    "payload_fraction": payload_fraction, "power": power,
                    "effective_bpshz": payload_fraction * np.log2(1 + power / data_noise),
                })
    return rows


def summarize(frames: pd.DataFrame) -> dict:
    frames["abs_error_mm"] = 1000 * abs(frames.estimated_x_m - frames.actual_x_m)
    frames["beam_error_mm"] = 1000 * abs(frames.beam_x_m - frames.actual_x_m)
    frames["accepted"] = frames.measurement_x_m.notna()
    frames["accurate_update"] = abs(frames.measurement_x_m - frames.actual_x_m) <= .010
    metrics, differences, periodic_comparisons = [], [], []
    rng = np.random.default_rng(181122)
    for condition, group in frames.groupby("condition", sort=False):
        for region, mask in (("all", np.ones(len(group), dtype=bool)),
                             ("deep", group.deep), ("other", ~group.deep)):
            selected = group.loc[mask]
            for method, subset in selected.groupby("method", sort=False):
                runs = group.loc[group.method == method].groupby(["scene", "run"], sort=False)
                streaks = []
                for _, series in runs:
                    gaps = ~series.accepted
                    if region != "all":
                        gaps = gaps & (series.deep if region == "deep" else ~series.deep)
                    if region == "all" or (series.deep if region == "deep" else ~series.deep).any():
                        streaks.append(find_longest_streak(gaps.to_numpy()))
                metrics.append({
                    "condition": condition, "region": region, "method": method,
                    "frames": len(subset), "detection_pct": float(100 * subset.detected.mean()),
                    "accepted_pct": float(100 * subset.accepted.mean()),
                    "accurate_update_pct": float(100 * subset.accurate_update.mean()),
                    "mae_mm": float(subset.abs_error_mm.mean()),
                    "p95_mm": float(subset.abs_error_mm.quantile(.95)),
                    "beam_mae_mm": float(subset.beam_error_mm.mean()),
                    "mean_longest_gap_ms": float(1000 * FRAME_S * np.mean(streaks)),
                    "max_gap_ms": float(1000 * FRAME_S * max(streaks)),
                    "scan_pct": float(100 * subset.scanned.mean()),
                    "payload_pct": float(100 * subset.payload_fraction.mean()),
                    "effective_bpshz": float(subset.effective_bpshz.mean()),
                    "outage_below_1_pct": float(100 * (subset.effective_bpshz < 1).mean()),
                })
            # Equal scene weights for the paired effect and its interval.
            by_scene = selected.groupby(["scene", "method"])["effective_bpshz"].mean().unstack()
            draws = rng.integers(len(by_scene), size=(10000, len(by_scene)))
            base = by_scene.dedicated.to_numpy()
            for method in METHODS[1:]:
                compare = by_scene[method].to_numpy()
                changes = (compare[draws].mean(axis=1) / base[draws].mean(axis=1) - 1) * 100
                differences.append({
                    "condition": condition, "region": region, "method": method,
                    "rate_change_pct": float(100 * (compare.mean() / base.mean() - 1)),
                    "paired_scene_bootstrap_95pct_ci_pct": np.quantile(changes, [.025, .975]).tolist(),
                })
            periodic = by_scene.periodic.to_numpy()
            recovery = by_scene.shared_recovery.to_numpy()
            changes = (recovery[draws].mean(axis=1) / periodic[draws].mean(axis=1) - 1) * 100
            periodic_comparisons.append({
                "condition": condition, "region": region, "method": "shared_recovery",
                "rate_change_pct": float(100 * (recovery.mean() / periodic.mean() - 1)),
                "paired_scene_bootstrap_95pct_ci_pct": np.quantile(changes, [.025, .975]).tolist(),
            })
    return {"metrics": metrics, "comparison_to_dedicated": differences,
            "comparison_to_periodic": periodic_comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-limit", type=int, help="Run a subset for a smoke check")
    args = parser.parse_args()
    scenes = json.loads((ROOT / "artifacts" / "general_scenes.json").read_text(encoding="utf-8"))
    scenes = [row for row in scenes if row["split"] == "test"][:args.scene_limit]
    model = GeneralSceneNetwork()
    rows = []
    for record in scenes:
        scene_id = record["scene_id"]
        scene = Scene(**{k: v for k, v in record.items() if k not in ("scene_id", "split")})
        scene = replace(scene, n_grid=8192, obstacle_step_m=.0015)
        beams = [direct_parameters(scene, model, float(x)) for x in GRID_X]
        nominal = propagation_tables(scene, scene, beams)
        mismatch = replace(scene, obstacle_x_m=scene.obstacle_x_m + .003,
                           obstacle_width_m=1.05 * scene.obstacle_width_m)
        truth_near = propagation_tables(scene, mismatch, beams)
        _, _, focal, angle = parameter_bounds(scene, .095)
        reference_power = float(aperture_power(AngularSpectrum(scene), (0., focal, angle),
                                               np.array([.095]), 1).sum())
        for condition in CONDITIONS_TO_TEST:
            perturbed = condition == "geometry_offset"
            rows.extend(simulate(scene_id, condition, mismatch if perturbed else scene, nominal,
                                 truth_near if perturbed else nominal,
                                 NOISE * (4 if condition == "high_pilot_noise" else 1),
                                 reference_power))
        print(f"Completed scene {scene_id}", flush=True)
    frames = pd.DataFrame(rows)
    result = summarize(frames)
    result.update({"scenes": len(scenes), "runs_per_scene": RUNS, "frames_per_run": 101,
                   "frame_ms": 1000 * FRAME_S, "pilot_ms": 1000 * TAU,
                   "feedback_ms": 1000 * CONTROL_S, "subarrays": SUBARRAYS,
                   "dedicated_beams": len(SCAN_X), "recovery_after_misses": RECOVERY_AFTER,
                   "periodic_sweep_interval_frames": PERIODIC_INTERVAL,
                   "nominal_pilot_noise_power": NOISE, "simulation_only": True})
    out = OUT if args.scene_limit is None else OUT / f"smoke_{args.scene_limit}"
    out.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out / "frames.csv", index=False, float_format="%.9g")
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {len(frames):,} frames to {out}", flush=True)


if __name__ == "__main__":
    main()
