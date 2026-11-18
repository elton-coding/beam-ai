"""End-to-end sensing plus one-pass general-network beam design on unseen scenes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.experiments.evaluate_general_scene import direct_parameters
from direct_network.core.general_model import GeneralSceneNetwork
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio, parameter_bounds
from direct_network.core.sensing import CausalPositionTracker
from direct_network.core.sensing_stage2 import CONDITIONS, PilotLibrary, PositionInstrument
from direct_network.core.paths import ROOT


OUT = ROOT / "results" / "stage3"
FRAME_S = .02
TAU = .0001
RUNS = 5


def score_table(scene: Scene, engine: AngularSpectrum, model: GeneralSceneNetwork,
                design_x: np.ndarray, actual_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(engine.x, actual_x - scene.rx_width_m / 2, side="left")
    right = np.searchsorted(engine.x, actual_x + scene.rx_width_m / 2, side="right")
    network = np.empty((len(design_x), len(actual_x)))
    focus = np.empty_like(network)
    for i, x in enumerate(design_x):
        _, _, f0, theta0 = parameter_bounds(scene, float(x))
        for output, params in ((focus, (0.0, f0, theta0)),
                               (network, direct_parameters(scene, model, float(x)))):
            field = engine.received_field(engine.source(*params))
            cumulative = np.r_[0.0, np.cumsum(np.abs(field) ** 2) * engine.dx]
            output[i] = cumulative[right] - cumulative[left]
    return network, focus


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((ROOT / "artifacts" / "general_scenes.json").read_text(encoding="utf-8"))
    model = GeneralSceneNetwork()
    actual_x = np.linspace(-.095, .095, 101)
    rows = []
    for record in scenes:
        if record["split"] != "test":
            continue
        scene_id = record["scene_id"]
        scene = Scene(**{k: v for k, v in record.items() if k not in ("scene_id", "split")})
        scene = replace(scene, n_grid=8192, obstacle_step_m=.0015)
        library = PilotLibrary(scene)
        network, focus = score_table(scene, library.engine, model, library.grid_x, actual_x)
        reference_index = int(np.argmin(np.abs(actual_x - .095)))
        reference_power = focus[np.argmin(np.abs(library.grid_x - .095)), reference_index]
        data_noise = reference_power / 10
        for mode, n_pilots in (("active", 9), ("passive", 21)):
            instrument = PositionInstrument(library, mode, CONDITIONS[1], n_pilots,
                                            calibration_seed=310000 + scene_id * 10 + n_pilots)
            fraction = ((FRAME_S - n_pilots * TAU) /
                        (FRAME_S + (64 if mode == "passive" else 0) * n_pilots * TAU / len(actual_x)))
            for run in range(RUNS):
                rng = np.random.default_rng(320000 + scene_id * 1000 + run * 10 + n_pilots)
                rho = 10 ** (rng.uniform(-6, 0) / 10) if mode == "passive" else 1.0
                tracker = CausalPositionTracker(dt_s=FRAME_S)
                instrument.reset()
                for t, x in enumerate(actual_x):
                    measured, peak = instrument.observe(float(x), rng, rho)
                    estimate, _ = tracker.step(measured)
                    design_i = int(np.clip(np.rint((estimate - library.grid_x[0]) / .001),
                                           0, len(library.grid_x) - 1))
                    p_network = network[design_i, t]
                    p_focus = focus[design_i, t]
                    rows.append({"scene_id": scene_id, "run": run, "mode": mode,
                                 "n_pilots": n_pilots, "frame": t, "actual_x_m": x,
                                 "deep_shadow": blockage_ratio(scene, float(x)) >= scene.activation_bl,
                                 "detected": measured is not None, "measured_x_m": measured,
                                 "estimated_x_m": estimate, "peak": peak,
                                 "power_network": p_network, "power_focus": p_focus,
                                 "effective_network_bpshz": fraction * np.log2(1 + p_network / data_noise),
                                 "effective_focus_bpshz": fraction * np.log2(1 + p_focus / data_noise),
                                 "fraction": fraction})
        print(f"completed held-out pipeline scene {scene_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "general_pipeline_frames.csv", index=False, float_format="%.12g")
    results = []
    rng = np.random.default_rng(20261106)
    for (mode, deep), group in frame.groupby(["mode", "deep_shadow"]):
        by_scene = group.groupby("scene_id").apply(
            lambda g: pd.Series({"network": g.effective_network_bpshz.mean(),
                                 "focus": g.effective_focus_bpshz.mean()}),
            include_groups=False)
        draws = rng.integers(0, len(by_scene), (10000, len(by_scene)))
        delta = (by_scene.network.to_numpy()[draws].mean(axis=1) /
                 by_scene.focus.to_numpy()[draws].mean(axis=1) - 1) * 100
        results.append({"mode": mode, "region": "deep" if deep else "other",
                        "scenes": group.scene_id.nunique(), "runs_per_scene": RUNS,
                        "frames": len(group), "detection_pct": float(100 * group.detected.mean()),
                        "tracked_mae_mm": float(1000 * np.mean(np.abs(group.estimated_x_m - group.actual_x_m))),
                        "rate_network_bpshz": float(group.effective_network_bpshz.mean()),
                        "rate_focus_bpshz": float(group.effective_focus_bpshz.mean()),
                        "network_vs_focus_pct": float(100 * (
                            group.effective_network_bpshz.mean() / group.effective_focus_bpshz.mean() - 1)),
                        "scene_bootstrap_95pct_ci_pct": np.quantile(delta, [.025, .975]).tolist()})
    summary = {"test_scenes": 16, "runs_per_scene": RUNS, "frames_per_run": len(actual_x),
               "communication_grid": 8192, "sensing_condition": "reflective",
               "pilot_duration_s": TAU, "simulation_only": True, "results": results}
    (OUT / "general_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
