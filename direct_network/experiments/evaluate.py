"""Evaluate one-shot NN inference on static paths and the delayed trajectory."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from direct_network.core.model import SceneNetwork
from direct_network.core.motion import predict_at_frame, receiver_trajectory
from direct_network.core.paths import ROOT
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio, features, gain_db, parameter_bounds


def beam_parameters(scene: Scene, network: SceneNetwork, x_m: float) -> tuple[np.ndarray, str, float]:
    """Return parameters using only geometry and one neural-network forward pass."""
    if (scene.frequency_hz, scene.tx_width_m, scene.rx_width_m) != (120e9, 0.20, 0.025):
        raise ValueError("Scene model requires 120 GHz, 20 cm TX, and 2.5 cm RX")
    bl = blockage_ratio(scene, x_m)
    lo, hi, focal0, theta0 = parameter_bounds(scene, x_m)
    if bl < scene.activation_bl:
        return np.array([0.0, focal0, theta0]), "focus", bl
    b, delta_f, delta_theta = network.predict(features(scene, x_m, bl))
    params = np.clip([b, focal0 + delta_f, theta0 + delta_theta], lo, hi)
    return params, "direct_nn", bl


def evaluate_static(scene: Scene, network: SceneNetwork,
                    engine: AngularSpectrum, filename: str) -> dict:
    rows = []
    trajectories = {
        "shadow_crossing": np.linspace(-0.10, 0.10, 31),
        "deep_shadow": np.linspace(-0.025, 0.025, 21),
    }
    for name, positions in trajectories.items():
        for x in positions:
            x = float(x)
            _, _, focal0, theta0 = parameter_bounds(scene, x)
            t0 = time.perf_counter()
            params, selected, bl = beam_parameters(scene, network, x)
            inference_ms = (time.perf_counter() - t0) * 1000
            power_focus = engine.received_power(x, 0.0, focal0, theta0)
            power_direct = engine.received_power(x, *params)
            rows.append({"trajectory": name, "receiver_x_m": x, "blockage_ratio": bl,
                         "selected": selected, "B_per_m": params[0], "F_m": params[1],
                         "theta_deg": params[2], "power_focus": power_focus,
                         "power_direct": power_direct, "gain_db": gain_db(power_direct, power_focus),
                         "inference_ms": inference_ms})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / filename, index=False, float_format="%.12g")
    summary = {}
    for name, group in table.groupby("trajectory", sort=False):
        active = group[group.selected == "direct_nn"]
        summary[name] = {"positions": len(group), "active_positions": len(active),
                         "mean_gain_db": float(group.gain_db.mean()),
                         "active_mean_gain_db": float(active.gain_db.mean()),
                         "active_min_gain_db": float(active.gain_db.min()),
                         "active_positive_fraction": float((active.gain_db > 0).mean()),
                         "active_mean_pointwise_power_change_pct": float(100 * ((active.power_direct / active.power_focus).mean() - 1)),
                         "mean_inference_ms": float(group.inference_ms.mean())}
    return summary


def evaluate_delayed(scene: Scene, network: SceneNetwork,
                     engine: AngularSpectrum, filename: str) -> dict:
    x = receiver_trajectory()
    rows = []
    for t in range(2, len(x)):
        actual = float(x[t])
        stale = float(x[t - 1])
        predicted = predict_at_frame(x[:t], t)
        row = {"time_index": t, "actual_x_m": actual, "stale_x_m": stale,
               "predicted_x_m": predicted}
        for name, design_x in (("stale", stale), ("predicted", predicted), ("oracle", actual)):
            params, selected, _ = beam_parameters(scene, network, design_x)
            _, _, focal0, theta0 = parameter_bounds(scene, design_x)
            row[f"selected_{name}"] = selected
            row[f"B_{name}"] = params[0]
            row[f"F_{name}"] = params[1]
            row[f"theta_{name}"] = params[2]
            row[f"power_{name}_direct"] = engine.received_power(actual, *params)
            row[f"power_{name}_focus"] = engine.received_power(actual, 0.0, focal0, theta0)
        row["gain_pred_vs_stale_db"] = gain_db(row["power_predicted_direct"], row["power_stale_direct"])
        row["gain_pred_vs_focus_db"] = gain_db(row["power_predicted_direct"], row["power_predicted_focus"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / filename, index=False, float_format="%.12g")
    def comparison(a: str, b: str) -> dict:
        ratio = table[a].to_numpy() / table[b].to_numpy()
        return {"mean_pointwise_power_change_pct": float(100 * (ratio.mean() - 1)),
                "total_power_change_pct": float(100 * (table[a].sum() / table[b].sum() - 1)),
                "mean_pointwise_db": float((10 * np.log10(ratio)).mean()),
                "better_frames": int(np.count_nonzero(ratio > 1))}
    return {"frames": len(table),
            "stale_position_mae_mm": float(1000 * np.abs(table.stale_x_m - table.actual_x_m).mean()),
            "predicted_position_mae_mm": float(1000 * np.abs(table.predicted_x_m - table.actual_x_m).mean()),
            "predicted_direct_vs_stale_direct": comparison("power_predicted_direct", "power_stale_direct"),
            "predicted_direct_vs_predicted_focus": comparison("power_predicted_direct", "power_predicted_focus"),
            "predicted_direct_vs_oracle_direct": comparison("power_predicted_direct", "power_oracle_direct")}


def main() -> None:
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=0.00075)
    engine = AngularSpectrum(scene)
    (ROOT / "results").mkdir(exist_ok=True)
    trained = SceneNetwork()
    result = {"scene": asdict(scene),
              "method": "Offline-trained, one-pass beam prediction; geometry-based focus activation",
              "static": evaluate_static(scene, trained, engine, "static_results.csv"),
              "delayed": evaluate_delayed(scene, trained, engine, "delayed_results.csv")}
    (ROOT / "results" / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
