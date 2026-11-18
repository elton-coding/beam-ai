"""Offline physical optimization labels for the fixed experimental geometry.

This script is intentionally separate from online inference. It never loads any
published neural-network weights or previously evaluated trajectory results.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from direct_network.core.paths import ROOT
from direct_network.core.physics import AngularSpectrum, Scene, blockage_ratio, features, parameter_bounds


def optimize_label(scene: Scene, engine: AngularSpectrum, x: float) -> dict:
    lo, hi, _, theta0 = parameter_bounds(scene, x)
    cache = {}

    def power(params: np.ndarray) -> float:
        params = np.clip(np.asarray(params, dtype=float), lo, hi)
        key = tuple(np.round(params, 10))
        if key not in cache:
            cache[key] = engine.received_power(x, *params)
        return cache[key]

    candidates = []
    for sign in (-1, 1):
        branch_lo, branch_hi = lo.copy(), hi.copy()
        if sign < 0:
            branch_hi[0] = -0.5
        else:
            branch_lo[0] = 0.5
        seeds = [np.array([sign * b, f, theta0 + dt])
                 for b in (3.5, 5.0) for f in (0.19, 0.26, 0.42) for dt in (-10.0, 10.0)]
        seed = max(seeds, key=power)

        def objective(unit: np.ndarray) -> float:
            params = branch_lo + np.clip(unit, 0, 1) * (branch_hi - branch_lo)
            return -power(params)

        result = minimize(objective, (seed - branch_lo) / (branch_hi - branch_lo),
                          method="Powell", bounds=[(0, 1)] * 3,
                          options={"maxfev": 70, "xtol": 0.01, "ftol": 0.0005})
        final = branch_lo + np.clip(result.x, 0, 1) * (branch_hi - branch_lo)
        power(final)
        branch_keys = [key for key in cache if key[0] * sign > 0]
        best_key = max(branch_keys, key=cache.get)
        candidates.append((cache[best_key], best_key))
    best_power, best = max(candidates, key=lambda item: item[0])
    bl = blockage_ratio(scene, x)
    return {"rx_x_m": x, "blockage_ratio": bl, "B_per_m": best[0], "F_m": best[1],
            "theta_deg": best[2], "power_best": best_power,
            "power_negative_branch": candidates[0][0], "power_positive_branch": candidates[1][0],
            "physics_evaluations": len(cache), **{f"feature_{i}": v for i, v in enumerate(features(scene, x, bl))}}


def main() -> None:
    scene = replace(Scene(), n_grid=16384, obstacle_step_m=0.00075)
    engine = AngularSpectrum(scene)
    rng = np.random.default_rng(20261005)
    # Positive half only; the negative half is obtained by exact scene symmetry.
    # Random continuous positions avoid copying any fixed test trajectory point.
    positions = np.sort(rng.uniform(0.00012, 0.0285, 120))
    rows = []
    for i, x in enumerate(positions, 1):
        rows.append(optimize_label(scene, engine, float(x)))
        if i % 10 == 0:
            print(f"offline labels {i}/{len(positions)}", flush=True)
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "scene_labels.csv", index=False, float_format="%.12g")
    (output / "scene.json").write_text(__import__("json").dumps(asdict(scene), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
