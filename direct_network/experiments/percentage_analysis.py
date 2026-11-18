"""Report arithmetic mean of pointwise received-power percentage changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from direct_network.core.paths import ROOT




def mean_pct(numerator: pd.Series, denominator: pd.Series) -> float:
    ratio = numerator.to_numpy(dtype=float) / denominator.to_numpy(dtype=float)
    if not np.all(np.isfinite(ratio)) or np.any(denominator.to_numpy() <= 0):
        raise ValueError("Power ratio is not finite and positive")
    return float(100 * np.mean(ratio - 1))


def main() -> None:
    static = pd.read_csv(ROOT / "results" / "static_results.csv")
    bound_static = pd.read_csv(ROOT / "results" / "bounds_static.csv")
    delayed = pd.read_csv(ROOT / "results" / "delayed_results.csv")
    bound_delayed = pd.read_csv(ROOT / "results" / "bounds_delayed.csv")
    scene = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))["scene"]
    crossing = static.loc[static.trajectory == "shadow_crossing"]
    deep_crossing = crossing.loc[crossing.blockage_ratio >= scene["activation_bl"]]
    other_crossing = crossing.loc[crossing.blockage_ratio < scene["activation_bl"]]
    deep = static.loc[static.trajectory == "deep_shadow"]
    assert len(crossing) == 31 and len(deep_crossing) == 9 and len(other_crossing) == 22
    assert len(deep) == len(bound_static) == 21 and len(delayed) == len(bound_delayed) == 39
    assert np.allclose(deep.receiver_x_m, bound_static.receiver_x_m)
    assert np.allclose(delayed.actual_x_m, bound_delayed.actual_x_m)
    result = {
        "definition": "100 * mean(P_scheme / P_baseline - 1), computed pointwise then averaged; not converted from mean dB",
        "crossing_path": {
            "deep_shadow": {"positions": len(deep_crossing),
                            "current_vs_focus_pct": mean_pct(deep_crossing.power_direct, deep_crossing.power_focus)},
            "other_positions": {"positions": len(other_crossing),
                                "current_vs_focus_pct": mean_pct(other_crossing.power_direct, other_crossing.power_focus)},
            "all": {"positions": len(crossing),
                    "current_vs_focus_pct": mean_pct(crossing.power_direct, crossing.power_focus)},
        },
        "dense_deep_shadow": {
            "positions": len(deep),
            "current_vs_focus_pct": mean_pct(bound_static.power_direct, bound_static.power_focus),
            "best_known_vs_focus_pct": mean_pct(bound_static.power_best_known_airy, bound_static.power_focus),
            "best_known_vs_current_pct": mean_pct(bound_static.power_best_known_airy, bound_static.power_direct),
            "theoretical_vs_focus_pct": mean_pct(bound_static.power_theoretical_ceiling, bound_static.power_focus),
            "theoretical_vs_current_pct": mean_pct(bound_static.power_theoretical_ceiling, bound_static.power_direct),
        },
        "delayed": {
            "frames": len(delayed),
            "current_vs_true_focus_pct": mean_pct(bound_delayed.power_predicted_direct, bound_delayed.power_focus_at_true_x),
            "best_known_vs_true_focus_pct": mean_pct(bound_delayed.power_best_known_airy, bound_delayed.power_focus_at_true_x),
            "best_known_vs_current_pct": mean_pct(bound_delayed.power_best_known_airy, bound_delayed.power_predicted_direct),
            "theoretical_vs_true_focus_pct": mean_pct(bound_delayed.power_theoretical_ceiling, bound_delayed.power_focus_at_true_x),
            "theoretical_vs_current_pct": mean_pct(bound_delayed.power_theoretical_ceiling, bound_delayed.power_predicted_direct),
            "current_vs_predicted_focus_pct": mean_pct(delayed.power_predicted_direct, delayed.power_predicted_focus),
            "predicted_vs_stale_pct": mean_pct(delayed.power_predicted_direct, delayed.power_stale_direct),
        },
    }
    target = ROOT / "results" / "percentage_analysis.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
