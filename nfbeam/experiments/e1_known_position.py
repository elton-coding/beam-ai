"""E1: beam families at known receiver positions on 16 unseen geometries.

Replaces the learned one-pass Airy predictor (whose cross-scene gain dropped to
48.7 % with a -74 % 5th percentile) with model-based online optimization on
the exact channel operator.  All methods are scored with the same operator.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from direct_network.core.paths import ARTIFACTS
from direct_network.core.physics import blockage_ratio
from direct_network.experiments.evaluate_general_scene import direct_parameters
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.scenes import RESULTS, fixed_scene, load_scenes, run_parallel, scene_bootstrap, write_json

OUT = RESULTS / "e1"


def positions(scene, labels: pd.DataFrame) -> list[tuple[str, float, object]]:
    pos = [("labeled", float(r.rx_x_m), r) for r in labels.itertuples(index=False)]
    grid = np.linspace(-.11, .11, 881)
    active = grid[[blockage_ratio(scene, float(x)) >= scene.activation_bl for x in grid]]
    for x in active[np.rint(np.linspace(0, len(active) - 1, 21)).astype(int)]:
        pos.append(("new_position", float(x) + 0.00025, None))
    return [p for p in pos if blockage_ratio(scene, p[1]) >= scene.activation_bl]


def evaluate_scene(item: tuple[int, object, pd.DataFrame]) -> list[dict]:
    from direct_network.core.general_model import GeneralSceneNetwork
    scene_id, scene, labels = item
    ch = Channel(scene)
    net = GeneralSceneNetwork()
    rows = []
    for group, x, label in positions(scene, labels):
        A = ch.gram_at(x)
        timings = {}
        t = time.perf_counter(); w_focus = D.focus_beam(ch, x); timings["focus"] = time.perf_counter() - t
        t = time.perf_counter(); w_legacy = D.airy_beam(ch, *direct_parameters(scene, net, x)); timings["legacy_net_airy"] = time.perf_counter() - t
        t = time.perf_counter(); w_airy = D.airy_search(ch, A, x); timings["airy_online"] = time.perf_counter() - t
        t = time.perf_counter(); w_phase = D.best_phase_only(ch, A, x, extra_seeds=[w_airy]); timings["phase_only"] = time.perf_counter() - t
        t = time.perf_counter(); w_2bit = D.best_phase_only(ch, A, x, bits=2, extra_seeds=[w_phase, w_airy]); timings["phase_2bit"] = time.perf_counter() - t
        ceil, _ = D.ceiling(ch, A)
        row = {"scene_id": scene_id, "group": group, "rx_x_m": x,
               "blockage_ratio": blockage_ratio(scene, x),
               "focus": D.quad(A, w_focus), "legacy_net_airy": D.quad(A, w_legacy),
               "airy_online": D.quad(A, w_airy), "phase_only": D.quad(A, w_phase),
               "phase_2bit": D.quad(A, w_2bit), "ceiling_amp_phase": ceil}
        if label is not None:
            row["offline_teacher_airy"] = D.quad(A, D.airy_beam(ch, label.B_per_m, label.F_m, label.theta_deg))
        row.update({f"ms_{k}": 1000 * v for k, v in timings.items()})
        rows.append(row)
    return rows


METHODS = ["legacy_net_airy", "offline_teacher_airy", "airy_online", "phase_only", "phase_2bit", "ceiling_amp_phase"]


def summarize(frame: pd.DataFrame) -> dict:
    out = {"positions": len(frame), "scenes": int(frame.scene_id.nunique()),
           "focus_mean_power": float(frame.focus.mean())}
    for m in METHODS:
        if m not in frame or frame[m].isna().all():
            continue
        sub = frame.dropna(subset=[m])
        ratio = sub[m] / sub.focus
        by_scene = (100 * (ratio - 1)).groupby(sub.scene_id).mean()
        out[m] = {"mean_power": float(sub[m].mean()),
                  "vs_focus_mean_pointwise_pct": float(100 * np.mean(ratio - 1)),
                  "vs_focus_mean_db": float(np.mean(10 * np.log10(ratio))),
                  "better_than_focus_pct": float(100 * np.mean(ratio > 1)),
                  "vs_focus_p05_pct": float(np.quantile(100 * (ratio - 1), .05)),
                  "worst_vs_focus_pct": float(100 * (ratio.min() - 1)),
                  "scene_bootstrap_95ci_pct": scene_bootstrap(by_scene.to_numpy()) if len(by_scene) > 1 else None,
                  "mean_design_ms": float(sub[f"ms_{m}"].mean()) if f"ms_{m}" in sub else None}
    return out


def main() -> None:
    labels = pd.read_csv(ARTIFACTS / "general_labels.csv")
    items = [(sid, sc, labels[labels.scene_id == sid]) for sid, sc in load_scenes("test")]
    fixed_labels = pd.read_csv(ARTIFACTS / "scene_labels.csv")
    rows = [r for chunk in run_parallel(evaluate_scene, items) for r in chunk]
    fixed_rows = evaluate_scene((-1, fixed_scene(), fixed_labels.iloc[0:0]))
    # Fixed scene: evaluate the deep-shadow labelled positions with their own teacher.
    frame = pd.DataFrame(rows)
    fixed = pd.DataFrame(fixed_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "positions.csv", index=False, float_format="%.9g")
    fixed.to_csv(OUT / "fixed_scene_positions.csv", index=False, float_format="%.9g")
    summary = {"definition": "Power at the true receiver position, unit transmitted power, element-wise "
                             "half-wavelength aperture; percentages are pointwise ratios to focusing.",
               "heldout_all": summarize(frame),
               "heldout_labeled": summarize(frame[frame.group == "labeled"]),
               "heldout_new_positions": summarize(frame[frame.group == "new_position"]),
               "fixed_scene_deep": summarize(fixed)}
    write_json(OUT / "summary.json", summary)
    for key in ("heldout_all", "fixed_scene_deep"):
        s = summary[key]
        print(key, s["positions"], "positions")
        for m in METHODS:
            if m in s:
                v = s[m]
                print(f"  {m:22s} +{v['vs_focus_mean_pointwise_pct']:7.1f}%  better {v['better_than_focus_pct']:5.1f}%  "
                      f"p05 {v['vs_focus_p05_pct']:7.1f}%  worst {v['worst_vs_focus_pct']:7.1f}%  ms {v['mean_design_ms']}")


if __name__ == "__main__":
    main()
