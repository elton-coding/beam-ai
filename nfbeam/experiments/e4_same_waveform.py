"""E4 (gap 1): non-cooperative receiver — dedicated illumination vs data-waveform echoes.

All policies use the same Bayes filter and the same robust phase-only data
beam; they differ only in how the transmitter senses the receiver:
  sweep21_focus            21 focused illumination beams every frame (old front end)
  sweep21_focus_nogating   same, without range gating (old proxy; needs a background)
  sweep9_shadow            9 shadow-penetrating illumination beams every frame
  same_waveform            echo of the data waveform every frame, 9-beam sweep only
                           when the belief is wider than 5 mm (and at start)
  same_waveform_nogating   same_waveform without range gating
Resources are equal: same frame, transmit power and reference/feedback slots;
dedicated illumination is paid for with payload time.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from nfbeam.link import Condition, RunRandom, SceneContext, run_passive, trajectory
from nfbeam.metrics import add_columns, paired_change, region_summary
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json
from nfbeam.sensing import EchoConfig

OUT = RESULTS / "e4"
RUNS = 5
BASE = EchoConfig()
CASES = {
    "matched": (Condition("matched", echo=BASE),
                ("sweep21_focus", "sweep9_shadow", "same_waveform", "same_waveform_nogating")),
    "no_gating_fresh_background": (Condition("no_gating_fresh_background", echo=replace(BASE, background="fresh")),
                                   ("sweep21_focus_nogating", "same_waveform_nogating")),
    "no_gating_stale_background": (Condition("no_gating_stale_background", echo=replace(BASE, background="stale")),
                                   ("sweep21_focus_nogating", "same_waveform_nogating", "same_waveform")),
    "bandwidth_1GHz": (Condition("bandwidth_1GHz", echo=replace(BASE, bandwidth_hz=1e9)),
                       ("sweep9_shadow", "same_waveform")),
    "geometry_mismatch": (Condition("geometry_mismatch", geometry="mismatch", echo=BASE),
                          ("sweep21_focus", "sweep9_shadow", "same_waveform")),
}


def work(item) -> list[dict]:
    case, sid, scene = item
    cond, policies = CASES[case]
    ctx = SceneContext(sid, scene, cond, seed=7000 + sid, need_legacy=False)
    rows = []
    for run in range(RUNS):
        path = trajectory(run)
        rr = RunRandom.draw(len(path), 181100 + 100 * sid + run, "none")
        rho = 10 ** (np.random.default_rng(900 + 100 * sid + run).uniform(-6, 0) / 10)
        for pol in policies:
            for r in run_passive(ctx, pol, path, rr, rho, seed=55000 + 100 * sid + run):
                r.update(condition=case, scene_id=sid, run=run, policy=pol, rho=rho)
                rows.append(r)
    return rows


def main() -> None:
    scenes = load_scenes("test")
    items = [(c, sid, sc) for c in CASES for sid, sc in scenes]
    frames = pd.DataFrame([r for chunk in run_parallel(work, items) for r in chunk])
    OUT.mkdir(parents=True, exist_ok=True)
    frames.to_csv(OUT / "frames.csv.gz", index=False, float_format="%.7g")
    f = add_columns(frames)
    summary = {"scenes": len(scenes), "runs_per_scene": RUNS,
               "echo_model": BASE.__dict__,
               "metrics": region_summary(f, ["condition", "policy"]),
               "vs_sweep21_focus": [r for p in ("sweep9_shadow", "same_waveform")
                                    for r in paired_change(f[f.condition.isin(["matched", "geometry_mismatch"])],
                                                           ["condition"], p, "sweep21_focus")]}
    write_json(OUT / "summary.json", summary)
    m = pd.DataFrame(summary["metrics"])
    print(m[m.region == "deep"][["condition", "policy", "se_mean", "low_rate_pct", "mae_mm", "p95_mm",
                                 "lock_lost_pct", "longest_lock_loss_ms_mean", "scan_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
