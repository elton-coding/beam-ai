"""E3: end-to-end cooperative tracking and beamforming on 16 unseen geometries.

Policies (paired: same trajectories, gains, noise, losses):
  legacy            previous scheme: Kalman + gated template match + recovery sweep + learned Airy net
  focus_bayes       new Bayes tracker, focused beam (ablation of the beam design)
  proposed          new tracker + robust phase-only design over position/geometry beliefs
  proposed_verified proposed + in-frame probe of {design, focus}, receiver picks the stronger
  proposed_airy     new tracker + robust online Airy search (for Airy-only hardware)
  codebook          classical beam training: 17-beam sweep every frame, best measured beam
  oracle            true position and true geometry, phase-only optimum, no sensing overhead
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfbeam.link import Condition, RunRandom, SceneContext, run_coop, trajectory
from nfbeam.metrics import add_columns, paired_change, region_summary
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json

OUT = RESULTS / "e3"
RUNS = 5
POLICIES = ("legacy", "focus_bayes", "proposed", "proposed_verified", "proposed_airy", "codebook", "oracle")
CONDITIONS = {
    "matched": Condition("matched"),
    "geometry_mismatch": Condition("geometry_mismatch", geometry="mismatch"),
    "strong_mismatch": Condition("strong_mismatch", geometry="strong"),
    "pilot_noise_+6dB": Condition("pilot_noise_+6dB", pilot_noise=4 * 3e-4),
}


def work(item) -> list[dict]:
    cname, sid, scene, policies, runs = item
    cond = CONDITIONS[cname]
    ctx = SceneContext(sid, scene, cond, seed=7000 + sid, need_codebook="codebook" in policies)
    rows = []
    for run in range(runs):
        path = trajectory(run)
        rr = RunRandom.draw(len(path), 181100 + 100 * sid + run, cond.loss)
        for pol in policies:
            for r in run_coop(ctx, pol, path, rr):
                r.update(condition=cname, scene_id=sid, run=run, policy=pol)
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--policies", nargs="*", default=list(POLICIES))
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    scenes = load_scenes("test")
    items = [(c, sid, sc, tuple(args.policies), args.runs) for c in args.conditions for sid, sc in scenes]
    frames = pd.DataFrame([r for chunk in run_parallel(work, items, args.workers) for r in chunk])
    OUT.mkdir(parents=True, exist_ok=True)
    frames.to_csv(OUT / "frames.csv.gz", index=False, float_format="%.7g")
    f = add_columns(frames)
    summary = {"scenes": len(scenes), "runs_per_scene": args.runs, "frames_per_run": 101,
               "metrics": region_summary(f, ["condition", "policy"]),
               "vs_legacy": [r for p in args.policies if p != "legacy"
                             for r in paired_change(f, ["condition"], p, "legacy")],
               "vs_focus_bayes": [r for p in ("proposed", "proposed_verified", "proposed_airy")
                                  if p in args.policies for r in paired_change(f, ["condition"], p, "focus_bayes")]}
    write_json(OUT / "summary.json", summary)
    m = pd.DataFrame(summary["metrics"])
    print(m[m.region == "deep"][["condition", "policy", "se_mean", "low_rate_pct", "mae_mm", "p95_mm",
                                 "lock_lost_pct", "scan_pct"]].round(3).to_string(index=False))
    c = pd.DataFrame(summary["vs_legacy"])
    print(c[c.region == "deep"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
