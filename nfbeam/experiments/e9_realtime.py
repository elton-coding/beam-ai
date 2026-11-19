"""E9: real-time beam design without giving up scene generality.

The full design (fresh robust Gram matrix, eigen/focus/previous seeds,
fixed-point iteration to convergence) takes tens of milliseconds, longer than
the 20 ms frame.  Two real-time variants, both computed from the channel model
of whatever scene is present (no offline training, nothing scene-specific
stored in advance):

  proposed_warm   matrix-free robust operator (receiver samples decimated to
                  at most 900 columns when the belief is wide) + 5 fixed-point
                  steps from the better of the previous frame's beam and the
                  focus beam; the first frame runs 20 steps from focus.
  proposed_table  same, but also seeded from a scene-adaptive table: shadow
                  beams on a 1 mm grid, built online once the obstacle
                  geometry is estimated (built per scene here; build time is
                  reported).

Compared against the full design (``proposed``) and the focused-beam Bayes
tracker on matched geometry, strong geometry mismatch and 3-frame delay with
bursty loss.  All timings are single-thread wall clock inside the simulation
processes (12 in parallel on a 24-core machine).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # single-thread timings
    os.environ.setdefault(_k, "1")

import argparse

import numpy as np
import pandas as pd

from nfbeam.link import TABLE_X, Condition, RunRandom, SceneContext, run_coop, trajectory
from nfbeam.metrics import add_columns, paired_change, region_summary
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json

OUT = RESULTS / "e9"
RUNS = 5
FRAME_MS = 20.0
POLICIES = ("focus_bayes", "proposed", "proposed_warm", "proposed_table")
CONDITIONS = {
    "matched": Condition("matched"),
    "strong_mismatch": Condition("strong_mismatch", geometry="strong"),
    "d3_burst": Condition("d3_burst", delay=3, loss="burst"),
}


def work(item) -> tuple[list[dict], dict]:
    cname, sid, scene, runs = item
    cond = CONDITIONS[cname]
    ctx = SceneContext(sid, scene, cond, seed=7000 + sid, need_legacy=False)
    build_s = ctx.build_table()
    rows = []
    for run in range(runs):
        path = trajectory(run)
        rr = RunRandom.draw(len(path), 181100 + 100 * sid + run, cond.loss)
        for pol in POLICIES:
            for r in run_coop(ctx, pol, path, rr):
                r.update(condition=cname, scene_id=sid, run=run, policy=pol)
                rows.append(r)
    return rows, {"condition": cname, "scene_id": sid, "table_build_s": build_s,
                  "table_entries": len(TABLE_X)}


SCALING_WIDTHS_M = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def scaling(scene, steps: int = 30) -> list[dict]:
    """Design time versus element count on one scene (single process, nothing else running).

    A receiver walks 0.6 mm per frame through the shadow with a 2 mm position
    belief and the 9 geometry hypotheses; full design versus one warm step.
    """
    import time
    from dataclasses import replace
    from nfbeam import design as D
    from nfbeam.channel import Channel
    from nfbeam.link import REALTIME_MAX_COLUMNS, WARM_ITERS
    from nfbeam.scenes import hypotheses
    from nfbeam.tracking import X_GRID
    out = []
    for width in SCALING_WIDTHS_M:
        hyps = [Channel(h) for h in hypotheses(replace(scene, tx_width_m=width))]
        nominal = hyps[len(hyps) // 2]
        pi = np.full(len(hyps), 0.5 / (len(hyps) - 1))
        pi[len(hyps) // 2] = 0.5
        prev, full_t, warm_t, ratio = None, [], [], []
        for x in -0.009 + 0.0006 * np.arange(steps):
            px = np.exp(-0.5 * ((X_GRID - x) / 0.002) ** 2)
            keep = px > 1e-4 * px.max()
            xs, ps = X_GRID[keep], px[keep] / px[keep].sum()
            t0 = time.perf_counter()
            A = D.robust_gram(hyps, pi, xs, ps)
            w_full = D.best_phase_only(nominal, A, x, extra_seeds=[prev] if prev is not None else None)
            t1 = time.perf_counter()
            op = D.PowerOperator(hyps, pi, xs, ps, max_columns=REALTIME_MAX_COLUMNS)
            seeds = [D.focus_beam(nominal, x)] + ([prev] if prev is not None else [])
            w_warm = D.warm_phase_only(nominal, op, seeds, iters=WARM_ITERS)
            t2 = time.perf_counter()
            if prev is not None:
                full_t.append(t1 - t0)
                warm_t.append(t2 - t1)
                ratio.append(D.quad(A, w_warm) / D.quad(A, w_full))
            prev = w_warm
        out.append({"tx_width_m": width, "n_elements": nominal.n_elements,
                    "full_ms_mean": 1e3 * float(np.mean(full_t)), "warm_ms_mean": 1e3 * float(np.mean(warm_t)),
                    "warm_ms_max": 1e3 * float(np.max(warm_t)),
                    "warm_over_full_power_mean": float(np.mean(ratio)),
                    "warm_over_full_power_min": float(np.min(ratio))})
    return out


def timing(f: pd.DataFrame) -> list[dict]:
    out = []
    for (cond, pol), g in f.groupby(["condition", "policy"], sort=False):
        steady = g[g.frame > 0].design_ms
        out.append({"condition": cond, "policy": pol,
                    "first_frame_ms_mean": float(g[g.frame == 0].design_ms.mean()),
                    "mean_ms": float(steady.mean()), "p50_ms": float(steady.median()),
                    "p99_ms": float(steady.quantile(.99)), "max_ms": float(steady.max()),
                    "over_frame_pct": float(100 * np.mean(steady > FRAME_MS))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scaling-only", action="store_true", help="only rerun the element-count timing")
    args = ap.parse_args()
    scenes = load_scenes("test")
    if args.scaling_only:
        prev = __import__("json").loads((OUT / "summary.json").read_text(encoding="utf-8"))
        prev["scaling"] = scaling(scenes[0][1])
        write_json(OUT / "summary.json", prev)
        print(pd.DataFrame(prev["scaling"]).round(3).to_string(index=False))
        return
    items = [(c, sid, sc, args.runs) for c in CONDITIONS for sid, sc in scenes]
    res = run_parallel(work, items, args.workers)
    frames = pd.DataFrame([r for rows, _ in res for r in rows])
    builds = pd.DataFrame([b for _, b in res])
    OUT.mkdir(parents=True, exist_ok=True)
    frames.to_csv(OUT / "frames.csv.gz", index=False, float_format="%.7g")
    f = add_columns(frames)
    summary = {"scenes": len(scenes), "runs_per_scene": args.runs, "frame_ms": FRAME_MS,
               "warm_iterations": 5,
               "table": {"entries": int(len(TABLE_X)), "grid_mm": 1.0,
                         "build_s_mean": float(builds.table_build_s.mean()),
                         "build_s_max": float(builds.table_build_s.max()),
                         "note": "single thread; entries are independent and parallelise"},
               "timing": timing(f),
               "metrics": region_summary(f, ["condition", "policy"]),
               "scaling": scaling(scenes[0][1]),
               "vs_full": [r for p in ("proposed_warm", "proposed_table", "focus_bayes")
                           for r in paired_change(f, ["condition"], p, "proposed")]}
    write_json(OUT / "summary.json", summary)
    print(pd.DataFrame(summary["timing"]).round(2).to_string(index=False))
    m = pd.DataFrame(summary["metrics"])
    print(m[m.region == "deep"][["condition", "policy", "se_mean", "low_rate_pct", "lock_lost_pct",
                                 "scan_pct"]].round(3).to_string(index=False))
    c = pd.DataFrame(summary["vs_full"])
    print(c[c.region != "other"].round(2).to_string(index=False))
    print(summary["table"])
    print(pd.DataFrame(summary["scaling"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
