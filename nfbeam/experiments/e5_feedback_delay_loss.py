"""E5 (gap 2): feedback delay and loss on the cooperative link.

Policies:
  legacy           previous scheme; late reports applied as if current
  proposed_naive   new tracker/design, late reports applied as if current
  proposed         new tracker/design with out-of-sequence re-filtering; the
                   belief widens while reports are missing, which both widens
                   the robust beam and triggers a re-acquisition sweep
Delays of 0/1/3 frames x losses none / i.i.d. 20 % / Gilbert-Elliott bursts.
"""

from __future__ import annotations

import pandas as pd

from nfbeam.link import Condition, RunRandom, SceneContext, run_coop, trajectory
from nfbeam.metrics import add_columns, paired_change, region_summary
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json

OUT = RESULTS / "e5"
RUNS = 5
DELAYS = (0, 1, 3)
LOSSES = ("none", "iid20", "burst")
POLICIES = ("legacy", "proposed_naive", "proposed")


def work(item) -> list[dict]:
    delay, loss, sid, scene = item
    cond = Condition(f"d{delay}_{loss}", delay=delay, loss=loss)
    ctx = SceneContext(sid, scene, cond, seed=7000 + sid)
    rows = []
    for run in range(RUNS):
        path = trajectory(run)
        rr = RunRandom.draw(len(path), 181100 + 100 * sid + run, loss)
        for pol in POLICIES:
            res = (run_coop(ctx, "proposed", path, rr, delay_aware=False) if pol == "proposed_naive"
                   else run_coop(ctx, pol, path, rr))
            for r in res:
                r.update(delay=delay, loss=loss, scene_id=sid, run=run, policy=pol)
                rows.append(r)
    return rows


def main() -> None:
    scenes = load_scenes("test")
    items = [(d, l, sid, sc) for d in DELAYS for l in LOSSES for sid, sc in scenes]
    frames = pd.DataFrame([r for chunk in run_parallel(work, items) for r in chunk])
    OUT.mkdir(parents=True, exist_ok=True)
    frames.to_csv(OUT / "frames.csv.gz", index=False, float_format="%.7g")
    f = add_columns(frames)
    summary = {"scenes": len(scenes), "runs_per_scene": RUNS,
               "loss_models": {"iid20": "independent 20 % report loss",
                               "burst": "Gilbert-Elliott, good->bad 3 %, bad->good 25 %, loss 90 % in bad / 1 % in good"},
               "metrics": region_summary(f, ["delay", "loss", "policy"]),
               "proposed_vs_legacy": paired_change(f, ["delay", "loss"], "proposed", "legacy"),
               "proposed_vs_naive": paired_change(f, ["delay", "loss"], "proposed", "proposed_naive")}
    write_json(OUT / "summary.json", summary)
    m = pd.DataFrame(summary["metrics"])
    print(m[m.region == "deep"][["delay", "loss", "policy", "se_mean", "low_rate_pct", "mae_mm",
                                 "lock_lost_pct", "longest_lock_loss_ms_mean", "scan_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
