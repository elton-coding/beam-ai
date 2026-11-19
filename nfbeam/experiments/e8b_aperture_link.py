"""E8b: end-to-end cooperative link for several aperture sizes in the same room.

Same physical scenes, trajectories and impairments as E3 (matched geometry),
with the transmit aperture set to 0.1 / 0.2 / 0.3 m.  Total transmit power and
the data-link noise are those of the 0.2 m reference link, so a larger array
only helps through its aperture, not through more power.  The spectral
efficiency under per-element power (total power growing with element count) is
recomputed from the same frames; pilots are left at the reference power, so
that column is a slightly pessimistic view of the larger arrays.

Policies: focus_bayes (Bayes tracker + focusing), proposed_warm (real-time
robust phase-only design, E9) and oracle (true position and geometry).
Deep shadow is reported both for the aperture in use and for the 0.2 m
reference (``deep_ref``), so the same positions can be compared across sizes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from direct_network.core.physics import blockage_ratio
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.link import Condition, RunRandom, SceneContext, run_coop, trajectory
from nfbeam.metrics import add_columns, paired_change, region_summary
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json

OUT = RESULTS / "e8b"
RUNS = 5
APERTURES_M = (0.1, 0.2, 0.3)
POLICIES = ("focus_bayes", "proposed_warm", "oracle")


def work(item) -> list[dict]:
    width, sid, scene = item
    ref = Channel(replace(scene, tx_width_m=0.2))
    noise = ref.power(D.focus_beam(ref, 0.095), 0.095) / 10
    sc = replace(scene, tx_width_m=width)
    ctx = SceneContext(sid, sc, Condition("matched"), seed=7000 + sid, need_legacy=False, data_noise=noise)
    ref_scene = replace(scene, tx_width_m=0.2)
    rows = []
    for run in range(RUNS):
        path = trajectory(run)
        rr = RunRandom.draw(len(path), 181100 + 100 * sid + run, "none")
        for pol in POLICIES:
            for r in run_coop(ctx, pol, path, rr):
                r.update(tx_width_m=width, n_elements=ctx.nominal.n_elements, scene_id=sid, run=run, policy=pol,
                         deep_ref=blockage_ratio(ref_scene, r["actual_x_m"]) >= scene.activation_bl,
                         se_per_element_power=r["payload_fraction"] * np.log2(
                             1 + r["power"] * ctx.nominal.n_elements / 160 / noise))
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    scenes = load_scenes("test")
    items = [(w, sid, sc) for w in sorted(APERTURES_M, reverse=True) for sid, sc in scenes]
    frames = pd.DataFrame([r for chunk in run_parallel(work, items, args.workers) for r in chunk])
    OUT.mkdir(parents=True, exist_ok=True)
    frames.to_csv(OUT / "frames.csv.gz", index=False, float_format="%.7g")
    f = add_columns(frames)
    by_ref = f.assign(deep=f.deep_ref)
    by_ref_elem = by_ref.assign(se_bpshz=by_ref.se_per_element_power)
    summary = {"scenes": len(scenes), "runs_per_scene": RUNS, "apertures_m": list(APERTURES_M),
               "note": "total transmit power and data noise fixed at the 0.2 m reference; 'deep' regions use the "
                       "aperture in use, 'deep_ref' summaries the 0.2 m deep-shadow positions",
               "metrics": region_summary(f, ["tx_width_m", "policy"]),
               "metrics_deep_ref": region_summary(by_ref, ["tx_width_m", "policy"]),
               "metrics_deep_ref_per_element_power": region_summary(by_ref_elem, ["tx_width_m", "policy"]),
               "warm_vs_focus_deep_ref": paired_change(by_ref, ["tx_width_m"], "proposed_warm", "focus_bayes"),
               "warm_vs_oracle_deep_ref": paired_change(by_ref, ["tx_width_m"], "proposed_warm", "oracle")}
    write_json(OUT / "summary.json", summary)
    m = pd.DataFrame(summary["metrics_deep_ref"])
    print(m[m.region == "deep"][["tx_width_m", "policy", "se_mean", "low_rate_pct", "lock_lost_pct",
                                 "scan_pct", "mae_mm"]].round(3).to_string(index=False))
    m = pd.DataFrame(summary["metrics_deep_ref_per_element_power"])
    print(m[m.region == "deep"][["tx_width_m", "policy", "se_mean", "low_rate_pct"]].round(3).to_string(index=False))
    c = pd.DataFrame(summary["warm_vs_focus_deep_ref"])
    print(c.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
