"""E2: robustness of beam designs to position error and obstacle-geometry mismatch.

Every design only knows the estimated position x_hat = x + e, e ~ N(0, sigma),
and the nominal geometry.  The truth geometry is the nominal one or a random
perturbation that is never on the hypothesis grid.  Power is scored on the
truth channel at the true position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from direct_network.core.paths import ARTIFACTS
from direct_network.experiments.evaluate_general_scene import direct_parameters
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.scenes import (RESULTS, hypotheses, load_scenes, random_mismatch, run_parallel,
                           scene_bootstrap, write_json)
from nfbeam.tracking import X_GRID

OUT = RESULTS / "e2"
SIGMAS_MM = (1.0, 3.0, 6.0)
GEOMETRIES = ("matched", "mismatch", "strong")
METHODS = ("focus", "legacy_net_airy", "nominal_phase_only", "robust_airy", "robust_phase_only")


def work(item) -> list[dict]:
    from direct_network.core.general_model import GeneralSceneNetwork
    sid, scene, xs = item
    net = GeneralSceneNetwork()
    hyps = [Channel(s) for s in hypotheses(scene)]
    nominal = hyps[len(hyps) // 2]
    pi = np.full(len(hyps), 0.5 / (len(hyps) - 1))
    pi[len(hyps) // 2] = 0.5
    rng = np.random.default_rng(4100 + sid)
    rows = []
    for geometry in GEOMETRIES:
        truths = ([nominal] if geometry == "matched" else
                  [Channel(random_mismatch(scene, rng, strong=geometry == "strong")) for _ in range(2)])
        for ti, truth in enumerate(truths):
            for sigma_mm in SIGMAS_MM:
                sigma = sigma_mm / 1000
                for x in xs:
                    xh = float(x + rng.normal(0, sigma))
                    px = np.exp(-0.5 * ((X_GRID - xh) / sigma) ** 2)
                    keep = px > 1e-4 * px.max()
                    A_rob = D.robust_gram(hyps, pi, X_GRID[keep], px[keep] / px[keep].sum())
                    A_nom = nominal.gram_at(xh)
                    beams = {"focus": D.focus_beam(nominal, xh),
                             "legacy_net_airy": D.airy_beam(nominal, *direct_parameters(scene, net, xh)),
                             "nominal_phase_only": D.best_phase_only(nominal, A_nom, xh),
                             "robust_airy": D.airy_search(nominal, A_rob, xh, max_eval=120),
                             "robust_phase_only": D.best_phase_only(nominal, A_rob, xh)}
                    row = {"scene_id": sid, "geometry": geometry, "truth_index": ti,
                           "sigma_mm": sigma_mm, "rx_x_m": float(x)}
                    row.update({k: truth.power(w, x) for k, w in beams.items()})
                    rows.append(row)
    return rows


def main() -> None:
    labels = pd.read_csv(ARTIFACTS / "general_labels.csv")
    items = []
    for sid, sc in load_scenes("test"):
        xs = np.sort(labels[labels.scene_id == sid].rx_x_m.to_numpy())[::3]   # 10 per scene
        items.append((sid, sc, xs))
    frame = pd.DataFrame([r for chunk in run_parallel(work, items) for r in chunk])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "positions.csv", index=False, float_format="%.9g")
    out = []
    for (geometry, sigma), g in frame.groupby(["geometry", "sigma_mm"]):
        for m in METHODS[1:]:
            ratio = g[m] / g.focus
            by_scene = (100 * (ratio - 1)).groupby(g.scene_id).mean().to_numpy()
            out.append({"geometry": geometry, "sigma_mm": sigma, "method": m, "positions": len(g),
                        "vs_focus_mean_pct": float(100 * np.mean(ratio - 1)),
                        "better_than_focus_pct": float(100 * np.mean(ratio > 1)),
                        "vs_focus_p05_pct": float(np.quantile(100 * (ratio - 1), .05)),
                        "worst_vs_focus_pct": float(100 * (ratio.min() - 1)),
                        "mean_power": float(g[m].mean()),
                        "scene_bootstrap_95ci_pct": scene_bootstrap(by_scene)})
    write_json(OUT / "summary.json", {"definition": "true power / power of focusing at the estimated position",
                                      "results": out})
    print(pd.DataFrame(out).drop(columns="scene_bootstrap_95ci_pct").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
