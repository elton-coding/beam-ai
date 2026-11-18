"""E7: how far is the deep-shadow link from 1 bit/s/Hz, and what does aperture buy?

Weakness 5 of the previous scheme was that ~64 % of deep-shadow frames stay
below 1 bit/s/Hz even with perfect position.  Beam shape alone cannot fix a
link-budget shortfall, so this experiment quantifies it:
  * the extra SNR margin each beam needs for 90 % of deep-shadow positions to
    reach 1 bit/s/Hz (known position, known geometry);
  * the effect of a larger transmit aperture at the same total transmit power
    and the same receiver noise (0.2 m reference).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from direct_network.core.paths import ARTIFACTS
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json

OUT = RESULTS / "e7"
WIDTHS_M = (0.2, 0.3, 0.4)
TARGET_SE = 1.0


def work(item) -> list[dict]:
    sid, scene, xs = item
    ref_ch = Channel(scene)
    noise = ref_ch.power(D.focus_beam(ref_ch, 0.095), 0.095) / 10
    rows = []
    for width in WIDTHS_M:
        ch = ref_ch if width == scene.tx_width_m else Channel(replace(scene, tx_width_m=width))
        for x in xs:
            A = ch.gram_at(x)
            w_airy = D.airy_search(ch, A, x)
            w_phase = D.best_phase_only(ch, A, x, extra_seeds=[w_airy])
            rows.append({"scene_id": sid, "tx_width_m": width, "rx_x_m": float(x), "noise": noise,
                         "focus": D.quad(A, D.focus_beam(ch, x)), "airy_online": D.quad(A, w_airy),
                         "phase_only": D.quad(A, w_phase)})
    return rows


def main() -> None:
    labels = pd.read_csv(ARTIFACTS / "general_labels.csv")
    items = [(sid, sc, np.sort(labels[labels.scene_id == sid].rx_x_m.to_numpy())[::3])
             for sid, sc in load_scenes("test")]
    frame = pd.DataFrame([r for chunk in run_parallel(work, items) for r in chunk])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "positions.csv", index=False, float_format="%.9g")
    need_snr = 2**TARGET_SE - 1
    out = []
    for width, g in frame.groupby("tx_width_m"):
        for m in ("focus", "airy_online", "phase_only"):
            snr = g[m] / g.noise
            se = np.log2(1 + snr)
            margin = 10 * np.log10(need_snr / snr)          # dB still missing per position
            out.append({"tx_width_m": width, "method": m, "positions": len(g),
                        "se_mean": float(se.mean()), "below_target_pct": float(100 * np.mean(se < TARGET_SE)),
                        "margin_for_90pct_db": float(np.quantile(margin, 0.9)),
                        "margin_median_db": float(np.median(margin))})
    write_json(OUT / "summary.json", {"target_se_bpshz": TARGET_SE,
                                      "definition": "known position and geometry, ideal SE log2(1+P/N) "
                                                    "without overhead; N fixed by the 0.2 m reference link",
                                      "results": out})
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
