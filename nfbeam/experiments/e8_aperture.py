"""E8: transmit aperture as an experimental dimension.

The reference paper (and this project so far) uses a 0.2 m aperture at
120 GHz: D/lambda = 80, with the receiver ~0.37 m away (z/D ~ 1.85).  Related
work spans D/lambda ~ 65-330 and z/D ~ 2-25.  Changing D alone also changes how
much of the aperture the obstacle hides, so three sweeps separate the effects:

  room      same physical scenes, only D changes (0.05-0.5 m).  Answers "what
            does a bigger array buy in the same room", but the shadow itself
            gets shallower as D grows; deep shadow is re-classified per D and
            the 0.2 m deep-shadow set is tracked separately.
  similar   every length (aperture, obstacle, distances, receiver positions and
            width) scaled with D: identical geometric blockage and z/D, only the
            electrical size D/lambda changes (20-200).
  range     D fixed at 0.2 m, every longitudinal length (obstacle depth and
            position, receiver distance) scaled by t: identical geometric
            blockage and D/lambda, only z/D changes (~0.9-15).

Per position: focusing, online Airy search (``airy_search_wide``: grid + Powell valid at any D and range),
element-wise phase-only, and the amplitude-and-phase ceiling.  Scale-free
metrics are reported for every sweep: gain over focusing and shadow loss
(power relative to the unobstructed focused link at the same position).  The
"room" sweep also reports absolute spectral efficiency under fixed total
transmit power and fixed per-element power, with noise fixed by the 0.2 m
reference link (as in E7).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from direct_network.core.physics import blockage_ratio
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json

OUT = RESULTS / "e8"
X_POS = np.linspace(-0.10, 0.10, 41)
REF_D = 0.20
ROOM_D = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
SIMILAR_S = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
RANGE_T = (0.5, 1.0, 2.0, 4.0, 8.0)
LOS_SNR_DB = 10.0
TARGET_SE = 1.0
METHODS = ("focus", "airy", "phase", "ceiling")

# Apertures reported in the reference paper and related work (see docs).
LITERATURE = [
    {"work": "Reference paper (Nat. Commun. 2025)", "f_ghz": 120, "D_m": 0.20, "z_m": 0.37},
    {"work": "Curving THz links (Commun. Eng. 2024), device", "f_ghz": 200, "D_m": 0.0975, "z_m": 0.18},
    {"work": "Zhao/Han/Bjornson closed form, arXiv 2603.13866", "f_ghz": 140, "D_m": 0.273, "z_m": 3.0},
    {"work": "Physics-guided neural Airy, arXiv 2608.04388", "f_ghz": 140, "D_m": 0.273, "z_m": 3.25},
    {"work": "Single to multi-Airy, arXiv 2607.07278", "f_ghz": 100, "D_m": 1.0, "z_m": 5.0},
    {"work": "Holographic Airy, arXiv 2605.16057", "f_ghz": 100, "D_m": 0.20, "z_m": 2.4},
    {"work": "Near-field beam shaping, arXiv 2609.28400", "f_ghz": 150, "D_m": 0.50, "z_m": 1.0},
]
for r in LITERATURE:
    r["D_over_lambda"] = r["D_m"] / (299_792_458.0 / (r["f_ghz"] * 1e9))
    r["z_over_D"] = r["z_m"] / r["D_m"]


def similar_scene(scene, s: float):
    return replace(scene, tx_width_m=REF_D * s, rx_width_m=scene.rx_width_m * s,
                   obstacle_z_m=scene.obstacle_z_m * s, obstacle_x_m=scene.obstacle_x_m * s,
                   obstacle_depth_m=scene.obstacle_depth_m * s, obstacle_width_m=scene.obstacle_width_m * s,
                   receiver_z_m=scene.receiver_z_m * s)


def range_scene(scene, t: float):
    return replace(scene, obstacle_z_m=scene.obstacle_z_m * t, obstacle_depth_m=scene.obstacle_depth_m * t,
                   receiver_z_m=scene.receiver_z_m * t)


def configure(sweep: str, value: float, scene):
    """Scene for one sweep point and the receiver positions it is evaluated at."""
    if sweep == "room":
        return replace(scene, tx_width_m=value), X_POS
    if sweep == "similar":
        return similar_scene(scene, value), X_POS * value
    if sweep == "range":
        return range_scene(scene, value), X_POS
    raise ValueError(sweep)


def work(item) -> list[dict]:
    sweep, value, sid, scene = item
    sc, xs = configure(sweep, value, scene)
    window = max(0.16, float(np.max(np.abs(xs))) + sc.rx_width_m)
    ch = Channel(sc, window_half_width_m=window)
    los = Channel(replace(sc, obstacle_width_m=0.0), window_half_width_m=window)
    ref_noise = np.nan
    if sweep == "room":     # absolute noise fixed by the 0.2 m reference link (E7 convention)
        ref = Channel(replace(scene, tx_width_m=REF_D))
        ref_noise = ref.power(D.focus_beam(ref, 0.095), 0.095) / 10
    rows = []
    for x_unit, x in zip(X_POS, xs):
        A = ch.gram_at(float(x))
        w_focus = D.focus_beam(ch, float(x))
        t0 = time.perf_counter()
        w_airy = D.airy_search_wide(ch, A, float(x))
        t1 = time.perf_counter()
        w_phase = D.best_phase_only(ch, A, float(x), extra_seeds=[w_airy])
        t2 = time.perf_counter()
        ceil, _ = D.ceiling(ch, A)
        bl = blockage_ratio(sc, float(x))
        rows.append({
            "sweep": sweep, "value": value, "scene_id": sid, "x_unit_m": float(x_unit), "rx_x_m": float(x),
            "D_m": sc.tx_width_m, "n_elements": ch.n_elements, "D_over_lambda": sc.tx_width_m / ch.wavelength,
            "z_over_D": sc.receiver_z_m / sc.tx_width_m, "blockage": bl, "deep": bl >= sc.activation_bl,
            "deep_ref": blockage_ratio(replace(scene, tx_width_m=REF_D), float(x_unit)) >= scene.activation_bl,
            "p_los": los.power(D.focus_beam(los, float(x)), float(x)),
            "focus": D.quad(A, w_focus), "airy": D.quad(A, w_airy), "phase": D.quad(A, w_phase),
            "ceiling": ceil, "t_airy_s": t1 - t0, "t_phase_s": t2 - t1, "ref_noise": ref_noise})
    return rows


def _stats(g: pd.DataFrame, sweep: str) -> dict:
    los_snr = 10 ** (LOS_SNR_DB / 10)
    out = {"positions": len(g)}
    for m in METHODS:
        ratio = g[m] / g.focus
        loss_db = 10 * np.log10(g[m] / g.p_los)
        se = np.log2(1 + los_snr * g[m] / g.p_los)
        out.update({f"{m}_gain_median_db": float(10 * np.log10(ratio).median()),
                    f"{m}_gain_mean_pct": float(100 * (ratio.mean() - 1)),
                    f"{m}_gain_p05_pct": float(100 * (ratio.quantile(.05) - 1)),
                    f"{m}_shadow_loss_median_db": float(loss_db.median()),
                    f"{m}_shadow_loss_p10_db": float(loss_db.quantile(.1)),
                    f"{m}_se_los10_mean": float(se.mean()),
                    f"{m}_below_target_los10_pct": float(100 * np.mean(se < TARGET_SE))})
        if sweep == "room":
            snr_tot = g[m] / g.ref_noise
            snr_el = snr_tot * g.n_elements / 160
            for tag, snr in (("total", snr_tot), ("per_element", snr_el)):
                se_abs = np.log2(1 + snr)
                margin = 10 * np.log10((2**TARGET_SE - 1) / snr)
                out.update({f"{m}_se_{tag}_power_mean": float(se_abs.mean()),
                            f"{m}_below_target_{tag}_power_pct": float(100 * np.mean(se_abs < TARGET_SE)),
                            f"{m}_margin90_{tag}_power_db": float(margin.quantile(.9))})
    out["phase_over_ceiling_mean"] = float((g.phase / g.ceiling).mean())
    out["airy_better_than_focus_pct"] = float(100 * np.mean(g.airy > g.focus * 1.001))
    out["t_airy_ms_mean"] = float(1e3 * g.t_airy_s.mean())
    out["t_phase_ms_mean"] = float(1e3 * g.t_phase_s.mean())
    out["los_fraction_mean"] = float((1 - g.blockage).mean())
    return out


def summarize(f: pd.DataFrame) -> list[dict]:
    out = []
    for (sweep, value), g in f.groupby(["sweep", "value"], sort=False):
        base = {"sweep": sweep, "value": value, "D_m": float(g.D_m.iloc[0]),
                "n_elements": int(g.n_elements.iloc[0]), "D_over_lambda": float(g.D_over_lambda.iloc[0]),
                "z_over_D_mean": float(g.z_over_D.mean()), "deep_fraction_pct": float(100 * g.deep.mean())}
        regions = [("all", np.ones(len(g), bool)), ("deep", g.deep.to_numpy())]
        if sweep == "room":
            regions.append(("deep_ref", g.deep_ref.to_numpy()))
        for region, mask in regions:
            if mask.sum() == 0:
                continue
            out.append(base | {"region": region} | _stats(g[mask], sweep))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="*", default=["room", "similar", "range"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true", help="rebuild summary.json from positions.csv")
    args = ap.parse_args()
    grid = {"room": ROOM_D, "similar": SIMILAR_S, "range": RANGE_T}
    scenes = load_scenes("test")
    items = [(sw, v, sid, sc) for sw in args.sweeps for v in grid[sw] for sid, sc in scenes]
    items.sort(key=lambda it: -it[1])       # largest apertures / ranges first: better load balance
    if args.summarize_only:
        frame = pd.read_csv(OUT / "positions.csv")
    else:
        frame = pd.DataFrame([r for chunk in run_parallel(work, items, args.workers) for r in chunk])
        OUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT / "positions.csv", index=False, float_format="%.9g")
    summary = {"definition": {
        "gain": "received power over focusing at the same position, known position and geometry",
        "shadow_loss": "received power over the unobstructed focused link at the same position (dB)",
        "se_los10": f"log2(1 + SNR) with the unobstructed focused link at {LOS_SNR_DB:.0f} dB SNR",
        "room_absolute": "noise fixed by the 0.2 m reference link at x = 0.095 m (SNR 10 dB there); "
                         "'total' keeps total transmit power, 'per_element' scales it with element count",
        "deep": "geometric blockage ratio >= 0.8 for that aperture; deep_ref uses the 0.2 m aperture"},
        "scenes": len(scenes), "positions_per_scene": len(X_POS),
        "literature": LITERATURE, "results": summarize(frame)}
    write_json(OUT / "summary.json", summary)
    s = pd.DataFrame(summary["results"])
    cols = ["sweep", "value", "region", "n_elements", "D_over_lambda", "z_over_D_mean", "deep_fraction_pct",
            "airy_gain_median_db", "phase_gain_median_db", "ceiling_gain_median_db", "phase_gain_p05_pct",
            "phase_shadow_loss_median_db",
            "focus_shadow_loss_median_db", "phase_over_ceiling_mean", "t_phase_ms_mean"]
    print(s[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
