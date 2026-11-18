"""E6 (gap 3): simulation-to-measurement comparison and model calibration.

THERE IS NO MEASURED DATA.  The "measurements" here come from a synthetic
digital twin (finer grid, hidden geometry offsets, per-element amplitude and
phase errors, 3-bit phase shifters).  The experiment validates the procedure
that real measurements will go through:

  1. campaign: 40 known receiver positions x 18 probe beams x 4 sub-apertures;
  2. calibrate on 20 positions, test on the other 20;
  3. compare model predictions with "measurements" (power error in dB) by
     blockage group, before and after calibration;
  4. design beams with the nominal and the calibrated model and score them on
     the twin (received power and spectral efficiency).

Real campaigns use ``nfbeam.calibration.load_measurements`` with the same
schema; see docs/实测对照与模型校准.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from direct_network.core.physics import blockage_ratio
from nfbeam import design as D
from nfbeam.calibration import (CalibratedModel, DigitalTwin, fit, predict_table, prediction_error_db,
                                save_measurements)
from nfbeam.link import SCAN_X
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, write_json
from nfbeam.tracking import X_GRID

OUT = RESULTS / "e6"
NOISE = 3e-4 / 16          # 16 pilots averaged per reading
N_POS = 40


def campaign(twin: DigitalTwin, beams: dict[int, np.ndarray], xs: np.ndarray, rng) -> pd.DataFrame:
    rows = []
    for x in xs:
        for b, ph in beams.items():
            for k, a in enumerate(twin.amplitudes(ph, float(x), rng, NOISE)):
                rows.append({"rx_x_m": float(x), "beam_id": b, "subaperture": k,
                             "amplitude": float(a), "noise_power": NOISE})
    return pd.DataFrame(rows)


def design_command(model: CalibratedModel, x: float) -> np.ndarray:
    ch = model.channel
    w = D.best_phase_only(ch, ch.gram_at(x), x)
    return model.command(np.angle(w))


def work(item) -> dict:
    sid, scene = item
    rng = np.random.default_rng(66000 + sid)
    twin = DigitalTwin(scene, rng)
    nominal = CalibratedModel(scene)
    ch = nominal.channel
    beams = {}
    for i, x in enumerate(SCAN_X):
        px = np.exp(-0.5 * ((X_GRID - x) / 0.0125) ** 2)
        beams[i] = np.angle(D.best_phase_only(ch, D.robust_gram([ch], np.array([1.0]), X_GRID, px / px.sum()), x))
        beams[9 + i] = np.angle(D.focus_beam(ch, x))
    xs = np.sort(rng.uniform(-0.10, 0.10, N_POS))
    meas = campaign(twin, beams, xs, rng)
    fit_x, test_x = xs[::2], xs[1::2]
    fit_m = meas[meas.rx_x_m.isin(fit_x)].reset_index(drop=True)
    test_m = meas[meas.rx_x_m.isin(test_x)].reset_index(drop=True)
    if sid == 64:
        save_measurements(OUT / "synthetic_campaign_scene64", meas, beams)

    # Nominal model only gets the least-squares gain; calibrated model gets everything.
    pred0 = predict_table(nominal, fit_m, beams)
    nominal.gain = float(max((pred0 @ fit_m.amplitude) / (pred0 @ pred0), 1e-3) ** 2)
    calibrated = fit(scene, fit_m, beams, log=lambda s: None)

    deep_mask = test_m.rx_x_m.map(lambda v: blockage_ratio(twin.channel.scene, v) >= scene.activation_bl).to_numpy()
    out = {"scene_id": sid,
           "hidden": {"dx_mm": 1000 * twin.params.dx_m, "width_scale": twin.params.width_scale,
                      "dz_mm": 1000 * twin.params.dz_m, "depth_scale": twin.params.depth_scale},
           "estimated": {"dx_mm": 1000 * calibrated.geo[0], "width_scale": 1 + calibrated.geo[1],
                         "dz_mm": 1000 * calibrated.geo[2], "depth_scale": 1 + calibrated.geo[3]}}
    for name, model in (("nominal", nominal), ("calibrated", calibrated)):
        pred = predict_table(model, test_m, beams)          # includes the fitted gain
        out[f"{name}_error_all"] = prediction_error_db(pred, test_m)
        out[f"{name}_error_deep"] = prediction_error_db(pred[deep_mask], test_m[deep_mask].reset_index(drop=True))
        out[f"{name}_error_other"] = prediction_error_db(pred[~deep_mask], test_m[~deep_mask].reset_index(drop=True))

    # Beam performance on the twin at held-out positions.
    ref =twin.power(np.angle(D.focus_beam(ch, 0.095)), 0.095)
    noise = ref / 10
    rows = []
    for x in test_x:
        deep = blockage_ratio(twin.channel.scene, float(x)) >= scene.activation_bl
        cmd = {"focus_nominal": np.angle(D.focus_beam(ch, x)),
               "phase_only_nominal": design_command(nominal, x),
               "phase_only_calibrated": design_command(calibrated, x)}
        w_tw = D.best_phase_only(twin.channel, twin.channel.gram_at(x), x)
        cmd["twin_oracle"] = np.angle(w_tw) - np.angle(twin.element_error)
        row = {"x": float(x), "deep": deep}
        for k, c in cmd.items():
            p = twin.power(c, x)
            row[k] = p
            row[f"se_{k}"] = float(np.log2(1 + p / noise))
            if k != "twin_oracle":
                model = calibrated if k.endswith("calibrated") else nominal
                pm = model.gain * model.channel.power(model.radiated(c), x)
                row[f"pred_se_{k}"] = float(np.log2(1 + pm / noise))
        rows.append(row)
    out["beams"] = rows
    return out


def main() -> None:
    results = run_parallel(work, load_scenes("test"))
    OUT.mkdir(parents=True, exist_ok=True)
    beams = pd.DataFrame([dict(scene_id=r["scene_id"], **b) for r in results for b in r["beams"]])
    beams.to_csv(OUT / "beam_scores.csv", index=False, float_format="%.8g")
    summary = {"synthetic": True,
               "note": "Digital-twin pseudo-measurements; no real hardware or channel data were available.",
               "per_scene": [{k: v for k, v in r.items() if k != "beams"} for r in results]}
    agg = {}
    for name in ("nominal", "calibrated"):
        for grp in ("all", "deep", "other"):
            vals = pd.DataFrame([r[f"{name}_error_{grp}"] for r in results])
            agg[f"{name}_{grp}"] = vals.mean().to_dict()
    summary["prediction_error_mean_over_scenes"] = agg
    est = pd.DataFrame([{**{f"hidden_{k}": v for k, v in r["hidden"].items()},
                         **{f"est_{k}": v for k, v in r["estimated"].items()}} for r in results])
    summary["geometry_recovery_abs_error"] = {
        "dx_mm": float((est.est_dx_mm - est.hidden_dx_mm).abs().mean()),
        "width_scale": float((est.est_width_scale - est.hidden_width_scale).abs().mean()),
        "dz_mm": float((est.est_dz_mm - est.hidden_dz_mm).abs().mean()),
        "depth_scale": float((est.est_depth_scale - est.hidden_depth_scale).abs().mean())}
    perf = {}
    for grp, g in (("deep", beams[beams.deep]), ("all", beams)):
        perf[grp] = {k: {"mean_power": float(g[k].mean()), "se_mean": float(g[f"se_{k}"].mean()),
                         "vs_focus_pct": float(100 * np.mean(g[k] / g.focus_nominal - 1))}
                     for k in ("focus_nominal", "phase_only_nominal", "phase_only_calibrated", "twin_oracle")}
        for k in ("focus_nominal", "phase_only_nominal", "phase_only_calibrated"):
            err = g[f"pred_se_{k}"] - g[f"se_{k}"]
            perf[grp][k]["predicted_minus_measured_se"] = {"bias": float(err.mean()),
                                                            "rmse": float(np.sqrt(np.mean(err**2)))}
    summary["beam_performance_on_twin"] = perf
    write_json(OUT / "summary.json", summary)
    print(pd.DataFrame(agg).T.round(2))
    print(summary["geometry_recovery_abs_error"])
    for grp in perf:
        print(grp, {k: (round(v["se_mean"], 3), round(v["vs_focus_pct"], 1)) for k, v in perf[grp].items()})


if __name__ == "__main__":
    main()
