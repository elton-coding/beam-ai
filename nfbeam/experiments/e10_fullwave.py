"""E10: full-wave validation of the angular-spectrum channel model.

Every earlier experiment designs and scores beams with the same scalar
angular-spectrum operator, which models the obstacle as stacked absorbing
masks.  E10 replaces the *truth* by an exact 2-D boundary-integral solution
(``nfbeam.fullwave``) for three physically different obstacles:

  tm        metal, E out of plane   (Dirichlet)
  te        metal, H out of plane   (Neumann)
  absorber  impedance-matched surface (Robin)

Parts:
  validation  solver vs Mie series, free space vs angular spectrum, mesh
              convergence, and an independent FDTD cross-check (TM metal);
  beams       16 unseen scenes x 41 positions: beams designed on the
              angular-spectrum model (as deployed) scored in the full-wave
              channel, against beams designed on the full-wave channel;
  link        end-to-end cooperative link with the full-wave channel as truth
              while sensing templates and designs keep the angular-spectrum model.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from direct_network.core.physics import blockage_ratio
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.fullwave import BCS, FullWaveChannel, Scatterer, aperture_field, circle, line_source_field, rectangle
from nfbeam.scenes import RESULTS, load_scenes, run_parallel, scene_bootstrap, write_json

OUT = RESULTS / "e10"
PPW = 12
X_POS = np.linspace(-0.10, 0.10, 41) + 0.00025
LINK_POLICIES = ("legacy", "focus_bayes", "proposed", "proposed_warm", "oracle")
LINK_TRUTHS = ("as",) + BCS
LINK_RUNS = 5


def region(scene, x: float) -> str:
    bl = blockage_ratio(scene, float(x))
    return "deep" if bl >= scene.activation_bl else ("partial" if bl > 0.2 else "clear")


# ----------------------------------------------------------------- validation
def _mie(bc: str, k: float, a: float, r: float, phi: np.ndarray, n_max: int = 160) -> np.ndarray:
    from scipy.special import h1vp, hankel1, jv, jvp
    n = np.arange(-n_max, n_max + 1)[:, None]
    ka = k * a
    if bc == "tm":
        c = -jv(n, ka) / hankel1(n, ka)
    elif bc == "te":
        c = -jvp(n, ka) / h1vp(n, ka)
    else:
        c = -(jvp(n, ka) + 1j * jv(n, ka)) / (h1vp(n, ka) + 1j * hankel1(n, ka))
    return np.sum((1j ** n) * c * hankel1(n, k * r) * np.exp(1j * n * phi), axis=0)


def _fdtd_job(with_obstacle: bool):
    from nfbeam.fdtd import run_tmz
    return run_tmz((-18, 18), (-3, 30), (0.0, 0.0), FDTD_RECT if with_obstacle else None, FDTD_PROBES)


FDTD_RECT = (-4.5, 7.5, 12.0, 16.0)        # wavelengths: 12 x 4 metal block, offset
FDTD_PROBES = np.stack([np.linspace(-14, 14, 113), np.full(113, 27.0)], axis=1)


def validation(run_fdtd: bool) -> dict:
    out: dict = {}
    sid, sc = load_scenes()[0]
    ch = Channel(sc)
    k, lam = ch.k, ch.wavelength

    # Free space: Rayleigh–Sommerfeld vs angular spectrum, per element and per beam.
    basis = np.zeros((ch.n_elements, sc.n_grid), complex)
    basis[:, ch._basis_cols] = ch._basis
    fs_as = Channel._propagate(basis, ch.engine.h_direct)[:, np.abs(ch.engine.x) <= 0.16]
    rx = np.stack([ch.x, np.full_like(ch.x, sc.receiver_z_m)], axis=1)
    fs_rs = aperture_field(rx, ch.centers, ch.pitch, k)
    beam_err = []
    for x in X_POS:
        w = D.focus_beam(ch, x)
        q = ch.sample_weights(np.array([x]), np.array([1.0]))
        p_as = np.sum(q * np.abs(w @ fs_as) ** 2) * ch.dx
        p_rs = np.sum(q * np.abs(w @ fs_rs) ** 2) * ch.dx
        beam_err.append(10 * np.log10(p_rs / p_as))
    out["free_space"] = {"element_field_rel_err": float(np.linalg.norm(fs_rs - fs_as) / np.linalg.norm(fs_as)),
                         "focus_power_abs_db_max": float(np.max(np.abs(beam_err))),
                         "note": "difference = element discretisation of the angular-spectrum grid"}

    # Mie series, plane wave, PEC / absorbing cylinder of radius 5 wavelengths.
    a = 5 * lam
    phi = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    obs = np.stack([3 * a * np.cos(phi), 3 * a * np.sin(phi)], axis=1)
    mie = []
    for ppw in (10, 20):
        for bc in BCS:
            s = Scatterer(circle(a, int(2 * np.pi * a / (lam / ppw))), k, bc, ppw)
            q = s.solve(np.exp(1j * k * s.rhs_points()[:, 0]))
            us = s.scattered(obs, q)
            ref = _mie(bc, k, a, 3 * a, phi)
            mie.append({"bc": bc, "points_per_wavelength": ppw, "unknowns": s.bnd.n,
                        "rel_err": float(np.linalg.norm(us - ref) / np.linalg.norm(ref))})
    out["mie"] = mie

    # Mesh convergence on a real scene (focused beams over the track).
    conv = []
    beams = [D.focus_beam(ch, x) for x in X_POS]
    for bc in BCS:
        powers = {}
        for ppw in (8, 12, 16):
            t0 = time.perf_counter()
            fw = FullWaveChannel(sc, bc, ppw)
            powers[ppw] = np.array([fw.power(w, x) for w, x in zip(beams, X_POS)])
            conv.append({"bc": bc, "points_per_wavelength": ppw, "unknowns": fw.n_unknowns,
                         "build_s": time.perf_counter() - t0})
        for ppw in (8, 12):
            conv.append({"bc": bc, "points_per_wavelength": ppw, "max_abs_db_vs_16":
                         float(np.max(np.abs(10 * np.log10(powers[ppw] / powers[16]))))})
    out["convergence"] = conv

    # Independent FDTD (TM metal rectangle, line source).
    if run_fdtd:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(2) as ex:
            tot, inc = list(ex.map(_fdtd_job, [True, False]))
        k1 = 2 * np.pi
        r = FDTD_RECT
        verts = rectangle((r[0] + r[1]) / 2, (r[2] + r[3]) / 2, r[1] - r[0], r[3] - r[2])
        s = Scatterer(verts, k1, "tm", 16)
        src = np.array([0.0, 0.0])
        q = s.solve(line_source_field(s.rhs_points(), src, k1))
        ui = line_source_field(FDTD_PROBES, src, k1)
        r_mom = (ui + s.scattered(FDTD_PROBES, q)) / ui
        r_fd = tot / inc
        db = lambda v: 10 * np.log10(np.abs(v) ** 2)
        shadow = np.abs(r_mom) ** 2 < 0.5
        out["fdtd"] = {"geometry_wavelengths": {"rect_x0_x1_z0_z1": r, "source": [0, 0], "probe_z": 27},
                       "cells_per_wavelength": 30,
                       "ratio_rel_err_all": float(np.linalg.norm(r_fd - r_mom) / np.linalg.norm(r_mom)),
                       "ratio_rel_err_shadow": float(np.linalg.norm((r_fd - r_mom)[shadow]) / np.linalg.norm(r_mom[shadow])),
                       "rms_db_diff": float(np.sqrt(np.mean((db(r_fd) - db(r_mom)) ** 2))),
                       "median_abs_db_diff": float(np.median(np.abs(db(r_fd) - db(r_mom)))),
                       "seconds": time.perf_counter() - t0}
        pd.DataFrame({"x_wl": FDTD_PROBES[:, 0], "fdtd_db": db(r_fd), "bie_db": db(r_mom)}).to_csv(
            OUT / "fdtd_profile.csv", index=False, float_format="%.5g")
    return out


# ---------------------------------------------------------------------- beams
def beams_work(item) -> list[dict]:
    sid, sc = item
    ch = Channel(sc)
    fws = {bc: FullWaveChannel(sc, bc, PPW) for bc in BCS}
    rows = []
    for x in X_POS:
        A = ch.gram_at(x)
        w_focus = D.focus_beam(ch, x)
        w_airy = D.airy_search(ch, A, x)
        w_phase = D.best_phase_only(ch, A, x, extra_seeds=[w_airy])
        base = {"scene_id": sid, "rx_x_m": float(x), "region": region(sc, x),
                "blockage_ratio": blockage_ratio(sc, float(x)),
                "as_focus": D.quad(A, w_focus), "as_airy": D.quad(A, w_airy), "as_phase": D.quad(A, w_phase),
                "as_ceiling": D.ceiling(ch, A)[0]}
        for bc, fw in fws.items():
            Af = fw.gram_at(x)
            w_pf = D.best_phase_only(fw, Af, x, extra_seeds=[w_phase, w_airy])
            row = dict(base, bc=bc, fw_focus=D.quad(Af, w_focus), fw_airy=D.quad(Af, w_airy),
                       fw_phase_as=D.quad(Af, w_phase), fw_phase_fw=D.quad(Af, w_pf),
                       fw_ceiling=D.ceiling(fw, Af)[0])
            rows.append(row)
    return rows


def _db(a, b):
    return 10 * np.log10(np.asarray(a) / np.asarray(b))


def beams_summary(f: pd.DataFrame) -> list[dict]:
    out = []
    for (bc, reg), g in [((bc, r), f[(f.bc == bc) & ((f.region == r) | (r == "all"))])
                          for bc in BCS for r in ("deep", "partial", "all")]:
        err_focus = _db(g.fw_focus, g.as_focus)
        err_phase = _db(g.fw_phase_as, g.as_phase)
        gain_fw = _db(g.fw_phase_as, g.fw_focus)
        gain_as = _db(g.as_phase, g.as_focus)
        gain_airy_fw = _db(g.fw_airy, g.fw_focus)
        mism = _db(g.fw_phase_as, g.fw_phase_fw)
        by_scene = pd.Series(gain_fw, index=g.index).groupby(g.scene_id).mean().to_numpy()
        out.append({
            "bc": bc, "region": reg, "positions": len(g),
            "model_err_focus_db_median": float(np.median(err_focus)),
            "model_err_focus_absdb_p90": float(np.quantile(np.abs(err_focus), .9)),
            "model_err_phase_db_median": float(np.median(err_phase)),
            "model_err_phase_absdb_p90": float(np.quantile(np.abs(err_phase), .9)),
            "gain_phase_as_db_predicted_median": float(np.median(gain_as)),
            "gain_phase_as_db_fullwave_median": float(np.median(gain_fw)),
            "gain_phase_as_db_fullwave_mean": float(np.mean(gain_fw)),
            "gain_phase_as_db_fullwave_scene_ci95": scene_bootstrap(by_scene),
            "phase_as_better_than_focus_pct": float(100 * np.mean(gain_fw > 0)),
            "phase_as_ge_airy_pct": float(100 * np.mean(g.fw_phase_as >= g.fw_airy * 0.999)),
            "gain_airy_db_fullwave_median": float(np.median(gain_airy_fw)),
            "gain_phase_fw_db_median": float(np.median(_db(g.fw_phase_fw, g.fw_focus))),
            "mismatch_loss_db_median": float(np.median(mism)),
            "mismatch_loss_db_p10": float(np.quantile(mism, .1)),
            "phase_fw_over_ceiling_median": float(np.median(g.fw_phase_fw / g.fw_ceiling)),
        })
    return out


# ----------------------------------------------------------------------- link
def link_work(item) -> list[dict]:
    from nfbeam.link import Condition, RunRandom, SceneContext, run_coop, trajectory
    truth_name, sid, sc, policies, runs = item
    cond = Condition("matched")
    base = SceneContext(sid, sc, cond, seed=7000 + sid, need_legacy=False)
    truth = None if truth_name == "as" else FullWaveChannel(sc, truth_name, PPW)
    ctx = SceneContext(sid, sc, cond, seed=7000 + sid, need_legacy="legacy" in policies,
                       data_noise=base.data_noise, truth=truth)
    rows = []
    for run in range(runs):
        path = trajectory(run)
        rr = RunRandom.draw(len(path), 181100 + 100 * sid + run, cond.loss)
        for pol in policies:
            for r in run_coop(ctx, pol, path, rr):
                r.update(truth=truth_name, condition="matched", scene_id=sid, run=run, policy=pol)
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["validation", "beams", "link"])
    ap.add_argument("--skip-fdtd", action="store_true")
    ap.add_argument("--runs", type=int, default=LINK_RUNS)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes("test")
    summary_path = OUT / "summary.json"
    import json
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    if "validation" in args.parts:
        summary["validation"] = validation(run_fdtd=not args.skip_fdtd)
        write_json(summary_path, summary)
        print(json.dumps(summary["validation"], indent=1)[:3000])

    if "beams" in args.parts:
        t0 = time.perf_counter()
        f = pd.DataFrame([r for chunk in run_parallel(beams_work, scenes, args.workers) for r in chunk])
        f.to_csv(OUT / "beams.csv", index=False, float_format="%.7g")
        summary["beams"] = {"scenes": len(scenes), "positions_per_scene": len(X_POS),
                            "points_per_wavelength": PPW, "seconds": time.perf_counter() - t0,
                            "metrics": beams_summary(f)}
        write_json(summary_path, summary)
        m = pd.DataFrame(summary["beams"]["metrics"])
        print(m.drop(columns=["gain_phase_as_db_fullwave_scene_ci95"]).round(2).to_string(index=False))

    if "link" in args.parts:
        from nfbeam.metrics import add_columns, paired_change, region_summary
        items = [(t, sid, sc, LINK_POLICIES, args.runs) for t in LINK_TRUTHS for sid, sc in scenes]
        frames = pd.DataFrame([r for chunk in run_parallel(link_work, items, args.workers) for r in chunk])
        frames.to_csv(OUT / "link_frames.csv.gz", index=False, float_format="%.7g")
        fr = add_columns(frames)
        summary["link"] = {"runs_per_scene": args.runs,
                           "metrics": region_summary(fr, ["truth", "policy"]),
                           "vs_legacy": [r for p in LINK_POLICIES if p != "legacy"
                                         for r in paired_change(fr, ["truth"], p, "legacy")],
                           "vs_focus_bayes": [r for p in ("proposed", "proposed_warm")
                                              for r in paired_change(fr, ["truth"], p, "focus_bayes")]}
        write_json(summary_path, summary)
        m = pd.DataFrame(summary["link"]["metrics"])
        print(m[m.region == "deep"][["truth", "policy", "se_mean", "low_rate_pct", "mae_mm",
                                     "lock_lost_pct", "scan_pct"]].round(3).to_string(index=False))
        c = pd.DataFrame(summary["link"]["vs_legacy"])
        print(c[c.region == "deep"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
