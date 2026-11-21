"""E10b: independent cross-check of the boundary-integral channel with MEEP (FDTD).

The boundary-integral solver (``nfbeam.fullwave``) was verified against Mie
series and an own FDTD on a small block.  Here the full E10 geometry (0.2 m
array, real obstacle, receiver plane at 0.36-0.40 m) is recomputed with MEEP,
a widely used open-source FDTD code, for metal obstacles in both polarizations.

Parts (``--parts``):
  export   build the cases on Windows: scenes x {focus, phase-AS} beams x
           {tm, te}, plus free-space calibration runs; store the beam weights
           and the boundary-integral field on the receiver line;
  run      run ``nfbeam/meep_runner.py`` for every case in WSL (env ``meep``),
           at most 12 processes;
  compare  calibrate MEEP's source constant on the free-space run of the same
           scene and polarization, then compare the receiver-line fields and
           the receiver-aperture powers used by E10.
  canonical  line source + 12 x 4 wavelength block, both solvers refined;
  near     scene 74 focus beam on a line 5 wavelengths behind the obstacle;
  near_sheet  the same line, but MEEP is driven by the exact incident field
           re-radiated from a plane 4 wavelengths in front of the obstacle, so
           it can be refined to 60 cells per wavelength (TE agrees to 2.1%;
           MEEP TE is not converged below that on this 40 x 15 wavelength block).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.experiments.e10_fullwave import PPW, X_POS, region
from nfbeam.fullwave import FullWaveChannel, aperture_field
from nfbeam.scenes import RESULTS, load_scenes, write_json

OUT = RESULTS / "e10_meep"
CASES = OUT / "cases"
RUNS = OUT / "runs"
SCENE_IDS = (64, 65, 74, 78)
X_HALF = 0.12
BCS = ("tm", "te")
WSL = ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc"]
MICROMAMBA = "MAMBA_ROOT_PREFIX=~/micromamba ~/.local/bin/micromamba run -n meep python"
REPO = Path(__file__).resolve().parents[2]


def wsl_path(p: Path) -> str:
    p = p.resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def design_position(sc) -> float:
    deep = [x for x in X_POS if region(sc, x) == "deep"]
    x_proj = sc.obstacle_x_m * sc.receiver_z_m / sc.obstacle_z_m
    return float(min(deep, key=lambda x: abs(x - x_proj)))


def export() -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    scenes = dict(load_scenes())
    index = []
    for sid in SCENE_IDS:
        sc = scenes[sid]
        ch = Channel(sc)
        x_d = design_position(sc)
        A = ch.gram_at(x_d)
        w_focus = D.focus_beam(ch, x_d)
        w_phase = D.best_phase_only(ch, A, x_d, extra_seeds=[D.airy_search(ch, A, x_d)])
        keep = np.abs(ch.x) <= X_HALF
        rx = np.stack([ch.x[keep], np.full(keep.sum(), sc.receiver_z_m)], axis=1)
        H_free = aperture_field(rx, ch.centers, ch.pitch, ch.k)
        rect = np.array([sc.obstacle_x_m, sc.obstacle_z_m, sc.obstacle_width_m, sc.obstacle_depth_m])
        fws = {bc: FullWaveChannel(sc, bc, PPW) for bc in BCS}
        for beam, w in (("focus", w_focus), ("phase", w_phase)):
            for bc in ("free",) + BCS:
                ref = w @ H_free if bc == "free" else w @ fws[bc].H[:, keep]
                for pol in (BCS if bc == "free" else (bc,)):
                    name = f"s{sid}_{beam}_{bc if bc != 'free' else 'free_' + pol}"
                    np.savez(CASES / f"{name}.npz", wavelength=ch.wavelength, centers=ch.centers,
                             pitch=ch.pitch, weights=w, z_rx=sc.receiver_z_m, x_half=X_HALF,
                             bc=(bc if bc != "free" else f"free_{pol}"),
                             obstacle=(np.array([]) if bc == "free" else rect))
                    np.savez(CASES / f"{name}.ref.npz", x=ch.x[keep], field=ref, dx=ch.dx,
                             rx_width=sc.rx_width_m, x_design=x_d)
                    index.append(dict(name=name, scene_id=sid, beam=beam, bc=bc, pol=pol,
                                      x_design=x_d, region=region(sc, x_d)))
        print(f"scene {sid}: design x = {x_d * 1e3:.2f} mm", flush=True)
    write_json(OUT / "cases.json", index)


def run(ppw: float, workers: int, only: str | None, mpi: int = 1) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if any(o in c["name"] for o in only)]
    else:
        # Calibration needs only the focus beam in free space.
        cases = [c for c in cases if not (c["bc"] == "free" and c["beam"] != "focus")]
    # Longest first (TE, obstacle) so the last wave is not one slow straggler.
    cases.sort(key=lambda c: (c["bc"] == "free", c["pol"] != "te"))
    tag = f"ppw{ppw:g}"

    def one(c):
        return _meep(CASES / f"{c['name']}.npz", RUNS / f"{c['name']}.{tag}.npz", ppw, mpi)

    t0 = time.time()
    with ThreadPoolExecutor(max(1, workers // max(mpi, 1))) as ex:
        for msg in ex.map(one, cases):
            print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)


def _meep(src: Path, dst: Path, ppw: float, mpi: int = 1) -> str:
    if dst.exists():
        return f"{dst.name}: cached"
    py = MICROMAMBA if mpi <= 1 else MICROMAMBA.replace("-n meep python", f"-n meep_mpi mpirun -np {mpi} python")
    cmd = f"{py} {wsl_path(REPO / 'nfbeam' / 'meep_runner.py')} {wsl_path(src)} {wsl_path(dst)} --ppw {ppw:g}"
    r = subprocess.run(WSL + [cmd], capture_output=True, text=True)
    return f"{dst.name}: FAILED\n{r.stderr[-2000:]}" if r.returncode else r.stdout.strip().splitlines()[0]


# Canonical tie-breaker: line source + metal block, as the E10 own-FDTD check
# (wavelength units; block 12 x 4 wavelengths), both polarizations, both
# solvers refined until converged.
CANON = OUT / "canonical"
CANON_RECT = (-4.5, 7.5, 12.0, 16.0)
CANON_Z, CANON_X = 27.0, 14.0
CANON_MEEP_PPW = (20, 30, 40, 60)
CANON_BIE_PPW = (8, 12, 16, 24, 32)


def canonical(workers: int) -> dict:
    from nfbeam.fullwave import Scatterer, line_source_field, rectangle
    CANON.mkdir(parents=True, exist_ok=True)
    x0, x1, z0, z1 = CANON_RECT
    rect_m = np.array([(x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0]) * 1e-3
    jobs = []
    for name in ("free_tm", "free_te", "tm", "te"):
        src = CANON / f"{name}.npz"
        np.savez(src, wavelength=1e-3, line_source=np.array([0.0, 0.0]), z_rx=CANON_Z * 1e-3,
                 x_half=CANON_X * 1e-3, bc=name, obstacle=(np.array([]) if "free" in name else rect_m))
        jobs += [(src, CANON / f"{name}.ppw{p}.out.npz", p) for p in CANON_MEEP_PPW]
    jobs.sort(key=lambda j: -j[2])
    t0 = time.time()
    with ThreadPoolExecutor(min(workers, 12)) as ex:
        for msg in ex.map(lambda j: _meep(*j), jobs):
            print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    k = 2 * np.pi
    xs = np.linspace(-12, 12, 241)                 # inside the DFT line, away from its ends
    probes = np.stack([xs, np.full_like(xs, CANON_Z)], axis=1)
    verts = rectangle((x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0)
    ui = line_source_field(probes, np.zeros(2), k)
    ratios = {}
    for bc in BCS:
        for p in CANON_BIE_PPW:
            s = Scatterer(verts, k, bc, p)
            q = s.solve(line_source_field(s.rhs_points(), np.zeros(2), k))
            ratios[f"bie_{bc}_{p}"] = (ui + s.scattered(probes, q)) / ui
        for p in CANON_MEEP_PPW:
            tot, inc = (np.load(CANON / f"{n}.ppw{p}.out.npz") for n in (bc, f"free_{bc}"))
            xm = tot["x"] * 1e3
            interp = lambda f: np.interp(xs, xm, f.real) + 1j * np.interp(xs, xm, f.imag)
            ratios[f"meep_{bc}_{p}"] = interp(tot["field"]) / interp(inc["field"])
    ref = {bc: ratios[f"bie_{bc}_{CANON_BIE_PPW[-1]}"] for bc in BCS}
    rows = []
    for key, r in ratios.items():
        solver, bc, p = key.split("_")
        d = 10 * np.log10(np.abs(r) ** 2) - 10 * np.log10(np.abs(ref[bc]) ** 2)
        rows.append(dict(solver=solver, bc=bc, ppw=int(p),
                         rel_err_vs_bie_finest=float(np.linalg.norm(r - ref[bc]) / np.linalg.norm(ref[bc])),
                         median_abs_db=float(np.median(np.abs(d)))))
    # Richardson-style check: MEEP's error should fall as ppw grows if it converges to the BIE answer.
    out = dict(geometry_wavelengths=dict(rect=CANON_RECT, source=[0, 0], probe_z=CANON_Z),
               bie_reference_ppw=CANON_BIE_PPW[-1], rows=rows)
    write_json(CANON / "summary.json", out)
    np.savez_compressed(CANON / "ratios.npz", x=xs, **ratios)
    for r in rows:
        print(f"{r['solver']:4s} {r['bc']} ppw {r['ppw']:2d}: rel {r['rel_err_vs_bie_finest']:.4f}  "
              f"|dB| med {r['median_abs_db']:.3f}")
    return out


# Near-field line: 5 wavelengths behind the obstacle.  The boundary-integral field
# anywhere follows from the same boundary density, and the rest of the path to the
# receiver is free-space Rayleigh-Sommerfeld (validated separately), so agreement
# here validates the TE solution while MEEP's accumulated dispersion is ~half.
NEAR = OUT / "near"
NEAR_SCENE = 74
NEAR_BEHIND_WL = 5.0
# Short-domain variant ("near_sheet"): the exact incident field is sampled on a
# plane SHEET_FRONT_WL in front of the obstacle and re-radiated by a dipole sheet
# (Rayleigh-Sommerfeld from that plane), so MEEP only spans ~26 wavelengths in z
# and can be refined further.  The free-space run checks the re-radiation.
NEAR_SHEET = OUT / "near_sheet"
SHEET_FRONT_WL = 4.0
SHEET_HALF = 0.14
SHEET_SAMPLES_PER_WL = 40


def near(ppws: list[float], mpi: int, workers: int, sheet: bool = False, bcs=BCS) -> dict:
    from nfbeam.fullwave import Scatterer, rectangle
    out_dir = NEAR_SHEET if sheet else NEAR
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = dict(load_scenes())[NEAR_SCENE]
    ch = Channel(sc)
    w = np.load(CASES / f"s{NEAR_SCENE}_focus_te.npz")["weights"]
    z_line = sc.obstacle_z_m + sc.obstacle_depth_m / 2 + NEAR_BEHIND_WL * ch.wavelength
    xs = ch.x[np.abs(ch.x) <= X_HALF]
    line = np.stack([xs, np.full_like(xs, z_line)], axis=1)
    rect = np.array([sc.obstacle_x_m, sc.obstacle_z_m, sc.obstacle_width_m, sc.obstacle_depth_m])
    verts = rectangle(*rect)
    inc = w @ aperture_field(line, ch.centers, ch.pitch, ch.k)
    refs = {"free": inc}
    for bc in BCS:
        s = Scatterer(verts, ch.k, bc, PPW)
        q = s.solve(w @ aperture_field(s.rhs_points(), ch.centers, ch.pitch, ch.k))
        refs[bc] = inc + s.scattered(line, q)
    src_kw = dict(centers=ch.centers, pitch=ch.pitch, weights=w)
    if sheet:
        z_sheet = sc.obstacle_z_m - sc.obstacle_depth_m / 2 - SHEET_FRONT_WL * ch.wavelength
        pitch_s = ch.wavelength / SHEET_SAMPLES_PER_WL
        n_s = int(np.ceil(2 * SHEET_HALF / pitch_s))
        c_s = (np.arange(n_s) - (n_s - 1) / 2) * pitch_s
        pts = np.stack([c_s, np.full_like(c_s, z_sheet)], axis=1)
        src_kw = dict(centers=c_s, pitch=pitch_s, weights=w @ aperture_field(pts, ch.centers, ch.pitch, ch.k),
                      sheet_z=z_sheet)
    jobs = []
    for name in [n for bc in bcs for n in (bc, f"free_{bc}")]:
        src = out_dir / f"{name}.npz"
        np.savez(src, wavelength=ch.wavelength, z_rx=z_line, x_half=X_HALF, bc=name,
                 obstacle=(np.array([]) if "free" in name else rect), **src_kw)
        jobs += [(src, out_dir / f"{name}.ppw{p:g}.out.npz", p, mpi) for p in ppws]
    jobs.sort(key=lambda j: (-j[2], "free" in j[0].name))
    t0 = time.time()
    with ThreadPoolExecutor(max(1, min(workers, 12) // mpi)) as ex:
        for msg in ex.map(lambda j: _meep(*j), jobs):
            print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    rows, fields = [], {"x": xs}
    for bc in bcs:
        fields[f"bie_{bc}"] = refs[bc]
        for p in ppws:
            got = {}
            for name in (bc, f"free_{bc}"):
                m = np.load(out_dir / f"{name}.ppw{p:g}.out.npz")
                got[name] = np.interp(xs, m["x"], m["field"].real) + 1j * np.interp(xs, m["x"], m["field"].imag)
            const = np.vdot(inc, got[f"free_{bc}"]) / np.vdot(inc, inc)
            um = got[bc] / const
            fields[f"meep_{bc}_{p:g}"] = um
            rows.append(dict(bc=bc, ppw=p, free_rel_err=float(np.linalg.norm(got[f"free_{bc}"] / const - inc) / np.linalg.norm(inc)),
                             rel_err=float(np.linalg.norm(um - refs[bc]) / np.linalg.norm(refs[bc]))))
        if len(ppws) >= 2:
            a, b = sorted(ppws)[-2:]
            ua, ub = fields[f"meep_{bc}_{a:g}"], fields[f"meep_{bc}_{b:g}"]
            ext = (b ** 2 * ub - a ** 2 * ua) / (b ** 2 - a ** 2)
            fields[f"meep_{bc}_richardson"] = ext
            rows.append(dict(bc=bc, ppw=f"richardson {a:g}/{b:g}",
                             self_change=float(np.linalg.norm(ub - ua) / np.linalg.norm(ub)),
                             rel_err=float(np.linalg.norm(ext - refs[bc]) / np.linalg.norm(refs[bc]))))
    out = dict(scene_id=NEAR_SCENE, z_line_m=float(z_line), behind_wavelengths=NEAR_BEHIND_WL,
               sheet_z_m=(float(z_sheet) if sheet else 0.0), rows=rows)
    write_json(out_dir / "summary.json", out)
    np.savez_compressed(out_dir / "fields.npz", **fields)
    for r in rows:
        print(r)
    return out


def _power(x, field, dx, width, x0):
    """Receiver-aperture power, the same cumulative rule as ``Channel.aperture_power``."""
    edges = x[0] - dx / 2 + np.arange(len(x) + 1) * dx
    cum = np.r_[0.0, np.cumsum(np.abs(field) ** 2) * dx]
    return float(np.diff(np.interp([x0 - width / 2, x0 + width / 2], edges, cum))[0])


def _db(a, b):
    return float(10 * np.log10(a / b))


def compare(ppw: float) -> dict:
    tag = f"ppw{ppw:g}"
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    load = {}
    for c in cases:
        f = RUNS / f"{c['name']}.{tag}.npz"
        if f.exists():
            m, r = np.load(f), np.load(CASES / f"{c['name']}.ref.npz")
            # MEEP samples on its own grid: interpolate onto the reference grid.
            u = np.interp(r["x"], m["x"], m["field"].real) + 1j * np.interp(r["x"], m["x"], m["field"].imag)
            load[c["name"]] = (c, u, r, float(m["seconds"]))
    rows, profiles = [], {}
    for name, (c, u, r, sec) in load.items():
        free = f"s{c['scene_id']}_focus_free_{c['pol']}"
        if free not in load:
            continue
        _, uf, rf, _ = load[free]
        const = np.vdot(rf["field"], uf) / np.vdot(rf["field"], rf["field"])     # source constant
        um, ub = u / const, r["field"]
        x, dx, wd, x0 = r["x"], float(r["dx"]), float(r["rx_width"]), float(r["x_design"])
        rel = float(np.linalg.norm(um - ub) / np.linalg.norm(ub))
        p_m, p_b = _power(x, um, dx, wd, x0), _power(x, ub, dx, wd, x0)
        sig = np.abs(ub) ** 2 > 1e-3 * np.max(np.abs(ub) ** 2)
        ddb = 10 * np.log10(np.abs(um[sig]) ** 2 / np.abs(ub[sig]) ** 2)
        rows.append(dict(name=name, scene_id=c["scene_id"], beam=c["beam"], bc=c["bc"], pol=c["pol"],
                         x_design_mm=1e3 * x0, line_rel_err=rel, line_absdb_median=float(np.median(np.abs(ddb))),
                         line_absdb_p90=float(np.quantile(np.abs(ddb), .9)),
                         power_design_meep=p_m, power_design_bie=p_b, power_design_db_diff=_db(p_m, p_b),
                         meep_seconds=sec))
        profiles[name] = dict(x=x, bie=ub, meep=um)
    # Headline E10 quantity per scene: phase-AS gain over focus at the design position.
    gains = []
    for sid in SCENE_IDS:
        for bc in BCS:
            get = {r["beam"]: r for r in rows if r["scene_id"] == sid and r["bc"] == bc}
            if {"focus", "phase"} <= set(get):
                gains.append(dict(scene_id=sid, bc=bc,
                                  gain_bie_db=_db(get["phase"]["power_design_bie"], get["focus"]["power_design_bie"]),
                                  gain_meep_db=_db(get["phase"]["power_design_meep"], get["focus"]["power_design_meep"])))
    out = dict(ppw=ppw, cases=rows, gains=gains)
    write_json(OUT / f"summary_{tag}.json", out)
    np.savez_compressed(OUT / f"profiles_{tag}.npz",
                        **{f"{k}__{q}": v[q] for k, v in profiles.items() for q in ("x", "bie", "meep")})
    for r in rows:
        print(f"{r['name']:24s} rel {r['line_rel_err']:.3f}  |dB| med {r['line_absdb_median']:.3f} "
              f"p90 {r['line_absdb_p90']:.3f}  P(x_d) diff {r['power_design_db_diff']:+.3f} dB  {r['meep_seconds']:.0f} s")
    for g in gains:
        print(f"scene {g['scene_id']} {g['bc']}: gain BIE {g['gain_bie_db']:.2f} dB, MEEP {g['gain_meep_db']:.2f} dB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["export", "run", "compare"])
    ap.add_argument("--ppw", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None, help="case names (substrings) for 'run'")
    ap.add_argument("--mpi", type=int, default=1, help="MPI processes per case (env meep_mpi)")
    ap.add_argument("--near-ppw", type=float, nargs="*", default=[30.0, 45.0])
    ap.add_argument("--near-bcs", nargs="*", default=list(BCS), help="polarizations for 'near'/'near_sheet'")
    a = ap.parse_args()
    if "export" in a.parts:
        export()
    if "run" in a.parts:
        run(a.ppw, min(a.workers, 12), a.only, a.mpi)
    if "compare" in a.parts:
        compare(a.ppw)
    if "canonical" in a.parts:
        canonical(a.workers)
    if "near" in a.parts:
        near(a.near_ppw, a.mpi, a.workers, bcs=a.near_bcs)
    if "near_sheet" in a.parts:
        near(a.near_ppw, a.mpi, a.workers, sheet=True, bcs=a.near_bcs)


if __name__ == "__main__":
    main()
