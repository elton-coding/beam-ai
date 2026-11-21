"""Standalone MEEP replica of one E10 full-wave case (runs in the WSL ``meep`` env).

Only numpy and meep are imported, so this file runs outside the project env:

    python nfbeam/meep_runner.py case.npz out.npz --ppw 30

Equivalence with ``nfbeam.fullwave``: the 2-D Rayleigh–Sommerfeld field of an
aperture distribution u0 is u = -2 d/dz (G * u0), a dipole sheet.  A sheet of
x-directed magnetic current radiates E_z proportional to d/dz (G * M) (TM), and a
sheet of x-directed electric current radiates H_z proportional to d/dz (G * J)
(TE).  In z > 0 both reproduce the Rayleigh–Sommerfeld field up to one complex
constant, which the comparison calibrates on a free-space run with the same
source and resolution.  The sheet is transparent, as in the boundary-integral
model (no ground plane).  Perfect conductor: E_z = 0 (Dirichlet) for TM and
dH_z/dn = 0 (Neumann) for TE.

Geometry (case file, metres): element centres and pitch on z = 0, complex
weights w, obstacle rectangle (x_c, z_c, width, depth) or none, receiver line
z = z_rx sampled over |x| <= x_half.  MEEP units: 1 mm, c = 1.
"""

from __future__ import annotations

import argparse
import time

import meep as mp
import numpy as np


def axis_corrected_frequency(wavelength: float, resolution: float, courant: float = 0.5) -> float:
    """Frequency whose Yee-grid wavenumber along a grid axis equals 2 pi / wavelength.

    2-D Yee dispersion: (sin(w dt/2)/dt)^2 = sum_i (sin(k_i h/2)/h)^2 with c = 1.
    Propagation here is within about 30 degrees of the z axis, so matching the
    axial wavenumber removes the mean phase drift; the residual off-axis error
    is about -12 degrees over 150 wavelengths at 15 degrees for 30 cells per
    wavelength.
    """
    h = 1.0 / resolution
    dt = courant * h
    k = 2 * np.pi / wavelength
    return float(2 / dt * np.arcsin(dt / h * np.sin(k * h / 2)) / (2 * np.pi))


def run(case: dict, ppw: float, pml_wl: float = 4.0, margin_mm: float = 5.0,
        fwidth_rel: float = 0.3, tol: float = 1e-6, correct: bool = True, field_map: bool = False) -> dict:
    lam = float(case["wavelength"]) * 1e3
    res = ppw / lam
    fcen = axis_corrected_frequency(lam, res) if correct else 1.0 / lam
    z_rx = float(case["z_rx"]) * 1e3
    x_half = float(case["x_half"]) * 1e3
    bc = str(case["bc"])
    pml = pml_wl * lam
    line_source = "line_source" in case        # canonical check: one line source at (x, z)
    # Optional: the dipole sheet sits on z = sheet_z instead of the array plane
    # (Huygens re-radiation of a sampled incident field, to shorten the domain).
    sheet_z = float(case["sheet_z"]) * 1e3 if "sheet_z" in case else 0.0
    src_extent = 0.0 if line_source else len(case["weights"]) * float(case["pitch"]) * 1e3 / 2

    sx = 2 * (max(x_half, src_extent) + margin_mm + pml)
    y_lo, y_hi = sheet_z - (margin_mm + pml), z_rx + margin_mm + pml
    sy = y_hi - y_lo
    yc = (y_lo + y_hi) / 2                       # meep y = physical z - yc

    comp_src, comp_out = (mp.Hx, mp.Ez) if bc in ("tm", "free_tm") else (mp.Ex, mp.Hz)
    pulse = mp.GaussianSource(fcen, fwidth=fwidth_rel * fcen)
    if line_source:
        xs0, zs0 = (float(v) * 1e3 for v in case["line_source"])
        sources = [mp.Source(pulse, component=comp_out, center=mp.Vector3(xs0, zs0 - yc))]
        src_half = abs(xs0)
    else:
        centers = np.asarray(case["centers"]) * 1e3
        pitch = float(case["pitch"]) * 1e3
        w = np.asarray(case["weights"], dtype=complex)
        lo = centers[0] - pitch / 2

        x_mid = (centers[0] + centers[-1]) / 2

        def amp(p: mp.Vector3) -> complex:       # amp_func receives coordinates relative to the source centre
            n = int(np.floor((p.x + x_mid - lo) / pitch))
            return complex(w[n]) if 0 <= n < len(w) else 0j

        sources = [mp.Source(pulse, component=comp_src, center=mp.Vector3(x_mid, sheet_z - yc),
                             size=mp.Vector3(len(w) * pitch, 0), amp_func=amp)]
        src_half = len(w) * pitch / 2
    geometry = []
    if case["obstacle"] is not None and len(case["obstacle"]):
        x_c, z_c, width, depth = (float(v) * 1e3 for v in case["obstacle"])
        geometry = [mp.Block(size=mp.Vector3(width, depth, mp.inf), center=mp.Vector3(x_c, z_c - yc),
                             material=mp.metal)]

    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), resolution=res, sources=sources,
                        geometry=geometry, boundary_layers=[mp.PML(pml)],
                        force_complex_fields=True, eps_averaging=False)
    line = sim.add_dft_fields([comp_out], [fcen], center=mp.Vector3(0, z_rx - yc),
                              size=mp.Vector3(2 * x_half, 0))
    fmap = sim.add_dft_fields([comp_out], [fcen], center=mp.Vector3(0, (y_lo + y_hi) / 2 - yc),
                              size=mp.Vector3(sx - 2 * pml, sy - 2 * pml)) if field_map else None
    t0 = time.time()
    # The DFT looks "decayed" before the wavefront reaches the receiver line, so
    # require at least the slowest direct path (array edge to far receiver corner).
    min_t = 1.1 * float(np.hypot(x_half + src_half, z_rx - sheet_z))
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=tol, minimum_run_time=min_t))
    field = np.asarray(sim.get_dft_array(line, comp_out, 0)).ravel()
    xs, _, _, _ = sim.get_array_metadata(dft_cell=line)
    extra = {}
    if fmap is not None:
        mx, my, _, _ = sim.get_array_metadata(dft_cell=fmap)
        extra = dict(map=np.asarray(sim.get_dft_array(fmap, comp_out, 0)).astype(np.complex64),
                     map_x=np.asarray(mx) * 1e-3, map_z=(np.asarray(my) + yc) * 1e-3)
    return dict(**extra, x=np.asarray(xs).ravel() * 1e-3, field=field, seconds=time.time() - t0,
                meep_time=sim.meep_time(), ppw=ppw, resolution=res, frequency=fcen,
                cells=int(round(sx * res)) * int(round(sy * res)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--ppw", type=float, default=30.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--map", action="store_true", help="also store the 2-D DFT field over the interior")
    a = ap.parse_args()
    mp.verbosity(0)
    d = np.load(a.case, allow_pickle=True)
    case = {k: d[k] for k in d.files}
    case["obstacle"] = None if case["obstacle"].size == 0 else case["obstacle"]
    out = run(case, a.ppw, tol=a.tol, field_map=a.map)
    if mp.am_master():                   # under mpirun every rank holds the full DFT array
        out["processes"] = mp.count_processors()
        np.savez(a.out, **out)
        print(f"{a.case}: {out['cells']:,} cells, {out['seconds']:.0f} s on {out['processes']} processes")


if __name__ == "__main__":
    main()
