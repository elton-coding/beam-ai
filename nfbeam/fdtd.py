"""Minimal 2-D TMz FDTD (Yee grid, split-field PML) for cross-checking.

Used only to cross-check the boundary-integral solver on a perfectly
conducting rectangle with a time-domain method that shares no code or
formulation with it.  Units: wavelength = 1, c = 1.  Returns the complex
phasor of Ez (exp(-i w t) convention) on requested points.
"""

from __future__ import annotations

import numpy as np


def run_tmz(extent_x: tuple[float, float], extent_z: tuple[float, float],
            source: tuple[float, float], pec_rect: tuple[float, float, float, float] | None,
            probes: np.ndarray, cells_per_wavelength: int = 30, pml_cells: int = 40,
            periods: int = 80, ramp_periods: int = 10, dft_periods: int = 6) -> np.ndarray:
    """Phasor of Ez at ``probes`` (n, 2) for a CW soft line source.

    ``pec_rect`` = (x0, x1, z0, z1): Ez is forced to zero on grid nodes inside.
    """
    h = 1.0 / cells_per_wavelength
    dt = 0.5 * h
    x0, x1 = extent_x[0] - pml_cells * h, extent_x[1] + pml_cells * h
    z0, z1 = extent_z[0] - pml_cells * h, extent_z[1] + pml_cells * h
    nx = int(round((x1 - x0) / h)) + 1
    nz = int(round((z1 - z0) / h)) + 1
    xs = x0 + h * np.arange(nx)
    zs = z0 + h * np.arange(nz)

    def profile(n: int, total: int) -> np.ndarray:
        idx = np.arange(total, dtype=float)
        d = np.maximum(np.maximum(pml_cells - idx, idx - (total - 1 - pml_cells)), 0) / pml_cells
        sigma_max = 0.8 * (3 + 1) / h / 1.0     # polynomial grading m = 3, eta = 1
        return sigma_max * d ** 3

    sx = profile(pml_cells, nx)[:, None] * np.ones((1, nz))
    sz = np.ones((nx, 1)) * profile(pml_cells, nz)[None, :]
    sx_h = 0.5 * (sx[:-1, :] + sx[1:, :])        # at Hz-staggered x positions (Hy)
    sz_h = 0.5 * (sz[:, :-1] + sz[:, 1:])        # at staggered z positions (Hx)

    def coeffs(sig):
        a = np.exp(-sig * dt)
        b = np.where(sig > 0, (1 - a) / np.maximum(sig, 1e-30), dt) / h
        return a, b

    ax_e, bx_e = coeffs(sx)
    az_e, bz_e = coeffs(sz)
    ax_h, bx_h = coeffs(sx_h)
    az_h, bz_h = coeffs(sz_h)

    ezx = np.zeros((nx, nz))
    ezz = np.zeros((nx, nz))
    hx = np.zeros((nx, nz - 1))     # H along x, staggered in z
    hz = np.zeros((nx - 1, nz))     # H along z, staggered in x

    mask = np.ones((nx, nz))
    if pec_rect is not None:
        px0, px1, pz0, pz1 = pec_rect
        mask[np.ix_((xs >= px0 - 1e-9) & (xs <= px1 + 1e-9), (zs >= pz0 - 1e-9) & (zs <= pz1 + 1e-9))] = 0
    si = int(round((source[0] - x0) / h))
    sk = int(round((source[1] - z0) / h))
    pi_ = np.rint((probes[:, 0] - x0) / h).astype(int)
    pk_ = np.rint((probes[:, 1] - z0) / h).astype(int)

    omega = 2 * np.pi
    steps = int(round(periods / dt))
    ramp = ramp_periods / 1.0
    dft_start = steps - int(round(dft_periods / dt))
    acc = np.zeros(len(probes), dtype=complex)
    for n in range(steps):
        t = n * dt
        # H updates (TMz: Hx = -dEz/dz, Hz = +dEz/dx in the x-z plane)
        ez = ezx + ezz
        hx = az_h * hx - bz_h * (ez[:, 1:] - ez[:, :-1])
        hz = ax_h * hz + bx_h * (ez[1:, :] - ez[:-1, :])
        # E updates
        ezx[1:-1, :] = ax_e[1:-1, :] * ezx[1:-1, :] + bx_e[1:-1, :] * (hz[1:, :] - hz[:-1, :])
        ezz[:, 1:-1] = az_e[:, 1:-1] * ezz[:, 1:-1] - bz_e[:, 1:-1] * (hx[:, 1:] - hx[:, :-1])
        env = 0.5 * (1 - np.cos(np.pi * min(t / ramp, 1.0)))
        src = env * np.sin(omega * t) * dt
        ezx[si, sk] += 0.5 * src
        ezz[si, sk] += 0.5 * src
        ezx *= mask
        ezz *= mask
        if n >= dft_start:
            acc += (ezx[pi_, pk_] + ezz[pi_, pk_]) * np.exp(1j * omega * t)
    return acc * 2 / (steps - dft_start)
