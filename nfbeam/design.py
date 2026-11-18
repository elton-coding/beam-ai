"""Beam families and model-based optimizers on the quadratic power form.

All designs maximize ``w^H A w`` for unit transmitted power, where ``A`` is a
receiver-power Gram matrix: either for one known position (``Channel.gram_at``)
or an expectation over a position posterior and several obstacle-geometry
hypotheses (``robust_gram``).  The second choice is what makes the online beam
robust to localisation and geometry errors.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from direct_network.core.physics import parameter_bounds
from nfbeam.channel import Channel


# ----------------------------------------------------------------- families
def airy_phase(ch: Channel, b_per_m: float, focal_m: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    fx, fz = focal_m * np.sin(theta), focal_m * np.cos(theta)
    c = ch.centers
    return ((2 * np.pi * b_per_m) ** 3 / 3) * c**3 - ch.k * (np.sqrt((c - fx) ** 2 + fz**2) - focal_m)


def airy_beam(ch: Channel, b_per_m: float, focal_m: float, theta_deg: float) -> np.ndarray:
    return ch.phase_beam(airy_phase(ch, b_per_m, focal_m, theta_deg))


def focus_beam(ch: Channel, x: float) -> np.ndarray:
    _, _, f0, th0 = parameter_bounds(ch.scene, float(x))
    return airy_beam(ch, 0.0, f0, th0)


def quantize(w: np.ndarray, bits: int | None) -> np.ndarray:
    if bits is None:
        return np.exp(1j * np.angle(w))
    step = 2 * np.pi / 2**bits
    return np.exp(1j * step * np.round(np.angle(w) / step))


def quad(A: np.ndarray, w: np.ndarray) -> float:
    return float(np.real(np.vdot(w, A @ w)))


# --------------------------------------------------------------- optimizers
def ceiling(ch: Channel, A: np.ndarray) -> tuple[float, np.ndarray]:
    """Best amplitude-and-phase aperture field (total-power constraint)."""
    vals, vecs = eigh((A + A.conj().T) / 2, ch.tx_gram())
    w = vecs[:, -1]
    return float(vals[-1]), ch.normalize(w)


def phase_only(ch: Channel, A: np.ndarray, seed: np.ndarray, bits: int | None = None,
               iters: int = 150, tol: float = 1e-7) -> np.ndarray:
    """Monotone fixed-point iteration w <- exp(j angle(A w)) for PSD ``A``.

    Each element keeps a constant amplitude (one power amplifier per element),
    so this is the relevant optimum for a phased array.
    """
    w = ch.normalize(quantize(seed, bits))
    best = quad(A, w)
    for _ in range(iters):
        cand = ch.normalize(quantize(A @ w, bits))
        value = quad(A, cand)
        if value <= best * (1 + tol):
            if value > best:
                w, best = cand, value
            break
        w, best = cand, value
    return w


def airy_search(ch: Channel, A: np.ndarray, x_ref: float, max_eval: int = 160) -> np.ndarray:
    """Bounded two-branch Powell search of (B, F, theta) on the quadratic form."""
    lo, hi, f0, th0 = parameter_bounds(ch.scene, float(x_ref))

    def value(p: np.ndarray) -> float:
        p = np.clip(p, lo, hi)
        return quad(A, airy_beam(ch, *p))

    best_w, best_v = focus_beam(ch, x_ref), quad(A, focus_beam(ch, x_ref))
    for sign in (-1, 1):
        blo, bhi = lo.copy(), hi.copy()
        if sign < 0:
            bhi[0] = -0.5
        else:
            blo[0] = 0.5
        seeds = [np.array([sign * b, f, th0 + dt]) for b in (3.5, 5.0)
                 for f in (0.19, 0.26, 0.42) for dt in (-10.0, 10.0)]
        seed = max(seeds, key=value)
        res = minimize(lambda u: -value(blo + np.clip(u, 0, 1) * (bhi - blo)),
                       (seed - blo) / (bhi - blo), method="Powell", bounds=[(0, 1)] * 3,
                       options={"maxfev": max_eval // 2, "xtol": 1e-3, "ftol": 1e-5})
        p = blo + np.clip(res.x, 0, 1) * (bhi - blo)
        if -res.fun > best_v:
            best_v, best_w = -res.fun, airy_beam(ch, *p)
    return best_w


def best_phase_only(ch: Channel, A: np.ndarray, x_ref: float, bits: int | None = None,
                    extra_seeds: list[np.ndarray] | None = None) -> np.ndarray:
    """Phase-only optimum from several seeds (eigen-beam, focus, given seeds)."""
    _, w_eig = ceiling(ch, A)
    seeds = [w_eig, focus_beam(ch, x_ref)] + list(extra_seeds or [])
    cands = [phase_only(ch, A, s, bits) for s in seeds]
    return max(cands, key=lambda w: quad(A, w))


# ------------------------------------------------------------ robust gram
def robust_gram(channels: list[Channel], weights: np.ndarray, xs: np.ndarray,
                probs: np.ndarray) -> np.ndarray:
    """E[w^H G w] over a position posterior and geometry hypotheses."""
    q = channels[0].sample_weights(xs, probs)
    A = np.zeros((channels[0].n_elements,) * 2, dtype=complex)
    for ch, pi in zip(channels, weights):
        if pi > 1e-4:
            A += pi * ch.gram(q)
    return A
