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


def quad(A, w: np.ndarray) -> float:
    if isinstance(A, PowerOperator):
        return A.quad(w)
    return float(np.real(np.vdot(w, A @ w)))


# --------------------------------------------------------------- optimizers
def ceiling(ch: Channel, A: np.ndarray) -> tuple[float, np.ndarray]:
    """Best amplitude-and-phase aperture field (total-power constraint)."""
    vals, vecs = eigh((A + A.conj().T) / 2, ch.tx_gram())
    w = vecs[:, -1]
    return float(vals[-1]), ch.normalize(w)


def phase_only(ch: Channel, A, seed: np.ndarray, bits: int | None = None,
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


REF_TX_WIDTH_M = 0.20
REF_RECEIVER_Z_M = 0.36


def airy_search(ch: Channel, A, x_ref: float, max_eval: int = 160, rescale: bool = False) -> np.ndarray:
    """Bounded two-branch Powell search of (B, F, theta) on the quadratic form.

    The (B, F) box and seeds were set for the 0.2 m aperture and a receiver
    0.36 m away.  With ``rescale`` they are scaled so that the cubic phase
    across the aperture stays the same (B ~ 1/D) and the focal range follows
    the receiver distance; ``airy_search_wide`` uses that.  The default keeps
    the original behaviour used by E1-E7.
    """
    lo, hi, f0, th0 = parameter_bounds(ch.scene, float(x_ref))
    sb = REF_TX_WIDTH_M / ch.scene.tx_width_m if rescale else 1.0
    sf = ch.scene.receiver_z_m / REF_RECEIVER_Z_M if rescale else 1.0
    lo, hi = lo * np.array([sb, sf, 1.0]), hi * np.array([sb, sf, 1.0])

    def value(p: np.ndarray) -> float:
        p = np.clip(p, lo, hi)
        return quad(A, airy_beam(ch, *p))

    best_w, best_v = focus_beam(ch, x_ref), quad(A, focus_beam(ch, x_ref))
    for sign in (-1, 1):
        blo, bhi = lo.copy(), hi.copy()
        if sign < 0:
            bhi[0] = -0.5 * sb
        else:
            blo[0] = 0.5 * sb
        seeds = [np.array([sign * b * sb, f * sf, th0 + dt]) for b in (3.5, 5.0)
                 for f in (0.19, 0.26, 0.42) for dt in (-10.0, 10.0)]
        seed = max(seeds, key=value)
        res = minimize(lambda u: -value(blo + np.clip(u, 0, 1) * (bhi - blo)),
                       (seed - blo) / (bhi - blo), method="Powell", bounds=[(0, 1)] * 3,
                       options={"maxfev": max_eval // 2, "xtol": 1e-3, "ftol": 1e-5})
        p = blo + np.clip(res.x, 0, 1) * (bhi - blo)
        if -res.fun > best_v:
            best_v, best_w = -res.fun, airy_beam(ch, *p)
    return best_w


def airy_search_wide(ch: Channel, A: np.ndarray, x_ref: float, max_eval: int = 160) -> np.ndarray:
    """Airy search that holds for any aperture and range (used by E8).

    ``airy_search`` seeds and bounds were tuned for the 0.2 m / 0.36 m
    geometry and miss the optimum far from it (e.g. 8x the range).  This one
    scores a vectorised log-spaced grid over |B| (0.05-20 x 0.2 m / D, both
    signs), F in 0.3-1.9 x the focal distance and theta in +-20 deg, then
    refines the best grid point of each sign with Powell in (log|B|, F, theta);
    the rescaled ``airy_search`` result is kept as a further candidate.
    """
    _, _, f0, th0 = parameter_bounds(ch.scene, float(x_ref))
    sb = REF_TX_WIDTH_M / ch.scene.tx_width_m
    T = ch.tx_gram()
    c = ch.centers

    def beams(b: np.ndarray, f: np.ndarray, th: np.ndarray) -> np.ndarray:
        th = np.deg2rad(th)
        fx, fz = (f * np.sin(th))[:, None], (f * np.cos(th))[:, None]
        ph = ((2 * np.pi * b[:, None]) ** 3 / 3) * c**3 - ch.k * (np.sqrt((c - fx) ** 2 + fz**2) - f[:, None])
        w = np.exp(1j * ph)
        return w / np.sqrt(np.real(np.sum(w.conj() * (w @ T.T), axis=1)))[:, None]

    def values(w: np.ndarray) -> np.ndarray:
        return np.real(np.sum(w.conj() * (w @ A.T), axis=1))

    log_lo, log_hi = -1.3, 1.3
    mags = sb * np.logspace(log_lo, log_hi, 12)
    grid = np.array([(sg * m, f0 * fr, th0 + dt) for sg in (-1, 1) for m in mags
                     for fr in (0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 1.9) for dt in (-20, -10, 0, 10, 20)])
    v = values(beams(grid[:, 0], grid[:, 1], grid[:, 2]))
    w_focus = focus_beam(ch, x_ref)
    best_w, best_v = w_focus, quad(A, w_focus)
    lo = np.array([log_lo - 0.2, 0.25, -45.0])
    hi = np.array([log_hi + 0.2, 2.0, 45.0])
    for sg in (-1, 1):
        idx = np.flatnonzero(np.sign(grid[:, 0]) == sg)
        g = grid[idx[np.argmax(v[idx])]]
        u0 = np.array([np.log10(abs(g[0]) / sb), g[1] / f0, g[2] - th0])

        def to_p(u: np.ndarray) -> np.ndarray:
            u = lo + np.clip(u, 0, 1) * (hi - lo)
            return np.array([sg * sb * 10 ** u[0], f0 * u[1], th0 + u[2]])

        def neg(u: np.ndarray) -> float:
            p = to_p(u)
            return -float(values(beams(p[:1], p[1:2], p[2:]))[0])

        res = minimize(neg, (u0 - lo) / (hi - lo), method="Powell", bounds=[(0, 1)] * 3,
                       options={"maxfev": max_eval // 2, "xtol": 1e-3, "ftol": 1e-5})
        if -res.fun > best_v:
            best_v, best_w = -res.fun, airy_beam(ch, *to_p(res.x))
    w_local = airy_search(ch, A, x_ref, max_eval, rescale=True)   # tuned local search as a candidate
    return w_local if quad(A, w_local) > best_v else best_w


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


# ------------------------------------------------------- real-time operator
class PowerOperator:
    """Matrix-free form of ``robust_gram``: ``A w`` without building ``A``.

    ``A = sum_h pi_h conj(H_h) diag(q dx) H_h^T`` restricted to the receiver
    samples with non-zero weight.  One product costs two thin matrix-vector
    products instead of the ``n x n x M`` Gram construction, which is what
    lets a warm-started design fit inside a frame.
    """

    def __init__(self, channels: list[Channel], weights: np.ndarray, xs: np.ndarray,
                 probs: np.ndarray, min_weight: float = 1e-4, max_columns: int | None = None):
        ch0 = channels[0]
        q = ch0.sample_weights(xs, probs)
        idx = np.flatnonzero(q > 1e-12)
        keep = [(float(pi), ch) for ch, pi in zip(channels, weights) if pi > min_weight]
        # A wide position belief spreads the weight over thousands of receiver
        # samples; with ``max_columns`` the samples are decimated (weights
        # scaled by the stride) so the cost stays bounded.  The quadrature is
        # coarser but the beam it produces is scored on the exact form.
        n_cols = (idx[-1] - idx[0] + 1) * len(keep)
        stride = 1 if max_columns is None else max(1, int(np.ceil(n_cols / max_columns)))
        sl = slice(int(idx[0]), int(idx[-1]) + 1, stride)
        qdx = q[sl] * ch0.dx * stride
        self.H = np.concatenate([ch.H[:, sl] for _, ch in keep], axis=1)
        self.s = np.concatenate([pi * qdx for pi, _ in keep])

    def __matmul__(self, w: np.ndarray) -> np.ndarray:
        return self.H.conj() @ (self.s * (w @ self.H))

    def quad(self, w: np.ndarray) -> float:
        return float(np.sum(self.s * np.abs(w @ self.H) ** 2))


def warm_phase_only(ch: Channel, op: PowerOperator, seeds: list[np.ndarray],
                    iters: int = 5, bits: int | None = None) -> np.ndarray:
    """Real-time design: a few fixed-point steps from the best of the given seeds.

    Used frame to frame with the previous beam (and optionally a table entry)
    as seeds; the receiver moves a fraction of a millimetre per frame, so the
    previous optimum is already close and a handful of monotone steps track it.
    """
    seed = max(seeds, key=lambda w: op.quad(w))
    return phase_only(ch, op, seed, bits, iters=iters)
