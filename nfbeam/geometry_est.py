"""Obstacle geometry estimation from monostatic echoes of the transmit array.

Stage 1 (this module): the transmitter scans near-field focused beams over a
grid of points around the expected obstacle plane and records the coherent
echo amplitude in the obstacle range bin (range gating isolates it from the
receiver and the background).  A physical-optics model of the front face

    y_b(theta) = Gamma * int_{x_c - w/2}^{x_c + w/2} (w_b^T h(x, z_f))^2 dx,

with h the free-space element field (Rayleigh–Sommerfeld) and Gamma an unknown
reflection coefficient, is fitted to the measured amplitudes |y_b|.  The
parameters theta = (x_c, w, z_f) are estimated by a coarse grid search and a
local refinement; their covariance is the Gauss–Newton (Laplace) estimate,
which at the true parameters equals the Cramér–Rao bound of the model.

The depth (thickness) of the obstacle is not observable from front-face echoes
and is left to its prior.  Stage 2 refines the geometry online from the
receiver feedback through the existing hypothesis posterior
(``sensing.update_geometry``), with hypotheses spaced by the stage-1 spread.

Derivation of the model scale: under physical optics a flat metal face carries
du/dn ~ 2 du_inc/dn, its scattered field is u_s ~ 2ik int G u_inc, and the
array's receive coupling int a_w G dx equals u_inc / (2ik) by reciprocity, so
y ~ -int u_inc^2 dx for metal (Gamma = -1) in the units of one-way power.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from direct_network.core.physics import Scene
from nfbeam.channel import Channel
from nfbeam.fullwave import aperture_field

SCAN_X = np.linspace(-0.08, 0.08, 41)
SCAN_Z = np.array([0.15, 0.17, 0.19, 0.21, 0.23])
DWELL_S = 1e-4                        # one beam = one 0.1 ms pilot slot
PRIOR = dict(x_c=(-0.04, 0.04), width=(0.05, 0.14), z_f=(0.14, 0.26))
PRIOR_DEPTH_M = 0.030
FACE_DX = 2.5e-4
MIN_PEAK_SNR_DB = 15.0
MAX_REL_RESIDUAL = 0.10


def in_prior(x_c: float, width: float, z_f: float) -> bool:
    return (PRIOR["x_c"][0] <= x_c <= PRIOR["x_c"][1] and PRIOR["width"][0] <= width <= PRIOR["width"][1]
            and PRIOR["z_f"][0] <= z_f <= PRIOR["z_f"][1])


def point_focus(ch: Channel, x: float, z: float) -> np.ndarray:
    d = np.hypot(ch.centers - x, z)
    return ch.normalize(np.exp(-1j * ch.k * d))


def scan_beams(ch: Channel) -> np.ndarray:
    return np.array([point_focus(ch, x, z) for z in SCAN_Z for x in SCAN_X])


@dataclass
class Estimate:
    x_c: float
    width: float
    z_f: float
    gamma: float
    cov: np.ndarray            # 3x3 for (x_c, width, z_f)
    cost: float
    peak_snr_db: float = float("nan")      # fitted echo peak power / noise
    rel_residual: float = float("nan")     # unexplained share of the echo energy
    valid: bool = True                     # passed the acceptance test

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov), 0))

    def scene(self, template: Scene, depth: float = PRIOR_DEPTH_M) -> Scene:
        return replace(template, obstacle_x_m=self.x_c, obstacle_width_m=self.width,
                       obstacle_z_m=self.z_f + depth / 2, obstacle_depth_m=depth)


class EchoEstimator:
    """Physical-optics echo model and its fit for one array and scan."""

    def __init__(self, ch: Channel, beams: np.ndarray, noise_power: float):
        self.ch, self.W = ch, beams
        self.noise_power = noise_power
        self.sigma2 = noise_power / 2            # amplitude noise variance at high SNR
        self.k = ch.k
        # Coarse tables on a z grid for the grid search.
        self.z_grid = np.arange(PRIOR["z_f"][0], PRIOR["z_f"][1] + 1e-9, 0.004)
        self.x_grid = np.arange(-0.12, 0.12 + 1e-9, 5e-4)
        self._cum = []
        for z in self.z_grid:
            pts = np.stack([self.x_grid, np.full_like(self.x_grid, z)], axis=1)
            f = self.W @ aperture_field(pts, ch.centers, ch.pitch, self.k)      # (n_beams, n_x)
            g = f ** 2
            cum = np.concatenate([np.zeros((len(self.W), 1)),
                                  np.cumsum(0.5 * (g[:, 1:] + g[:, :-1]), axis=1) * 5e-4], axis=1)
            self._cum.append(cum)

    # ------------------------------------------------------------- model
    def model(self, x_c: float, width: float, z_f: float) -> np.ndarray:
        """Complex face integral per beam (without Gamma), midpoint rule at lambda/10."""
        n = max(8, int(np.ceil(width / FACE_DX)))
        h = width / n
        xs = x_c - width / 2 + h * (np.arange(n) + 0.5)
        pts = np.stack([xs, np.full(n, z_f)], axis=1)
        f = self.W @ aperture_field(pts, self.ch.centers, self.ch.pitch, self.k, nq=2)
        return (f ** 2).sum(axis=1) * h

    @staticmethod
    def _fit_gain(amp: np.ndarray, m: np.ndarray) -> tuple[float, float]:
        mm = np.abs(m)
        g = max(float(amp @ mm / max(mm @ mm, 1e-300)), 0.0)
        return g, float(np.sum((amp - g * mm) ** 2))

    def _coarse(self, amp: np.ndarray):
        xc = np.arange(PRIOR["x_c"][0], PRIOR["x_c"][1] + 1e-9, 0.002)
        wd = np.arange(PRIOR["width"][0], PRIOR["width"][1] + 1e-9, 0.002)
        XC, WD = np.meshgrid(xc, wd, indexing="ij")
        lo = (XC - WD / 2).ravel()
        hi = (XC + WD / 2).ravel()
        best = (np.inf, None)
        for zi, cum in enumerate(self._cum):
            m = np.empty((len(self.W), len(lo)), complex)
            for b in range(len(self.W)):
                m[b] = np.interp(hi, self.x_grid, cum[b]) - np.interp(lo, self.x_grid, cum[b])
            mm = np.abs(m)
            g = np.maximum((amp @ mm) / np.maximum(np.sum(mm ** 2, 0), 1e-300), 0)
            cost = np.sum((amp[:, None] - g[None, :] * mm) ** 2, axis=0)
            i = int(np.argmin(cost))
            if cost[i] < best[0]:
                best = (cost[i], (lo[i] + hi[i]) / 2, hi[i] - lo[i], self.z_grid[zi])
        return best

    def fit(self, amp: np.ndarray) -> Estimate:
        _, x0, w0, z0 = self._coarse(amp)
        scale = np.array([1e-3, 1e-3, 1e-3])

        def cost(p):
            x_c, width, z_f = p * scale
            if not in_prior(x_c, width, z_f):
                return 1e9
            return self._fit_gain(amp, self.model(x_c, width, z_f))[1] / self.sigma2

        res = minimize(cost, np.array([x0, w0, z0]) / scale, method="Nelder-Mead",
                       options=dict(xatol=0.02, fatol=1e-3, maxiter=400))
        x_c, width, z_f = res.x * scale
        m = self.model(x_c, width, z_f)
        g, resid = self._fit_gain(amp, m)
        cov = self.covariance(np.array([x_c, width, z_f]), g)
        peak = float(10 * np.log10(max(g ** 2 * np.max(np.abs(m) ** 2), 1e-300) / self.noise_power))
        rel = float(resid / max(np.sum(amp ** 2), 1e-300))
        # Acceptance: a detectable specular echo that the face model explains.
        # Matched absorbers (no specular echo) and low-SNR threshold failures
        # are rejected; the link then falls back to the prior.
        valid = bool(peak >= MIN_PEAK_SNR_DB and rel <= MAX_REL_RESIDUAL)
        return Estimate(float(x_c), float(width), float(z_f), g, cov, float(res.fun), peak, rel, valid)

    def covariance(self, theta: np.ndarray, gain: float, step: float = 2e-4) -> np.ndarray:
        """Gauss–Newton covariance of (x_c, width, z_f) with the gain as a nuisance.

        At the true parameters this is the Cramér–Rao bound of the model.
        """
        cols = []
        base = gain * np.abs(self.model(*theta))
        for i in range(3):
            d = np.zeros(3)
            d[i] = step
            plus = gain * np.abs(self.model(*(theta + d)))
            minus = gain * np.abs(self.model(*(theta - d)))
            cols.append((plus - minus) / (2 * step))
        cols.append(base / max(gain, 1e-300))           # derivative w.r.t. the gain
        J = np.stack(cols, axis=1)
        info = J.T @ J / self.sigma2
        return np.linalg.pinv(info)[:3, :3]


def measure(H_echo: np.ndarray, beams: np.ndarray, pitch: float, noise_power: float,
            rng: np.random.Generator) -> np.ndarray:
    """Range-gated echo amplitudes |pitch * w^T H_echo w + n| for each scan beam."""
    y = pitch * np.einsum("bi,ij,bj->b", beams, H_echo, beams)
    n = np.sqrt(noise_power / 2) * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
    return np.abs(y + n)


def hypothesis_scenes(nominal: Scene, est: Estimate | None,
                      min_shift: float = 0.003, min_scale: float = 0.05) -> list[Scene]:
    """3 x 3 (shift, width) hypotheses around the nominal, spaced by the stage-1 spread."""
    from nfbeam.channel import perturbed
    if est is None:
        sx, sw = min_shift, min_scale
    else:
        sx = max(min_shift, float(est.std[0]))
        sw = max(min_scale, float(est.std[1]) / est.width)
    return [perturbed(nominal, dx_m=dx, width_scale=ws)
            for dx in (-sx, 0.0, sx) for ws in (1 - sw, 1.0, 1 + sw)]
