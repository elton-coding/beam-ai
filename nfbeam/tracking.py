"""Grid Bayes filter over (static auxiliary, velocity, position) with delayed updates.

Compared with the previous threshold detector + Kalman tracker this filter
- keeps multi-modal beliefs, so a missed or ambiguous frame widens the belief
  instead of forcing a hard decision;
- turns every observation (a scan, the reference symbols of a data frame, or
  the echo of the data waveform) into a likelihood, with no detection threshold;
- carries an auxiliary static dimension (unknown target reflectivity for
  passive sensing);
- re-filters from the stored prior when a feedback report arrives late
  (out-of-sequence measurement), instead of applying it to the wrong frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve1d


X_GRID = np.round(np.arange(-0.11, 0.11 + 1e-9, 0.0005), 6)


@dataclass
class FilterConfig:
    dt_s: float = 0.02
    v_max: float = 0.6
    dv: float = 0.03
    accel_std: float = 1.5
    position_diffusion_m: float = 0.0003
    uniform_mix: float = 1e-4
    history: int = 12


class GridFilter:
    def __init__(self, cfg: FilterConfig, n_aux: int = 1, x_grid: np.ndarray = X_GRID,
                 initial_x: np.ndarray | None = None):
        self.cfg = cfg
        self.x = x_grid
        self.dx = float(x_grid[1] - x_grid[0])
        self.v = np.arange(-cfg.v_max, cfg.v_max + 1e-12, cfg.dv)
        p = np.ones((n_aux, len(self.v), len(self.x)))
        if initial_x is not None:
            p *= initial_x[None, None, :]
        vel_prior = np.exp(-0.5 * (self.v / 0.2) ** 2)
        p *= vel_prior[None, :, None]
        self.p = p / p.sum()
        shift = self.v * cfg.dt_s / self.dx
        self._k = np.floor(shift).astype(int)
        self._f = shift - self._k
        sv = cfg.accel_std * cfg.dt_s / cfg.dv
        r = int(np.ceil(4 * max(sv, 0.3)))
        kv = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sv, 1e-6)) ** 2)
        self._kv = kv / kv.sum()
        sx = cfg.position_diffusion_m / self.dx
        r = int(np.ceil(4 * max(sx, 0.3)))
        kx = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sx, 1e-6)) ** 2)
        self._kx = kx / kx.sum()

    # ---------------------------------------------------------------- dynamics
    @staticmethod
    def _shift(a: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(a)
        n = a.shape[-1]
        if k >= 0:
            out[..., k:] = a[..., :n - k] if k < n else 0
        else:
            out[..., :n + k] = a[..., -k:]
        return out

    def predict(self) -> None:
        p = self.p
        new = np.empty_like(p)
        for i in range(len(self.v)):
            row = p[:, i, :]
            new[:, i, :] = ((1 - self._f[i]) * self._shift(row, self._k[i])
                            + self._f[i] * self._shift(row, self._k[i] + 1))
        new = convolve1d(new, self._kv, axis=1, mode="constant")
        new = convolve1d(new, self._kx, axis=2, mode="constant")
        s = new.sum()
        new = new / s if s > 0 else np.ones_like(new) / new.size
        u = self.cfg.uniform_mix
        marg_v = new.sum(axis=2, keepdims=True)
        self.p = (1 - u) * new + u * marg_v / len(self.x)

    def update(self, loglik: np.ndarray) -> float:
        """Multiply by exp(loglik); loglik has shape (X,) or (aux, X). Returns log evidence."""
        ll = np.asarray(loglik, dtype=float)
        if ll.ndim == 1:
            ll = ll[None, :]
        m = ll.max()
        like = np.exp(ll - m)[:, None, :]
        post = self.p * like
        z = post.sum()
        if not np.isfinite(z) or z <= 0:
            return -np.inf
        self.p = post / z
        return float(np.log(z) + m)

    # ----------------------------------------------------------------- outputs
    def marginal_x(self) -> np.ndarray:
        return self.p.sum(axis=(0, 1))

    def marginal_aux(self) -> np.ndarray:
        return self.p.sum(axis=(1, 2))

    def mean_std(self) -> tuple[float, float]:
        px = self.marginal_x()
        mu = float(px @ self.x)
        return mu, float(np.sqrt(max(px @ (self.x - mu) ** 2, 0)))

    def predicted_marginal(self) -> np.ndarray:
        """Position belief one frame ahead, without mutating the filter."""
        saved = self.p.copy()
        self.predict()
        out = self.marginal_x()
        self.p = saved
        return out


@dataclass
class DelayedFilter:
    """Wraps GridFilter with a per-frame history for out-of-sequence updates."""
    filt: GridFilter
    aware: bool = True
    priors: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    t: int = -1

    def advance(self) -> None:
        """Move to the next frame (predict unless this is the first frame)."""
        if self.t >= 0:
            self.filt.predict()
        self.t += 1
        self.priors[self.t] = self.filt.p.copy()
        for old in [k for k in self.priors if k < self.t - self.filt.cfg.history]:
            self.priors.pop(old, None)
            self.measurements.pop(old, None)

    def add(self, frame: int, loglik: np.ndarray) -> None:
        """Apply a likelihood that describes the state at ``frame``."""
        if not self.aware or frame == self.t or frame not in self.priors:
            self.filt.update(loglik)
            self.measurements.setdefault(self.t, []).append(loglik)
            return
        self.measurements.setdefault(frame, []).append(loglik)
        self.filt.p = self.priors[frame].copy()
        for k in range(frame, self.t + 1):
            if k > frame:
                self.filt.predict()
                self.priors[k] = self.filt.p.copy()
            for ll in self.measurements.get(k, []):
                self.filt.update(ll)
