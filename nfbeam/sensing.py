"""Observation models (truth side) and their likelihoods (estimator side).

Cooperative receiver: each probed beam gives four sub-aperture amplitudes,
with per-frame common gain, a coherent secondary path and complex noise —
identical to the previous pilot-reuse proxy so results stay comparable.

Non-cooperative receiver: the transmitter observes the echo of its own
transmission.  New against the previous proxy:
- range gating with a finite bandwidth (Hann-windowed range response): the
  obstacle echo and the transmit/receive leakage fall in other range bins and
  leak into the target bin only through sidelobes;
- the echo of the data waveform can be used ("same-waveform sensing"), whose
  matched-filter noise shrinks with the payload duration;
- the unknown target reflectivity is estimated as an auxiliary filter state.
The echo power is still the square of the one-way aperture power (two-way
reciprocal proxy), not a full radar channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from nfbeam.channel import Channel
from nfbeam.tracking import X_GRID


C0 = 299_792_458.0
N_SUB = 4
PILOT_S = 1e-4


@dataclass(frozen=True)
class Impairments:
    gain_std_db: float = 0.75
    multipath: float = 0.15
    multipath_offset_m: float = 0.020
    pilot_noise: float = 3e-4          # complex noise power of one 0.1 ms pilot


def complex_normal(rng: np.random.Generator, shape=()) -> np.ndarray:
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


# ------------------------------------------------------------- cooperative
def coop_template(ch: Channel, w: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(ch.aperture_power(w, X_GRID, N_SUB), 0))


def coop_observe(truth: Channel, w: np.ndarray, x: float, gain: float, phase: float,
                 noise: np.ndarray, imp: Impairments) -> np.ndarray:
    main = np.sqrt(np.maximum(truth.aperture_power(w, x, N_SUB), 0))
    extra = np.sqrt(np.maximum(truth.aperture_power(w, x + imp.multipath_offset_m, N_SUB), 0))
    s = np.sqrt(gain) * (main + imp.multipath * extra * np.exp(1j * phase)) + np.sqrt(imp.pilot_noise) * noise
    return np.abs(s)


def payload_power(truth: Channel, w: np.ndarray, x: float, gain: float, phase: float,
                  imp: Impairments) -> float:
    main = np.sqrt(np.maximum(truth.aperture_power(w, x, N_SUB), 0))
    extra = np.sqrt(np.maximum(truth.aperture_power(w, x + imp.multipath_offset_m, N_SUB), 0))
    return float(gain * np.sum(np.abs(main + imp.multipath * extra * np.exp(1j * phase)) ** 2))


def _kappa2(imp: Impairments) -> float:
    g = np.log(10) / 20 * imp.gain_std_db
    return g**2 + imp.multipath**2 / 2


def coop_loglik(templates: list[np.ndarray], y: np.ndarray, imp: Impairments,
                noise_power: float) -> np.ndarray:
    """Per-hypothesis log-likelihood over X_GRID, shape (n_hyp, X)."""
    out = []
    k2 = _kappa2(imp)
    for t in templates:                       # t: (n_beams*N_SUB, X)
        mu = np.sqrt(t**2 + noise_power)
        var = noise_power / 2 + k2 * t**2 + 1e-12
        out.append(np.sum(-(y[:, None] - mu) ** 2 / (2 * var) - 0.5 * np.log(var), axis=0))
    return np.array(out)


def combine_hypotheses(ll_h: np.ndarray, log_pi: np.ndarray, floor: float = 8.0) -> np.ndarray:
    """Mixture over geometry hypotheses plus a heavy tail against model error."""
    ll = logsumexp(ll_h + log_pi[:, None], axis=0)
    return np.logaddexp(ll, ll.max() - floor)


def update_geometry(log_pi: np.ndarray, ll_h: np.ndarray, prior_x: np.ndarray,
                    forget: float = 0.97, floor: float = np.log(1e-3)) -> np.ndarray:
    """Posterior weights of geometry hypotheses (with forgetting)."""
    ev = logsumexp(ll_h + np.log(prior_x + 1e-300)[None, :], axis=1)
    lp = forget * log_pi + ev
    lp -= logsumexp(lp)
    lp = np.maximum(lp, floor)
    return lp - logsumexp(lp)


# ----------------------------------------------------------------- passive
def hann_range_response(delta_bins: float) -> float:
    d = float(delta_bins)
    r = 0.5 * np.sinc(d) + 0.25 * np.sinc(d - 1) + 0.25 * np.sinc(d + 1)
    return abs(r) / 0.5


@dataclass(frozen=True)
class EchoConfig:
    bandwidth_hz: float = 4e9             # 0 disables range gating
    sidelobe_floor_db: float = -40.0
    obstacle_reflectivity: float = 0.1     # power reflectivity of the obstacle face
    leakage_db: float = -40.0              # TX->RX isolation, amplitude rel. unit transmit
    clutter_drift_std: float = 0.15
    background: str = "none"               # none | fresh | stale
    stale_offset: float = 0.5              # relative clutter change after the background aged


def gating_factor(cfg: EchoConfig, range_offset_m: float) -> float:
    if cfg.bandwidth_hz <= 0:
        return 1.0
    res = C0 / (2 * cfg.bandwidth_hz)
    return max(hann_range_response(range_offset_m / res), 10 ** (cfg.sidelobe_floor_db / 20))


class EchoModel:
    """Truth-side echo generator and estimator-side likelihood for one scene."""

    def __init__(self, truth: Channel, nominal: Channel, cfg: EchoConfig, imp: Impairments,
                 rng: np.random.Generator):
        self.truth, self.nominal, self.cfg, self.imp = truth, nominal, cfg, imp
        sc = truth.scene
        front = sc.obstacle_z_m - sc.obstacle_depth_m / 2
        self.g_clutter = gating_factor(cfg, sc.receiver_z_m - front)
        self.g_leak = gating_factor(cfg, sc.receiver_z_m)
        self.leak = 10 ** (cfg.leakage_db / 20) * self.g_leak
        self.drift = 0j
        self.rng = rng
        # Background estimate error: fresh = 64-snapshot calibration noise only,
        # stale = the static clutter changed by ``stale_offset`` since calibration.
        self.bg_error = {"none": None, "fresh": 0.0, "stale": cfg.stale_offset}[cfg.background]

    def _clutter(self, ch: Channel, w: np.ndarray) -> complex:
        f = np.asarray(w) @ ch.H_obstacle
        inc = np.sum(np.abs(f) ** 2) * ch.dx
        phase = np.angle(np.sum(f)) * 2
        return np.sqrt(self.cfg.obstacle_reflectivity) * inc * np.exp(1j * phase) * self.g_clutter

    def observe(self, w: np.ndarray, x: float, rho: float, integration_s: float) -> float:
        imp, rng = self.imp, self.rng
        p = self.truth.power(w, x)
        p2 = self.truth.power(w, x + imp.multipath_offset_m)
        target = np.sqrt(rho) * (p + imp.multipath * p2 * np.exp(1j * rng.uniform(-np.pi, np.pi)))
        target *= np.exp(1j * rng.uniform(-np.pi, np.pi))
        self.drift = 0.9 * self.drift + np.sqrt(1 - 0.81) * self.cfg.clutter_drift_std * complex_normal(rng)
        c = self._clutter(self.truth, w)
        residual = c * (1 + self.drift)
        if self.bg_error is not None:
            residual -= c * (1 + self.bg_error * np.exp(1j * 1.3))
        noise_power = imp.pilot_noise * PILOT_S / integration_s
        s = target + residual + self.leak * np.exp(1j * 0.7) + np.sqrt(noise_power) * complex_normal(rng)
        return float(abs(s))

    def interference_power(self, w: np.ndarray, integration_s: float) -> float:
        """Estimator's model of non-target power in the target range bin."""
        c = abs(self._clutter(self.nominal, w))
        if self.bg_error is None:
            clutter = c**2 * (1 + self.cfg.clutter_drift_std**2)
        else:
            clutter = c**2 * self.cfg.clutter_drift_std**2
        return clutter + self.leak**2 + self.imp.pilot_noise * PILOT_S / integration_s

    def loglik(self, templates_p: np.ndarray, y: float, interference: float,
               rho_grid: np.ndarray) -> np.ndarray:
        """templates_p: one-way power over X_GRID; returns (n_rho, X)."""
        k2 = _kappa2(self.imp)
        a2 = rho_grid[:, None] * templates_p[None, :] ** 2
        mu = np.sqrt(a2 + interference)
        var = interference / 2 + k2 * a2 + 1e-14
        return -(y - mu) ** 2 / (2 * var) - 0.5 * np.log(var)


def echo_template(ch: Channel, w: np.ndarray) -> np.ndarray:
    return np.maximum(ch.aperture_power(w, X_GRID)[0], 0)
