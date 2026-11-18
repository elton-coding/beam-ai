"""Linear channel operator of a half-wavelength phased aperture.

The scalar angular-spectrum propagator with absorbing obstacle slices
(``direct_network.core.physics.AngularSpectrum``) is linear in the source
field.  Propagating one unit excitation per array element once gives the
matrix ``H`` (elements x receiver-plane samples).  Any beam ``w`` then yields
the receiver-plane field ``w @ H`` without another propagation, and the power
collected by an aperture centred at ``x`` is the quadratic form
``w^H G(x) w``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from direct_network.core.physics import AngularSpectrum, Scene


WINDOW_HALF_WIDTH_M = 0.16


class Channel:
    """Element-to-receiver-plane operator for one obstacle geometry."""

    def __init__(self, scene: Scene, pitch_wavelengths: float = 0.5,
                 window_half_width_m: float = WINDOW_HALF_WIDTH_M):
        self.scene = scene
        engine = AngularSpectrum(scene)
        self.engine = engine
        self.k = engine.k
        self.wavelength = engine.wavelength
        self.dx = float(engine.dx)
        pitch = pitch_wavelengths * engine.wavelength
        n = int(round(scene.tx_width_m / pitch))
        self.pitch = scene.tx_width_m / n
        self.centers = (np.arange(n) - (n - 1) / 2) * self.pitch
        self.n_elements = n

        # Fractional overlap of each grid cell with each element: every
        # element covers exactly one pitch of aperture.
        cell_lo = engine.x - self.dx / 2
        cell_hi = engine.x + self.dx / 2
        cols = np.flatnonzero((cell_hi > -scene.tx_width_m / 2) & (cell_lo < scene.tx_width_m / 2))
        lo = self.centers[:, None] - self.pitch / 2
        hi = self.centers[:, None] + self.pitch / 2
        overlap = np.clip(np.minimum(hi, cell_hi[cols]) - np.maximum(lo, cell_lo[cols]), 0, None) / self.dx
        self._basis_cols = cols
        self._basis = overlap                      # (n, len(cols)) real
        basis = np.zeros((n, scene.n_grid), dtype=complex)
        basis[:, cols] = overlap
        field = self._propagate_blocked(basis)
        keep = np.abs(engine.x) <= window_half_width_m
        self.x = engine.x[keep].copy()
        self.H = np.ascontiguousarray(field[:, keep])   # (n, M)
        # Incident field on the obstacle front face, used by the echo model.
        front = self._propagate(basis, engine.h_to_block)
        on_obstacle = np.abs(engine.x - scene.obstacle_x_m) <= scene.obstacle_width_m / 2
        self.H_obstacle = np.ascontiguousarray(front[:, on_obstacle])
        self._edges = self.x[0] - self.dx / 2 + np.arange(len(self.x) + 1) * self.dx

    # ------------------------------------------------------------------ physics
    @staticmethod
    def _propagate(field: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(field, axis=-1) * transfer, axis=-1)

    def _propagate_blocked(self, field: np.ndarray) -> np.ndarray:
        e = self.engine
        f = self._propagate(field, e.h_to_block) * e.block_mask
        for _ in range(e.n_steps):
            f = self._propagate(f, e.h_step) * e.block_mask
        return self._propagate(f, e.h_to_rx)

    # -------------------------------------------------------------------- beams
    def tx_power(self, w: np.ndarray) -> float:
        src = np.asarray(w) @ self._basis
        return float(np.sum(np.abs(src) ** 2) * self.dx)

    def normalize(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=complex)
        return w / np.sqrt(self.tx_power(w))

    def tx_gram(self) -> np.ndarray:
        """Gram matrix T with tx power w^H T w."""
        return (self._basis @ self._basis.T) * self.dx

    def phase_beam(self, phase: np.ndarray) -> np.ndarray:
        return self.normalize(np.exp(1j * np.asarray(phase)))

    # ------------------------------------------------------------------ scoring
    def field(self, w: np.ndarray) -> np.ndarray:
        return np.asarray(w) @ self.H

    def _cumulative(self, w: np.ndarray) -> np.ndarray:
        f = self.field(w)
        return np.r_[0.0, np.cumsum(np.abs(f) ** 2) * self.dx]

    def aperture_power(self, w: np.ndarray, x: np.ndarray | float,
                       n_sub: int = 1, width: float | None = None) -> np.ndarray:
        """Power in ``n_sub`` equal sub-apertures centred at each ``x``.

        Returns shape (n_sub, len(x)); a scalar ``x`` gives shape (n_sub,).
        """
        width = self.scene.rx_width_m if width is None else width
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        cum = self._cumulative(w)
        offsets = np.linspace(-width / 2, width / 2, n_sub + 1)
        c = np.interp(xs[:, None] + offsets, self._edges, cum)
        out = np.diff(c, axis=1).T
        return out[:, 0] if np.ndim(x) == 0 else out

    def power(self, w: np.ndarray, x: float) -> float:
        return float(self.aperture_power(w, x)[0])

    def obstacle_incident_power(self, w: np.ndarray) -> float:
        f = np.asarray(w) @ self.H_obstacle
        return float(np.sum(np.abs(f) ** 2) * self.dx)

    # ----------------------------------------------------------- quadratic form
    def sample_weights(self, xs: np.ndarray, probs: np.ndarray,
                       width: float | None = None) -> np.ndarray:
        """Per receiver-plane sample weight q_m = sum_i p_i * overlap(m, aperture_i)."""
        width = self.scene.rx_width_m if width is None else width
        xs = np.atleast_1d(xs)
        probs = np.atleast_1d(probs)
        lo = np.clip(self.x[None, :] - self.dx / 2, xs[:, None] - width / 2, None)
        hi = np.clip(self.x[None, :] + self.dx / 2, None, xs[:, None] + width / 2)
        frac = np.clip(hi - lo, 0, None) / self.dx
        return probs @ frac

    def gram(self, q: np.ndarray) -> np.ndarray:
        """G = sum_m q_m conj(h_m) h_m^T dx, so that power = w^H G w."""
        m = q > 1e-12
        h = self.H[:, m]
        return (h.conj() * (q[m] * self.dx)) @ h.T

    def gram_at(self, x: float) -> np.ndarray:
        return self.gram(self.sample_weights(np.array([x]), np.array([1.0])))


def perturbed(scene: Scene, dx_m: float = 0.0, width_scale: float = 1.0,
              dz_m: float = 0.0, depth_scale: float = 1.0) -> Scene:
    return replace(scene, obstacle_x_m=scene.obstacle_x_m + dx_m,
                   obstacle_width_m=scene.obstacle_width_m * width_scale,
                   obstacle_z_m=scene.obstacle_z_m + dz_m,
                   obstacle_depth_m=scene.obstacle_depth_m * depth_scale)
