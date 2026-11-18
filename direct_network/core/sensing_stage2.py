"""Two controlled position-observation hypotheses for the fixed Airy scene.

The passive channel adds a coherent, calibratable obstacle echo and residual
drift to a square-law target-return proxy. The active channel uses a reciprocal
one-way pilot proxy. Neither is a full radar or radio hardware model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from direct_network.core.physics import AngularSpectrum, Scene, parameter_bounds


@dataclass(frozen=True)
class Condition:
    name: str
    noise_power: float
    passive_reflectivity_db_low: float
    passive_reflectivity_db_high: float
    obstacle_reflectivity: float
    clutter_drift_std: float
    multipath_amplitude: float


CONDITIONS = (
    Condition("clean", 3e-4, 0.0, 0.0, 0.0, 0.0, 0.0),
    Condition("reflective", 3e-4, -6.0, 0.0, 0.1, 0.15, 0.15),
    Condition("reflective_high_noise", 6e-4, -6.0, 0.0, 0.1, 0.15, 0.15),
)

ABLATIONS = (
    Condition("weak_target_only", 3e-4, -6.0, 0.0, 0.0, 0.0, 0.0),
    Condition("obstacle_echo_only", 3e-4, 0.0, 0.0, 0.1, 0.15, 0.0),
    Condition("multipath_only", 3e-4, 0.0, 0.0, 0.0, 0.0, 0.15),
)


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


class PilotLibrary:
    """Fixed physical one-way templates for the 21 candidate directions."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.engine = AngularSpectrum(scene)
        self.pointing_x = np.linspace(-0.10, 0.10, 21)
        self.grid_x = np.linspace(-0.11, 0.11, 221)
        self.one_way = np.empty((21, len(self.grid_x)))
        self.obstacle_incident = np.empty(21)
        edges = self.engine.x[0] - self.engine.dx / 2 + np.arange(scene.n_grid + 1) * self.engine.dx
        rx_left = self.grid_x - scene.rx_width_m / 2
        rx_right = self.grid_x + scene.rx_width_m / 2
        mask = np.abs(self.engine.x - scene.obstacle_x_m) <= scene.obstacle_width_m / 2
        for j, x in enumerate(self.pointing_x):
            _, _, focal, theta = parameter_bounds(scene, float(x))
            source = self.engine.source(0.0, focal, theta)
            front = self.engine._propagate(source, self.engine.h_to_block)
            self.obstacle_incident[j] = float(np.sum(np.abs(front[mask]) ** 2) * self.engine.dx)
            received = self.engine.received_field(source)
            cumulative = np.r_[0.0, np.cumsum(np.abs(received) ** 2) * self.engine.dx]
            self.one_way[j] = np.interp(rx_right, edges, cumulative) - np.interp(rx_left, edges, cumulative)

    @staticmethod
    def indices(n_pilots: int) -> np.ndarray:
        if n_pilots not in (5, 9, 21):
            raise ValueError("Supported sweep sizes are 5, 9 and 21")
        return np.rint(np.linspace(0, 20, n_pilots)).astype(int)


class PositionInstrument:
    """Observe one moving target with a fixed subset of the pilot library."""

    def __init__(self, library: PilotLibrary, mode: str, condition: Condition,
                 n_pilots: int, calibration_seed: int, *,
                 background_snapshots: int = 64,
                 truth_library: PilotLibrary | None = None):
        if mode not in ("passive", "active"):
            raise ValueError(mode)
        self.library = library
        self.mode = mode
        self.condition = condition
        self.background_snapshots = background_snapshots
        self.truth_library = truth_library or library
        self.indices = library.indices(n_pilots)
        self.grid_x = library.grid_x
        self.noise_power = condition.noise_power
        self.template = library.one_way[self.indices] ** (2 if mode == "passive" else 1)
        self.template_norm = np.sum(self.template**2, axis=0) + 1e-20
        self.direct_amplitude = np.sqrt(self.template)
        self.true_direct_amplitude = np.sqrt(
            self.truth_library.one_way[self.indices] ** (2 if mode == "passive" else 1))
        phase = (2 * self.truth_library.engine.k
                 * np.hypot(self.truth_library.pointing_x[self.indices],
                            self.truth_library.scene.obstacle_z_m))
        self.clutter = (np.sqrt(condition.obstacle_reflectivity)
                        * self.truth_library.obstacle_incident[self.indices]
                        * np.exp(1j * phase)) if mode == "passive" else np.zeros(n_pilots, dtype=complex)
        rng = np.random.default_rng(calibration_seed)
        # Passive sensing uses 64 no-target snapshots to estimate a static
        # background. Active pilot correlation has no obstacle-only return.
        if mode == "passive" and background_snapshots > 0:
            calibration_noise = np.sqrt(self.noise_power) * complex_normal(rng, (background_snapshots, n_pilots))
            self.background_estimate = self.clutter + np.mean(calibration_noise, axis=0)
        else:
            self.background_estimate = np.zeros(n_pilots, dtype=complex)
        self.threshold = self._null_threshold(rng, 4000)
        self.threshold_cache = {tuple(range(n_pilots)): self.threshold}
        self.last_threshold = self.threshold
        validation = self._null_peaks(rng, 10000)
        self.no_target_false_alarm = float(np.mean(validation > self.threshold))
        self.drift = 0j

    def _null_peaks(self, rng: np.random.Generator, n: int,
                    subset: np.ndarray | None = None) -> np.ndarray:
        subset = np.arange(len(self.indices)) if subset is None else subset
        drift = self.condition.clutter_drift_std * complex_normal(rng, (n, 1))
        noise = np.sqrt(self.noise_power) * complex_normal(rng, (n, len(subset)))
        residual = self.clutter[subset][None, :] * (1 + drift) + noise - self.background_estimate[subset][None, :]
        return np.max(np.abs(residual) ** 2, axis=1)

    def _null_threshold(self, rng: np.random.Generator, n: int) -> float:
        return float(np.quantile(self._null_peaks(rng, n), 0.99))

    def reset(self) -> None:
        self.drift = 0j

    def threshold_for(self, subset: np.ndarray) -> float:
        key = tuple(int(v) for v in subset)
        if key not in self.threshold_cache:
            seed = 20261104 + sum((i + 1) * (v + 1) for i, v in enumerate(key))
            null = self._null_peaks(np.random.default_rng(seed), 4000, subset)
            self.threshold_cache[key] = float(np.quantile(null, .99))
        return self.threshold_cache[key]

    def observe(self, actual_x: float, rng: np.random.Generator,
                target_reflectivity: float, *,
                selected_local_indices: np.ndarray | None = None) -> tuple[float | None, float]:
        subset = (np.arange(len(self.indices)) if selected_local_indices is None
                  else np.asarray(selected_local_indices, dtype=int))
        if len(subset) == 0 or np.any(subset < 0) or np.any(subset >= len(self.indices)):
            raise ValueError("Invalid pilot subset")
        direct = np.array([np.interp(actual_x, self.grid_x, row)
                           for row in self.true_direct_amplitude[subset]])
        shifted_x = float(np.clip(actual_x + 0.020, self.grid_x[0], self.grid_x[-1]))
        secondary = np.array([np.interp(shifted_x, self.grid_x, row)
                              for row in self.true_direct_amplitude[subset]])
        target = (np.sqrt(target_reflectivity)
                  * (direct + self.condition.multipath_amplitude * secondary * np.exp(1j * 1.7))
                  * np.exp(1j * rng.uniform(-np.pi, np.pi)))
        self.drift = (0.9 * self.drift + np.sqrt(1 - 0.9**2)
                      * self.condition.clutter_drift_std * complex_normal(rng, ()))
        noise = np.sqrt(self.noise_power) * complex_normal(rng, (len(subset),))
        residual = target + self.clutter[subset] * (1 + self.drift) + noise - self.background_estimate[subset]
        measured_power = np.abs(residual) ** 2
        peak = float(measured_power.max())
        self.last_threshold = self.threshold_for(subset)
        if peak <= self.last_threshold:
            return None, peak
        # The estimator knows neither target strength nor the extra path.
        data = np.maximum(measured_power - self.noise_power, 0)
        template = self.template[subset]
        template_norm = np.sum(template**2, axis=0) + 1e-20
        projected = template.T @ data
        scale = np.clip(projected / template_norm, 0.1, 3.0)
        error = np.sum(data**2) - 2 * scale * projected + scale**2 * template_norm
        return float(self.grid_x[int(np.argmin(error))]), peak
