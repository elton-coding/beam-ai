"""Causal reflected-pilot position observations for the fixed 2-D scene.

This is a declared two-way *proxy*, not a full electromagnetic radar model:
the monostatic echo power is proportional to the square of the one-way
aperture power. Complex receiver noise, a constant reflectivity, and a
scan-wide false-alarm threshold turn that power into intermittent positions.
Only ``observe`` sees the simulated true position; the tracker sees its output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from direct_network.core.physics import AngularSpectrum, Scene, parameter_bounds


@dataclass(frozen=True)
class SensingConfig:
    scan_min_m: float = -0.10
    scan_max_m: float = 0.10
    scan_beams: int = 21
    grid_step_m: float = 0.001
    reflectivity: float = 1.0
    complex_noise_power: float = 3e-4
    scan_false_alarm_probability: float = 0.01
    seed: int = 20261007


@dataclass(frozen=True)
class Observation:
    measured_x_m: float | None
    detected: bool
    peak_echo_power: float
    threshold_power: float


class ReflectionScanner:
    """Offline calibrated focus-beam sweep; online observations use fresh noise."""

    def __init__(self, scene: Scene, config: SensingConfig = SensingConfig()):
        self.scene = scene
        self.config = config
        self.engine = AngularSpectrum(scene)
        self.scan_x = np.linspace(config.scan_min_m, config.scan_max_m, config.scan_beams)
        self.grid_x = np.arange(config.scan_min_m - .01,
                                config.scan_max_m + .0100001, config.grid_step_m)
        self.one_way_power = self._calibrate()
        self.echo_template = config.reflectivity * self.one_way_power**2
        n = config.scan_beams
        pfa = config.scan_false_alarm_probability
        # P(max |noise|^2 > threshold) = pfa for n independent complex samples.
        self.threshold = -config.complex_noise_power * np.log(1 - (1 - pfa) ** (1 / n))

    def _calibrate(self) -> np.ndarray:
        x_edges = self.engine.x[0] - self.engine.dx / 2 + np.arange(self.scene.n_grid + 1) * self.engine.dx
        left = self.grid_x - self.scene.rx_width_m / 2
        right = self.grid_x + self.scene.rx_width_m / 2
        power = np.empty((len(self.scan_x), len(self.grid_x)))
        for j, pointing_x in enumerate(self.scan_x):
            _, _, f0, theta0 = parameter_bounds(self.scene, float(pointing_x))
            field = self.engine.received_field(self.engine.source(0.0, f0, theta0))
            cumulative = np.r_[0.0, np.cumsum(np.abs(field)**2) * self.engine.dx]
            power[j] = np.interp(right, x_edges, cumulative) - np.interp(left, x_edges, cumulative)
        return power

    def observe(self, actual_x_m: float, rng: np.random.Generator) -> Observation:
        """Generate a scan and return only a detected location, never true x."""
        signal_power = np.array([np.interp(actual_x_m, self.grid_x, row)
                                 for row in self.echo_template])
        phase = rng.uniform(0, 2 * np.pi)
        noise_std = np.sqrt(self.config.complex_noise_power / 2)
        noise = noise_std * (rng.normal(size=len(self.scan_x))
                             + 1j * rng.normal(size=len(self.scan_x)))
        scan = np.abs(np.sqrt(signal_power) * np.exp(1j * phase) + noise) ** 2
        peak = float(np.max(scan))
        if peak <= self.threshold:
            return Observation(None, False, peak, self.threshold)
        denoised = np.maximum(scan - self.config.complex_noise_power, 0)
        # Correlate the whole scan with fixed-scene fingerprints. Reflectivity
        # is known in this simulation; no true position enters this estimate.
        error = np.mean((self.echo_template - denoised[:, None]) ** 2, axis=0)
        measured = float(self.grid_x[int(np.argmin(error))])
        return Observation(measured, True, peak, self.threshold)


class CausalPositionTracker:
    """Constant-velocity Kalman predictor with updates only on detections."""

    def __init__(self, dt_s: float = 0.02, measurement_std_m: float = 0.003,
                 acceleration_std_m_s2: float = 5.0,
                 velocity_retention_without_detection: float = 0.9):
        self.dt = dt_s
        self.measurement_var = measurement_std_m**2
        self.acceleration_var = acceleration_std_m_s2**2
        self.velocity_retention_without_detection = velocity_retention_without_detection
        self.state = np.array([0.0, 0.0], dtype=float)
        self.cov = np.diag([0.10**2, 1.0**2])
        self.initialized = False

    def step(self, measurement_x_m: float | None) -> tuple[float, float]:
        if not self.initialized and measurement_x_m is not None:
            self.state = np.array([measurement_x_m, 0.0], dtype=float)
            self.cov = np.diag([self.measurement_var, 1.0**2])
            self.initialized = True
        elif self.initialized:
            dt = self.dt
            retention = 1.0 if measurement_x_m is not None else self.velocity_retention_without_detection
            F = np.array([[1.0, dt], [0.0, retention]])
            G = np.array([0.5 * dt**2, dt])
            self.state = F @ self.state
            self.cov = F @ self.cov @ F.T + self.acceleration_var * np.outer(G, G)
            if measurement_x_m is not None:
                innovation_var = self.cov[0, 0] + self.measurement_var
                K = self.cov[:, 0] / innovation_var
                self.state += K * (measurement_x_m - self.state[0])
                self.cov -= np.outer(K, self.cov[0, :].copy())
        self.state[0] = float(np.clip(self.state[0], -0.11, 0.11))
        return float(self.state[0]), float(np.sqrt(max(self.cov[0, 0], 0)))
