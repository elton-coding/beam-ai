"""Shared scene geometry and scalar angular-spectrum scoring model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


C0 = 299_792_458.0


@dataclass(frozen=True)
class Scene:
    frequency_hz: float = 120e9
    tx_width_m: float = 0.20
    rx_width_m: float = 0.025
    obstacle_z_m: float = 0.20
    obstacle_x_m: float = 0.0
    obstacle_depth_m: float = 0.030
    obstacle_width_m: float = 0.085
    receiver_z_m: float = 0.36
    field_half_width_m: float = 2.0
    n_grid: int = 8192
    obstacle_step_m: float = 0.003
    activation_bl: float = 0.80


def blockage_ratio(scene: Scene, rx_x_m: float) -> float:
    source_x = np.linspace(-scene.tx_width_m / 2, scene.tx_width_m / 2, 201)
    za = scene.obstacle_z_m - scene.obstacle_depth_m / 2
    zb = scene.obstacle_z_m + scene.obstacle_depth_m / 2
    xa = source_x + (rx_x_m - source_x) * (za / scene.receiver_z_m)
    xb = source_x + (rx_x_m - source_x) * (zb / scene.receiver_z_m)
    lo = scene.obstacle_x_m - scene.obstacle_width_m / 2
    hi = scene.obstacle_x_m + scene.obstacle_width_m / 2
    blocked = (np.maximum(xa, xb) >= lo) & (np.minimum(xa, xb) <= hi)
    return float(np.mean(blocked))


def features(scene: Scene, rx_x_m: float, bl: float) -> np.ndarray:
    dx = rx_x_m - scene.obstacle_x_m * scene.receiver_z_m / scene.obstacle_z_m
    return np.array([scene.obstacle_z_m, scene.obstacle_x_m, scene.obstacle_depth_m,
                     scene.obstacle_width_m, scene.receiver_z_m, rx_x_m, bl, dx], dtype=float)


def parameter_bounds(scene: Scene, rx_x_m: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    focal0 = float(np.hypot(scene.receiver_z_m, rx_x_m))
    theta0 = float(np.rad2deg(np.arctan2(rx_x_m, scene.receiver_z_m)))
    low = np.array([-10.0, 0.09, theta0 - 45.0])
    high = np.array([10.0, 0.54, theta0 + 45.0])
    return low, high, focal0, theta0


def gain_db(power: float, reference: float) -> float:
    return float(10 * np.log10(max(power, 1e-30) / max(reference, 1e-30)))


class AngularSpectrum:
    """1D scalar angular-spectrum propagation with a binary absorber."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.x = np.linspace(-scene.field_half_width_m, scene.field_half_width_m, scene.n_grid, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.aperture = (np.abs(self.x) <= scene.tx_width_m / 2).astype(np.float64)
        self.amplitude = 1.0 / math.sqrt(float(self.aperture.sum() * self.dx))
        self.block_mask = np.abs(self.x - scene.obstacle_x_m) > scene.obstacle_width_m / 2
        self.wavelength = C0 / scene.frequency_hz
        self.k = 2 * np.pi / self.wavelength
        fx = np.fft.fftfreq(scene.n_grid, d=self.dx)
        self.kz = self.k * np.sqrt((1 - (self.wavelength * fx) ** 2).astype(np.complex128))
        za = scene.obstacle_z_m - scene.obstacle_depth_m / 2
        zb = scene.obstacle_z_m + scene.obstacle_depth_m / 2
        self.n_steps = int(math.ceil((zb - za) / scene.obstacle_step_m))
        self.actual_step = (zb - za) / self.n_steps
        self.h_to_block = np.exp(1j * self.kz * za)
        self.h_step = np.exp(1j * self.kz * self.actual_step)
        self.h_to_rx = np.exp(1j * self.kz * (scene.receiver_z_m - zb))
        self.h_direct = np.exp(1j * self.kz * scene.receiver_z_m)

    @staticmethod
    def _propagate(field: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.fft.fft(field) * transfer)

    def source(self, curvature_per_m: float, focal_m: float, theta_deg: float) -> np.ndarray:
        theta = np.deg2rad(theta_deg)
        focal_x = focal_m * np.sin(theta)
        focal_z = focal_m * np.cos(theta)
        distance = np.sqrt((self.x - focal_x) ** 2 + focal_z**2)
        cubic = ((2 * np.pi * curvature_per_m) ** 3 / 3) * self.x**3
        phase = cubic - self.k * (distance - focal_m)
        return self.amplitude * self.aperture * np.exp(1j * phase)

    def received_field(self, source: np.ndarray, blocked: bool = True) -> np.ndarray:
        if not blocked:
            return self._propagate(source, self.h_direct)
        field = self._propagate(source, self.h_to_block)
        field *= self.block_mask
        for _ in range(self.n_steps):
            field = self._propagate(field, self.h_step)
            field *= self.block_mask
        return self._propagate(field, self.h_to_rx)

    def received_power(self, rx_x_m: float, curvature_per_m: float, focal_m: float,
                       theta_deg: float, *, blocked: bool = True) -> float:
        source = self.source(curvature_per_m, focal_m, theta_deg)
        field = self.received_field(source, blocked)
        rx_mask = np.abs(self.x - rx_x_m) <= self.scene.rx_width_m / 2
        return float(np.sum(np.abs(field[rx_mask]) ** 2) * self.dx)
