"""Measurement schema, digital twin and model calibration (gap 3).

No measured channel, echo or hardware data exist in this project.  This module
provides what a measurement campaign needs:

1. A file schema (``MEASUREMENT_COLUMNS`` + a beam table) and a loader that
   validates it, so real measurements can be dropped in.
2. ``fit`` — maximum-likelihood calibration of the propagation model from
   measured pilot amplitudes: obstacle centre/width/depth/position offsets,
   a smooth (Legendre) element phase-error profile and a global gain.
3. ``DigitalTwin`` — a *synthetic* stand-in for the hardware used only to test
   the pipeline: finer propagation grid, hidden geometry offsets, random and
   smooth per-element amplitude/phase errors and 3-bit phase shifters.  Its
   outputs are labelled synthetic everywhere and are not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import legendre
from scipy.optimize import least_squares, minimize

from direct_network.core.physics import Scene
from nfbeam.channel import Channel, perturbed
from nfbeam.sensing import N_SUB, complex_normal


MEASUREMENT_COLUMNS = ("rx_x_m", "beam_id", "subaperture", "amplitude", "noise_power")
BEAM_COLUMNS = ("beam_id", "element", "phase_rad")
PHASE_BITS = 3
N_LEGENDRE = 5


def load_measurements(directory: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Read ``pilot_amplitudes.csv`` and ``beams.csv`` from a campaign directory."""
    meas = pd.read_csv(Path(directory) / "pilot_amplitudes.csv")
    beams = pd.read_csv(Path(directory) / "beams.csv")
    for cols, df, name in ((MEASUREMENT_COLUMNS, meas, "pilot_amplitudes.csv"), (BEAM_COLUMNS, beams, "beams.csv")):
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} lacks columns {sorted(missing)}")
    table = {int(b): g.sort_values("element").phase_rad.to_numpy() for b, g in beams.groupby("beam_id")}
    return meas, table


def save_measurements(directory: Path, meas: pd.DataFrame, beams: dict[int, np.ndarray]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meas.to_csv(directory / "pilot_amplitudes.csv", index=False, float_format="%.8g")
    rows = [{"beam_id": b, "element": i, "phase_rad": float(p)} for b, ph in beams.items() for i, p in enumerate(ph)]
    pd.DataFrame(rows).to_csv(directory / "beams.csv", index=False, float_format="%.8g")


def quantize_phase(phase: np.ndarray, bits: int = PHASE_BITS) -> np.ndarray:
    step = 2 * np.pi / 2**bits
    return step * np.round(np.asarray(phase) / step)


def legendre_profile(n_elements: int, coeffs: np.ndarray) -> np.ndarray:
    u = np.linspace(-1, 1, n_elements)
    return legendre.legval(u, np.r_[0.0, coeffs])      # no constant term (absorbed by phase)


# ------------------------------------------------------------- digital twin
@dataclass(frozen=True)
class TwinParams:
    dx_m: float
    width_scale: float
    dz_m: float
    depth_scale: float
    smooth_phase: np.ndarray        # Legendre coefficients (rad)
    random_phase: np.ndarray        # per element (rad)
    amp_error: np.ndarray           # per element, linear amplitude factor


class DigitalTwin:
    """Synthetic hardware stand-in (NOT measured data)."""

    def __init__(self, nominal: Scene, rng: np.random.Generator):
        n_el = Channel(nominal).n_elements
        self.params = TwinParams(dx_m=rng.uniform(-0.005, 0.005), width_scale=rng.uniform(0.92, 1.08),
                                 dz_m=rng.uniform(-0.004, 0.004), depth_scale=rng.uniform(0.85, 1.15),
                                 smooth_phase=rng.normal(0, 0.6, N_LEGENDRE),
                                 random_phase=rng.normal(0, np.deg2rad(10), n_el),
                                 amp_error=10 ** (rng.normal(0, 1.0, n_el) / 20))
        p = self.params
        scene = replace(perturbed(nominal, p.dx_m, p.width_scale, p.dz_m, p.depth_scale),
                        n_grid=16384, obstacle_step_m=0.00075)
        self.channel = Channel(scene)
        self.element_error = (p.amp_error * np.exp(1j * (legendre_profile(n_el, p.smooth_phase) + p.random_phase)))

    def radiated(self, phase: np.ndarray) -> np.ndarray:
        w = np.exp(1j * quantize_phase(phase)) * self.element_error
        return w / np.sqrt(self.channel.tx_power(np.exp(1j * quantize_phase(phase))))

    def amplitudes(self, phase: np.ndarray, x: float, rng: np.random.Generator,
                   noise_power: float, gain_std_db: float = 0.75) -> np.ndarray:
        w = self.radiated(phase)
        a = np.sqrt(np.maximum(self.channel.aperture_power(w, x, N_SUB), 0))
        g = 10 ** (rng.normal(0, gain_std_db) / 10)
        return np.abs(np.sqrt(g) * a + np.sqrt(noise_power) * complex_normal(rng, (N_SUB,)))

    def power(self, phase: np.ndarray, x: float) -> float:
        return self.channel.power(self.radiated(phase), x)


# ---------------------------------------------------------------- the model
class CalibratedModel:
    """Propagation model with geometry offsets, smooth element phase error and gain."""

    def __init__(self, nominal: Scene, geo: np.ndarray | None = None, coeffs: np.ndarray | None = None,
                 gain: float = 1.0):
        self.nominal = nominal
        self.geo = np.zeros(4) if geo is None else np.asarray(geo, float)   # dx, width scale-1, dz, depth scale-1
        self.coeffs = np.zeros(N_LEGENDRE) if coeffs is None else np.asarray(coeffs, float)
        self.gain = gain
        self._ch = None

    def scene(self) -> Scene:
        dx, ws, dz, ds = self.geo
        return perturbed(self.nominal, dx, 1 + ws, dz, 1 + ds)

    @property
    def channel(self) -> Channel:
        if self._ch is None:
            self._ch = Channel(self.scene())
        return self._ch

    def element_phase(self) -> np.ndarray:
        return legendre_profile(self.channel.n_elements, self.coeffs)

    def radiated(self, phase: np.ndarray) -> np.ndarray:
        q = np.exp(1j * quantize_phase(phase))
        return self.channel.normalize(q) * np.exp(1j * self.element_phase())

    def amplitudes(self, phase: np.ndarray, xs: np.ndarray) -> np.ndarray:
        w = self.radiated(phase)
        return np.sqrt(self.gain) * np.sqrt(np.maximum(self.channel.aperture_power(w, xs, N_SUB), 0))

    def command(self, design_phase: np.ndarray) -> np.ndarray:
        """Phases to command so that the radiated phases match the design."""
        return design_phase - self.element_phase()


def predict_table(model: CalibratedModel, meas: pd.DataFrame, beams: dict[int, np.ndarray]) -> np.ndarray:
    pred = np.empty(len(meas))
    for b, g in meas.groupby("beam_id"):
        xs = np.sort(g.rx_x_m.unique())
        amp = model.amplitudes(beams[int(b)], xs)                      # (N_SUB, len(xs))
        col = np.searchsorted(xs, g.rx_x_m.to_numpy())
        pred[g.index.to_numpy()] = amp[g.subaperture.to_numpy(), col]
    return pred


def residuals(pred: np.ndarray, meas: pd.DataFrame) -> np.ndarray:
    """Amplitude residuals scaled by the expected noise + gain spread."""
    y = meas.amplitude.to_numpy()
    mu = np.sqrt(pred**2 + meas.noise_power.to_numpy())
    sigma = np.sqrt(meas.noise_power.to_numpy() / 2 + (0.05 * pred) ** 2)
    return (y - mu) / sigma


def prediction_error_db(pred: np.ndarray, meas: pd.DataFrame) -> dict:
    p_pred = pred**2 + meas.noise_power.to_numpy()
    p_meas = np.maximum(meas.amplitude.to_numpy() ** 2, 1e-12)
    err = 10 * np.log10(p_pred / p_meas)
    return {"bias_db": float(np.mean(err)), "rmse_db": float(np.sqrt(np.mean(err**2))),
            "p90_abs_db": float(np.quantile(np.abs(err), 0.9))}


def _scored(nominal: Scene, geo: np.ndarray, coeffs: np.ndarray, meas: pd.DataFrame,
            beams: dict[int, np.ndarray], channel: Channel | None = None) -> tuple[CalibratedModel, np.ndarray]:
    """Model with the least-squares global gain, and its residual vector."""
    m = CalibratedModel(nominal, geo, coeffs)
    if channel is not None:
        m._ch = channel
    pred = predict_table(m, meas, beams)
    y = meas.amplitude.to_numpy()
    m.gain = float(max((pred @ y) / max(pred @ pred, 1e-20), 1e-3) ** 2)
    return m, residuals(np.sqrt(m.gain) * pred, meas)


def fit(nominal: Scene, meas: pd.DataFrame, beams: dict[int, np.ndarray], log=print) -> CalibratedModel:
    """ML calibration: coarse geometry grid, then alternate element-phase and geometry refinement."""
    coeffs = np.zeros(N_LEGENDRE)
    grid = [np.array([dx, ws, 0.0, 0.0]) for dx in np.linspace(-0.006, 0.006, 7)
            for ws in (-0.08, -0.04, 0.0, 0.04, 0.08)]
    costs = [np.sum(_scored(nominal, g, coeffs, meas, beams)[1] ** 2) for g in grid]
    geo = grid[int(np.argmin(costs))]
    scale = np.array([0.002, 0.02, 0.002, 0.05])
    for _ in range(2):
        channel = CalibratedModel(nominal, geo).channel        # geometry fixed: build once
        coeffs = least_squares(lambda c: _scored(nominal, geo, c, meas, beams, channel)[1],
                               coeffs, x_scale=0.5, max_nfev=60).x
        opt = minimize(lambda u: float(np.sum(_scored(nominal, geo + u * scale, coeffs, meas, beams)[1] ** 2)),
                       np.zeros(4), method="Nelder-Mead",
                       options={"maxfev": 45, "xatol": 0.05, "fatol": 1e-3,
                                "initial_simplex": np.vstack([np.zeros(4), np.eye(4)])})
        geo = geo + opt.x * scale
        log(f"  geometry {np.round(geo, 4)}  cost {opt.fun:.1f}")
    return _scored(nominal, geo, coeffs, meas, beams)[0]
