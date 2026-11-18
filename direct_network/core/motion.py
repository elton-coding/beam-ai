"""Causal one-frame receiver position extrapolation and evaluation path."""

from __future__ import annotations

import numpy as np


def receiver_trajectory() -> np.ndarray:
    t = np.arange(41)
    return 0.022 * np.sin(2 * np.pi * t / 40) + 0.003 * np.sin(4 * np.pi * t / 40 + 0.4)


def predict_at_frame(history: np.ndarray, t: int) -> float:
    if t < 2 or len(history) != t:
        raise ValueError("Need exactly t historical positions and t >= 2")
    return float(2 * history[-1] - history[-2])
