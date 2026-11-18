"""One-pass Airy predictor for the fixed experimental geometry (stages 3.1-3.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from direct_network.core.paths import ROOT


SCENE_MODEL_FILE = ROOT / "artifacts" / "scene_model.npz"


class SceneNetwork:
    """One-pass branch-aware model fitted to independently optimized scene labels."""

    def __init__(self, path: Path = SCENE_MODEL_FILE):
        with np.load(path) as saved:
            self.xmean = saved["xmean"]
            self.xstd = saved["xstd"]
            self.ymean = saved["ymean"]
            self.ystd = saved["ystd"]
            self.weights = [(saved[f"w{i}"].astype(np.float32), saved[f"b{i}"].astype(np.float32)) for i in range(3)]

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=np.float64)).copy()
        if x.shape[1] != 8:
            raise ValueError("Expected eight environment features")
        if np.any(np.abs(x[:, [0, 1, 2, 3, 4]] - self.xmean[[0, 1, 2, 3, 4]]) > 1e-9):
            raise ValueError("Scene model is trained for the fixed experimental geometry only")
        flip = x[:, 5] < 0
        x[flip, 1] *= -1
        x[flip, 5] *= -1
        x[flip, 7] *= -1
        h = ((x - self.xmean) / self.xstd).astype(np.float32)
        for i, (weight, bias) in enumerate(self.weights):
            h = h @ weight + bias
            if i < 2:
                h = np.maximum(h, 0)
        choose_positive = h[:, 0] >= 0
        y_normalized = np.where(choose_positive[:, None], h[:, 4:7], h[:, 1:4])
        y = y_normalized.astype(np.float64) * self.ystd + self.ymean
        y[flip, 0] *= -1
        y[flip, 2] *= -1
        return y[0] if np.asarray(features).ndim == 1 else y
