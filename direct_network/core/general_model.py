"""One-pass Airy predictor trained across obstacle and receiver geometries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from direct_network.core.paths import ROOT


MODEL_FILE = ROOT / "artifacts" / "general_scene_model.npz"


class GeneralSceneNetwork:
    def __init__(self, path: Path = MODEL_FILE):
        with np.load(path) as saved:
            self.xmean = saved["xmean"].astype(float)
            self.xstd = saved["xstd"].astype(float)
            self.ymean = saved["ymean"].astype(float)
            self.ystd = saved["ystd"].astype(float)
            self.weights = [(saved[f"w{i}"].astype(np.float32),
                             saved[f"b{i}"].astype(np.float32)) for i in range(3)]

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features, dtype=float)
        x = np.atleast_2d(raw)
        if x.shape[1] != 8:
            raise ValueError("Expected eight scene features")
        h = ((x - self.xmean) / self.xstd).astype(np.float32)
        for i, (weight, bias) in enumerate(self.weights):
            h = h @ weight + bias
            if i < 2:
                h = np.maximum(h, 0)
        choose_positive = h[:, 0] >= 0
        y = np.where(choose_positive[:, None], h[:, 4:7], h[:, 1:4])
        output = y.astype(float) * self.ystd + self.ymean
        return output[0] if raw.ndim == 1 else output
