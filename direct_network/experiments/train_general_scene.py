"""Train the general Airy predictor with scene-disjoint data splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from direct_network.core.general_model import MODEL_FILE
from direct_network.core.paths import ROOT


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(20261104)
    np.random.seed(20261104)
    source = ROOT / "artifacts" / "general_labels.csv"
    data = pd.read_csv(source)
    if len(data) != 2400 or data.scene_id.nunique() != 80:
        raise ValueError("Expected 2,400 labels from 80 scenes")
    x = data[[f"feature_{i}" for i in range(8)]].to_numpy(float)
    focal = np.hypot(x[:, 4], x[:, 5])
    theta = np.rad2deg(np.arctan2(x[:, 5], x[:, 4]))
    y = np.column_stack((data.B_per_m, data.F_m - focal, data.theta_deg - theta))
    labels = (y[:, 0] > 0).astype(np.float32)
    train = np.flatnonzero(data.split == "train")
    validation = np.flatnonzero(data.split == "validation")
    test = np.flatnonzero(data.split == "test")
    if (len(train), len(validation), len(test)) != (1440, 480, 480):
        raise AssertionError("Scene partitions changed")
    if len(set(data.scene_id.iloc[train]) & set(data.scene_id.iloc[test])):
        raise AssertionError("Train and test share geometry")
    xmean, xstd = x[train].mean(0), x[train].std(0)
    xstd[xstd < 1e-12] = 1
    ymean, ystd = y[train].mean(0), y[train].std(0)
    ystd[ystd < 1e-12] = 1
    xn = torch.from_numpy(((x - xmean) / xstd).astype(np.float32))
    yn = torch.from_numpy(((y - ymean) / ystd).astype(np.float32))
    branch = torch.from_numpy(labels)
    model = torch.nn.Sequential(torch.nn.Linear(8, 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 7))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    positive = int(labels[train].sum())
    weight = torch.tensor([(len(train) - positive) / max(positive, 1)], dtype=torch.float32)

    def loss(indices: np.ndarray) -> torch.Tensor:
        out = model(xn[indices])
        chosen = torch.where(branch[indices, None] > 0, out[:, 4:7], out[:, 1:4])
        regression = torch.nn.functional.smooth_l1_loss(chosen, yn[indices])
        classification = torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], branch[indices], pos_weight=weight)
        return regression + .3 * classification

    best = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    generator = torch.Generator().manual_seed(20261104)
    for epoch in range(1, 3001):
        model.train()
        for indices in torch.randperm(len(train), generator=generator).split(128):
            batch = train[indices.numpy()]
            optimizer.zero_grad()
            value = loss(batch)
            value.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(loss(validation))
        if value < best - 1e-5:
            best = value
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 100 == 0:
            print(f"epoch={epoch} validation_loss={value:.5f} best={best:.5f}", flush=True)
        if patience >= 250:
            break
    if best_state is None:
        raise AssertionError("Training failed")
    model.load_state_dict(best_state)
    model.eval()
    arrays = {"xmean": xmean, "xstd": xstd, "ymean": ymean, "ystd": ystd}
    for i, layer in enumerate((model[0], model[2], model[4])):
        arrays[f"w{i}"] = layer.weight.detach().numpy().T
        arrays[f"b{i}"] = layer.bias.detach().numpy()
    np.savez_compressed(MODEL_FILE, **arrays)

    metrics = {"labels_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "total_labels": len(data), "scene_count": 80,
               "train_scenes": 48, "validation_scenes": 16, "test_scenes": 16,
               "train_labels": len(train), "validation_labels": len(validation),
               "test_labels": len(test), "architecture": "8-128-128-7 branch-aware",
               "best_epoch": best_epoch, "validation_loss": best}
    with torch.no_grad():
        out = model(xn).numpy()
    predicted_branch = out[:, 0] >= 0
    predicted_y = np.where(predicted_branch[:, None], out[:, 4:7], out[:, 1:4]) * ystd + ymean
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        metrics[name] = {"branch_accuracy": float(np.mean(predicted_branch[indices] == labels[indices])),
                         "parameter_mae": np.abs(predicted_y[indices] - y[indices]).mean(0).tolist()}
    target = ROOT / "artifacts" / "general_training_metrics.json"
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
