"""Train one-pass branch-aware Airy predictor on newly generated scene labels."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from direct_network.core.paths import ROOT


def main() -> None:
    torch.manual_seed(20261005)
    np.random.seed(20261005)
    torch.set_num_threads(4)
    source = ROOT / "artifacts" / "scene_labels.csv"
    data = pd.read_csv(source)
    if len(data) != 120 or not np.isfinite(data.select_dtypes(include="number")).all().all():
        raise ValueError("Unexpected label data")
    x = data[[f"feature_{i}" for i in range(8)]].to_numpy(float)
    f0 = np.hypot(x[:, 4], x[:, 5])
    theta0 = np.rad2deg(np.arctan2(x[:, 5], x[:, 4]))
    y = np.column_stack((data.B_per_m, data.F_m - f0, data.theta_deg - theta0))
    labels = (y[:, 0] > 0).astype(np.float32)
    train_idx, val_idx = train_test_split(np.arange(len(x)), test_size=0.2, random_state=42, stratify=labels)
    xmean, xstd = x[train_idx].mean(0), x[train_idx].std(0)
    xstd[xstd < 1e-12] = 1.0
    ymean, ystd = y[train_idx].mean(0), y[train_idx].std(0)
    ystd[ystd < 1e-12] = 1.0
    xn = torch.from_numpy(((x - xmean) / xstd).astype(np.float32))
    yn = torch.from_numpy(((y - ymean) / ystd).astype(np.float32))
    branch = torch.from_numpy(labels)
    model = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 7))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    pos_count = int(labels[train_idx].sum())
    pos_weight = torch.tensor([(len(train_idx) - pos_count) / pos_count], dtype=torch.float32)

    def loss(indices: np.ndarray) -> torch.Tensor:
        out = model(xn[indices])
        pick = torch.where(branch[indices, None] > 0, out[:, 4:7], out[:, 1:4])
        regression = torch.nn.functional.smooth_l1_loss(pick, yn[indices])
        classification = torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], branch[indices], pos_weight=pos_weight)
        return regression + 0.3 * classification

    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, 4001):
        model.train()
        optimizer.zero_grad()
        train_loss = loss(train_idx)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss(val_idx))
        if val_loss < best - 1e-6:
            best, best_epoch = val_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 500 == 0:
            print(f"epoch={epoch} train={float(train_loss):.5f} val={val_loss:.5f} best={best:.5f}", flush=True)
        if stale >= 500:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(xn[val_idx]).numpy()
    predicted_branch = out[:, 0] >= 0
    pred_norm = np.where(predicted_branch[:, None], out[:, 4:7], out[:, 1:4])
    predicted_y = pred_norm * ystd + ymean
    arrays = {"xmean": xmean, "xstd": xstd, "ymean": ymean, "ystd": ystd}
    for i, layer in enumerate((model[0], model[2], model[4])):
        arrays[f"w{i}"] = layer.weight.detach().numpy().T
        arrays[f"b{i}"] = layer.bias.detach().numpy()
    output = ROOT / "artifacts" / "scene_model.npz"
    np.savez_compressed(output, **arrays)
    metrics = {"labels_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "training_samples": len(train_idx), "validation_samples": len(val_idx),
               "architecture": "8-64-64-7 with branch classifier and two 3-parameter regression heads",
               "split_seed": 42, "best_epoch": best_epoch, "validation_loss": best,
               "validation_branch_accuracy": float(np.mean(predicted_branch == labels[val_idx])),
               "validation_parameter_mae": np.abs(predicted_y - y[val_idx]).mean(0).tolist(),
               "note": "Trained only for the fixed scene; labels from offline physical search. Inference has no search."}
    (ROOT / "artifacts" / "scene_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
