"""Frame-table summaries shared by the dynamic experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfbeam.link import LOCK_ERROR_M
from nfbeam.scenes import scene_bootstrap


LOW_RATE = 1.0


def longest_streak(mask: np.ndarray) -> int:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1), initial=0))


def add_columns(frames: pd.DataFrame) -> pd.DataFrame:
    f = frames.copy()
    f["abs_error_mm"] = 1000 * (f.estimate_x_m - f.actual_x_m).abs()
    f["lock_lost"] = f.abs_error_mm > 1000 * LOCK_ERROR_M
    f["low_rate"] = f.se_bpshz < LOW_RATE
    return f


def region_summary(f: pd.DataFrame, keys: list[str]) -> list[dict]:
    out = []
    for region, mask in (("all", np.ones(len(f), bool)), ("deep", f.deep.to_numpy()),
                         ("other", ~f.deep.to_numpy())):
        sub = f[mask]
        for key, g in sub.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            streaks = []
            for _, run in f[f[keys].apply(tuple, axis=1) == key].groupby(["scene_id", "run"]):
                m = run.lock_lost.to_numpy()
                if region != "all":
                    m = m & (run.deep.to_numpy() if region == "deep" else ~run.deep.to_numpy())
                streaks.append(longest_streak(m))
            row = dict(zip(keys, key))
            row.update({"region": region, "frames": len(g),
                        "se_mean": float(g.se_bpshz.mean()),
                        "se_p05": float(g.se_bpshz.quantile(.05)),
                        "low_rate_pct": float(100 * g.low_rate.mean()),
                        "scan_pct": float(100 * g.scanned.mean()),
                        "mae_mm": float(g.abs_error_mm.mean()),
                        "p95_mm": float(g.abs_error_mm.quantile(.95)),
                        "lock_lost_pct": float(100 * g.lock_lost.mean()),
                        "longest_lock_loss_ms_mean": float(20 * np.mean(streaks)) if streaks else 0.0})
            out.append(row)
    return out


def paired_change(f: pd.DataFrame, group_keys: list[str], method: str, baseline: str,
                  method_col: str = "policy") -> list[dict]:
    """Scene-level paired relative change of mean SE, with scene bootstrap CI."""
    out = []
    for key, g in f.groupby(group_keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for region, mask in (("all", np.ones(len(g), bool)), ("deep", g.deep.to_numpy()),
                             ("other", ~g.deep.to_numpy())):
            sub = g[mask]
            a = sub[sub[method_col] == method].groupby("scene_id").se_bpshz.mean()
            b = sub[sub[method_col] == baseline].groupby("scene_id").se_bpshz.mean()
            common = a.index.intersection(b.index)
            if not len(common):
                continue
            a, b = a[common].to_numpy(), b[common].to_numpy()
            rng = np.random.default_rng(20261118)
            draws = rng.integers(0, len(a), (10000, len(a)))
            delta = 100 * (a[draws].mean(1) / b[draws].mean(1) - 1)
            row = dict(zip(group_keys, key))
            row.update({"region": region, "method": method, "baseline": baseline,
                        "se_change_pct": float(100 * (a.mean() / b.mean() - 1)),
                        "ci95_pct": np.quantile(delta, [.025, .975]).tolist()})
            out.append(row)
    return out
