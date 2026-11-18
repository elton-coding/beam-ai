"""Figures for the paired passive and active position-acquisition experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from direct_network.core.paths import ROOT


RESULTS = ROOT / "results" / "stage2"
OUT = ROOT / "figures"
COLORS = {"clean": "#4774B7", "reflective": "#B46C28", "reflective_high_noise": "#A84755"}
MODES = {"passive": "#B46C28", "active": "#167A75"}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .22})


def sensing_figure(rows: pd.DataFrame) -> None:
    data = rows[(rows.motion == "variable") & (rows.region == "deep")
                & rows.condition.isin(COLORS)]
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex="col", constrained_layout=True)
    for col, mode in enumerate(("passive", "active")):
        subset = data[data["mode"] == mode]
        for condition, color in COLORS.items():
            one = subset[subset.condition == condition].sort_values("n_pilots")
            ax[0, col].plot(one.n_pilots, 100 * one.detection_rate, "o-", color=color,
                            lw=1.8, label=condition.replace("_", " "))
            ax[1, col].plot(one.n_pilots, one.tracked_mae_mm, "o-", color=color, lw=1.8)
        ax[0, col].set_title(f"{mode.capitalize()} observation")
        ax[1, col].set_xlabel("Pilot beams per frame")
        ax[1, col].set_xticks([5, 9, 21])
        ax[0, col].set_ylim(-4, 105)
    ax[0, 0].set_ylabel("Deep-shadow detection (%)")
    ax[1, 0].set_ylabel("Tracking MAE (mm)")
    ax[0, 0].legend(fontsize=7, loc="upper right")
    fig.savefig(OUT / "stage2_sensing_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def rate_figure(rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
    for axis, region in zip(axes, ("all", "deep")):
        data = rows[(rows.motion == "variable") & (rows.condition == "reflective")
                    & (rows.region == region)]
        for mode, color in MODES.items():
            one = data[data["mode"] == mode].sort_values("n_pilots")
            for duration, linestyle in (("0.05", ":"), ("0.10", "-"), ("0.20", "--")):
                axis.plot(one.n_pilots, one[f"rate_{duration}ms.network_mean_bpshz"],
                          marker="o", markersize=3, lw=1.7, ls=linestyle, color=color,
                          label=f"{mode}, {duration} ms/pilot")
        axis.set(title="Entire crossing" if region == "all" else "Deep shadow only",
                 xlabel="Pilot beams per frame", ylabel="Effective spectral efficiency (bit/s/Hz)")
        axis.set_xticks([5, 9, 21])
    axes[1].legend(fontsize=6.8, loc="lower right")
    fig.savefig(OUT / "stage2_rate_tradeoff.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def trajectory_figure(frame: pd.DataFrame, summary: dict) -> None:
    subset = frame[(frame.condition == "reflective") & (frame.motion == "variable")
                   & (frame.run == 0)]
    passive = subset[(subset["mode"] == "passive") & (subset.n_pilots == 21)]
    active = subset[(subset["mode"] == "active") & (subset.n_pilots == 9)]
    assert len(passive) == len(active) == 101
    deep = passive[passive.deep_shadow]
    first, last = int(deep.frame.min()), int(deep.frame.max())
    fig, ax = plt.subplots(2, 2, figsize=(9.8, 6.2), constrained_layout=True)
    for axis in ax.flat:
        axis.axvspan(first, last, color="#E7A14A", alpha=.15, lw=0)
        axis.set_xlim(0, 100)
    ax[0, 0].plot(passive.frame, 1000 * passive.actual_x_m, color="black", lw=1.6, label="True x (scoring only)")
    for one, mode in ((passive, "passive"), (active, "active")):
        ax[0, 0].plot(one.frame, 1000 * one.estimated_x_m, color=MODES[mode], lw=1.4,
                      label=f"{mode} estimate")
        margin = 10 * np.log10(one.peak_echo_power / one.detection_threshold_power)
        ax[0, 1].plot(one.frame, margin, color=MODES[mode], lw=1.3, label=mode)
        ax[1, 0].plot(one.frame, 1000 * np.abs(one.estimated_x_m - one.actual_x_m),
                      color=MODES[mode], lw=1.3, label=mode)
        row = next(item for item in summary["rows"] if item["condition"] == "reflective"
                   and item["mode"] == mode and item["motion"] == "variable"
                   and item["region"] == "all" and item["n_pilots"] == (21 if mode == "passive" else 9))
        fraction = row["rate_0.10ms"]["session_data_fraction_including_calibration"]
        noise = summary["data_noise_power_at_10db_reference"]
        rate = fraction * np.log2(1 + one.power_network / noise)
        ax[1, 1].plot(one.frame, rate, color=MODES[mode], lw=1.3, label=mode)
    ax[0, 0].set(title="Causal position estimates", ylabel="Receiver x (mm)")
    ax[0, 1].axhline(0, color="#555555", ls="--", lw=.8)
    ax[0, 1].set(title="Strongest pilot / threshold", ylabel="Detection margin (dB)")
    ax[1, 0].set(title="Position error", xlabel="Frame", ylabel="Absolute error (mm)")
    ax[1, 1].set(title="Communication after pilot time", xlabel="Frame", ylabel="Effective bit/s/Hz")
    for axis in ax.flat:
        axis.legend(fontsize=7, loc="best")
    fig.savefig(OUT / "stage2_trajectory.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    rows = pd.json_normalize(summary["rows"])
    frame = pd.read_csv(RESULTS / "frames.csv")
    sensing_figure(rows)
    rate_figure(rows)
    trajectory_figure(frame, summary)
    print("saved stage2 figures")


if __name__ == "__main__":
    main()
