"""Plots for the new reflection-observation stage."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from direct_network.core.physics import Scene, blockage_ratio
from direct_network.core.sensing import ReflectionScanner, SensingConfig
from direct_network.core.paths import ROOT


OUT = ROOT / "figures"


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .22})


def trajectory_figure() -> None:
    frame = pd.read_csv(ROOT / "results" / "sensing_frames.csv")
    frame = frame.loc[frame.run == 0]
    deep = frame.loc[frame.deep_shadow]
    first, last = int(deep.frame.min()), int(deep.frame.max())
    fig, ax = plt.subplots(2, 2, figsize=(10.0, 6.7), constrained_layout=True)
    for axes in ax.flat:
        axes.axvspan(first, last, color="#E7A14A", alpha=.15, lw=0)
        axes.set_xlim(frame.frame.min(), frame.frame.max())
    ax[0, 0].plot(frame.frame, frame.actual_x_m * 1000, color="black", lw=1.7, label="Actual (scoring only)")
    detected = frame.loc[frame.detected]
    ax[0, 0].scatter(detected.frame, detected.measured_x_m * 1000, s=15,
                     color="#18877F", label="Reflected-pilot detection", zorder=3)
    ax[0, 0].plot(frame.frame, frame.estimated_x_m * 1000, color="#2458A6", lw=1.3,
                  label="Causal estimate")
    ax[0, 0].set(ylabel="Receiver x (mm)", title="Position acquisition and tracking")
    ax[0, 0].legend(fontsize=7, loc="upper left")
    echo_margin = 10 * np.log10(frame.peak_echo_power / frame.detection_threshold_power)
    ax[0, 1].plot(frame.frame, echo_margin, color="#7051A3", lw=1.4)
    ax[0, 1].axhline(0, color="#A6373A", ls="--", lw=1, label="Detection threshold")
    ax[0, 1].set(ylabel="Peak echo / threshold (dB)", title="Missed echoes in deep shadow")
    ax[0, 1].legend(fontsize=7)
    ax[1, 0].plot(frame.frame, 1000 * np.abs(frame.estimated_x_m - frame.actual_x_m),
                  color="#2458A6", label="Causal tracker")
    ax[1, 0].plot(frame.frame, 1000 * np.abs(frame.hold_x_m - frame.actual_x_m),
                  color="#B66825", label="Hold last detection", alpha=.85)
    ax[1, 0].set(xlabel="Frame", ylabel="Absolute position error (mm)", title="Error during observation gaps")
    ax[1, 0].legend(fontsize=7)
    for column, label, color in (("power_tracker", "Tracker + direct NN", "#2458A6"),
                                 ("power_hold", "Hold + direct NN", "#B66825"),
                                 ("power_oracle", "True-position reference", "#397D74")):
        ax[1, 1].plot(frame.frame, 100 * (frame[column] / frame.power_true_focus - 1),
                      lw=1.25, label=label, color=color)
    ax[1, 1].axhline(0, color="#777777", lw=.8)
    ax[1, 1].set(xlabel="Frame", ylabel="Power increase vs true-position focus (%)",
                  title="Downstream beam impact")
    ax[1, 1].legend(fontsize=7, loc="upper left")
    fig.savefig(OUT / "sensing_tracking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def echo_figure() -> None:
    summary = json.loads((ROOT / "results" / "sensing_summary.json").read_text(encoding="utf-8"))
    scene = Scene(**summary["scene"])
    config = SensingConfig(**summary["sensing"])
    scanner = ReflectionScanner(scene, config)
    echo_db = 10 * np.log10(np.maximum(scanner.echo_template / scanner.threshold, 1e-12))
    max_echo_db = np.max(echo_db, axis=0)
    deep = np.array([blockage_ratio(scene, float(x)) >= scene.activation_bl for x in scanner.grid_x])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    mesh = axes[0].pcolormesh(scanner.grid_x * 1000, scanner.scan_x * 1000,
                              np.clip(echo_db, -30, 30), cmap="coolwarm", vmin=-30, vmax=30,
                              shading="nearest", rasterized=True)
    axes[0].set(xlabel="Receiver position x (mm)", ylabel="Pilot pointing x (mm)",
                title="Expected two-way echo vs threshold")
    cbar = fig.colorbar(mesh, ax=axes[0], pad=.02)
    cbar.set_label("Echo / threshold (dB)")
    axes[1].plot(scanner.grid_x * 1000, max_echo_db, color="#2458A6", lw=1.8)
    axes[1].axhline(0, color="#A6373A", ls="--", lw=1.2, label="Nominal detection limit")
    axes[1].fill_between(scanner.grid_x * 1000, -35, 35, where=deep,
                         color="#E7A14A", alpha=.15, label="Deep-shadow geometry")
    axes[1].set(xlabel="Receiver position x (mm)", ylabel="Strongest expected echo / threshold (dB)",
                title="Why the center becomes hard to locate", ylim=(-35, 35))
    axes[1].legend(fontsize=7, loc="lower left")
    fig.savefig(OUT / "sensing_echo_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    trajectory_figure()
    echo_figure()


if __name__ == "__main__":
    main()
