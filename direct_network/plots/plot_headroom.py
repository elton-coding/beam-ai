"""Compare one-shot beam power with offline Airy references."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from direct_network.core.paths import ROOT




def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.22})
    static = pd.read_csv(ROOT / "results" / "bounds_static.csv")
    delayed = pd.read_csv(ROOT / "results" / "bounds_delayed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    x = static.receiver_x_m * 1000
    direct = 100 * (static.power_direct / static.power_focus - 1)
    reference = 100 * (static.power_best_known_airy / static.power_focus - 1)
    axes[0].plot(x, reference, color="#B66825", lw=1.8,
                 label="Best-known Airy reference")
    axes[0].plot(x, direct, color="#2458A6", lw=1.7, marker="o", ms=3.2,
                 label="Current direct NN")
    axes[0].set(xlabel="Receiver position (mm)", ylabel="Power increase vs focus (%)",
                title="Deep-shadow static path")
    axes[0].legend(fontsize=8, loc="lower center")
    remaining = 100 * (delayed.power_best_known_airy / delayed.power_predicted_direct - 1)
    axes[1].plot(delayed.time_index, remaining,
                 color="#457B76", lw=1.5, marker=".", ms=4.0)
    axes[1].axhline(0, color="#666666", lw=0.9)
    axes[1].set(xlabel="Frame index", ylabel="Reference above current (%)",
                title="Delayed-position remaining gap")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "headroom_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
