"""Plot effective spectral efficiency, tracking errors and sweep overhead."""

from __future__ import annotations

import json

import numpy as np

from direct_network.plots.plot_stage3 import ROOT, plt, style


def main() -> None:
    result = json.loads((ROOT / "results/pilot_reuse/summary.json").read_text(encoding="utf-8"))
    metrics = {(r["condition"], r["region"], r["method"]): r for r in result["metrics"]}
    effects = {(r["condition"], r["region"]): r for r in result["comparison_to_dedicated"]
               if r["method"] == "shared_recovery"}
    conditions = ("matched", "geometry_offset", "high_pilot_noise")
    labels = ("基准条件", "几何失配", "导频噪声 +6 dB")
    methods = (
        ("dedicated", "每帧扫描", "#596978", ""),
        ("periodic", "定期扫描", "#4F91B4", "//"),
        ("shared", "仅复用", "#BD6434", "\\\\"),
        ("shared_recovery", "复用＋按需补扫", "#23795F", ".."),
    )
    style()
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, .92))
    x = np.arange(3)
    for ax, title, ylabel, region, key in (
        (axes[0, 0], "a  深遮挡通信性能", "有效频谱效率 (bit/s/Hz)", "deep", "effective_bpshz"),
        (axes[0, 1], "b  深遮挡跟踪误差", "绝对位置误差第 95 百分位 (mm)", "deep", "p95_mm"),
        (axes[1, 0], "c  全程扫描开销", "执行九束扫描的帧 (%)", "all", "scan_pct"),
    ):
        for i, (method, label, color, hatch) in enumerate(methods):
            values = [metrics[condition, region, method][key] for condition in conditions]
            ax.bar(x + (i - 1.5) * .19, values, width=.18, label=label,
                   color=color, hatch=hatch, edgecolor="#24333B", linewidth=.35)
        ax.set(xticks=x, xticklabels=labels, ylabel=ylabel)
        ax.set_title(title, loc="left")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncols=4, frameon=False)

    ax = axes[1, 1]
    ax.axvline(0, color="#62717D", lw=.8, ls="--")
    for region, label, offset, marker, color in (
        ("all", "全程", -.12, "o", "#23795F"), ("deep", "深遮挡", .12, "D", "#BD6434"),
    ):
        records = [effects[condition, region] for condition in conditions]
        change = np.array([r["rate_change_pct"] for r in records])
        ends = np.array([r["paired_scene_bootstrap_95pct_ci_pct"] for r in records]).T
        ax.errorbar(change, x + offset, xerr=[change - ends[0], ends[1] - change],
                    marker=marker, color=color, linestyle="none", ms=4, capsize=3, label=label)
    ax.invert_yaxis()
    ax.set(yticks=x, yticklabels=labels, xlabel="相对每帧扫描的频谱效率变化 (%)")
    ax.set_title("d  复用＋按需补扫的收益", loc="left")
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False, loc="upper left")
    target = ROOT / "figures/pilot_reuse_comparison"
    fig.savefig(target.with_suffix(".png"), dpi=300)
    fig.savefig(target.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
