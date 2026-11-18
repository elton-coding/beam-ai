"""Figures for the expanded simulation report; outputs PNG and SVG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from direct_network.core.paths import ROOT


FIG = ROOT / "figures"
RES = ROOT / "results" / "stage3"
COLORS = {"active": "#1B6A8F", "passive": "#C46136", "focus": "#6A737D"}


def style() -> None:
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#20252B", "text.color": "#20252B",
                         "axes.edgecolor": "#8A9199", "savefig.dpi": 300})


def save(fig, name: str) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def dataset() -> None:
    scenes = pd.DataFrame(json.loads((ROOT / "artifacts" / "general_scenes.json").read_text(encoding="utf-8")))
    labels = pd.read_csv(ROOT / "artifacts" / "general_labels.csv")
    assert len(labels) == 2400 and scenes.scene_id.nunique() == 80
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for split, name, color, marker in (("train", "训练 48", "#1B6A8F", "o"),
                                       ("validation", "验证 16", "#C46136", "s"),
                                       ("test", "测试 16", "#677649", "^")):
        sub = scenes[scenes.split == split]
        ax.scatter(1000 * sub.obstacle_width_m, 1000 * sub.receiver_z_m,
                   s=32, alpha=.85, label=name, c=color, marker=marker)
    ax.set(xlabel="障碍物宽度 (mm)", ylabel="接收机纵向位置 (mm)",
           title="80 个独立几何场景和按场景划分的数据集")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(alpha=.17)
    save(fig, "stage3_dataset_scenes")


def general_gain() -> None:
    frame = pd.read_csv(RES / "general_scene_frames.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    for group, name, color in (("labeled", "有离线标签位置 480", "#1B6A8F"),
                                ("new_position", "额外位置 336", "#C46136")):
        sub = frame[frame.group == group]
        gain = 100 * (sub.power_network / sub.power_focus - 1)
        xs = np.sort(gain)
        axes[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs),
                     label=name, color=color, lw=1.9)
    axes[0].axvline(0, color="#6A737D", ls="--", lw=1)
    axes[0].set(xlabel="网络相对聚焦功率提升 (%)", ylabel="累计样本比例",
                title="未见过的场景与位置")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    axes[0].grid(alpha=.15)
    labeled = frame[frame.group == "labeled"]
    teacher_gap = 100 * (labeled.power_offline_teacher / labeled.power_network - 1)
    visible = teacher_gap[(teacher_gap >= -100) & (teacher_gap <= 300)]
    axes[1].hist(visible, bins=28, range=(-100, 300),
                 color="#5D829C", edgecolor="white", alpha=.9)
    axes[1].axvline(0, color="#6A737D", ls="--", lw=1)
    axes[1].set(xlabel="离线可行参照相对网络的功率差 (%)", ylabel="位置数",
                title="剩余空间分布（横轴显示 −100 至 300%）")
    axes[1].text(.98, .98, f"超出横轴 {len(teacher_gap)-len(visible)} 点",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=7.5)
    axes[1].grid(axis="y", alpha=.15)
    save(fig, "stage3_heldout_gain")


def pipeline() -> None:
    summary = json.loads((RES / "general_pipeline_summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    for j, region in enumerate(("other", "deep")):
        records = {r["mode"]: r for r in summary["results"] if r["region"] == region}
        x = np.arange(2)
        for i, mode in enumerate(("active", "passive")):
            row = records[mode]
            axes[j].bar(i - .18, row["rate_network_bpshz"], width=.36,
                        color=COLORS[mode], label="网络" if i == 0 else None)
            axes[j].bar(i + .18, row["rate_focus_bpshz"], width=.36,
                        color=COLORS[mode], alpha=.45, hatch="//", label="聚焦" if i == 0 else None)
            axes[j].text(i, max(row["rate_network_bpshz"], row["rate_focus_bpshz"]) + .04,
                         f"{row['network_vs_focus_pct']:+.1f}%", ha="center", fontsize=8)
        axes[j].set_xticks(x, ["主动 9 束", "被动 21 束"])
        axes[j].set(ylabel="有效频谱效率 (bit/s/Hz)",
                    title="其他位置" if region == "other" else "深遮挡位置")
        axes[j].set_ylim(bottom=0)
        axes[j].grid(axis="y", alpha=.15)
    save(fig, "stage3_heldout_pipeline")


def roc() -> None:
    roc = pd.read_csv(RES / "roc_deep.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for mode, name, marker in (("active", "主动 9 束", "o"),
                               ("passive", "被动 21 束", "s")):
        sub = roc[roc["mode"] == mode].sort_values("empirical_false_alarm_pct")
        ax.plot(sub.empirical_false_alarm_pct, sub.deep_detection_pct,
                marker=marker, color=COLORS[mode], label=name)
    ax.set(xlabel="无目标误警率 (%)", ylabel="深遮挡检出率 (%)",
           title="相同反射条件下的检出与误警权衡", xscale="log", ylim=(0, 105))
    ax.legend(frameon=False)
    ax.grid(alpha=.18, which="both")
    save(fig, "stage3_roc")


def stress() -> None:
    summary = json.loads((RES / "stress_summary.json").read_text(encoding="utf-8"))
    values = {r["case"]: r for r in summary["summary"] if r["region"] == "deep"}
    names = ["baseline_passive_21", "adaptive_passive_9",
             "no_background_passive_21", "stale_background_passive_21",
             "mismatch_passive_21", "baseline_active_9", "adaptive_active_9",
             "mismatch_active_9"]
    labels = ["被动 21", "被动自适应 9", "被动无校准", "被动校准过期",
              "被动几何失配", "主动 9", "主动自适应 9", "主动几何失配"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), sharey=True)
    y = np.arange(len(names))
    colors = [COLORS["passive"]] * 5 + [COLORS["active"]] * 3
    axes[0].barh(y, [values[n]["detection_pct"] for n in names], color=colors)
    axes[0].set(xlabel="深遮挡检出率 (%)", xlim=(0, 105))
    axes[1].barh(y, [values[n]["rate_mean_bpshz"] for n in names], color=colors)
    axes[1].set(xlabel="深遮挡有效频谱效率 (bit/s/Hz)")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("独立随机重复的定位链路压力测试（每组 2020 帧）", fontsize=10)
    save(fig, "stage3_stress")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    style()
    dataset()
    general_gain()
    pipeline()
    roc()
    stress()
    print("saved five stage3 figures")


if __name__ == "__main__":
    main()
