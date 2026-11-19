"""Figures for E1-E9 from the saved summaries.  Run: python -m nfbeam.plots"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nfbeam.scenes import FIGURES, RESULTS

# Validated categorical slots (light surface); identity is also given by labels.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8985"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def load(name: str) -> dict:
    return json.loads((RESULTS / name / "summary.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def hbars(ax, labels, values, colors, fmt="{:+.0f}%"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, lw=0.8)
    span = max(abs(v) for v in values) or 1
    for yi, v in zip(y, values):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), yi, fmt.format(v),
                va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    lo = min(0, min(values))
    ax.set_xlim(lo - 0.25 * span * (lo < 0), max(values) + 0.22 * span)


def fig_e1() -> None:
    s = load("e1")["heldout_all"]
    names = {"legacy_net_airy": "原方案：学习型 Airy 网络", "airy_online": "在线 Airy 优化",
             "phase_2bit": "逐阵元相位（2 bit）", "phase_only": "逐阵元纯相位"}
    colors = [GRAY, ORANGE, AQUA, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    hbars(axes[0], list(names.values()), [s[k]["vs_focus_mean_pointwise_pct"] for k in names], colors)
    axes[0].set_title("平均逐点功率提升（相对聚焦）")
    hbars(axes[1], list(names.values()), [s[k]["vs_focus_p05_pct"] for k in names], colors)
    axes[1].set_title("最差 5% 分位（相对聚焦）")
    fig.suptitle(f"E1 已知位置：16 个未见场景、{s['positions']} 个深遮挡位置", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e1_known_position")


def fig_e2() -> None:
    r = pd.DataFrame(load("e2")["results"])
    series = [("legacy_net_airy", "原方案网络", GRAY), ("nominal_phase_only", "名义模型纯相位", ORANGE),
              ("robust_airy", "稳健 Airy", AQUA), ("robust_phase_only", "稳健纯相位", BLUE)]
    titles = {"matched": "几何匹配", "mismatch": "几何失配（±4 mm、±6%）", "strong": "强失配（含纵向与厚度）"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (geo, title) in zip(axes, titles.items()):
        g = r[r.geometry == geo]
        for key, label, color in series:
            d = g[g.method == key].sort_values("sigma_mm")
            ax.plot(d.sigma_mm, d.vs_focus_p05_pct, color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("定位误差标准差 (mm)")
        ax.set_xticks(sorted(g.sigma_mm.unique()))
    axes[0].set_ylabel("相对聚焦的 5% 分位 (%)")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("E2 稳健性：失败尾部（越高越好，0 = 与聚焦持平）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e2_robustness")


def grouped(ax, groups, series, values, fmt="{:.2f}"):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (label, color) in enumerate(series):
        v = values[i]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, v, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for p, val in zip(pos, v):
            ax.text(p, val, fmt.format(val), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def fig_e3() -> None:
    m = pd.DataFrame(load("e3")["metrics"])
    m = m[m.region == "deep"]
    conds = {"matched": "基准", "geometry_mismatch": "几何失配", "strong_mismatch": "强失配",
             "pilot_noise_+6dB": "导频噪声 +6 dB"}
    pols = [("legacy", "原方案", GRAY), ("codebook", "码本波束训练", ORANGE),
            ("proposed", "新方案", BLUE), ("oracle", "真实位置参照", AQUA)]
    vals = [[float(m[(m.condition == c) & (m.policy == p)].se_mean.iloc[0]) for c in conds] for p, _, _ in pols]
    low = [[float(m[(m.condition == c) & (m.policy == p)].low_rate_pct.iloc[0]) for c in conds] for p, _, _ in pols]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    grouped(axes[0], list(conds.values()), [(l, c) for _, l, c in pols], vals)
    axes[0].set_title("深遮挡有效频谱效率 (bit/s/Hz)")
    grouped(axes[1], list(conds.values()), [(l, c) for _, l, c in pols], low, fmt="{:.0f}")
    axes[1].set_title("深遮挡低速率帧比例 (<1 bit/s/Hz, %)")
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.suptitle("E3 协作接收机端到端（16 个未见场景 × 5 条轨迹）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e3_cooperative")


def fig_e4() -> None:
    m = pd.DataFrame(load("e4")["metrics"])
    m = m[(m.region == "deep") & (m.condition == "matched")]
    pols = {"sweep21_focus": "专用照射：21 束聚焦", "sweep9_shadow": "专用照射：9 束绕射",
            "same_waveform_nogating": "数据波形回波（无距离门）", "same_waveform": "数据波形回波 + 距离门"}
    colors = [GRAY, ORANGE, YELLOW, BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), sharey=True)
    get = lambda p, k: float(m[m.policy == p][k].iloc[0])
    hbars(axes[0], list(pols.values()), [get(p, "se_mean") for p in pols], colors, fmt="{:.2f}")
    axes[0].set_title("深遮挡频谱效率 (bit/s/Hz)")
    hbars(axes[1], list(pols.values()), [get(p, "mae_mm") for p in pols], colors, fmt="{:.1f}")
    axes[1].set_title("深遮挡位置平均误差 (mm)")
    hbars(axes[2], list(pols.values()), [get(p, "lock_lost_pct") for p in pols], colors, fmt="{:.1f}")
    axes[2].set_title("深遮挡失锁帧比例 (%)")
    fig.suptitle("E4 非协作接收机：专用照射 vs 数据波形回波（同发射资源）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e4_same_waveform")


def fig_e5() -> None:
    m = pd.DataFrame(load("e5")["metrics"])
    m = m[m.region == "deep"]
    losses = {"none": "无丢包", "iid20": "独立丢包 20%", "burst": "突发丢包"}
    series = [("legacy", "原方案", GRAY), ("proposed_naive", "新方案（不处理时延）", ORANGE),
              ("proposed", "新方案（时延感知）", BLUE)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for j, (loss, title) in enumerate(losses.items()):
        g = m[m.loss == loss]
        for key, label, color in series:
            d = g[g.policy == key].sort_values("delay")
            for i, col in enumerate(("se_mean", "longest_lock_loss_ms_mean")):
                axes[i, j].plot(d.delay, d[col], color=color, lw=2, marker="o", ms=6,
                                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        axes[0, j].set_title(title)
        axes[1, j].set_xlabel("反馈时延（帧，20 ms/帧）")
        axes[1, j].set_xticks(sorted(g.delay.unique()))
    axes[0, 0].set_ylabel("深遮挡频谱效率\n(bit/s/Hz)")
    axes[1, 0].set_ylabel("深遮挡最长失锁\n(ms，每次运行平均)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("E5 反馈时延与丢包", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e5_feedback")


def fig_e6() -> None:
    s = load("e6")
    err = s["prediction_error_mean_over_scenes"]
    groups = {"all": "全部", "deep": "深遮挡", "other": "其他位置"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    grouped(axes[0], list(groups.values()), [("名义模型", GRAY), ("校准后模型", BLUE)],
            [[err[f"nominal_{g}"]["rmse_db"] for g in groups], [err[f"calibrated_{g}"]["rmse_db"] for g in groups]],
            fmt="{:.2f}")
    axes[0].set_title("留出位置的功率预测误差 RMSE (dB)")
    perf = s["beam_performance_on_twin"]["deep"]
    names = {"focus_nominal": "聚焦", "phase_only_nominal": "名义模型设计", "phase_only_calibrated": "校准后设计",
             "twin_oracle": "完全已知硬件"}
    hbars(axes[1], list(names.values()), [perf[k]["se_mean"] for k in names], [GRAY, ORANGE, BLUE, AQUA], fmt="{:.2f}")
    axes[1].set_title("深遮挡频谱效率 (bit/s/Hz)，在数字孪生上测得")
    fig.suptitle("E6 仿真与（合成）测量对照、模型校准 —— 数据来自数字孪生，非实测", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e6_calibration")


def fig_e7() -> None:
    r = pd.DataFrame(load("e7")["results"])
    names = {"focus": ("聚焦", GRAY), "airy_online": ("在线 Airy", ORANGE), "phase_only": ("逐阵元纯相位", BLUE)}
    base = r[r.tx_width_m == 0.2].set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    hbars(axes[0], [v[0] for v in names.values()], [float(base.loc[k, "margin_for_90pct_db"]) for k in names],
          [v[1] for v in names.values()], fmt="{:.1f} dB")
    axes[0].set_title("0.2 m 孔径：90% 深遮挡位置达到 1 bit/s/Hz 还差的链路余量")
    widths = sorted(r.tx_width_m.unique())
    grouped(axes[1], [f"{w:.1f} m" for w in widths], [(v[0], v[1]) for v in names.values()],
            [[float(r[(r.tx_width_m == w) & (r.method == k)].se_mean.iloc[0]) for w in widths] for k in names])
    axes[1].set_title("发射孔径宽度与深遮挡频谱效率 (bit/s/Hz)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("E7 链路预算：已知位置、理想速率（不计开销），噪声按 0.2 m 参考链路固定", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e7_link_budget")


def fig_e8() -> None:
    s = load("e8")
    r = pd.DataFrame(s["results"])
    meth = [("focus", "聚焦", GRAY), ("airy", "在线 Airy", ORANGE), ("phase", "逐阵元纯相位", BLUE),
            ("ceiling", "幅相上界", AQUA)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    sweeps = [("room", "同一房间：只改孔径", "发射孔径 D (m)", "D_m"),
              ("similar", "等比例缩放：只改电尺寸", "D/λ", "D_over_lambda"),
              ("range", "D = 0.2 m：只改距离孔径比", "z/D", "z_over_D_mean")]
    for j, (sw, title, xl, xcol) in enumerate(sweeps):
        g = r[(r.sweep == sw) & (r.region == ("deep_ref" if sw == "room" else "deep"))].sort_values(xcol)
        for key, label, color in meth[1:]:
            axes[0, j].plot(g[xcol], g[f"{key}_gain_median_db"], color=color, lw=2, marker="o", ms=6,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        axes[0, j].axhline(0, color=INK2, lw=0.8)
        axes[0, j].set_title(title)
        for key, label, color in meth:
            axes[1, j].plot(g[xcol], g[f"{key}_shadow_loss_median_db"], color=color, lw=2, marker="o", ms=6,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        axes[1, j].set_xlabel(xl)
        if sw == "range":
            axes[0, j].set_xscale("log")
            axes[1, j].set_xscale("log")
    axes[0, 0].set_ylabel("相对聚焦的增益中位数 (dB)")
    axes[1, 0].set_ylabel("遮挡损耗中位数 (dB)，\n相对无遮挡聚焦链路")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1, 0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("E8 孔径维度：深遮挡位置（同一房间一栏用 0.2 m 孔径下的深遮挡位置）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e8_aperture")

    # Where this project and the literature sit in (D/lambda, z/D).
    lit = pd.DataFrame(s["literature"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for sw, label, color in (("similar", "本实验：等比例缩放", BLUE), ("range", "本实验：距离缩放", ORANGE)):
        g = r[(r.sweep == sw) & (r.region == "all")].sort_values("D_over_lambda")
        ax.plot(g.D_over_lambda, g.z_over_D_mean, color=color, lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.scatter(lit.D_over_lambda, lit.z_over_D, s=60, color=INK2, marker="D", edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="相关文献")
    for _, row in lit.iterrows():
        name = row.work.split(",")[0].split(" (")[0]
        off, ha = {"Curving": ((-6, -11), "right"), "Zhao/Han/Bjornson closed form": ((6, -10), "left")}.get(
            name if name.startswith("Zhao") else name.split()[0], ((6, 4), "left"))
        ax.annotate(name, (row.D_over_lambda, row.z_over_D), fontsize=7.5, color=INK2, xytext=off, ha=ha,
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("发射孔径电尺寸 D/λ")
    ax.set_ylabel("接收距离与孔径之比 z/D")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("E8 扫描覆盖的参数区间与相关文献的位置")
    fig.tight_layout()
    save(fig, "e8_literature_map")


def fig_e8b() -> None:
    m = pd.DataFrame(load("e8b")["metrics_deep_ref"])
    m = m[m.region == "deep"]
    widths = sorted(m.tx_width_m.unique())
    pols = [("focus_bayes", "贝叶斯跟踪 + 聚焦", GRAY), ("proposed_warm", "新方案（实时版）", BLUE),
            ("oracle", "真实位置参照", AQUA)]
    get = lambda w, p, k: float(m[(m.tx_width_m == w) & (m.policy == p)][k].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    grouped(axes[0], [f"{w:.1f} m" for w in widths], [(l, c) for _, l, c in pols],
            [[get(w, p, "se_mean") for w in widths] for p, _, _ in pols])
    axes[0].set_title("有效频谱效率 (bit/s/Hz)")
    grouped(axes[1], [f"{w:.1f} m" for w in widths], [(l, c) for _, l, c in pols],
            [[get(w, p, "low_rate_pct") for w in widths] for p, _, _ in pols], fmt="{:.0f}")
    axes[1].set_title("低速率帧比例 (<1 bit/s/Hz, %)")
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.suptitle("E8b 不同孔径的端到端链路（0.2 m 孔径下的深遮挡位置；总发射功率固定）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e8b_aperture_link")


def fig_e9() -> None:
    s = load("e9")
    t = pd.DataFrame(s["timing"])
    m = pd.DataFrame(s["metrics"])
    m = m[m.region == "deep"]
    conds = {"matched": "基准", "strong_mismatch": "强失配", "d3_burst": "时延 3 帧 + 突发丢包"}
    pols = [("proposed", "完整设计", ORANGE), ("proposed_warm", "热启动", BLUE), ("proposed_table", "现场建表 + 热启动", AQUA)]
    get = lambda df, c, p, k: float(df[(df.condition == c) & (df.policy == p)][k].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    grouped(axes[0], list(conds.values()), [(l, c) for _, l, c in pols],
            [[get(t, c, p, "mean_ms") for c in conds] for p, _, _ in pols], fmt="{:.1f}")
    axes[0].axhline(s["frame_ms"], color=INK2, lw=0.8, ls="--")
    axes[0].set_title("每帧设计耗时均值 (ms，虚线 = 帧长)")
    grouped(axes[1], list(conds.values()), [(l, c) for _, l, c in pols],
            [[get(t, c, p, "p99_ms") for c in conds] for p, _, _ in pols], fmt="{:.0f}")
    axes[1].axhline(s["frame_ms"], color=INK2, lw=0.8, ls="--")
    axes[1].set_title("每帧设计耗时 99% 分位 (ms)")
    grouped(axes[2], list(conds.values()), [(l, c) for _, l, c in pols],
            [[get(m, c, p, "se_mean") for c in conds] for p, _, _ in pols])
    axes[2].set_title("深遮挡有效频谱效率 (bit/s/Hz)")
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.suptitle("E9 实时化：单线程耗时与链路效果（16 个未见场景 × 5 条轨迹）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e9_realtime")


def main() -> None:
    for name, fn in (("e1", fig_e1), ("e2", fig_e2), ("e3", fig_e3), ("e4", fig_e4), ("e5", fig_e5),
                     ("e6", fig_e6), ("e7", fig_e7), ("e8", fig_e8), ("e8b", fig_e8b), ("e9", fig_e9)):
        if (RESULTS / name / "summary.json").exists():
            fn()
            print("figure", name)


if __name__ == "__main__":
    main()
