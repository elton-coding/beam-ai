"""Figures for E1-E11 from the saved summaries.  Run: python -m nfbeam.plots"""

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


BC_LABEL = {"as": "角谱模型", "tm": "全波·金属 TM", "te": "全波·金属 TE", "absorber": "全波·吸收体"}


def fig_e10() -> None:
    s = load("e10")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    # (a) independent FDTD check of the boundary-integral solver
    ax = axes[0, 0]
    prof = RESULTS / "e10" / "fdtd_profile.csv"
    if prof.exists():
        p = pd.read_csv(prof)
        ax.plot(p.x_wl, p.fdtd_db, color=ORANGE, lw=2, label="FDTD（时域，独立实现）")
        ax.plot(p.x_wl, p.bie_db, color=BLUE, lw=1.2, ls="--", label="边界积分（本文求解器）")
        ax.set_xlabel("观测线位置 (波长)")
        ax.set_ylabel("总场 / 入射场 (dB)")
        v = s["validation"]["fdtd"]
        ax.set_title(f"求解器交叉验证：金属矩形，复数误差 {100 * v['ratio_rel_err_all']:.1f}%")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    # (b) gains over focus in the deep shadow
    m = pd.DataFrame(s["beams"]["metrics"])
    d = m[m.region == "deep"].set_index("bc")
    groups = ["as", "tm", "te", "absorber"]
    series = [("在线 Airy（角谱设计）", YELLOW), ("逐阵元纯相位（角谱设计）", BLUE), ("逐阵元纯相位（全波设计）", AQUA)]
    beams = pd.read_csv(RESULTS / "e10" / "beams.csv")
    deep = beams[(beams.region == "deep") & (beams.bc == "tm")]
    as_airy = float(np.median(10 * np.log10(deep.as_airy / deep.as_focus)))
    vals = [[as_airy] + [float(d.loc[b, "gain_airy_db_fullwave_median"]) for b in groups[1:]],
            [float(d.loc["tm", "gain_phase_as_db_predicted_median"])] +
            [float(d.loc[b, "gain_phase_as_db_fullwave_median"]) for b in groups[1:]],
            [float("nan")] + [float(d.loc[b, "gain_phase_fw_db_median"]) for b in groups[1:]]]
    grouped(axes[0, 1], [BC_LABEL[g] for g in groups], series, vals, fmt="{:.1f}")
    axes[0, 1].set_title("深遮挡相对聚焦的增益中位数 (dB)")
    axes[0, 1].legend(frameon=False, fontsize=8.5, loc="upper left")
    # (c) model error and design mismatch loss
    ax = axes[1, 0]
    bcs = groups[1:]
    vals = [[float(d.loc[b, "model_err_focus_db_median"]) for b in bcs],
            [float(d.loc[b, "model_err_phase_db_median"]) for b in bcs],
            [float(d.loc[b, "mismatch_loss_db_median"]) for b in bcs]]
    grouped(ax, [BC_LABEL[b] for b in bcs],
            [("角谱预测误差·聚焦", GRAY), ("角谱预测误差·纯相位", BLUE), ("角谱设计相对全波设计的损失", ORANGE)],
            vals, fmt="{:+.2f}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("深遮挡：全波 − 角谱 (dB，中位数)")
    ax.set_ylim(-0.9, 3.4)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    # (d) end-to-end link with the full-wave channel as truth
    if "link" in s:
        lm = pd.DataFrame(s["link"]["metrics"])
        lm = lm[lm.region == "deep"]
        truths = ["as", "tm", "te", "absorber"]
        pols = [("legacy", "原方案", GRAY), ("focus_bayes", "新跟踪 + 聚焦", YELLOW),
                ("proposed_warm", "新方案（实时）", BLUE), ("oracle", "真实位置参照", AQUA)]
        vals = [[float(lm[(lm.truth == t) & (lm.policy == p)].se_mean.iloc[0]) for t in truths] for p, _, _ in pols]
        grouped(axes[1, 1], [BC_LABEL[t] for t in truths], [(l, c) for _, l, c in pols], vals)
        axes[1, 1].set_title("端到端深遮挡有效频谱效率 (bit/s/Hz)")
        axes[1, 1].set_ylim(0, 2.75)
        axes[1, 1].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
    fig.suptitle("E10 全波验证：设计仍用角谱模型，真实信道换成二维全波解（16 个未见场景）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e10_fullwave")


def fig_e10_meep() -> None:
    """E10b: MEEP (FDTD) cross-check of the boundary-integral solver."""
    base = RESULTS / "e10_meep"
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    # (a) canonical case: both solvers converge to the same field
    ax = axes[0, 0]
    c = pd.DataFrame(json.loads((base / "canonical" / "summary.json").read_text(encoding="utf-8"))["rows"])
    c = c[c.rel_err_vs_bie_finest > 0]
    for solver, color, name in (("bie", BLUE, "边界积分"), ("meep", ORANGE, "MEEP")):
        for bc, ls, mk in (("te", "-", "o"), ("tm", "--", "s")):
            r = c[(c.solver == solver) & (c.bc == bc)]
            ax.plot(r.ppw, 100 * r.rel_err_vs_bie_finest, ls=ls, marker=mk, color=color, lw=1.8, ms=6,
                    label=f"{name} {bc.upper()}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([8, 12, 16, 24, 32, 40, 60], ["8", "12", "16", "24", "32", "40", "60"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("每波长网格点数")
    ax.set_ylabel("相对边界积分 32 点/波长的误差 (%)")
    ax.set_title("规范算例（线源 + 12λ×4λ 金属块）：两种求解器收敛到同一解")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")
    # (b) scene 74, line 5 wavelengths behind the obstacle, TE
    ax = axes[0, 1]
    f = np.load(base / "near_sheet" / "fields.npz")
    x = 1e3 * f["x"]
    db = lambda u: 20 * np.log10(np.abs(u) + 1e-9)
    ax.plot(x, db(f["meep_te_30"]), color=GRAY, lw=0.9, label="MEEP 30 点/波长")
    ax.plot(x, db(f["bie_te"]), color=BLUE, lw=2.4, label="边界积分（本文求解器）")
    ax.plot(x, db(f["meep_te_60"]), color=ORANGE, lw=1.1, label="MEEP 60 点/波长")
    from nfbeam.scenes import load_scenes
    sc = dict(load_scenes())[74]
    for e in (sc.obstacle_x_m - sc.obstacle_width_m / 2, sc.obstacle_x_m + sc.obstacle_width_m / 2):
        ax.axvline(1e3 * e, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(-45, 27)
    ax.set_xlabel("横向位置 x (mm)")
    ax.set_ylabel("|Hz| (dB)")
    ax.set_title("场景 74，金属 TE，障碍物后 5λ 处的场（虚线为障碍物两侧边）")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    # (c) MEEP's difference from the boundary integral vs its own resolution
    ax = axes[1, 0]
    for sub, bc, color, mk, fill, name in (("near_sheet", "te", ORANGE, "o", True, "TE，短计算域"),
                                           ("near", "te", ORANGE, "o", False, "TE，完整计算域"),
                                           ("near_sheet", "tm", AQUA, "s", True, "TM，短计算域"),
                                           ("near", "tm", AQUA, "s", False, "TM，完整计算域")):
        rows = [r for r in json.loads((base / sub / "summary.json").read_text(encoding="utf-8"))["rows"]
                if r["bc"] == bc and isinstance(r["ppw"], (int, float))]
        if not rows:
            continue
        ppw = [r["ppw"] for r in rows]
        err = [100 * r["rel_err"] for r in rows]
        ax.plot(ppw, err, marker=mk, color=color, lw=1.6, ms=7, ls="-" if fill else "--",
                mfc=color if fill else SURFACE, label=name)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([20, 30, 45, 60])
    ax.set_xlabel("MEEP 每波长网格点数")
    ax.set_ylabel("与边界积分的相对差 (%)")
    ax.set_title("同一条线：MEEP 的 TE 到 60 点/波长才收敛到边界积分解")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    # (d) receiver line, TM: phase-only gain over focus at the design point
    ax = axes[1, 1]
    g = pd.DataFrame(json.loads((base / "summary_ppw30.json").read_text(encoding="utf-8"))["gains"])
    g = g[g.bc == "tm"]
    grouped(ax, [f"场景 {i}" for i in g.scene_id], [("边界积分", BLUE), ("MEEP 30 点/波长", ORANGE)],
            [list(g.gain_bie_db), list(g.gain_meep_db)], fmt="{:.2f}")
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("dB")
    ax.set_title("接收线，金属 TM：逐阵元纯相位相对聚焦的增益")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("E10b MEEP 独立复核：开源 FDTD 与本文边界积分求解器对比", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e10_meep")


TRUTH_LABEL = {"metal": "金属 TM", "metal_te": "金属 TE", "lossy": "有损体 (反射 0.1)", "absorber": "匹配吸收体"}


def fig_e11() -> None:
    s = load("e11")
    e = pd.DataFrame(s["estimation"])
    e = e[e.snr_offset_db == 0]
    truths = [t for t in TRUTH_LABEL if t in set(e.truth)]
    g = e.set_index("truth")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    vals = [[float(g.loc[t, "edge_median_abs_mm"]) for t in truths],
            [float(g.loc[t, "edge_p90_abs_mm"]) for t in truths],
            [float(g.loc[t, "z_f_median_abs_mm"]) for t in truths]]
    acc = {t: float(g.loc[t, "accepted_pct"]) for t in truths}
    grouped(axes[0], [TRUTH_LABEL[t] + ("\n（估计被拒绝）" if acc[t] == 0 else "") for t in truths],
            [("边缘误差中位数", BLUE), ("边缘误差 90% 分位", ORANGE), ("前表面距离误差中位数", AQUA)],
            vals, fmt="{:.1f}")
    axes[0].set_yscale("log")
    axes[0].set_title("第一阶段回波估计误差 (mm，对数轴，标称回波信噪比)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    if "link" in s:
        lm = pd.DataFrame(s["link"]["metrics"])
        lm = lm[lm.region == "deep"]
        lt = [t for t in TRUTH_LABEL if t in set(lm.truth)]
        pols = [("proposed_warm@prior", "无估计（先验）", GRAY), ("proposed_warm@echo", "回波估计", BLUE),
                ("proposed_warm@known", "已知几何", YELLOW), ("oracle", "真实位置参照", AQUA)]
        vals = [[float(lm[(lm.truth == t) & (lm.policy == p)].se_mean.iloc[0]) for t in lt] for p, _, _ in pols]
        grouped(axes[1], [TRUTH_LABEL[t] + ("\n（回波估计被拒绝 → 先验）" if acc.get(t, 100) == 0 else "") for t in lt],
                [(l, c) for _, l, c in pols], vals)
        axes[1].set_title("新方案（实时）深遮挡有效频谱效率 (bit/s/Hz)")
        axes[1].set_ylim(0, 1.95)
        axes[1].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
    fig.suptitle("E11 障碍物几何估计：回波扫描 + 在线假设后验（全波信道为真值）", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "e11_geometry")


def main() -> None:
    for name, fn in (("e1", fig_e1), ("e2", fig_e2), ("e3", fig_e3), ("e4", fig_e4), ("e5", fig_e5),
                     ("e6", fig_e6), ("e7", fig_e7), ("e8", fig_e8), ("e8b", fig_e8b), ("e9", fig_e9),
                     ("e10", fig_e10), ("e11", fig_e11)):
        if (RESULTS / name / "summary.json").exists():
            fn()
            print("figure", name)
    if (RESULTS / "e10_meep" / "near_sheet" / "summary.json").exists():
        fig_e10_meep()
        print("figure e10_meep")


if __name__ == "__main__":
    main()
