"""Rebuild the complete experiment report from the saved simulation results."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from direct_network.report.report_utils import caption, font, image, paragraph, table
from direct_network.core.paths import REPORTS, ROOT


RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
OUT = REPORTS / "全实验逐项分析与效果报告.docx"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


EARLY = read(RESULTS / "summary.json")
PCT = read(RESULTS / "percentage_analysis.json")
BOUNDS = read(RESULTS / "bounds_summary.json")
SENSITIVITY = read(RESULTS / "sensitivity.json")
SENSING = read(RESULTS / "sensing_summary.json")
ROBUST = {name: read(RESULTS / f"sensing_{name}_summary.json")
          for name in ("steady", "alternate", "high_noise")}
STAGE2 = read(RESULTS / "stage2" / "summary.json")
STAGE3 = RESULTS / "stage3"
TRAIN_FIXED = read(ARTIFACTS / "scene_training_metrics.json")
TRAIN_GENERAL = read(ARTIFACTS / "general_training_metrics.json")
GENERAL = read(STAGE3 / "general_scene_summary.json")
PIPELINE = read(STAGE3 / "general_pipeline_summary.json")
STRESS = read(STAGE3 / "stress_summary.json")
ANALYSIS = read(STAGE3 / "analysis_summary.json")


def pick(rows: list[dict], **criteria) -> dict:
    return next(row for row in rows if all(row.get(key) == value for key, value in criteria.items()))


def s2(condition: str, mode: str, beams: int, region: str = "deep",
       motion: str = "variable") -> dict:
    return pick(STAGE2["rows"], condition=condition, mode=mode, n_pilots=beams,
                region=region, motion=motion)


def f1(value: float) -> str:
    return f"{value:.1f}"


def pct(value: float) -> str:
    return f"{value:+.1f}%"


def configure(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(.58), Inches(.56)
    sec.left_margin, sec.right_margin = Inches(.78), Inches(.78)
    styles = doc.styles
    font(styles["Normal"], 10.1)
    styles["Normal"].paragraph_format.line_spacing = 1.15
    styles["Normal"].paragraph_format.space_after = Pt(5)
    font(styles["Title"], 17.5, True)
    styles["Title"].paragraph_format.space_after = Pt(8)
    title_props = styles["Title"]._element.get_or_add_pPr()
    for child in list(title_props):
        if child.tag == qn("w:pBdr"):
            title_props.remove(child)
    for name, size, before, after in (("Heading 1", 12.2, 10, 5),
                                      ("Heading 2", 10.8, 8, 4)):
        font(styles[name], size, True)
        styles[name].paragraph_format.space_before = Pt(before)
        styles[name].paragraph_format.space_after = Pt(after)
        styles[name].paragraph_format.keep_with_next = True
    font(styles["Caption"], 8.8)
    styles["Caption"].paragraph_format.space_before = Pt(4)
    styles["Caption"].paragraph_format.space_after = Pt(4)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("全实验逐项分析  ·  2026 年 11 月 5 日  ·  第 ").font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    footer.add_run(" 页").font.size = Pt(8)


def experiment(doc: Document, number: int, title: str, setting_result: str,
               analysis: str) -> None:
    doc.add_heading(f"实验 {number:02d}  {title}", 2)
    setting = doc.add_paragraph(f"设置与结果：{setting_result}", style="Normal")
    setting.paragraph_format.keep_with_next = True
    paragraph(doc, f"分析总结：{analysis}")


def main() -> None:
    doc = Document()
    configure(doc)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Airy 波束直接预测与位置获取全实验分析报告")
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("当前算法全阶段仿真复核  2026 年 11 月 5 日").italic = True

    doc.add_heading("摘要", 1)
    paragraph(doc,
        "本报告逐项整理当前 direct_network 方案的 24 组实验，覆盖固定场景波束设计、位置延迟、"
        "反射探测、探测资源与失配、跨场景训练和端到端有效通信。每项均给出设置、可追溯结果与分析。"
        "离线物理搜索只生成标签或评估参照；运行时由因果位置估计和神经网络一次前向计算输出波束参数。"
        "早期固定场景实验保留为能力与缺陷的证据，当前跨场景模型是最后阶段的方案。")
    paragraph(doc,
        f"当前最有代表性的结果是：80 个几何场景的 2400 个标签训练后，16 个未见场景的"
        f" 480 个深遮挡标签位置相对真实位置聚焦平均逐点功率提升 "
        f"{pct(GENERAL['labeled_positions']['network_vs_focus_mean_pointwise_pct'])}；"
        f"端到端深遮挡条件下，主动 9 束方案相对相同估计位置聚焦的有效效率提升 "
        f"{pct(pick(PIPELINE['results'], mode='active', region='deep')['network_vs_focus_pct'])}。"
        f"但最差 5% 位置相对聚焦低于 {pct(GENERAL['labeled_positions']['network_gain_p05_pct'])}，"
        "被动定位在深遮挡中也频繁失检。全部数值来自仿真，没有实测回波、信道或硬件能耗数据。")

    doc.add_heading("统一口径与阶段区别", 1)
    paragraph(doc,
        "深遮挡定义为几何遮挡比例不低于 0.80。归一化接收功率是接收孔径内场强平方的积分；"
        "功率增益 dB=10 log₁₀(P方案/P对照)，逐点百分比=100×平均(P方案/P对照−1)。"
        "“减去 1”使相等功率对应 0%。端到端有效效率采用"
        "数据时间比例×log₂(1+接收功率/通信噪声功率)，单位 bit/s/Hz；"
        "探测资源与跨场景端到端实验中的该指标扣除了导频时间，被动方案还计入无目标背景扫描。"
        "真实位置聚焦、同估计位置聚焦、过期位置网络和离线可行参照是不同对照，不能混算。")
    table(doc, ["阶段", "数据与场景", "核心检验"], [
        ["一  已知位置和延迟", "固定几何；120 个训练标签", "波束本身增益、理论空间、数值精度"],
        ["二  反射测位", "固定几何；含失检的轨迹", "位置获取与因果跟踪"],
        ["三  探测资源和压力", "固定几何；主动与被动两假设", "有效效率、误警、校准、失配"],
        ["四  跨场景推广", "80 个几何；2400 个标签", "未见场景功率和端到端通信"],
    ], [1.42, 2.15, 3.33])

    doc.add_page_break()
    doc.add_heading("一  固定场景的波束设计", 1)
    scene = EARLY["scene"]
    paragraph(doc,
        f"共同场景为 {scene['frequency_hz']/1e9:.0f} GHz、发射孔径 "
        f"{scene['tx_width_m']*100:.0f} cm、接收孔径 {scene['rx_width_m']*100:.1f} cm；"
        f"障碍物中心 z={scene['obstacle_z_m']:.2f} m，宽 "
        f"{scene['obstacle_width_m']*100:.1f} cm、厚 {scene['obstacle_depth_m']*100:.0f} cm，"
        f"接收机 z={scene['receiver_z_m']:.2f} m。传播为二维截面上的一维标量角谱仿真。")
    experiment(doc, 1, "固定场景离线标签与直接预测网络",
        f"在 120 个接收位置用两曲率方向的离线有界搜索产生标签，合计 18,267 次物理功率评价；"
        f"96 个标签训练、24 个验证。网络为 8–64–64–7，验证集曲率方向准确率 "
        f"{TRAIN_FIXED['validation_branch_accuracy']:.1%}，在线只取一个分支。",
        "这证明自行训练的网络能把离线物理优化结果压缩为一次参数预测。120 个位置来自单一几何，"
        "因此固定场景的高增益不能证明换障碍物或接收机纵向位置后仍有效。")
    crossing = PCT["crossing_path"]
    experiment(doc, 2, "已知位置的静态穿越",
        f"31 个位置中有 9 个深遮挡点调用 Airy 网络，其他 22 点使用聚焦。"
        f"深遮挡平均逐点功率提升 {pct(crossing['deep_shadow']['current_vs_focus_pct'])}，"
        f"其他位置为 0%，全路径为 {pct(crossing['all']['current_vs_focus_pct'])}。",
        "增益集中在障碍后的深遮挡区域；全路径均值因聚焦切换而降低。"
        "这里使用真实位置，不包含定位误差，适合作为波束设计能力的上游基准。")
    dense = PCT["dense_deep_shadow"]
    experiment(doc, 3, "密集深遮挡与传播热力图",
        f"21 个深遮挡位置全部调用网络，平均逐点功率提升 "
        f"{pct(dense['current_vs_focus_pct'])}，平均 dB 增益 "
        f"{BOUNDS['static_deep_shadow']['direct_mean_gain_vs_focus_db']:.2f} dB；"
        f"21 点全部胜过聚焦，最小增益 {EARLY['static']['deep_shadow']['active_min_gain_db']:.2f} dB。",
        "固定几何下，Airy 波束能把更多能量绕过吸收障碍送入接收孔径。热力图中的绿色标记是接收机孔径，"
        "颜色表示空间功率分布，最终增益仍由孔径积分计算。")
    caption(doc, "图 1  固定场景的波束传播热力图  绿色标记为接收孔径")
    image(doc, "propagation_heatmap.png", 5.1)
    experiment(doc, 4, "离线可行参照与严格理论上界",
        f"密集深遮挡组的离线最佳已知 Airy 参数比网络平均逐点功率高 "
        f"{pct(dense['best_known_vs_current_pct'])}，平均 dB 差 "
        f"{BOUNDS['static_deep_shadow']['gap_to_best_known']['mean_db']:.3f}；"
        f"归一化接收功率≤1 的严格上界相对网络为 "
        f"{pct(dense['theoretical_vs_current_pct'])}。",
        "离线参照说明同一 Airy 参数族内仍有可改进空间，但搜索结果不保证全局最优。"
        "严格上界允许任意复波前，包含不可避免的孔径与吸收损失，因此不能把其大数值当成实际可获得增益。")

    experiment(doc, 5, "一帧位置延迟和线性外推",
        f"39 个深遮挡帧只有真实历史位置可用。直接沿用上一帧位置的 MAE 为 "
        f"{EARLY['delayed']['stale_position_mae_mm']:.2f} mm，两帧线性外推后为 "
        f"{EARLY['delayed']['predicted_position_mae_mm']:.2f} mm。"
        f"外推位置网络比同位置聚焦高 {pct(PCT['delayed']['current_vs_predicted_focus_pct'])}，"
        f"比过期位置网络高 {pct(PCT['delayed']['predicted_vs_stale_pct'])}。",
        "外推显著降低这条轨迹的位置误差，但相对直接使用过期位置的波束收益仅约 2.6%，"
        "且只在 20/39 帧更强。该实验隔离时延影响，尚未模拟反射噪声或失检。")
    experiment(doc, 6, "传播网格与障碍积分精度",
        f"保持波束参数不变，将横向网格由 16,384 加密到 "
        f"{SENSITIVITY['fine_grid_points']:,}，障碍积分步长由 0.75 mm 减为 "
        f"{SENSITIVITY['fine_obstacle_step_mm']:.3f} mm；密集深遮挡平均增益变化 "
        f"{SENSITIVITY['deep_shadow_mean_gain_change_db']:+.3f} dB，单点最大绝对变化 "
        f"{SENSITIVITY['deep_shadow_max_abs_gain_change_db']:.3f} dB。",
        "主要增益不是粗网格积分误差造成的。这是对数值离散化的检查，不能证明标量传播模型与真实环境一致。")

    doc.add_heading("二  初始被动反射位置获取", 1)
    deep = SENSING["deep_shadow"]
    other = SENSING["other_positions"]
    experiment(doc, 7, "反射扫描与因果跟踪主实验",
        f"101 帧变速穿越、20 次噪声重复；每帧 21 个聚焦探测方向。"
        f"深遮挡 {deep['samples']} 帧检出率 {deep['detection_rate']:.1%}，"
        f"其余 {other['samples']} 帧为 {other['detection_rate']:.1%}；"
        f"深遮挡跟踪 MAE {deep['tracker_position_mae_mm']:.2f} mm，"
        f"保持最近一次观测的 MAE {deep['hold_position_mae_mm']:.2f} mm。",
        f"遮挡使双程回波下降并产生连续失检，最长 {deep['max_consecutive_missed_frames']} 帧。"
        f"即使这样，网络相对同估计位置聚焦的深遮挡功率仍提升 "
        f"{pct(deep['tracker_vs_estimated_focus_power_change_pct'])}。"
        "本阶段尚未扣除 21 束扫描占用的通信时间，不能直接与后续有效效率比较。")
    caption(doc, "图 2  被动反射探测中的真实位置  检出与跟踪轨迹")
    image(doc, "sensing_tracking.png", 5.8)
    robust_rows = []
    for key, name in (("steady", "匀速"), ("alternate", "交替运动"),
                      ("high_noise", "噪声加倍")):
        row = ROBUST[key]["deep_shadow"]
        robust_rows.append([name, f"{row['detection_rate']:.1%}",
                            f"{row['tracker_position_mae_mm']:.2f}",
                            pct(row["tracker_vs_estimated_focus_power_change_pct"])])
    table(doc, ["条件", "深遮挡检出", "跟踪 MAE mm", "网络相对估计聚焦"],
          robust_rows, [1.35, 1.45, 1.55, 2.55])
    experiment(doc, 8, "运动形式与回波噪声敏感性",
        "匀速、交替运动和回波噪声加倍分别做 5 次重复。"
        f"深遮挡检出率为 {ROBUST['steady']['deep_shadow']['detection_rate']:.1%}、"
        f"{ROBUST['alternate']['deep_shadow']['detection_rate']:.1%}、"
        f"{ROBUST['high_noise']['deep_shadow']['detection_rate']:.1%}；"
        f"噪声加倍时跟踪 MAE 为 "
        f"{ROBUST['high_noise']['deep_shadow']['tracker_position_mae_mm']:.2f} mm。",
        "噪声增大使反射探测更脆弱；不同运动的轨迹与深遮挡位置分布不完全相同，"
        "所以组间差异不能全部归因于跟踪方法。部分轨迹中保持最近位置优于速度衰减跟踪，"
        "提示运动先验需要自适应。")

    doc.add_heading("三  更实际的观测代理与探测开销", 1)
    clean_p = s2("clean", "passive", 21)
    refl_p = s2("reflective", "passive", 21)
    noisy_p = s2("reflective_high_noise", "passive", 21)
    refl_a = s2("reflective", "active", 9)
    refl_pa = s2("reflective", "passive", 21, "all")
    refl_aa = s2("reflective", "active", 9, "all")
    experiment(doc, 9, "简化回波与完整反射环境对比",
        f"在相同固定几何中加入目标反射强度变化、障碍相干回波与漂移、多径和噪声。"
        f"被动 21 束深遮挡检出率从简化条件的 {clean_p['detection_rate']:.1%} "
        f"降到完整反射的 {refl_p['detection_rate']:.1%}，噪声加倍后为 "
        f"{noisy_p['detection_rate']:.1%}。",
        "早期简化回波高估了被动定位的稳定性；障碍物回波并非弱于目标即可忽略，"
        "其残差和漂移仍会抬高检测门限。实际回波参数未由测量校准，结论限于设定条件。")
    ci = pick(STAGE2["paired_run_bootstrap"], region="all",
              comparison="active_9_minus_passive_21")
    experiment(doc, 10, "主动导频与被动反射主对比",
        f"完整反射条件、单束 0.10 ms、20 次变速穿越下，主动 9 束深遮挡检出率 "
        f"{refl_a['detection_rate']:.1%}，被动 21 束 {refl_p['detection_rate']:.1%}；"
        f"全程有效效率分别为 {refl_aa['rate_0.10ms']['network_mean_bpshz']:.3f} 与 "
        f"{refl_pa['rate_0.10ms']['network_mean_bpshz']:.3f} bit/s/Hz。",
        f"主动方案优势为 {ci['mean_difference_bpshz']:.3f} bit/s/Hz，"
        f"配对重复 bootstrap 95% 区间 "
        f"[{ci['bootstrap_95pct_ci_bpshz'][0]:.3f}, "
        f"{ci['bootstrap_95pct_ci_bpshz'][1]:.3f}]。"
        "优势依赖接收机能够发送导频的假设，尚未计入两种设备的真实发射能量。")
    caption(doc, "图 3  三种反射条件下的深遮挡检出率与跟踪误差")
    image(doc, "stage2_sensing_comparison.png", 5.9)
    rate_rows = []
    for mode, label in (("active", "主动"), ("passive", "被动")):
        for n in (5, 9, 21):
            all_row = s2("reflective", mode, n, "all")
            deep_row = s2("reflective", mode, n)
            rate_rows.append([f"{label} {n} 束", f"{deep_row['detection_rate']:.1%}",
                              *[f"{all_row[f'rate_{tau}ms']['network_mean_bpshz']:.3f}"
                                for tau in ("0.05", "0.10", "0.20")]])
    table(doc, ["扫描", "深遮挡检出", "0.05 ms", "0.10 ms", "0.20 ms"],
          rate_rows, [1.4, 1.35, 1.35, 1.35, 1.45])
    experiment(doc, 11, "扫描束数与单束时长",
        "两条观测链路均比较 5、9、21 束及单束 0.05、0.10、0.20 ms；"
        "表中数值为全程有效效率 bit/s/Hz，已扣除扫描时隙，被动方案另分摊 64 次无目标校准。"
        "在 0.10 ms 下主动 9 束为 2.180，被动 21 束为 1.858。",
        "更多束可改善覆盖与定位，却压缩通信时间。主动方案在 9 束取得当前设置的较好折中；"
        "被动方案需 21 束才部分弥补深遮挡失检。导频时长是仿真假设，不是硬件实测。")
    caption(doc, "图 4  扫描束数与单束时长对全程有效效率的影响")
    image(doc, "stage2_rate_tradeoff.png", 6.0)

    snr_a = refl_aa["rate_0.10ms_snr_sweep"]
    snr_p = refl_pa["rate_0.10ms_snr_sweep"]
    table(doc, ["参考 SNR", "主动 9 束", "被动 21 束"], [
        [f"{db} dB", f"{snr_a[f'{db}db_network_mean_bpshz']:.3f}",
         f"{snr_p[f'{db}db_network_mean_bpshz']:.3f}"] for db in (0, 10, 20)
    ], [2.3, 2.3, 2.3])
    experiment(doc, 12, "通信参考信噪比敏感性",
        "保持观测数据与波束不变，把通信参考 SNR 设为 0、10、20 dB；"
        "表中为全程有效效率 bit/s/Hz。主动 9 束依次为 0.524、2.180、4.879，"
        "被动 21 束依次为 0.450、1.858、4.122。",
        "参考 SNR 越高，速率代理值越大，但这是对通信噪声的后处理扫描，"
        "并未同步改变位置探测噪声或硬件功耗，不能视为不同设备测量结果。")
    experiment(doc, 13, "被动回波因素消融",
        "单独降低目标反射、加入障碍回波、加入附加路径，并比较三因素合并及噪声加倍。"
        "深遮挡检出率依次为 27.7%、20.8%、41.5%、8.5%、8.3%；简化基准为 39.1%。",
        "在当前参数下，目标变弱和障碍回波是主要不利因素；附加路径单独作用较小。"
        "合并时检出率不能由单因素数值线性推算，说明背景、目标和噪声的相互作用需要一起评估。")
    rel_a = pick(ANALYSIS["reliability"], mode="active", n_pilots=9, region="deep")
    rel_p = pick(ANALYSIS["reliability"], mode="passive", n_pilots=21, region="deep")
    table(doc, ["深遮挡可靠性", "主动 9 束", "被动 21 束"], [
        ["跟踪误差 95% 分位", f"{rel_a['tracked_p95_mm']:.1f} mm",
         f"{rel_p['tracked_p95_mm']:.1f} mm"],
        ["每次运行最长失检段均值", f"{rel_a['mean_max_miss_streak_frames']:.1f} 帧",
         f"{rel_p['mean_max_miss_streak_frames']:.1f} 帧"],
        ["速率低于 1 bit/s/Hz 的帧", f"{rel_a['outage_below_1bpshz_pct']:.1f}%",
         f"{rel_p['outage_below_1bpshz_pct']:.1f}%"],
        ["网络相对保持位置的效率", pct(rel_a["network_vs_hold_pct"]),
         pct(rel_p["network_vs_hold_pct"])],
    ], [2.6, 2.1, 2.2])
    experiment(doc, 14, "误差尾部  连续失检与保持位置消融",
        "在完整反射条件的 20 次固定场景穿越中，主动 9 束与被动 21 束的"
        "跟踪误差 95% 分位分别为 5.2 和 40.1 mm；最长失检段均值分别为 0 与 28.9 帧。"
        "失检时用最近一次观测替代因果跟踪位置作为消融对照。",
        "被动链路的主要损失来自长时间缺乏可信位置。主动探测虽使定位可靠，"
        "深遮挡仍有约 64% 的帧低于本报告指定的 1 bit/s/Hz 阈值，表明遮挡本身仍限制通信。"
        "该阈值仅用于仿真分析，不是业务协议门限。")

    roc_p = pick(ANALYSIS["roc"], mode="passive", beams=21,
                 empirical_false_alarm_pct=1.0)
    roc_a = pick(ANALYSIS["roc"], mode="active", beams=9,
                 empirical_false_alarm_pct=1.0)
    experiment(doc, 15, "误警与检出门限权衡",
        f"50,000 次独立无目标扫描形成门限，660 个深遮挡目标帧评价检出。"
        f"在 1% 误警处，主动 9 束检出 {roc_a['deep_detection_pct']:.1f}%，"
        f"被动 21 束 {roc_p['deep_detection_pct']:.1f}%；"
        "被动容许 10% 误警时检出升至 31.5%。",
        "放宽门限不能彻底解决被动深遮挡失检，而且会显著增加假目标风险。"
        "此处的目标帧与上一实验使用独立随机流，1% 门限下的检出率不必等于其 8.5%。")
    caption(doc, "图 5  深遮挡目标检出率与无目标误警率")
    image(doc, "stage3_roc.png", 5.3)
    base_passive9 = pick(STRESS["summary"], case="baseline_passive_9", region="deep")
    adap_passive9 = pick(STRESS["summary"], case="adaptive_passive_9", region="deep")
    base_active9 = pick(STRESS["summary"], case="baseline_active_9", region="deep")
    adap_active9 = pick(STRESS["summary"], case="adaptive_active_9", region="deep")
    experiment(doc, 16, "固定与自适应方向扫描",
        f"每帧同用 9 束，自适应方式根据前一帧跟踪位置在 21 个候选方向中选 9 束。"
        f"被动深遮挡有效效率由固定 9 束的 {base_passive9['rate_mean_bpshz']:.3f} "
        f"升至 {adap_passive9['rate_mean_bpshz']:.3f} bit/s/Hz；"
        f"主动由 {base_active9['rate_mean_bpshz']:.3f} 升至 "
        f"{adap_active9['rate_mean_bpshz']:.3f}。",
        "局部扫描在本轨迹上有收益，但实验为不同随机重复，且按子集设置门限的离线校准成本未完整计入。"
        "因此它是探索性改进，尚不能确定可替换当前固定扫描主方案。")

    doc.add_heading("四  背景校准  模型失配与导频强度", 1)
    stress_cases = [
        ("baseline_passive_21", "被动基准"),
        ("no_background_passive_21", "被动无校准"),
        ("stale_background_passive_21", "被动校准过期"),
        ("mismatch_passive_21", "被动观测失配"),
        ("baseline_active_9", "主动基准"),
        ("mismatch_active_9", "主动观测失配"),
    ]
    table(doc, ["压力条件", "深遮挡检出", "无目标误警", "跟踪 MAE mm", "有效效率"], [
        [label,
         f"{pick(STRESS['summary'], case=key, region='deep')['detection_pct']:.1f}%",
         f"{100*pick(STRESS['null_false_alarm'], case=key, phase='drifted')['false_alarm_rate']:.1f}%",
         f"{pick(STRESS['summary'], case=key, region='deep')['tracked_mae_mm']:.2f}",
         f"{pick(STRESS['summary'], case=key, region='deep')['rate_mean_bpshz']:.3f}"]
        for key, label in stress_cases
    ], [1.95, 1.18, 1.18, 1.45, 1.14])
    experiment(doc, 17, "缺少无目标背景校准",
        "被动 21 束去掉 64 次无目标扫描，改用噪声门限。"
        "表面深遮挡检出率变为 100%，但无目标误警也达到 100%，"
        "跟踪 MAE 为 20.48 mm，有效效率仅 0.471 bit/s/Hz。",
        "所谓全部检出主要是残留障碍物回波触发的虚警，不能视为正确定位。"
        "真实系统若无法获得干净背景，必须设计更可靠的杂波抑制与目标确认方法。")
    experiment(doc, 18, "背景校准过期",
        "在第 50 帧改变障碍回波复振幅而沿用旧背景。"
        "无目标误警从约 1% 升至 87.3%，表面深遮挡检出率 55.2%，"
        "有效效率 0.612 bit/s/Hz。",
        "检出率上升被背景变化造成的虚警污染，不能说明定位变好。"
        "持续使用旧校准是被动方案的重要实际风险，需加入背景变化检测和重新校准策略。")
    experiment(doc, 19, "独立观测生成模型的几何失配",
        "用偏移 3 mm、宽度增加 5%、厚度增加 10% 且传播网格改变的真值库生成观测，"
        "估计器仍使用原库。被动 21 束深遮挡检出从本组基准 12.9% 降至 5.8%；"
        "主动 9 束虽保持 100% 检出，跟踪 MAE 从 1.79 增至 5.51 mm。",
        "几何误差可在检出率未下降时损害测位精度。此压力实验只改变观测生成模型，"
        "通信波束评分仍用原场景，因此尚不是整条物理链路的全面失配验证。")
    caption(doc, "图 6  各压力条件的深遮挡检出与有效效率  无校准高检出含虚警")
    image(doc, "stage3_stress.png", 6.1)
    experiment(doc, 20, "主动导频功率代理敏感性",
        "同为 9 束且单束 0.10 ms，只把目标信号幅度对应的归一化功率从 −12 扫到 +6 dB。"
        "深遮挡检出率为 52.0%、98.8%、100%、100%；有效效率为 "
        "0.866、0.899、0.911、0.897 bit/s/Hz。",
        "较弱导频明显损害定位；更强导频在本模型中没有单调提高通信效率。"
        "这是固定噪声、固定时长的归一化强度代理，不是主动与被动方案的真实同能耗比较。")

    doc.add_heading("五  扩大数据集后的跨场景验证", 1)
    experiment(doc, 21, "80 场景数据集和新网络训练",
        f"改变障碍物宽度、厚度、横向和纵向位置以及接收机纵向位置，"
        f"80 个独立场景各取 30 个深遮挡位置，共 2400 个标签，"
        f"较原 120 标签扩大 20 倍。按场景划分 48/16/16 组训练、验证、测试；"
        f"新网络为 8–128–128–7，测试曲率方向准确率 "
        f"{TRAIN_GENERAL['test']['branch_accuracy']:.1%}。",
        "场景隔离比随机拆分同一几何的位置更能检验推广。前 38 个场景连续抽样，"
        "其余 42 个从 0.25 mm 有效网格抽样以处理不连通遮挡区，位置分布仍有限制。"
        "测试集分支准确率低于训练集 91.4%，显示几何推广仍困难。")
    caption(doc, "图 7  80 个几何场景按场景而非按位置划分")
    image(doc, "stage3_dataset_scenes.png", 5.7)
    labeled = GENERAL["labeled_positions"]
    extra = GENERAL["new_positions"]
    table(doc, ["留出场景位置", "平均逐点功率提升", "5% 分位提升", "胜过聚焦比例"], [
        ["480 标签位置", pct(labeled["network_vs_focus_mean_pointwise_pct"]),
         pct(labeled["network_gain_p05_pct"]), f"{labeled['network_better_fraction']:.1%}"],
        ["336 额外位置", pct(extra["network_vs_focus_mean_pointwise_pct"]),
         pct(extra["network_gain_p05_pct"]), f"{extra['network_better_fraction']:.1%}"],
    ], [1.6, 1.9, 1.55, 1.85])
    experiment(doc, 22, "未见场景的高精度波束功率",
        f"16 个未参与训练的场景采用 16,384 网格与 0.75 mm 障碍步长复评。"
        f"480 个标签位置相对真实位置聚焦平均逐点提升 "
        f"{pct(labeled['network_vs_focus_mean_pointwise_pct'])}，"
        f"场景级 bootstrap 95% 区间 "
        f"[{labeled['scene_mean_bootstrap_95pct_ci_pct'][0]:+.1f}%, "
        f"{labeled['scene_mean_bootstrap_95pct_ci_pct'][1]:+.1f}%]；"
        f"336 个额外位置提升 {pct(extra['network_vs_focus_mean_pointwise_pct'])}。",
        f"平均增益证明跨几何迁移已有一定效果，但仅 {labeled['network_better_fraction']:.1%} "
        f"的位置胜过聚焦，最差 5% 低于 {pct(labeled['network_gain_p05_pct'])}。"
        "这些大幅退化是当前网络的首要缺陷，平均值不足以代表可靠性。")
    experiment(doc, 23, "跨场景离线参照与理论剩余空间",
        f"在 480 个标签位置，用更细传播网格复评离线可行搜索参数。"
        f"其平均功率比网络高 "
        f"{pct(GENERAL['offline_teacher_vs_network']['mean_power_ratio_pct'])}，"
        f"有 {GENERAL['offline_teacher_vs_network']['teacher_better_fraction']:.1%} 的位置更高；"
        f"归一化功率 1 对网络平均功率的宽松剩余比例为 "
        f"{labeled['strict_unit_power_upper_bound_headroom_pct']:.1f}%。",
        "离线参照说明网络在 Airy 参数预测上还有可改善空间，但它仍是粗网格可行解，"
        "不是经证明的最优值。严格功率 1 上界包含不可达的传播损失，不能作为可实现提升目标。")
    caption(doc, "图 8  留出场景的增益分布和离线可行参照差距")
    image(doc, "stage3_heldout_gain.png", 6.2)

    active_deep = pick(PIPELINE["results"], mode="active", region="deep")
    passive_deep = pick(PIPELINE["results"], mode="passive", region="deep")
    active_other = pick(PIPELINE["results"], mode="active", region="other")
    passive_other = pick(PIPELINE["results"], mode="passive", region="other")
    table(doc, ["留出场景链路", "检出率", "跟踪 MAE", "有效效率", "对同估计聚焦提升"], [
        ["主动 9 束 深遮挡", f"{active_deep['detection_pct']:.1f}%",
         f"{active_deep['tracked_mae_mm']:.2f} mm", f"{active_deep['rate_network_bpshz']:.3f}",
         pct(active_deep["network_vs_focus_pct"])],
        ["被动 21 束 深遮挡", f"{passive_deep['detection_pct']:.1f}%",
         f"{passive_deep['tracked_mae_mm']:.2f} mm", f"{passive_deep['rate_network_bpshz']:.3f}",
         pct(passive_deep["network_vs_focus_pct"])],
        ["主动 9 束 其他", f"{active_other['detection_pct']:.1f}%",
         f"{active_other['tracked_mae_mm']:.2f} mm", f"{active_other['rate_network_bpshz']:.3f}",
         pct(active_other["network_vs_focus_pct"])],
        ["被动 21 束 其他", f"{passive_other['detection_pct']:.1f}%",
         f"{passive_other['tracked_mae_mm']:.2f} mm", f"{passive_other['rate_network_bpshz']:.3f}",
         pct(passive_other["network_vs_focus_pct"])],
    ], [1.7, .9, 1.3, 1.0, 2.0])
    experiment(doc, 24, "未见场景的位置获取与端到端有效通信",
        "16 个留出场景各进行 5 次、每次 101 帧连续穿越；每束 0.10 ms，"
        "通信噪声按每个场景的无遮挡端点参考 10 dB 校准。"
        f"深遮挡主动 9 束检出 {active_deep['detection_pct']:.1f}%，"
        f"有效效率 {active_deep['rate_network_bpshz']:.3f} bit/s/Hz，"
        f"相对同估计位置聚焦提升 {pct(active_deep['network_vs_focus_pct'])}；"
        f"被动 21 束分别为 {passive_deep['detection_pct']:.1f}%、"
        f"{passive_deep['rate_network_bpshz']:.3f} 与 "
        f"{pct(passive_deep['network_vs_focus_pct'])}。",
        f"主动深遮挡提升的按场景 bootstrap 95% 区间为 "
        f"[{active_deep['scene_bootstrap_95pct_ci_pct'][0]:+.1f}%, "
        f"{active_deep['scene_bootstrap_95pct_ci_pct'][1]:+.1f}%]。"
        "在非深遮挡区策略接近聚焦，增益很小。主动与被动之差同时包含定位质量和不同导频开销，"
        "不应解释成神经网络本身的差异。")
    caption(doc, "图 9  留出场景的端到端有效效率  实色为网络  斜纹为同估计位置聚焦")
    image(doc, "stage3_heldout_pipeline.png", 6.2)

    doc.add_heading("综合判断与仍缺少的证据", 1)
    paragraph(doc,
        "24 项实验形成了从波束参数、位置获取到通信开销的仿真证据链。固定场景中，已知位置时网络能稳定改善深遮挡功率；"
        "更实际的反射条件揭示被动定位长时间失检，主动导频在既定功率代理下更可靠；"
        "扩大数据集后，平均收益可以跨未见几何保持，但少数位置发生严重退化。"
        "因此当前模型适合继续研究和仿真对比，尚不能宣称可靠部署。")
    paragraph(doc,
        "下一步应优先降低波束预测的最差情形损失，并用新的独立几何复测；"
        "同时为被动链路设计无需理想无目标时段的杂波与虚警处理。"
        "仍缺少真实回波与信道数据、导频能量与切换时延、完整双程电磁效应、二维位置、多目标及场景外障碍物。"
        "所有有效效率是 AWGN 代理值，不是实测吞吐量。")
    doc.add_heading("结果文件与复现", 1)
    paragraph(doc,
        "固定场景结果见 results/summary.json、percentage_analysis.json、bounds_summary.json；"
        "初始测位见 results/sensing*_summary.json；资源扫描见 results/stage2/summary.json；"
        "跨场景与压力测试见 results/stage3/，标签和权重见 artifacts/general_labels.csv、general_scene_model.npz。")
    paragraph(doc,
        "上述路径均相对于 direct_network 目录。报告可由 report/build_all_experiments_20261105.py"
        " 从已保存结果重建；各实验脚本与命令见仓库根目录 README.md。")

    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
