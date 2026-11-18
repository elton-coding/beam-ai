# Airy 波束参数直接预测（120 GHz 近场遮挡通信仿真）

本项目复现并扩展论文 *A physics-informed Airy beam learning framework for blockage avoidance in sub-terahertz wireless networks*（Nature Communications，DOI 10.1038/s41467-025-62443-0）。研究对象是移动接收机经过障碍物后方时，发射端如何用 Airy 波束把能量送进接收孔径，并在位置有时延、需要测位、探测占用资源的条件下维持链路。

系统建模、各阶段的研究问题和对比实验设计见 [docs/项目系统建模与对比实验.md](docs/项目系统建模与对比实验.md)。

**方法概要。** 离线用二维标量角谱传播和有界搜索生成 Airy 参数标签，训练分支网络。在线流程是位置探测、因果卡尔曼跟踪，再做一次网络前向得到 $(B,F,\theta)$，不做在线搜索。几何遮挡率低于 0.80 时直接使用聚焦波束。

## 目录结构

```
beam_ai/
├── direct_network/          当前方案的代码与数据（Python 包）
│   ├── core/                共用模型：传播与几何、网络推理、运动、测位、路径
│   ├── experiments/         标签生成、训练、各阶段评估脚本
│   ├── plots/               由已保存结果作图
│   ├── report/              总报告 DOCX 生成器
│   ├── artifacts/           场景、标签、训练好的权重与训练指标
│   ├── results/             各实验逐帧数据与汇总 JSON
│   └── figures/             PNG / SVG 图
├── docs/                    建模文档、实验说明
│   └── reports/             最新总报告（DOCX、PDF 预览）；archive/ 为各阶段旧报告
└── reference/               论文、补充材料、原作者 MATLAB/TensorFlow 代码（只读）
```

`reference/数据结果/` 是论文绘图的 MAT 数据，约 542 MB，只保存在本地，不纳入 git。

## 实验阶段与代码对应

阶段编号与建模文档第 3 节一致。`results/` 下的 `stage2/`、`stage3/` 是历史目录名，与文档阶段编号不一一对应，以下表为准。

| 文档阶段 | 内容 | 主要脚本（`direct_network/`） | 结果 |
| --- | --- | --- | --- |
| 3.1 | 已知真实位置的静态遮挡 | `experiments/generate_scene_labels.py`、`train_scene.py`、`evaluate.py`、`best_known.py`、`percentage_analysis.py`、`sensitivity.py`；`plots/plot_propagation.py`、`plot_headroom.py` | `results/summary.json`、`bounds_*`、`percentage_analysis.json`、`sensitivity.json` |
| 3.2 | 位置真实但有时延 | `experiments/evaluate.py`（delayed 部分）、`core/motion.py` | `results/delayed_results.csv` |
| 3.3 | 被动反射测位 | `core/sensing.py`、`experiments/evaluate_sensing.py`、`plots/plot_sensing.py` | `results/sensing_*` |
| 3.4 | 主动/被动观测与探测开销、压力测试 | `core/sensing_stage2.py`、`experiments/evaluate_stage2.py`、`evaluate_stage3_stress.py`、`analyze_stage3.py`；`plots/plot_stage2.py` | `results/stage2/`、`results/stage3/stress_*`、`roc_deep.csv` 等 |
| 3.5 | 未见几何场景推广 | `experiments/generate_general_labels.py`、`train_general_scene.py`、`evaluate_general_scene.py`、`evaluate_general_pipeline.py`；`plots/plot_stage3.py` | `results/stage3/general_*` |
| 3.6 | 通信导频复用与按需补扫 | `experiments/evaluate_pilot_reuse.py`、`plots/plot_pilot_reuse.py` | `results/pilot_reuse/` |

阶段 3.1–3.4 使用固定几何网络 `core/model.py`（`artifacts/scene_model.npz`）。阶段 3.5–3.6 使用跨场景网络 `core/general_model.py`（`artifacts/general_scene_model.npz`）。

## 环境与复现

```powershell
pip install -r requirements.txt
```

所有脚本以模块方式从仓库根目录运行。标签生成需要大量传播仿真，耗时较长，并会持续保存进度。

```powershell
# 3.1–3.2 固定场景
python -m direct_network.experiments.generate_scene_labels
python -m direct_network.experiments.train_scene
python -m direct_network.experiments.evaluate
python -m direct_network.experiments.best_known
python -m direct_network.experiments.percentage_analysis
python -m direct_network.experiments.sensitivity
python -m direct_network.plots.plot_propagation
python -m direct_network.plots.plot_headroom

# 3.3–3.4 测位与探测开销
python -m direct_network.experiments.evaluate_sensing
python -m direct_network.experiments.evaluate_stage2
python -m direct_network.experiments.evaluate_stage3_stress
python -m direct_network.experiments.analyze_stage3
python -m direct_network.plots.plot_sensing
python -m direct_network.plots.plot_stage2

# 3.5 跨场景
python -m direct_network.experiments.generate_general_labels
python -m direct_network.experiments.train_general_scene
python -m direct_network.experiments.evaluate_general_scene
python -m direct_network.experiments.evaluate_general_pipeline
python -m direct_network.plots.plot_stage3

# 3.6 导频复用
python -m direct_network.experiments.evaluate_pilot_reuse
python -m direct_network.plots.plot_pilot_reuse

# 总报告（输出到 docs/reports/）
python -m direct_network.report.build_all_experiments
```

`evaluate_sensing` 的稳态、交替运动和高噪声补充结果需带参数运行，参数见 `python -m direct_network.experiments.evaluate_sensing --help`。

## 最新结果

- **跨场景推广（3.5）**：16 个留出场景、480 个标签位置，网络相对真实位置聚焦的平均逐点功率提升 **48.7%**。但 24% 的位置低于聚焦，最差 5% 低于 **−73.9%**。
- **端到端（3.5）**：深遮挡下，主动 9 束的检出率为 **99.4%**，有效效率 **1.004 bit/s/Hz**，比相同估计位置的聚焦高 **15.8%**。被动 21 束的检出率只有 **16.5%**。
- **导频复用（3.6）**：九束扫描帧占比从 100% 降到 **8.51%**，全程有效效率提升 **3.68%**，深遮挡提升 **1.22%**。几何失配后，深遮挡收益的区间跨过零。见 [docs/通信导频复用与按需补扫实验.md](docs/通信导频复用与按需补扫实验.md)。

逐项分析见 [docs/reports/全实验逐项分析与效果报告.docx](docs/reports/全实验逐项分析与效果报告.docx)。

## 结论边界

- 全部结果来自二维标量仿真和回波、噪声、速率的代理模型，没有实测数据。
- 离线搜索得到的是可行参照，不是 Airy 参数的全局最优证明。归一化接收功率不超过 1 是严格但宽松的物理上界，不能直接解释为可实现增益。
- 接收机能否发送导频尚未确定，因此主动和被动观测两条路线都保留。
