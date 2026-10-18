# 计算机视觉博士级知识库

> **英文原文：** [cv-knowledge-base/INDEX.md](../cv-knowledge-base/INDEX.md)（本中文版供参考，内容以英文原文为准）

> **面向博士阶段计算机视觉研究的动态参考资料，重点覆盖 2024–2026 年前沿：视觉-语言-动作（VLA）模型、世界模型、具身智能与机器人。**
>
> **最后更新：** 2026年10月
>
> **中文版说明：** 本文件夹是英文知识库 [`cv-knowledge-base/`](../cv-knowledge-base/INDEX.md) 的完整简体中文参考译本，目录结构与文件名与英文版一一对应；每个文件开头均附有指向英文原文的链接。

每个 Markdown 文件都是一个自成体系、经过深入调研的参考模块：包括架构描述、数学公式、真实基准数据、关键论文表、优缺点分析、开放问题以及精选延伸阅读。文件之间通过相对 Markdown 链接相互引用。

---

## 📍 构建状态

本知识库按优先级分阶段构建。机器人/具身智能与 2024–2026 前沿两个板块优先级最高。

**图例：** `[x]` 已完成 · `[ ]` 尚未撰写

### 第一阶段 — 核心机器人与前沿（最高优先级）
- [x] `06_robotics_and_embodied_ai/00_overview_roadmap.md`
- [x] `06_robotics_and_embodied_ai/01_vla_models.md`
- [x] `06_robotics_and_embodied_ai/02_world_models.md`
- [x] `06_robotics_and_embodied_ai/03_diffusion_policy.md`
- [x] `06_robotics_and_embodied_ai/04_cross_embodiment.md`
- [x] `06_robotics_and_embodied_ai/06_manipulation_navigation.md`
- [x] `06_robotics_and_embodied_ai/08_perception_for_robotics.md`
- [x] `06_robotics_and_embodied_ai/11_generalist_vs_specialist.md`
- [x] `12_research_frontier_2024_2026/07_future_trends.md`
- [x] `06_robotics_and_embodied_ai/05_simulation_and_data.md`
- [x] `06_robotics_and_embodied_ai/07_pros_cons_roadmaps.md`
- [x] `06_robotics_and_embodied_ai/09_robot_data_and_teleoperation.md`
- [x] `06_robotics_and_embodied_ai/10_language_conditioned_control.md`
- [x] `06_robotics_and_embodied_ai/12_hardware_robot_platforms.md`

> **第一阶段完成** ✅ — `06_robotics_and_embodied_ai/` 全部 13 个文件及 `12/07_future_trends.md` 均已完成。

### 第二阶段 — 最新进展（2024–2026 前沿）
- [x] `12_research_frontier_2024_2026/00_overview_latest.md`
- [x] `12_research_frontier_2024_2026/01_scaling_laws_vision.md`
- [x] `12_research_frontier_2024_2026/02_any_model_paradigm.md`
- [x] `12_research_frontier_2024_2026/03_multimodal_llm_2025_2026.md`
- [x] `12_research_frontier_2024_2026/04_vla_embodied_2025_2026.md`
- [x] `12_research_frontier_2024_2026/05_4d_and_dynamic_scenes.md`
- [x] `12_research_frontier_2024_2026/06_spatial_intelligence.md`
- [x] `05_multimodal_vision_language/02_large_vision_language_models.md`

> **第二阶段完成** ✅ — `12_research_frontier_2024_2026/` 全部 8 个文件及大型视觉-语言模型模块均已完成。

### 第三阶段 — 核心技术深度 ✅ **已完成**
- [x] `02_core_tasks/00_image_classification.md`
- [x] `02_core_tasks/01_object_detection.md`
- [x] `02_core_tasks/02_semantic_segmentation.md`
- [x] `02_core_tasks/03_instance_panoptic_segmentation.md`
- [x] `02_core_tasks/04_pose_estimation.md`
- [x] `02_core_tasks/05_depth_estimation.md`
- [x] `02_core_tasks/06_optical_flow_tracking.md`
- [x] `02_core_tasks/07_image_generation.md`
- [x] `02_core_tasks/08_video_understanding.md`
- [x] `02_core_tasks/09_3d_reconstruction.md`
- [x] `03_architectures/00_cnn_architectures.md`
- [x] `03_architectures/01_vision_transformers.md`
- [x] `03_architectures/02_hybrid_architectures.md`
- [x] `03_architectures/03_self_supervised_learning.md`
- [x] `03_architectures/04_multimodal_architectures.md`
- [x] `03_architectures/05_state_space_models.md`
- [x] `04_3d_vision_and_scene/00_3d_representations.md`
- [x] `04_3d_vision_and_scene/01_nerf_and_variants.md`
- [x] `04_3d_vision_and_scene/02_gaussian_splatting.md`
- [x] `04_3d_vision_and_scene/03_sfm_and_slam.md`
- [x] `04_3d_vision_and_scene/04_3d_generation.md`
- [x] `05_multimodal_vision_language/00_overview.md`
- [x] `05_multimodal_vision_language/01_vlp_models.md`
- [x] `05_multimodal_vision_language/03_open_vocabulary_detection.md`
- [x] `05_multimodal_vision_language/04_document_and_ocr.md`
- [x] `10_generative_vision/00_gans_and_vaes.md`
- [x] `10_generative_vision/01_diffusion_models.md`
- [x] `10_generative_vision/02_video_generation.md`
- [x] `10_generative_vision/03_3d_generation.md`
- [x] `10_generative_vision/04_evaluation_and_safety.md`

### 第四阶段 — 支撑板块 ✅ **已完成**
- [x] `00_foundations/` — 概述、成像原理、图像处理、特征工程、经典几何（5 个文件）
- [x] `01_history/` — 时间线，以及前深度学习时代、深度学习革命、检测/分割时代、生成时代、Transformer 时代、基础模型时代（7 个文件）
- [x] `07_autonomous_driving/` — 概述、感知、占用预测、端到端、仿真/数据（5 个文件）
- [x] `08_medical_and_scientific_cv/` — 医学影像、医学基础模型、科学成像（3 个文件）
- [x] `09_efficiency_and_deployment/` — 模型压缩、高效架构、硬件加速、边缘部署（4 个文件）
- [x] `11_datasets_and_benchmarks/` — 核心、视频、3D/机器人、多模态数据集，评测指标（5 个文件）
- [x] `13_resources_and_meta/` — 关键论文、会议期刊、重要实验室、软件工具、博士学习指南（5 个文件）

---

## ✅ 知识库已完成

全部 **86 个内容模块**、**14 个板块**均已完成（另加本 INDEX）。每个文件都遵循统一模板——博士级概述、包含 LaTeX 公式和 Mermaid 图的技术章节、作者/会议/年份经核实的关键论文表、带真实数据的基准表（若无公开数据则明确标注“未公开报告”）、优缺点、5–7 个开放问题以及精选延伸阅读——并在全文中相互链接。按照项目优先级，2024–2026 前沿与机器人/具身智能板块内容最为深入。

---

## 🗂️ 目录结构

```
Chinese translation/                  # 与 cv-knowledge-base/ 结构完全一致
├── INDEX.md                          # 本文件 — 总导航
├── 00_foundations/                   # 严谨的基础知识（成像原理、图像处理、几何）
├── 01_history/                       # 计算机视觉史：感知机 → 基础模型
├── 02_core_tasks/                    # 分类、检测、分割、深度、光流、三维
├── 03_architectures/                 # CNN、ViT、混合架构、自监督、多模态、状态空间模型
├── 04_3d_vision_and_scene/           # 三维表示、NeRF、3DGS、SfM/SLAM、三维生成
├── 05_multimodal_vision_language/    # 视觉-语言预训练、大型 VLM、开放词汇检测、文档/OCR
├── 06_robotics_and_embodied_ai/      # ★ VLA、世界模型、扩散策略、跨具身学习
├── 07_autonomous_driving/            # BEV 感知、占用预测、端到端驾驶
├── 08_medical_and_scientific_cv/     # 医学影像、医学基础模型、科学计算机视觉
├── 09_efficiency_and_deployment/     # 压缩、高效架构、硬件、边缘部署
├── 10_generative_vision/             # GAN/VAE、扩散模型、视频/三维生成、评估与安全
├── 11_datasets_and_benchmarks/       # 核心/视频/三维/多模态数据集，评测指标
├── 12_research_frontier_2024_2026/   # ★ 最新 SOTA、缩放定律、空间智能、趋势
└── 13_resources_and_meta/            # 关键论文、会议、实验室、工具、博士学习指南
```

★ = 最高优先级板块

---

## 🚀 从这里开始 — 博士新生推荐阅读路径

1. **把握领域全貌：** [具身智能概述与路线图](06_robotics_and_embodied_ai/00_overview_roadmap.md) — 感知→推理→行动的流水线，以及领域正在收敛的方向。
2. **定义前沿的控制范式：** [VLA 模型](06_robotics_and_embodied_ai/01_vla_models.md) — RT-1 → RT-2 → OpenVLA → π₀ → GR00T N1 → π₀.₅ / π\*₀.₆ → Gemini Robotics 1.5 → π₀.₇ → Gemini Robotics 2 / GR00T N1.7（2026）。
3. **主要的替代路线：** [世界模型](06_robotics_and_embodied_ai/02_world_models.md) — DreamerV3 / Dreamer 4、V-JEPA 2、Genie 3、Cosmos 3，以及世界-动作模型（World Action Model）。
4. **核心学术争论：** [通才 vs. 专才](06_robotics_and_embodied_ai/11_generalist_vs_specialist.md)。
5. **未来走向：** [未来趋势](12_research_frontier_2024_2026/07_future_trends.md)。
6. **深入支撑内容：** [扩散策略](06_robotics_and_embodied_ai/03_diffusion_policy.md)、[跨具身学习](06_robotics_and_embodied_ai/04_cross_embodiment.md)、[操作与导航](06_robotics_and_embodied_ai/06_manipulation_navigation.md)、[机器人感知](06_robotics_and_embodied_ai/08_perception_for_robotics.md)。

---

## 🔥 最新进展（第 12 板块）

[研究前沿 2024–2026](12_research_frontier_2024_2026/) 板块追踪最前沿进展（全部 8 个模块已完成，并于 2026年10月更新）：
- [领域现状](12_research_frontier_2024_2026/00_overview_latest.md) — VLM、VLA 与世界模型的最新 SOTA；开源模型格局；产业背景；CVPR 2026 / ECCV 2026 亮点。
- [视觉中的缩放定律](12_research_frontier_2024_2026/01_scaling_laws_vision.md) · [“Any-Model”范式](12_research_frontier_2024_2026/02_any_model_paradigm.md) · [多模态大语言模型 2025–2026](12_research_frontier_2024_2026/03_multimodal_llm_2025_2026.md)
- [VLA 与具身智能 2025–2026](12_research_frontier_2024_2026/04_vla_embodied_2025_2026.md) · [4D 与动态场景](12_research_frontier_2024_2026/05_4d_and_dynamic_scenes.md) · [空间智能](12_research_frontier_2024_2026/06_spatial_intelligence.md)
- [未来趋势](12_research_frontier_2024_2026/07_future_trends.md) — 具身基础模型、空间智能、4D 场景、视频作为数据引擎、触觉融合、边缘端 VLA、类脑（神经形态）视觉、具身安全。

---

## 🆕 2026年10月更新

所有模块均已对照 2025 年中至 2026 年 10 月的研究与产业进展进行了复核。每个文件现均标注 `最后更新：2026年10月`，并在开头附有简短的 **🆕 2026年10月更新** 提示框，概括本次改动；新的模型、论文、基准数据和产业事实均已融入正文（关键论文表、基准表、开放问题、延伸阅读）。文件夹名称（如 `12_research_frontier_2024_2026/`）保持不变，以确保所有交叉链接继续有效。

本中文文件夹即为整个知识库的完整简体中文参考译本，目录结构与英文版一致，且每个文件均链接回对应的英文原文。

---

## ⚖️ 核心争论

- **通才 vs. 专才** — 单一的“机器人 GPT” vs. 模块化的感知/规划/控制：[11_generalist_vs_specialist.md](06_robotics_and_embodied_ai/11_generalist_vs_specialist.md)
- **VLA vs. 世界模型** — 端到端动作预测 vs. 学习得到的潜在动力学：[01_vla_models.md](06_robotics_and_embodied_ai/01_vla_models.md) vs. [02_world_models.md](06_robotics_and_embodied_ai/02_world_models.md)
- **像素空间 vs. 潜在空间预测** — 扩散式世界模型 vs. JEPA：[02_world_models.md](06_robotics_and_embodied_ai/02_world_models.md)
- **离散 token vs. 流匹配（flow matching）生成动作：** [01_vla_models.md](06_robotics_and_embodied_ai/01_vla_models.md)、[03_diffusion_policy.md](06_robotics_and_embodied_ai/03_diffusion_policy.md)

---

## ✍️ 贡献 / 维护说明

每个文件都遵循统一模板：概述 → 定制化技术章节 → 关键论文表 → 基准性能 → 优缺点 → 开放问题与研究空白 → 延伸阅读。引用采用 `[AuthorYear]` 格式；公式置于代码块中；流程图使用 Mermaid。本 INDEX 随各阶段完成而更新。修改英文模块时，请同步更新 `Chinese translation/` 下相同相对路径的对应中文文件。
