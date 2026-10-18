# 计算机视觉研究的重要会议与期刊

> **英文原文：** [cv-knowledge-base/13_resources_and_meta/01_key_venues.md](../../cv-knowledge-base/13_resources_and_meta/01_key_venues.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 基础
> **相关章节：**
> - [博士学习指南](./04_phd_study_guide.md)
> - [关键论文](./00_key_papers.md)
> - [重要实验室与研究组](./02_key_labs_and_groups.md)

> **🆕 2026年10月更新：**
> - **CVPR 2026**（丹佛，6月3–7日）创下纪录：投稿 16,092 篇（同比增长 24%），录用 4,089 篇（约 25.4%），注册人数约 12,200。最佳论文为 D4RT，最佳学生论文为 TRELLIS.2。
> - 新增近期结果：**ICCV 2025**（檀香山；投稿约 1.12 万篇，录用约 2.7 千篇；Marr 奖授予 BrickGPT）和 **ECCV 2026**（马尔默；据报道投稿约 1.05 万篇，录用约 2.9 千篇；最佳论文为帝国理工学院的 HKTex）。
> - 新增**即将举行的会议日程表**：CVPR 2027 西雅图（论文截止日期 2026年11月中旬）、ICLR 2027 旧金山、ICCV 2027 香港、NeurIPS 2026 悉尼（另有亚特兰大和巴黎分会场）、CoRL 2026 奥斯汀、RSS 2027 雅典（据报道）。
> - 会议对比表已更新为最新录用数据。Papers With Code 条目已更新（2025年7月关停；数据已归档至 Hugging Face）。

---

## 概述

计算机视觉研究发表在一小批竞争激烈的会议和少数存档期刊上，领域的重心明确落在**会议**而非期刊上。这一文化源于领域的快速发展：6–12 个月的期刊周期往往太慢。理解会议与期刊格局对博士生至关重要，它决定了投稿去向、阅读内容、如何校准创新性门槛，以及如何解读一篇论文的“出身”。三大“视觉原生”会议（CVPR、ICCV、ECCV）是领域的支柱；“学习类”会议（NeurIPS、ICLR、ICML）承载更偏理论和方法论的工作；随着具身智能（Embodied AI）的兴起，机器人会议（CoRL、RSS、ICRA、IROS）也变得不可或缺（参见 [VLA 模型](../06_robotics_and_embodied_ai/01_vla_models.md)）。

有两个结构性事实塑造了这一格局。第一，**录用率低且仍在下降**：在投稿量激增的背景下，顶级会议的录用率通常为 20–27%，使发表竞争激烈，审稿质量也参差不齐。第二，**会议级别不等于影响力**：许多被引最多的工作（CLIP、SAM、Stable Diffusion）无论发表在哪里都产生了巨大影响，arXiv 预印本也越来越多地在正式发表之前就设定了议程。本文梳理主要会议与期刊的范围、选择性以及各自青睐的工作类型，为投稿和阅读策略提供参考（参见[博士学习指南](./04_phd_study_guide.md)）。

---

## 视觉原生会议

- **CVPR**（IEEE/CVF Conference on Computer Vision and Pattern Recognition）— 旗舰会议，每年 6 月举办。规模最大、影响力最强的 CV 会议；范围广泛；录用率约 22–27%，每年录用数千篇论文。举办的研讨会、教程和挑战赛最多。CVPR 2026（丹佛）收到 16,092 篇投稿，是五年前的两倍多，录用 4,089 篇（约 25.4%）。CVPR 2027 将在西雅图举办（2027年6月）。
- **ICCV**（International Conference on Computer Vision）— 两年一届（奇数年），与 CVPR 并列的顶级国际 CV 会议；选择性和声望相当。设有 Marr 奖（最佳论文）。ICCV 2025（檀香山，2025年10月）在约 1.12 万篇投稿中录用约 2.7 千篇（约 24%）。CVF 公布的 ICCV 2027 地点为香港，时间为 2027年10月2–8日。
- **ECCV**（European Conference on Computer Vision）— 两年一届（偶数年），欧洲的对应会议；以对理论更友好著称；论文集由 Springer LNCS 出版。ECCV 2026 在瑞典马尔默举办（2026年9月8–12日；主会 9月10–12日），据报道投稿约 1.05 万篇，录用约 2.9 千篇。

CVPR/ICCV/ECCV 合称“三大会”，发表了绝大多数经典 CV 工作。

## 机器学习会议

- **NeurIPS** — 规模最大的机器学习会议；接收学习理论、生成模型和基础性工作，并设有重要的数据集与基准（Datasets & Benchmarks）赛道。注意：NeurIPS 的审稿和格式惯例逐年变化。NeurIPS 已变为**多会场**举办：2025 年在圣迭戈和墨西哥城同时举行，2026 年以悉尼为主会场（12月），并在亚特兰大和巴黎设有分会场。
- **ICLR** — 聚焦表示学习与深度学习；完全公开评审（OpenReview）；节奏快，偏好方法与实证洞见。ICLR 2026 在里约热内卢举办（2026年4月）。ICLR 2027 将在旧金山举办（2027年4月26–30日），其投稿截止日期（2026年9月下旬）已过。
- **ICML** — 覆盖广泛的机器学习理论与方法；在优化、生成模型和学习理论方面实力突出。ICML 2026 在首尔举办（2026年7月）。

学习/方法论贡献突出的视觉工作（ViT、MAE、扩散理论）常以这些会议为目标。

## 机器人与图形学会议

- **CoRL**（Conference on Robot Learning）— 顶级机器人学习会议；RT-1/2、Diffusion Policy、OpenVLA 均发表于此（参见 [VLA 模型](../06_robotics_and_embodied_ai/01_vla_models.md)）。CoRL 2025 在首尔举办（2025年9月）；CoRL 2026 在美国得克萨斯州奥斯汀举办（2026年11月9–12日）。
- **RSS**（Robotics: Science and Systems）— 单轨制，选择性极高；兼顾系统与理论；ALOHA、UMI、DROID 均发表于此。RSS 2026 在悉尼举办（2026年7月13–17日）。RSS 2027 地点列为雅典，采用两阶段投稿，第一阶段截止于 2026年12月初（依据社区截止日期追踪工具；请在 roboticsconference.org 上确认）。
- **ICRA / IROS** — 大型 IEEE 机器人会议；范围广泛，涵盖感知与操作。
- **SIGGRAPH / SIGGRAPH Asia** — 顶级图形学会议；3D Gaussian Splatting 以及渲染/生成类工作的发表地（参见[高斯泼溅](../04_3d_vision_and_scene/02_gaussian_splatting.md)）。

## 存档期刊

- **TPAMI**（IEEE Transactions on Pattern Analysis and Machine Intelligence）— 最负盛名的 CV 期刊；刊载扩展版/定稿版论文、综述和严格的评估。
- **IJCV**（International Journal of Computer Vision）— Springer 出版；声望相当；篇幅更长的论文。
- **JMLR**、**TMLR** — 机器学习期刊；TMLR（Transactions on Machine Learning Research）越来越多地用于发表扎实但不必承受会议创新性门槛压力的工作（DINOv2、CoCa 均发表于 TMLR）。

---

## 会议与期刊对比

| 会议/期刊 | 类型 | 频率 | 大致录用率 | 最适合 |
|-------|------|---------|--------------------|----------|
| CVPR | 会议 | 每年（6月） | ~22–27%（2026：16,092 篇中录用 25.4%） | 广泛的 CV 方向，旗舰 |
| ICCV | 会议 | 两年一届（奇数年） | ~24–26%（2025：约 1.12 万篇中录用约 2.7 千篇） | 广泛的 CV 方向，国际化 |
| ECCV | 会议 | 两年一届（偶数年） | ~25–28%（2026：据报道约 1.05 万篇中录用约 2.9 千篇） | CV，偏理论 |
| NeurIPS | 会议 | 每年（12月；自 2025 年起多会场） | ~25% | 学习、生成模型、数据集 |
| ICLR | 会议 | 每年（4/5月） | ~30% | 表示学习/深度学习 |
| ICML | 会议 | 每年（7月） | ~26% | 机器学习理论与方法 |
| CoRL | 会议 | 每年 | ~30% | 机器人学习 |
| RSS | 会议 | 每年 | ~25–30% | 机器人系统，单轨制 |
| TPAMI | 期刊 | 滚动投稿 | 不适用（审稿严格） | 定稿版/综述 |

*录用率为近似值，逐年有所变化。*

---

## 近期奖项（2025–2026）

| 会议 | 奖项 | 论文 | 机构 |
|-------|-------|-------|-------------|
| CVPR 2025（纳什维尔） | 最佳论文 | VGGT: Visual Geometry Grounded Transformer | Oxford VGG / Meta |
| ICCV 2025（檀香山） | 最佳论文（Marr 奖） | Generating Physically Stable and Buildable Brick Structures from Text (BrickGPT) | CMU |
| ICCV 2025 | 最佳学生论文 | FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models | Technion |
| ICCV 2025 | 最佳论文荣誉提名 | Spatially-Varying Autofocus；RayZer: A Self-supervised Large View Synthesis Model | CMU（autofocus）；多机构合作（RayZer） |
| NeurIPS 2025 | 时间检验奖 | Faster R-CNN（Ren, He, Girshick, Sun, 2015） | Microsoft Research（论文发表时） |
| CVPR 2026（丹佛） | 最佳论文 | Efficiently Reconstructing Dynamic Scenes One D4RT at a Time | Google DeepMind / Oxford VGG / UCL |
| CVPR 2026 | 最佳学生论文 | Native and Compact Structured Latents for 3D Generation (TRELLIS.2) | Tsinghua / Microsoft / USTC |
| CVPR 2026 | 最佳论文荣誉提名 | NitroGen（开放的游戏智能体基础模型）；SAM 3D: 3Dfy Anything in Images | NVIDIA et al.；Meta |
| ECCV 2026（马尔默） | 最佳论文 | Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat | Imperial College London |

从 2025–2026 年的获奖论文中可以看出两大趋势：**前馈式三维/四维重建**（VGGT、D4RT）和**三维生成**（TRELLIS.2、SAM 3D、BrickGPT）。关注清单见[关键论文](./00_key_papers.md)。

## 即将举行的会议（截至 2026年10月）

| 会议 | 日期 | 地点 | 关键截止日期（请以官网为准） |
|-------|-------|----------|--------------------------------------------|
| CoRL 2026 | 2026年11月9–12日 | 美国得克萨斯州奥斯汀 | 已截止（2026年5月） |
| NeurIPS 2026 | 2026年12月 | 悉尼（主会场）+ 亚特兰大和巴黎分会场 | 已截止（2026年5月） |
| ICLR 2027 | 2027年4月26–30日 | 美国旧金山 | 已截止（2026年9月25日）；2026年11月进行 rebuttal |
| CVPR 2027 | 约 2027年6月20–25日 | 美国西雅图 | 论文截止约 2026年11月中旬（摘要提前数日） |
| RSS 2027 | 2027年7月（据报道） | 希腊雅典（据报道） | 第一阶段约 2026年12月初（据报道） |
| ICCV 2027 | 2027年10月2–8日 | 香港 | 约 2027年3月（待公布） |

*日期来自 CVF 即将举办会议列表和社区截止日期追踪工具（ccf-deadlines、Hugging Face ai-deadlines），以官方征稿通知为准。*

---

## 优缺点（会议与期刊）

| 方面 | 会议 | 期刊 |
|--------|-----------|---------|
| 速度 | 快（数月）；契合 CV 的节奏 | 慢（常超过 1 年） |
| 声望 | CV 领域的主要“硬通货” | TPAMI/IJCV 声望高；总体居次要地位 |
| 评审 | 质量参差；有 rebuttal 阶段 | 更为彻底，多轮修改 |
| 篇幅/深度 | 有页数限制 | 有空间呈现完整细节、消融实验和综述 |

---

## 开放问题与研究空白（元层面）

- **审稿可扩展性。** 投稿量激增，给审稿质量和一致性带来压力。CVPR 投稿量五年内翻了一番多（2026 年为 16,092 篇），会议现在要求作者担任审稿人，并需要甄别由 LLM 撰写的审稿意见。
- **评审保密性。** 2025年11月 OpenReview 的一个 API 漏洞曾短暂暴露 ICLR 2026 投稿的审稿人身份，凸显出双盲评审基础设施的脆弱性。
- **地域可达性。** 多会场会议（NeurIPS 2025 在圣迭戈和墨西哥城；NeurIPS 2026 在悉尼、亚特兰大和巴黎）是对签证和出行障碍的回应，但也使学术社区更加分散。
- **arXiv 与同行评审。** 预印本越来越多地在正式评审之前（甚至完全绕开评审）设定议程，引发了对质量控制的质疑。
- **可复现性。** 尽管规范在改善，代码/数据的发布仍不一致（参见[软件工具](./03_software_tools.md)）。
- **创新性偏好。** 会议重新颖、轻严谨与复现；负面结果发表不足。
- **会议“通胀”。** 在“三大会”发表的压力扭曲了激励机制和研究时间线。
- **基准饱和。** 在已饱和的数据集上刷榜，挤占了发现新问题的空间。

---

## 延伸阅读

- [CVF Open Access](https://openaccess.thecvf.com/) — 免费的 CVPR/ICCV/ECCV 论文集
- [OpenReview](https://openreview.net/) — ICLR/NeurIPS 的评审意见与投稿
- [Papers With Code 存档](https://huggingface.co/datasets/pwc-archive/evaluation-tables) — 原网站已于2025年7月关停；其数据转储（论文、代码链接、评测表）已归档至 Hugging Face
- [PMLR](https://proceedings.mlr.press/) — ICML/CoRL/AISTATS 论文集
- [Robotics Proceedings (RSS)](https://www.roboticsproceedings.org/) — RSS 论文存档
- [CVPR 2026 闭幕新闻稿](https://cvpr.thecvf.com/Conferences/2026/News/Closing) — 2026 年统计数据与奖项
- [CVF 即将举办的会议](https://www.thecvf.com/?cat=2) — 官方的 CVPR/ICCV/WACV 日期与地点
- [ccf-deadlines](https://github.com/ccfddl/ccf-deadlines) 与 [Hugging Face AI Deadlines](https://huggingface.co/spaces/huggingface/ai-deadlines) — 社区维护的截止日期追踪工具
