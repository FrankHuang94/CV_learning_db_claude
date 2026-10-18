# 50 篇计算机视觉必读论文

> **英文原文：** [cv-knowledge-base/13_resources_and_meta/00_key_papers.md](../../cv-knowledge-base/13_resources_and_meta/00_key_papers.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 基础
> **相关章节：**
> - [时间线概览](../01_history/00_timeline_overview.md)
> - [重要会议与期刊](./01_key_venues.md)
> - [博士学习指南](./04_phd_study_guide.md)
> - [最新进展](../12_research_frontier_2024_2026/00_overview_latest.md)

> **🆕 2026年10月更新：**
> - 新增 **2025–2026 关注清单**（不计入五十篇），收录获奖论文和很可能进入经典的工作：VGGT（CVPR 2025 最佳论文）、D4RT（CVPR 2026 最佳论文）、TRELLIS.2（CVPR 2026 最佳学生论文）、SAM 3D 与 NitroGen（CVPR 2026 最佳论文荣誉提名）、BrickGPT 与 FlowEdit（ICCV 2025）、HKTex（ECCV 2026 最佳论文）。
> - 补充了 2025 年的基础模型继任者，它们已取代前代成为默认必读：**SAM 3**（概念分割，2025年11月；SAM 3.1 于2026年3月发布）、**DINOv3**（2025年8月）、**V-JEPA 2**（2025年6月；V-JEPA 2.1 于2026年3月发布）、**π₀.₅** 和 **Genie 3**。
> - 注明 **Faster R-CNN** 获得 NeurIPS 2025 时间检验奖（Test of Time Award）。
> - 来源：CVPR 2026 闭幕新闻稿、CVF/TCPAMI 奖项列表、帝国理工学院关于 ECCV 2026 的公告，以及官方 GitHub README（facebookresearch/sam3、dinov3、vggt、vjepa2；microsoft/TRELLIS.2；Physical-Intelligence/openpi）。

---

## 概述

这是一份精选的五十篇论文阅读清单，是计算机视觉博士生应当了解的论文。它们不一定是被引最多的五十篇，而是最能体现该领域概念结构与发展轨迹的五十篇，覆盖从深度学习革命到具身智能（Embodied AI）前沿的全过程。论文按主题组织，每篇给出标题、作者、年份、发表会议/期刊，以及一句话说明其重要性。清单有意偏重架构类和范式转变类工作，因为它们在本知识库中作为基础构件反复出现；如需各篇的技术细节，请查阅链接的章节文件。

阅读策略：新手应先读**分类骨干网络**（1–6）和 **Transformer/CLIP 系列**（16–20），以理解现代技术底座，再根据兴趣延伸到各任务与前沿方向。2024–2026 年的条目较新，且部分为非存档文献（公司技术报告）；在得到复现之前，应把其报告结果视为初步结论（参见[最新进展](../12_research_frontier_2024_2026/00_overview_latest.md)）。末尾单独列出一份 **2025–2026 关注清单**，汇集近期获奖论文和基础模型继任者（SAM 3、DINOv3、VGGT、TRELLIS.2、D4RT）。它们是进入经典的有力候选，但哪些能经受住时间考验，目前还难以判断。

---

## 分类与骨干网络

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 1 | ImageNet | Deng et al. | 2009 | CVPR | 使深度学习在视觉中成为可能的数据集 |
| 2 | AlexNet | Krizhevsky, Sutskever, Hinton | 2012 | NeurIPS | 开启了深度学习革命 |
| 3 | VGGNet | Simonyan, Zisserman | 2015 | ICLR | 通过统一的 3×3 卷积实现网络加深 |
| 4 | GoogLeNet/Inception | Szegedy et al. | 2015 | CVPR | 多分支结构带来的效率 |
| 5 | ResNet | He et al. | 2016 | CVPR | 残差连接；使超深网络可训练 |
| 6 | EfficientNet | Tan, Le | 2019 | ICML | 对深度/宽度/分辨率的复合缩放 |
| 7 | ConvNeXt | Liu et al. | 2022 | CVPR | 现代化的 CNN，性能比肩 Transformer |

## 检测与分割

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 8 | R-CNN | Girshick et al. | 2014 | CVPR | 将 CNN 特征用于检测 |
| 9 | Faster R-CNN | Ren et al. | 2015 | NeurIPS | 可学习的区域提议；两阶段检测范式模板；NeurIPS 2025 时间检验奖 |
| 10 | YOLO | Redmon et al. | 2016 | CVPR | 实时单阶段检测 |
| 11 | Focal Loss / RetinaNet | Lin et al. | 2017 | ICCV | 解决前景-背景类别不平衡 |
| 12 | DETR | Carion et al. | 2020 | ECCV | 将端到端（end-to-end）检测表述为集合预测 |
| 13 | FCN | Long et al. | 2015 | CVPR | 全卷积分割 |
| 14 | U-Net | Ronneberger et al. | 2015 | MICCAI | 编码器-解码器结构；医学影像标准 |
| 15 | Mask R-CNN | He et al. | 2017 | ICCV | 统一检测与实例分割 |

## Transformer、CLIP 与自监督

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 16 | Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | Transformer |
| 17 | ViT | Dosovitskiy et al. | 2021 | ICLR | 大规模下用于视觉的纯 Transformer |
| 18 | Swin Transformer | Liu et al. | 2021 | ICCV | 层次化移位窗口注意力 |
| 19 | CLIP | Radford et al. | 2021 | ICML | 对比式视觉-语言学习；零样本迁移 |
| 20 | MAE | He et al. | 2022 | CVPR | 掩码自编码器；可扩展的自监督学习（Self-Supervised Learning, SSL） |
| 21 | DINO | Caron et al. | 2021 | ICCV | 自蒸馏；涌现出分割能力 |
| 22 | DINOv2 | Oquab et al. | 2023 | TMLR | 通用自监督特征（DINOv3 于 2025 年进一步扩展规模；见关注清单） |
| 23 | MoCo | He et al. | 2020 | CVPR | 动量对比；大规模自监督学习 |
| 24 | SimCLR | Chen et al. | 2020 | ICML | 简洁的对比学习框架 |
| 25 | Mamba | Gu, Dao | 2023 | COLM 2024 | 线性时间的选择性状态空间模型 |

## 三维视觉与场景

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 26 | NeRF | Mildenhall et al. | 2020 | ECCV | 神经辐射场（NeRF）；新视角合成 |
| 27 | Instant-NGP | Müller et al. | 2022 | SIGGRAPH | 哈希编码；约 5 分钟完成 NeRF 训练 |
| 28 | 3D Gaussian Splatting | Kerbl et al. | 2023 | SIGGRAPH | 实时显式三维表示；在实践中取代了 NeRF |
| 29 | COLMAP | Schönberger, Frahm | 2016 | CVPR | 标准运动恢复结构（SfM）工具 |
| 30 | DUSt3R | Wang et al. | 2024 | CVPR | 点图回归；学习式多视图立体 |
| 31 | PointNet | Qi et al. | 2017 | CVPR | 直接在原始点云上进行深度学习 |

## 生成式视觉

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 32 | GAN | Goodfellow et al. | 2014 | NeurIPS | 对抗式生成 |
| 33 | StyleGAN | Karras et al. | 2019 | CVPR | 照片级真实、可控的图像合成 |
| 34 | DDPM | Ho et al. | 2020 | NeurIPS | 去噪扩散 |
| 35 | Latent Diffusion (SD) | Rombach et al. | 2022 | CVPR | 在潜空间中做扩散；Stable Diffusion |
| 36 | Classifier-Free Guidance | Ho, Salimans | 2022 | NeurIPS WS | 扩散模型的核心条件控制技术 |
| 37 | Sora（技术报告） | OpenAI | 2024 | — | 基于时空块的 DiT 视频生成 |

## 多模态与视觉-语言模型（VLM）

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 38 | Flamingo | Alayrac et al. | 2022 | NeurIPS | 少样本、图文交错的视觉-语言模型 |
| 39 | BLIP-2 | Li et al. | 2023 | ICML | 用 Q-Former 连接冻结的大语言模型 |
| 40 | LLaVA | Liu et al. | 2023 | NeurIPS | 视觉指令微调 |
| 41 | GLIP | Li et al. | 2022 | CVPR | 基于定位的语言-图像预训练 |

## 基础模型、VLA 与世界模型（2023–2026）

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| 42 | Segment Anything (SAM) | Kirillov et al. | 2023 | ICCV | 可提示分割 + 十亿级掩码数据引擎（演进脉络：SAM 2 → SAM 3 / SAM 3D，2025） |
| 43 | Depth Anything V2 | Yang et al. | 2024 | NeurIPS | 单目深度估计基础模型 |
| 44 | RT-1 | Brohan et al. | 2022 | RSS 2023 | 大规模机器人 Transformer |
| 45 | RT-2 | Brohan et al. | 2023 | CoRL | 视觉-语言-动作模型（Vision-Language-Action, VLA）：将网络知识迁移到机器人控制 |
| 46 | Open X-Embodiment | OXE Collaboration | 2023 | ICRA 2024 | 跨具身（cross-embodiment）数据集与 RT-X |
| 47 | π₀ | Black et al. | 2024 | arXiv (PI) | 基于流匹配（flow matching）的 VLA；高灵巧度 |
| 48 | Diffusion Policy | Chi et al. | 2023 | RSS | 在动作轨迹上做扩散的扩散策略（Diffusion Policy） |
| 49 | DreamerV3 | Hafner et al. | 2023 | ICLR 2024 | 基于世界模型（World Model）的强化学习；在 Minecraft 中获取钻石 |
| 50 | Genie | Bruce et al. | 2024 | ICML | 从视频中学习的生成式交互环境 |

---

## 关注清单：2025–2026 获奖论文与可能入选经典的工作

以下工作不计入五十篇之列，是 2026 年下半年博士生最可能被问到的近期论文。获奖论文标注了所获奖项；公司报告标注为非存档文献。

| # | 论文 | 作者 | 年份 | 会议/期刊 | 重要性 |
|---|-------|---------|------|-------|----------------|
| W1 | VGGT: Visual Geometry Grounded Transformer | Wang et al. (Oxford VGG / Meta) | 2025 | CVPR（最佳论文） | 以前馈方式在数秒内从一张到数百张视图中推断相机、深度、点图和轨迹；后继工作 VGGT-Omega（2026年5月） |
| W2 | DINOv3 | Siméoni et al. (Meta) | 2025 | arXiv 2508.10104 | 扩大规模的自监督学习，并获得高质量的稠密特征；新的默认冻结骨干网络（timm ≥ 1.0.20、transformers ≥ 4.56 已支持） |
| W3 | SAM 3: Segment Anything with Concepts | Carion et al. (Meta) | 2025 | arXiv 2511.16719 | 可提示的*概念*分割（文本或示例 → 所有实例），并附 SA-Co 基准；SAM 3.1 Object Multiplex（2026年3月） |
| W4 | SAM 3D: 3Dfy Anything in Images | SAM 3D Team (Meta) | 2025 | CVPR 2026（最佳论文荣誉提名） | 从单张图像重建三维形状、纹理和布局，借助人在回路的数据引擎训练 |
| W5 | Native and Compact Structured Latents for 3D Generation (TRELLIS.2) | Xiang et al. (Tsinghua / Microsoft / USTC) | 2025 | CVPR 2026（最佳学生论文） | O-Voxel 稀疏潜表示；40 亿参数的图像到三维模型，支持 PBR 材质；MIT 许可证 |
| W6 | Efficiently Reconstructing Dynamic Scenes One D4RT at a Time | Google DeepMind / Oxford VGG / UCL（含 Zisserman） | 2026 | CVPR（最佳论文） | 对视频编码一次，即可查询深度、点轨迹和相机位姿：统一的前馈式四维重建 |
| W7 | NitroGen: An Open Foundation Model for Generalist Gaming Agents | NVIDIA et al. | 2026 | CVPR（最佳论文荣誉提名） | 在 1,000 多款游戏、约 4 万小时游戏视频上进行“视频到动作”预训练 |
| W8 | Generating Physically Stable and Buildable Brick Structures from Text (BrickGPT) | Pun et al. (CMU) | 2025 | ICCV（Marr 奖） | 考虑物理约束的自回归三维生成，生成结果可实际搭建 |
| W9 | FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models | Kulikov et al. (Technion) | 2025 | ICCV（最佳学生论文） | 基于流匹配模型、无需训练、无需反演的文本驱动编辑 |
| W10 | Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat (HKTex) | Foti, Korkmaz, Zafeiriou, Birdal (Imperial) | 2026 | ECCV（最佳论文） | 用网格上的各向异性热核替代 UV 映射和泼溅来表示表面纹理 |
| W11 | Why Diffusion Models Don't Memorize | Bonnaire et al. | 2025 | NeurIPS（最佳论文） | 从训练动力学角度解释扩散模型何时泛化、何时记忆 |
| W12 | V-JEPA 2 | Assran et al. (Meta) | 2025 | arXiv | 自监督视频世界模型，支持动作条件规划；V-JEPA 2.1（2026年3月） |
| W13 | π₀.₅ | Physical Intelligence | 2025 | arXiv (PI) | 通过异构数据联合训练实现开放世界 VLA 泛化；权重在 openpi 中开放 |
| W14 | Genie 3 | Google DeepMind | 2025 | 博客（非存档） | 实时交互式世界模型，可保持数分钟一致性；据报道是 Waymo World Model（2026）的基础 |

---

## 优缺点（使用经典清单）

| 方面 | 优点 | 缺点 |
|--------|------|------|
| 精选经典 | 高效入门；共享术语体系 | 存在经典偏差；遗漏小众佳作 |
| 按主题组织 | 便于按兴趣延伸 | 跨领域工作难以归类 |
| 纳入新近工作 | 反映活跃前沿 | 2024–26 年的条目可能无法复现 |
| 基于奖项的关注清单 | 同行评选的质量信号 | 奖项奖励的是优雅与新颖，而非长期影响 |

---

## 开放问题与研究空白（元层面）

- **经典偏差。** 高被引的骨干网络论文挤占了重要的应用、数据集和理论工作的位置。
- **新近性与持久性。** 2024–2026 年的哪些工作会成为奠基之作尚不可知。
- **代表性不足的领域。** 公平性、鲁棒性、视频和低资源视觉在大多数经典清单中权重偏低。
- **可复现性。** 若干里程碑式成果（闭源 VLM/VLA）无法被独立验证。
- **地域/实验室集中。** 经典清单过度代表少数几个实验室。
- **综述空白。** 快速发展的子领域（VLA、世界模型）缺乏公认的权威综述。
- **继任者更替。** 基础模型的演进脉络（SAM → SAM 2 → SAM 3；DINOv2 → DINOv3；DUSt3R → VGGT → D4RT）如今在 12–18 个月内就会更替。读原始论文是为了理解思想，读最新版本是为了了解数字。
- **奖项与影响力。** 最佳论文的评选结果（如 CVPR/ICCV/ECCV 2025–2026）难以预测引用影响力。关注清单应每年重新审视。

---

## 延伸阅读

- [CVF Open Access](https://openaccess.thecvf.com/) — 清单中大多数论文的免费会议论文集
- [Semantic Scholar](https://www.semanticscholar.org/) — 引用图谱与影响力分析
- [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent) — 海量预印本
- [时间线概览](../01_history/00_timeline_overview.md) — 本清单的历史背景
- [最新进展](../12_research_frontier_2024_2026/00_overview_latest.md) — 2024–2026 年的前沿
- [CVPR 2026 闭幕新闻稿（奖项与统计）](https://cvpr.thecvf.com/Conferences/2026/News/Closing) — D4RT、TRELLIS.2、SAM 3D、NitroGen
- [CVF/TCPAMI 奖项列表](https://tc.computer.org/tcpami/awards/iccv-paper-awards/) — 历届 CVPR/ICCV/ECCV 论文奖
- [SAM 3 (arXiv 2511.16719)](https://arxiv.org/abs/2511.16719) · [DINOv3 (arXiv 2508.10104)](https://arxiv.org/abs/2508.10104) · [TRELLIS.2 (arXiv 2512.14692)](https://arxiv.org/abs/2512.14692) · [HKTex (arXiv 2609.07557)](https://arxiv.org/abs/2609.07557)
