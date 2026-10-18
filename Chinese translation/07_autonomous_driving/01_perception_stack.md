# 自动驾驶感知栈：BEV 感知、多相机融合与 3D 检测

> **英文原文：** [cv-knowledge-base/07_autonomous_driving/01_perception_stack.md](../../cv-knowledge-base/07_autonomous_driving/01_perception_stack.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：** [自动驾驶技术栈概览](./00_overview.md) · [占用预测](./02_occupancy_prediction.md) · [端到端驾驶](./03_end_to_end_driving.md) · [仿真与数据](./04_simulation_and_data.md)

> **🆕 2026年10月更新：**
> - 新增 **“2025–2026：基础主干网络、稀疏查询与驾驶基础模型中的感知”** 一节：DINOv3/SAM 3 作为主干网络和自动标注工具，Point Transformer V3 / Sonata 用于激光雷达，稀疏远距离检测器，以及 EMMA 式“以文本生成完成检测”。
> - 更新传感器之争：Waymo 第 6 代驾驶系统（13 个相机、4 个激光雷达、6 个毫米波雷达），Tesla 与小鹏的纯视觉 Robotaxi 分别处于运营与测试阶段，以及 Waymo 的生成式激光雷达仿真。
> - 修正 Waymo 传感器套件与里程数据；新增关键论文、开放问题和延伸阅读条目。

---

## 概述

现代自动驾驶车辆的感知栈将原始多模态传感器输入——主要是环视相机、机械旋转式激光雷达和角雷达——转换为适合下游预测与规划的结构化环境表征。十多年来，主导范式是*透视空间*目标检测：在每张相机图像中检测 2D 边界框，借助几何假设将其提升到 3D，并关联激光雷达点作为深度锚点 [Chen2016MV3D]。这种方法存在根本性局限：透视投影丢弃了度量深度信息，多相机关联需要显式的、考虑标定的融合，而得到的自车坐标系 3D 估计噪声较大。

向**鸟瞰图（Bird's-Eye View, BEV）**表征的转变，由 2020 年的 Lift-Splat-Shoot [Philion2020] 推动，并经 BEVFormer [Li2022] 和 BEVFusion [Liu2022] 走向成熟。它通过在统一的、以自车为参考的规范坐标系中构建自顶向下的空间特征图，基本解决了多相机融合问题。BEV 表征天然具有度量性——特征体中 $(x, y, z)$ 处的体素对应真实世界中的一个 3D 位置——因此可直接与地图坐标和规划算法兼容。此外，BEV 空间是异构传感器的天然融合目标：相机提取的语义特征和激光雷达几何信息可以在同一张量中融合，而无需针对特定传感器的后处理。

BEV 范式中的时序推理已成为第二个关键改进方向。BEVFormer [Li2022] 在对齐后的历史 BEV 网格上引入可变形注意力，使网络能够隐式估计速度、解决短时遮挡，并为远处或点数较少的目标累积证据。在典型高速公路条件下，利用 500 ms 的 BEV 历史，纯视觉系统的有效检测距离大约可翻倍。这种时序融合机制类似于经典的概率占用栅格，但作用于学习得到的特征表征，而非人工设计的传感器模型。

---

## Lift-Splat-Shoot (LSS)：奠基性的相机到 BEV 投影

LSS [Philion2020] 提出了第一个无需显式深度监督、即可将多相机图像特征投影到 BEV 的完全可微流水线。该架构分为三个阶段：

**Lift（提升）**：对于相机 $c$ 中的每个像素 $(u, v)$，由共享主干网络预测其在 $D$ 个离散深度区间上的深度分布 $\hat{\mathbf{d}} \in \mathbb{R}^D$。像素特征 $\mathbf{f}_{u,v}$ 沿深度射线广播，生成视锥点云：

$$\mathbf{F}_c = \left\{ \mathbf{f}_{u,v} \cdot \hat{d}_k \;\middle|\; u, v \in \text{pixels},\; k \in [D] \right\}$$

**Splat（泼溅）**：视锥中的每个点通过相机内参 $K$ 和外参 $T_{c \to \text{ego}}$ 投影到 3D，然后使用求和池化将其体素池化到 BEV 特征网格中（后来在 BEVDepth [Li2022BEVDepth] 中升级为基于 cumsum 的高效柱体池化）。

**Shoot（射出）**：BEV 特征图由特定任务的头部（分割、检测）使用。这一名称指的是其预期应用：在地图中“射出”规划轨迹。

LSS 证明了**隐式深度分布**可以仅凭 3D 检测监督学习得到，而无需激光雷达深度标签。其主要局限在于深度估计不精确：预测的深度分布较为弥散，导致 BEV 中出现特征“拖影”，降低了度量定位精度。BEVDepth [Li2022BEVDepth] 通过加入来自投影激光雷达点的显式深度监督作为辅助损失解决了这一问题，相比 LSS 基线将 nuScenes NDS 提高了约 4–6 个点。

---

## BEVFormer：面向 BEV 感知的时空 Transformer

BEVFormer [Li2022]（ECCV 2022，arXiv:2203.17270）为从多相机图像学习 BEV 表征引入了两种新的注意力机制：

**空间交叉注意力（Spatial Cross-Attention, SCA）**：在规则网格上定义一组可学习的 BEV 查询 $Q \in \mathbb{R}^{H \times W \times C}$。每个查询将其世界坐标通过各相机的投影矩阵投影得到 3D 参考点，并在所有相机图像的这些参考点处采样特征。这里使用可变形注意力 [Zhu2020]，使每个查询只关注参考点周围少量学习得到的偏移位置，从而实现高效的多尺度处理。

**时序自注意力（Temporal Self-Attention, TSA）**：利用自车运动变换 $\Delta T_{t-1 \to t}$ 对上一时刻的 BEV 特征 $B_{t-1}$ 进行变换对齐，再将当前 BEV 查询与之融合：

$$B_t^{\text{tmp}} = \text{Warp}(B_{t-1}, \Delta T_{t-1 \to t})$$

$$B_t = \text{SelfAttn}(Q_t, [B_t^{\text{spa}}, B_t^{\text{tmp}}])$$

这种循环式 BEV 传播无需显式时序建模模块，即可实现隐式速度估计和遮挡恢复。

**主要结果**：BEVFormer 在 nuScenes 测试集（纯视觉）上达到 **56.9% NDS** 和 **48.1% mAP**，比此前最佳纯视觉方法高 9.0 个 NDS 点，并与当时许多纯激光雷达基线持平。BEVFormer v2 [Yang2023BEVFormerV2] 通过加入透视视角 3D 检测辅助任务扩展了该架构，进一步提升了性能。

---

## BEVFusion：统一的激光雷达-相机 BEV 融合

BEVFusion [Liu2022]（MIT，ICRA 2023，arXiv:2205.13542）认为，激光雷达-相机融合的关键挑战在于**在统一表征空间中进行特征级对齐**，而非结果级融合（后融合）。其方法如下：

1. **相机分支**：通过 LSS 衍生的体素提升，将多相机图像特征投影到 BEV。
2. **激光雷达分支**：使用基于体素的编码器（VoxelNet/PointPillars 主干）处理点云。
3. **融合**：在同一网格中拼接相机 BEV 和激光雷达 BEV 特征图，然后使用卷积融合头进行联合检测和地图分割。

关键洞见在于，激光雷达 BEV 特征与相机 BEV 特征在同一度量空间中**天然对齐**，从而避免了前融合（原始传感器对齐）或后融合（独立 3D 框合并）中的几何不一致。作者报告：
- **nuScenes 验证集上 73.4% NDS**
- 在测试集上 **mAP 和 NDS 均比 TransFusion 高 1.3%**（此前的 SOTA）
- BEV 地图分割 **mIoU 高 13.6%**
- 在精度相当时，**MACs 比 TransFusion 降低 1.9 倍**

北京理工大学的一项同期工作也名为 BEVFusion [Liang2022]，提出了类似思路；MIT 版本是被引用更广泛的版本。

---

## 激光雷达-相机融合 vs. 纯视觉：Tesla 之争

量产自动驾驶行业在传感器路线上存在分歧：

**纯视觉（Tesla）**：Tesla 认为人类驾驶员仅靠视觉驾驶，因此只要训练数据规模足够，相机就已足够。FSD v12 通过大型 Transformer 主干处理八路相机输入，任何阶段都不使用激光雷达。该系统已在 **1,000 万个视频片段**上训练 [Tesla2024]，并展现出涌现的泛化能力。其关键工程优势在于成本：一个量产激光雷达会使物料成本增加 200 至 2,000 美元以上。其关键科学弱点是**度量深度歧义**：任何单目深度估计都存在固有的尺度歧义，只能通过多帧时序融合或几何先验来消除，这在深度快速变化的场景中（如加塞、急停）会引入失效模式。

**以激光雷达为主（Waymo）**：Waymo 第 5 代驾驶系统（Jaguar I-PACE）配备 5 个激光雷达、29 个相机和 6 个毫米波雷达。第 6 代驾驶系统（2024 年发布，搭载于极氪和 Hyundai Ioniq 5 平台）将其减少为 13 个相机、4 个激光雷达和 6 个毫米波雷达，在保持重叠冗余的同时降低成本。激光雷达以 10 Hz 提供无歧义的度量 3D 坐标和 360° 覆盖；FMCW 激光雷达还能为每个点提供径向速度。Waymo 公开的安全数据从 710 万英里纯乘客（无安全员）里程 [Waymo2023Safety] 增长到 2025 年 6 月的约 9,600 万英里。2026 年，Waymo 每周运营约 50 万次付费出行。随着车规级激光雷达价格下降（尤其是禾赛 Hesai、速腾聚创 RoboSense 等中国供应商），激光雷达在成本与机械可靠性方面的历史劣势已逐渐减弱。

**2026 年的争论现状。** 问题已从 *“相机的感知能力够不够？”* 转变为 *“纯视觉技术栈能否在车队规模上达到无人驾驶级可靠性？”*。Tesla 于 2026 年 1 月在 Austin 开始提供无安全员 Robotaxi 出行，小鹏的 VLA 2.0 Robotaxi（无激光雷达、无高精地图）于 2026 年 3 月进入公开道路测试。Waymo、Zoox、百度、Pony.ai 和 WeRide 都保留了激光雷达。目前尚无纯视觉运营方发布可与 Waymo 相比的无人驾驶安全数据。

nuScenes 上的基准证据清楚表明，在架构复杂度相当的情况下，激光雷达-相机融合比纯视觉方案高出 **15–20 个 NDS 点**，这说明度量深度确实是对感知有用的信号。不过，随着更大规模训练数据推动相机深度估计的改进，这一差距可能会缩小。

---

## 3D 目标检测：PointPillars 与 CenterPoint

### PointPillars [Lang2019]

PointPillars（CVPR 2019）将 3D 点云组织为竖直的*柱体*（沿 $z$ 方向高度无限的列），而非 3D 体素，从而将 3D 卷积简化为 2D 运算：

$$\text{Pillar feature} = \text{PointNet}(\{p_i - \bar{p} \mid p_i \in \text{pillar}\})$$

柱体伪图像由 2D 主干网络（类似 SSD/SECOND）处理以完成检测。PointPillars 在保持合理精度的同时实现了实时性能（单 GPU 约 62 Hz）。在 Waymo Open Dataset 上，PointPillars 在 Level 1 达到 63.3 / 62.7（mAP / mAPH），在 Level 2 达到 55.2 / 54.7。它仍是标准基线，并在许多量产系统中被用作主要检测主干。

### CenterPoint [Yin2021]

CenterPoint（CVPR 2021，arXiv:2006.11275）受 CenterNet 启发，用**中心热力图**范式取代了基于锚框的边界框回归。该检测器：

1. 通过 VoxelNet 或 PointPillars 主干将点云编码为 BEV 特征图。
2. 以高斯热力图形式检测目标中心：$\hat{Y} \in \mathbb{R}^{H \times W \times K}$，其中 $K$ 为类别数。
3. 从检测到的中心回归偏移、高度、尺寸和旋转。
4. 通过简单的基于速度的卡尔曼滤波器传播中心轨迹（CenterPoint 两阶段版本在裁剪特征上增加了第二阶段细化）。

CenterPoint 在 **Waymo 上达到 71.9 mAPH**（单模型、两阶段），在 Level 2 上精度几乎是 PointPillars 的**两倍**（行人 + 车辆合计 66.4 mAPH）。它在多届 Waymo 挑战赛的纯激光雷达提交中名列第一。CenterPoint 如今已是事实上的标准激光雷达检测头，被用作 BEVFusion 及许多其他多模态架构中的激光雷达分支。

---

## 时序融合与远距离感知

除了 BEVFormer 的循环式 BEV 传播外，研究者还探索了多种时序聚合策略：

- **BEV 特征堆叠**：拼接多个经变换对齐的 BEV 帧 [BEVDet4D]，简单而有效。
- **Transformer 时序注意力**：在当前与过去 BEV 帧之间进行带学习对齐的交叉注意力，见于 DETR3D 和 PETRv2。
- **占用流**：预测每个体素的速度向量以变换未来占用预测，从而实现如 UniAD [Hu2023UniAD] 中的时空联合预测。

远距离感知（>100 m 的相机检测）需要仔细处理透视压缩和量化伪影。SparseBEV 和 StreamPETR 表明，基于稀疏查询的方法比稠密 BEV 网格方法更易扩展到更远距离，后者需要代价过高的精细空间分辨率。

---

## 2025–2026：基础主干网络、稀疏查询与驾驶基础模型中的感知

**视觉基础主干网络。** 自监督主干网络如今主导了模型初始化。DINOv2 和 **DINOv3**（Meta，2025 年 8 月；一个采用 Gram 锚定以获得稠密特征的 70 亿参数教师模型，并蒸馏为 ViT/ConvNeXt 变体）为深度估计、分割和 BEV 提升提供了强大的冻结稠密特征。驾驶专用版本通常通过在车队视频上继续预训练获得。可提示的开放词汇（open-vocabulary）分割器（**SAM 3**，Meta，2025 年 11 月，可分割某一名词短语概念的所有实例）主要在**离线**场景中使用，作为长尾类别（散落物、施工设备、动物）的自动标注工具，而非部署在车端。

**激光雷达主干网络。** Point Transformer V3 [Wu2024PTv3]（CVPR 2024）以序列化（空间填充曲线）点排序取代精确的邻域搜索，在 nuScenes 和 Waymo 的分割与检测上取得了强劲结果，同时吞吐量大幅提高。**Sonata** [Wu2025Sonata]（CVPR 2025）在 PTv3 基础上提供自监督点云预训练，相当于“点云版 DINO”，可用于标签高效的激光雷达感知。

**稀疏与远距离相机检测。** 稠密 BEV 范式正让位于**完全稀疏查询**检测器（SparseBEV、Sparse4D v3、StreamPETR），它们避免了 $O(HW)$ 的 BEV 网格。Far3D [Jiang2024Far3D]（AAAI 2024）通过从 2D 检测和深度先验生成 3D 查询，在 Argoverse 2 上将相机检测范围扩展到约 150 m。稀疏查询也能自然地与端到端规划器（SparseDrive、DiffusionDrive）集成。

**作为驾驶基础模型副产品的感知。** Waymo 的 EMMA [Hwang2024EMMA] 通过基于 Gemini 的多模态大模型*以文本形式生成边界框坐标*来完成纯视觉 3D 检测，并在 Waymo Open Dataset 上具有竞争力。对规划、检测和道路图估计的协同训练提升了三者的表现。量产 VLA 和世界行为技术栈（小鹏 VLA 2.0、华为 ADS 4、Tesla FSD v14）不再将经典的边界框列表作为主要接口；显式感知输出主要作为辅助头和可视化而保留。在仿真方面，**Waymo World Model**（构建于 Genie 3 之上，2026 年 2 月）可*同时*生成相机视频和 Waymo 专属的激光雷达回波，从而能在合成的长尾场景上对感知进行压力测试（参见[仿真与数据](./04_simulation_and_data.md)）。

---

## 关键论文

| 论文 | 作者 | 年份 | 发表平台 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| Lift, Splat, Shoot | Philion & Fidler | 2020 | ECCV | 通过深度分布实现可微的相机到 BEV 投影；奠基性的 LSS |
| BEVFormer | Li et al. | 2022 | ECCV | 空间交叉注意力 + 时序自注意力；纯视觉 56.9% NDS |
| BEVFusion (MIT) | Liu et al. | 2022 | ICRA 2023 | 统一的激光雷达-相机 BEV 融合；73.4% NDS |
| BEVDepth | Li et al. | 2022 | AAAI 2023 | 显式深度监督；相比纯 LSS 显著提升 NDS |
| PointPillars | Lang et al. | 2019 | CVPR | 基于柱体伪图像的实时激光雷达检测 |
| CenterPoint | Yin et al. | 2021 | CVPR | 中心热力图 3D 检测；Waymo 上 71.9 mAPH |
| PETR | Liu et al. | 2022 | ECCV | 用于纯视觉检测的 3D 位置编码，无需 BEV 网格 |
| StreamPETR | Wang et al. | 2023 | ICCV | 通过稀疏传播查询实现长时序融合 |
| Far3D | Jiang et al. | 2024 | AAAI | 通过 2D 引导的 3D 查询实现远距离（约 150 m）相机 3D 检测 |
| Point Transformer V3 | Wu et al. | 2024 | CVPR | 序列化点 Transformer；快速、可扩展的激光雷达主干 |
| Sonata | Wu et al. | 2025 | CVPR | 基于 PTv3 的自监督点云预训练 |
| EMMA | Hwang et al. (Waymo) | 2024 | arXiv 2410.23262 | 基于多模态大模型的驾驶模型；以文本生成实现纯视觉 3D 检测 |
| DINOv3 | Siméoni et al. (Meta) | 2025 | arXiv | 70 亿参数自监督视觉主干，稠密特征强；广泛用于冻结特征感知 |

---

## 基准性能

| 模型 | 数据集 | 指标 | 分数 | 备注 |
|-------|---------|--------|-------|-------|
| BEVFormer | nuScenes test | NDS | 56.9% | 纯视觉；ECCV 2022 |
| BEVFormer v2 | nuScenes test | NDS | ~63.4% | 纯视觉 + 透视视角检测头 |
| BEVFusion (MIT) | nuScenes test | NDS | 73.9% | 激光雷达 + 相机 |
| BEVDepth | nuScenes val | NDS | ~60.0% | 纯视觉 + 激光雷达深度监督 |
| CenterPoint | Waymo val | mAPH L2 | ~66.4% | 纯激光雷达（车辆 + 行人） |
| PointPillars | Waymo val | mAP L1 | 63.3% | 纯激光雷达；实时基线 |

---

## 优缺点

| 方法 | 优点 | 缺点 |
|--------|------|------|
| 纯视觉 BEV（LSS/BEVFormer） | 硬件成本低；语义特征稠密；随规模提升 | 度量深度歧义；夜间/雨天性能下降 |
| 纯激光雷达（CenterPoint、PointPillars） | 精确的度量 3D；可通过多普勒获取速度；对光照鲁棒 | 无语义纹理；硬件昂贵；远距离回波稀疏 |
| 激光雷达-相机融合（BEVFusion） | 精度最佳；模态互补；基准上达到最先进水平 | 系统复杂度高；对标定敏感；成本高 |

---

## 开放问题与研究空白

- **深度估计质量**：纯视觉 NDS 比激光雷达-相机方案落后约 15 个点；借助规模、自监督和雷达集成来弥合这一差距是活跃的前沿方向。
- **实时 BEV 分辨率权衡**：高分辨率 BEV 网格（例如在 100 m 范围内 0.1 m/体素）在计算上不可行；需要高效的稀疏表征。
- **跨域泛化**：在某一地域训练的模型（例如 nuScenes：Boston + 新加坡）在未见过的城市中性能显著下降；域自适应对部署至关重要。
- **恶劣天气鲁棒性**：雾、雨和雪会同时削弱所有传感器模态；学习型传感器退化模型尚处于起步阶段。
- **罕见目标类别**：标准基准聚焦于车辆、行人、骑行者；对施工设备、应急车辆和新型障碍物的检测缺乏充分的基准评测。
- **4D 雷达集成**：新兴的 4D 成像雷达可提供稠密、全天候的点云；如何在 BEV 中将其与相机和激光雷达最优融合仍是开放研究问题。
- **远距离检测**：用相机在 150–300 m 处检测和分类目标仍具挑战性；该距离范围对高速汇入和高速场景至关重要。
- **VLA 技术栈中的感知可观测性**：当感知隐含在 VLA 或世界行为模型中时，标准感知指标（mAP、NDS）已无法衡量规划器“看到”了什么。用于审计隐式感知的探测方法和辅助头方法尚不成熟。
- **基础模型自动标注质量**：开放词汇自动标注工具（SAM 3、VLM）可扩展长尾标注，但其系统性错误可能传播到训练数据中；车队规模的标签噪声审计仍未解决。

---

## 延伸阅读

- [arXiv:2203.17270 BEVFormer](https://arxiv.org/abs/2203.17270) — 官方论文
- [arXiv:2205.13542 BEVFusion MIT](https://arxiv.org/abs/2205.13542) — 统一 BEV 融合
- [arXiv:2006.11275 CenterPoint](https://arxiv.org/abs/2006.11275) — 基于中心的 3D 检测
- [nuScenes 3D 检测排行榜](https://nuscenes.org/object-detection) — 实时基准
- [Waymo Open Dataset 挑战赛](https://waymo.com/open/challenges/) — 感知挑战赛结果
- [arXiv:2401.06542 鲁棒性感知 3D 检测综述](https://arxiv.org/abs/2401.06542) — 自动驾驶检测鲁棒性综述
- [Pointcept / Point Transformer V3 GitHub](https://github.com/Pointcept/PointTransformerV3) — PTv3 与 Sonata 代码
- [arXiv:2410.23262 EMMA](https://arxiv.org/abs/2410.23262) — 在单一多模态大模型中完成检测与规划
- [Meta DINOv3](https://ai.meta.com/dinov3/) — 自监督视觉主干网络（2025年8月）
- [Waymo 第 6 代驾驶系统博客](https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/) — 传感器套件重新设计
