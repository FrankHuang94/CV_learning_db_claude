# 计算机视觉：基础与概述

> **英文原文：** [cv-knowledge-base/00_foundations/00_overview.md](../../cv-knowledge-base/00_foundations/00_overview.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 基础
> **相关章节：** [图像形成](./01_image_formation.md) · [CV 时间线](../01_history/00_timeline_overview.md) · [特征工程](./03_feature_engineering.md) · [CNN 架构](../03_architectures/00_cnn_architectures.md)

> **🆕 2026年10月更新：**
> - “手工设计 → 学习”的叙事延续到了 2025–2026 年的基础模型（Foundation Model）：用于自监督稠密特征的 DINOv3（2025年8月）、用于开放词汇（open-vocabulary）“可提示概念分割”的 SAM 3（2025年11月；SAM 3.1 于2026年3月发布），以及前馈式三维几何网络（VGGT，CVPR 2025 最佳论文；MapAnything；Depth Anything 3）。
> - 关键论文表与“视觉分析层次”分类表新增 DINOv3、SAM 3 和 VGGT。
> - 针对前馈式几何重新审视了 Marr 框架（计算层目标沿用经典视觉，算法层改为学习得到），并新增一条关于学习式几何可靠性保证的开放问题。

---

## 概述

计算机视觉（Computer Vision, CV）是一门致力于让机器解释和理解来自世界的视觉信息的科学学科。这些视觉信息包括图像、视频以及其他编码空间与光度结构的传感器模态。从本质上讲，CV 求解的是一个固有的不适定逆问题：给定一个二维投影（即图像），恢复其背后三维场景的属性，包括几何、材质构成、光照和语义。之所以不适定，是因为无穷多种三维构型都可能产生同一幅图像；要消解这种歧义，需要先验、约束，以及（在现代系统中）从海量数据中提取出的统计规律。

计算机视觉的范围涵盖低层信号处理（降噪、去模糊）、中层场景分析（边缘检测、分割、深度估计）和高层语义推断（物体识别、动作理解、视觉问答）。它汲取了射影几何、线性代数、概率论、最优化，以及自 2000 年代后期以来日益重要的深度学习等领域的成果。该领域与机器人、自动驾驶、医学影像、遥感和增强现实交叉，是人工智能中实际影响最为深远的分支之一。

要严谨地理解 CV，既需要掌握描述图像形成的数学模型，也需要掌握对其求逆的统计/计算机制。这些描述层次之间的区别，即*需要计算什么*与*如何计算*，由 David Marr 著名地加以系统化。即便在端到端（end-to-end）学习系统占主导的时代，这一区分仍是有用的概念框架。

---

## 视觉处理流水线

现代 CV 系统可以理解为从原始传感器数据到高层推断的一连串变换。尽管深度网络模糊了其中许多边界，但这种概念上的分解仍然很有启发性：

```
世界（三维场景）
      │  光照、几何、材质
      ▼
图像形成（Image Formation）        ← 光学、传感器、数字化
      │  I(x,y)：像素强度
      ▼
预处理（Pre-processing）           ← 去噪、色彩校正、归一化
      │
      ▼
特征提取（Feature Extraction）     ← 边缘、角点、斑点或学习得到的嵌入
      │
      ▼
中层分析（Mid-level Analysis）     ← 分割、光流、立体视觉、深度
      │
      ▼
高层推断（High-level Inference）   ← 识别、检测、三维重建、图像描述
      │
      ▼
任务输出（Task Output）            ← 边界框、位姿、描述文本、点云
```

每个阶段都会引入假设与近似。**图像形成**阶段（见 [01_image_formation.md](./01_image_formation.md)）通过相机的光学与电子系统，将三维辐射转换为二维离散阵列。**预处理**为下游算法对原始信号进行规范化。**特征提取**在历史上依赖手工设计的算子（Sobel、SIFT、HOG）；现代系统则用基于卷积或 Transformer 的学习特征来替代或增强它们。**中层分析**恢复几何与运动线索，为**高层推断**提供信息，后者产生语义标签、三维结构或语言描述。

---

## Marr 的三个分析层次

关于视觉计算最有影响力的理论框架，由 David Marr 在其身后出版的专著 *Vision*（MIT Press，1982）中提出。Marr 认为，任何信息处理系统（无论是生物的还是人工的）都应在三个不同且在很大程度上相互独立的层次上加以理解：

1. **计算层（Computational level）**：计算的目标是*什么*，以及*为什么*这个目标是恰当的？这一层关注被计算的函数、要解决的问题，以及使求解成为可能的约束条件。以立体视觉为例，其计算理论表述为：给定从略微不同视点拍摄的同一场景的两幅图像，利用双目视差恢复深度图。这一计算的逻辑源自投影几何以及场景刚性的假设。

2. **算法层（Algorithmic level）**：计算是*如何*执行的？输入和输出采用何种表示，又由何种过程将前者变换为后者？对于立体视觉，算法必须规定匹配准则（例如归一化互相关、互信息或基于特征的匹配）、搜索策略（例如沿极线约束搜索），以及用于填补被遮挡区域的正则化方案。

3. **实现层（Implementational level）**：*算法如何在物理上实现？* 在生物视觉中，这涉及神经回路；在人工系统中，这涉及硬件、内存布局和数值精度。一个经过 GPU 优化的立体块匹配实现，与一种基于脉冲的神经形态方法在实现上差别巨大，但二者可以实例化同一个算法。

Marr 的关键洞见在于这些层次*相对独立*：人们可以在不了解算法的情况下理解计算目标，也可以在不了解实现的情况下理解算法。在实践中，各层次之间存在相互作用：硬件约束影响哪些算法可行，进而影响问题的表述方式。尽管如此，该框架阐明了为什么看似迥异的系统（人类视觉皮层、传统 CV 流水线和深度网络）能够解决相同的感知问题，而其机制却截然不同。

---

## 从手工设计表示到学习表示

在该领域的大部分历史中（大约 1960–2010 年），CV 流水线由手工设计的组件构建而成。研究者手动设计特征检测器（Canny，1986；Harris & Stephens，1988）、描述子（SIFT，Lowe 2004；HOG，Dalal & Triggs 2005）和分类器（SVM、提升级联）。这种方法需要深厚的领域专业知识，所得系统也较为脆弱，跨领域泛化能力差。

**深度学习革命**以 AlexNet 在 2012 年 ImageNet 上的决定性胜利为催化剂，将范式转向**学习表示**。卷积神经网络（CNN）直接从带标注数据中学习层次化特征：低层检测边缘与纹理，高层编码语义概念。这种数据驱动的方法消除了手工特征工程，并在几乎所有 CV 基准上带来了显著的精度提升。

随后的十年中，Transformer（最初来自 NLP）通过视觉 Transformer（ViT，Dosovitskiy 等，2020）进入 CV 领域，实现了对图像块的全局自注意力，并进一步提升了性能，尤其是在大规模预训练条件下。对比学习与自监督学习（Self-Supervised Learning, SSL）方法（SimCLR、MoCo、DINO、MAE）随后证明，无需人工标注即可学到强大的视觉表示；视觉-语言模型（CLIP、Flamingo、GPT-4V）则学会了在大规模上对齐视觉与文本语义。

**2025–2026：基础模型规模的视觉已延伸到流水线的每个阶段。** 以下三类发布说明了学习表示已深入到曾经专属于手工设计模块的领域：

- **自监督稠密特征。** DINOv3 [Simeoni2025] 将 DINO 式自蒸馏扩展到一个 7B 参数的 ViT，在一个经过筛选的 17 亿图像语料库（LVD-1689M）上训练，并引入了 *Gram 锚定（Gram anchoring）*：一种在长时间训练中保持图像块级特征相似性结构稳定的正则项，用以阻止早期大规模训练中出现的稠密特征图退化。7B 教师模型被蒸馏为 ViT-S/B/L/H+ 和 ConvNeXt 学生模型，另外还发布了在卫星影像（SAT-493M）上训练的变体。这些冻结特征可作为通用的低/中层描述子（见[特征工程](./03_feature_engineering.md)）。
- **开放词汇分割。** SAM 3 [Carion2025] 将可提示分割范式从“分割*这个*物体”（点/框提示）扩展到*可提示概念分割（Promptable Concept Segmentation）*：给定一个简短名词短语或图像示例，在图像和视频中检测、分割并跟踪该概念的*每一个*实例。它在共享主干网络（约 8.48 亿参数）上结合了检测器和基于记忆的跟踪器，并在新的 SA-Co 基准（约 27 万个不同概念）上进行评测。SAM 3.1（2026年3月）加入了更快的多目标联合跟踪。
- **前馈式几何。** VGGT [Wang2025]（CVPR 2025 最佳论文）在单次前向传播中，为一到数百个视图预测相机内参/外参、深度图、点图和点轨迹（作者报告典型场景的重建耗时不到一秒）。后续工作如 MapAnything [Keetha2025]（度量尺度输出，可选输入内参/位姿/深度）和 Depth Anything 3（ByteDance Seed，2025年11月；在朴素的 DINO 式 Transformer 上使用单一的深度-射线表示）沿着同一思路继续推进。值得注意的是，VGGT 的输出仍可导出到 COLMAP 并通过光束法平差加以优化：[经典几何](./04_classical_geometry.md)中的经典机制由此成为一个优化与验证层，而不再是整个流水线。

用 Marr 的术语来说，这些系统保留了经典视觉的*计算层*规范（从图像中恢复深度、位姿和物体范围），但用一种学习得到的、摊销式的推断过程取代了手工设计的*算法层*。

理解*经典*流水线（手工特征、显式几何模型、基于优化的推断）仍然至关重要，原因有以下几点：（1）它提供可解释、具有理论基础的模型，其失效模式是已知的；（2）经典几何是现代三维重建、SLAM 和相机标定流水线的基础；（3）许多对部署要求严格的系统（自动驾驶传感器、医疗设备）采用混合流水线；（4）经典方法常常作为学习系统的目标或先验。从手工设计到学习表示的转变，是 Marr 意义上*算法层*的变化；*计算层*的目标和*物理*约束保持不变。

---

## 关键论文

| 论文 | 作者 | 年份 | 发表场所 | 关键贡献 |
|-------|---------|------|-------|-----------------|
| *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information* | D. Marr | 1982 | MIT Press（专著） | 视觉计算的三层次框架；初始草图（primal sketch）、2.5D 草图、三维模型 |
| "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet) | A. Krizhevsky, I. Sutskever, G. Hinton | 2012 | NeurIPS | 证明深度 CNN 在大规模识别上决定性地超越手工设计流水线 |
| "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (ViT) | A. Dosovitskiy et al. | 2020 | ICLR 2021 | 将 Transformer 自注意力直接应用于图像块；无需卷积即可获得全局上下文 |
| "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) | A. Radford et al. | 2021 | ICML | 对比式视觉-语言预训练，实现零样本视觉识别 |
| "Masked Autoencoders Are Scalable Vision Learners" (MAE) | K. He et al. | 2022 | CVPR | 通过重建被遮蔽图像块进行自监督 ViT 预训练 |
| "Rich Feature Hierarchies for Accurate Object Detection" (R-CNN) | R. Girshick et al. | 2014 | CVPR | 将经典区域候选与 CNN 特征结合；开启现代目标检测 |
| "Fully Convolutional Networks for Semantic Segmentation" | J. Long, E. Shelhamer, T. Darrell | 2015 | CVPR | 通过全卷积架构实现端到端逐像素分类 |
| "VGGT: Visual Geometry Grounded Transformer" | J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny | 2025 | CVPR（最佳论文） | 单个前馈网络从多视图预测相机、深度、点图和轨迹 |
| "DINOv3" | O. Siméoni et al. | 2025 | arXiv 2508.10104 | 采用 Gram 锚定的 7B 自监督 ViT；高质量稠密特征；蒸馏模型家族 |
| "SAM 3: Segment Anything with Concepts" | N. Carion et al. | 2025 | arXiv 2511.16719 | 图像与视频中的可提示概念分割；SA-Co 基准 |

---

## 关键公式：一般性求逆问题

计算机视觉从根本上说是图像形成过程的求逆。设 $\mathbf{x} \in \mathbb{R}^3$ 为三维场景点，$\mathbf{I}$ 为观测图像，$\mathcal{F}$ 为前向成像模型：

```latex
\mathbf{I} = \mathcal{F}(\mathbf{x}, \boldsymbol{\theta}) + \boldsymbol{\eta}
```

其中 $\boldsymbol{\theta}$ 编码相机参数、光照和材质属性，$\boldsymbol{\eta}$ 为观测噪声。目标是从 $\mathbf{I}$ 中恢复 $\mathbf{x}$（可能还包括 $\boldsymbol{\theta}$）。这一逆问题是不适定的（Hadamard，1902）：解可能不存在、可能不唯一，或可能不连续依赖于数据。因此正则化，即编码关于场景的先验知识，是必不可少的：

```latex
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \; \underbrace{\|\mathbf{I} - \mathcal{F}(\mathbf{x}, \boldsymbol{\theta})\|^2}_{\text{data fidelity}} + \lambda \underbrace{\mathcal{R}(\mathbf{x})}_{\text{regularizer}}
```

（data fidelity：数据保真项；regularizer：正则项。）在贝叶斯表述中，正则项对应于对数先验 $-\log p(\mathbf{x})$，因此在高斯似然下，最大后验（MAP）估计等价于正则化最小二乘。

---

## 视觉分析层次（分类）

| 层次 | 目标 | 经典算子 | 学习式对应方法 |
|-------|------|-------------------|-------------------|
| 低层 | 信号清理、基本结构 | Gaussian 滤波、Sobel、Canny | 前几层卷积层 |
| 中层 | 几何/结构分组 | Hough 变换、RANSAC、光流 | FPN、可变形卷积 |
| 高层 | 语义理解 | SVM + HoG、词袋模型 | ResNet、ViT、DINO/DINOv3、SAM 3 |
| 三维 / 场景 | 几何、深度、运动 | 对极几何、SfM | NeRF、3DGS、DUSt3R、VGGT、MapAnything |

---

## 开放问题与研究空白

- **鲁棒性与分布偏移。** 在精心整理的基准上训练的深度模型，在分布外输入（天气变化、对抗扰动、传感器伪影）上会出现灾难性失效。开发可证明鲁棒的 CV 系统仍是一个开放问题。
- **数据效率与小样本泛化。** 生物视觉能从极少样本中实现出色的泛化；当前的学习系统所需数据量要多出几个数量级。
- **因果与组合式理解。** 模型擅长基于相关性的模式匹配，但在因果推理上表现不佳，例如预测物体被移动、移除或遮挡后会发生什么。
- **统一几何与语义。** 以有原则的端到端框架，将经典几何三维重建（SfM、SLAM）与语义场景理解结合起来，仍是一个活跃的研究挑战。
- **可解释性与失效模式理解。** 即使是最先进的网络在很大程度上仍不透明；理解模型*为何*成功或失败，对于安全攸关的部署至关重要。
- **大规模高效推理。** 视觉基础模型（SAM/SAM 3、CLIP、DINOv3-7B、GPT-4V 级别的 VLM）计算开销巨大；在不损失能力的前提下将其压缩以部署到边缘设备，是一个尚未解决的工程与算法挑战。蒸馏模型家族（例如 DINOv3 从 ViT-S 到 ViT-H+ 的学生模型）只是部分答案。
- **学习式几何的可靠性保证。** 前馈式三维模型（VGGT、MapAnything、Depth Anything 3）如今在许多基准上已可与基于优化的流水线相媲美，但它们没有与多视图几何中的可辨识性和退化性结论相对应的理论结果；何时可以信任其相机与深度预测，以及如何最好地将其与光束法平差结合，仍是开放问题。
- **具身与主动感知。** 大多数 CV 研究处理的是被动识别；将其扩展到能够控制自身传感器、在环境中移动以降低不确定性的*主动*智能体，仍处于起步阶段。

---

## 延伸阅读

- [Marr, D. (1982). *Vision*. MIT Press.](https://direct.mit.edu/books/monograph/3299/VisionA-Computational-Investigation-into-the-Human)：经典的理论基础。
- [Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press.](https://www.robots.ox.ac.uk/~vgg/hzbook/)：几何 CV 的权威参考书。
- [Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed. Springer.](https://szeliski.org/Book/)：全面的现代教科书（可在线免费获取）。
- [LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.](https://www.nature.com/articles/nature14539)：深度学习革命概览。
- [Dosovitskiy, A. et al. (2021). An image is worth 16×16 words. *ICLR 2021*.](https://arxiv.org/abs/2010.11929)：视觉 Transformer。
- [McClamrock, R. Marr's Three Levels: A Re-evaluation.](https://www.albany.edu/~ron/papers/marrlevl.html)：对 Marr 框架的批判性哲学分析。
- [Siméoni, O. et al. (2025). DINOv3. arXiv:2508.10104.](https://arxiv.org/abs/2508.10104)：采用 Gram 锚定的自监督基础特征；代码与权重见 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)。
- [Carion, N. et al. (2025). SAM 3: Segment Anything with Concepts. arXiv:2511.16719.](https://arxiv.org/abs/2511.16719)：可提示概念分割；代码见 [facebookresearch/sam3](https://github.com/facebookresearch/sam3)。
- [Wang, J. et al. (2025). VGGT: Visual Geometry Grounded Transformer. *CVPR 2025*.](https://arxiv.org/abs/2503.11651)：前馈式多视图几何；CVPR 2025 最佳论文。
