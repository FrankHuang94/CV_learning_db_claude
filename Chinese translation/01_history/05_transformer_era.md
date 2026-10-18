# Transformer 时代（2020–2023）

> **英文原文：** [cv-knowledge-base/01_history/05_transformer_era.md](../../cv-knowledge-base/01_history/05_transformer_era.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 基础
> **相关章节：**
> - [生成时代](./04_generative_era.md)
> - [视觉 Transformer](../03_architectures/01_vision_transformers.md)
> - [自监督学习](../03_architectures/03_self_supervised_learning.md)
> - [基础模型时代](./06_foundation_model_era.md)

> **🆕 2026年10月更新：**
> - 补充了自蒸馏路线在 2025 年的延续：**DINOv3**（2025 年 8 月）将 DINO 扩展到 70 亿参数的 ViT，并用 Gram 锚定解决了稠密特征退化问题。
> - 指出纯 ViT 主干如今支撑着前馈式三维/四维重建（VGGT，CVPR 2025 最佳论文；D4RT，CVPR 2026 最佳论文），将该时代"一种算子通吃"的论点扩展到了几何领域。
> - 更新了 Mermaid 谱系图、关键论文（DINOv3）和延伸阅读。

---

## 概述

Transformer 时代标志着注意力机制——自 2017 年起主导 NLP——向计算机视觉的迁移，随之而来的是从特定于架构的归纳偏置转向通用的、可扩展的序列模型。视觉 Transformer 在 2020 年提出的关键论断是：卷积并非必不可少——给定足够的预训练数据，作用于图像块的纯 Transformer 能够达到甚至超过最好的 CNN。这不仅仅是一次架构替换。它将视觉与语言统一在同一个计算基元（注意力）之下，催生了定义这一时期的多模态模型（CLIP）和自监督方法（MAE、DINO），并确立了通向[基础模型时代](./06_foundation_model_era.md)的发展轨迹。

这个时代的思想内涵有两方面。第一是**可扩展性**：ViT 随数据和参数平滑提升——不像 CNN 较早就出现饱和——这使视觉能够采用曾经改变 NLP 的缩放定律方法论（参见[视觉中的缩放定律](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)）。第二是**自监督**：同一个注意力主干支持无标签的预训练目标——对比式（CLIP、DINO）和掩码重建式（MAE、BEiT）——产生的表示具有监督式 CNN 无法获得的涌现特性（如 DINO 的无监督物体分割）。第三个较为隐性的后果是**架构统一**：由于两种模态现在使用相同的算子，多模态融合变成了拼接 token 流的问题，而不再需要设计特定于模态的桥接结构，这种简化使后来的视觉-语言-动作模型在架构上顺理成章。本文叙述这一转变；技术细节见[视觉 Transformer](../03_architectures/01_vision_transformers.md) 和[自监督学习](../03_architectures/03_self_supervised_learning.md)。

---

## 注意力进入视觉

**Transformer** [Vaswani2017] 引入了缩放点积自注意力，在 NLP 中取代了循环结构。**ViT** [Dosovitskiy2021]（ICLR 2021）将其应用于视觉：把图像切分为 16×16 的图像块 token，在前面加上一个 `[CLS]` token，添加可学习的位置嵌入，并运行标准的 Transformer 编码器——在 JFT-300M 上预训练时可与 CNN 相媲美。**DeiT** [Touvron2021] 通过蒸馏实现了仅用 ImageNet-1K 的训练；**Swin** [Liu2021] 通过移位窗口注意力重新引入了层次性和局部性，使 Transformer 成为可用于稠密预测的主干网络。

## 视觉-语言与对比学习

**CLIP** [Radford2021]（ICML 2021）在 4 亿网络图文对上对比式地对齐图像编码器与文本编码器，产生了一个*以语言为语义基础*的视觉主干，并通过将类别名称嵌入为文本实现零样本分类。CLIP 可以说是这个时代最具影响力的单一模型：它成为文本到图像生成、开放词汇检测和大型 VLM 的感知前端（参见[多模态架构](../03_architectures/04_multimodal_architectures.md)）。

## 自监督预训练

两类无标签方法走向成熟。**掩码图像建模**：**MAE** [He2022]（CVPR 2022）掩盖约 75% 的图像块，并用非对称的编码器-解码器重建像素，具有高效的可扩展性；**BEiT** 则预测离散的视觉 token。**自蒸馏**：**DINO** [Caron2021] 和 **DINOv2** [Oquab2023] 产生具有*涌现*语义结构的特征——无需任何分割标签，注意力图就能分割出物体。

自蒸馏路线随着 **DINOv3** [Simeoni2025]（arXiv:2508.10104，2025 年 8 月）达到了新的规模：一个 70 亿参数的 ViT 教师模型，在约 17 亿张经过筛选的图像上训练，随后被蒸馏为更小的 ViT 和 ConvNeXt 学生模型。规模的扩大暴露了早期论文仅有所暗示的问题——即使全局特征在提升，稠密的图像块特征在长时间训练中也会*退化*——DINOv3 用 **Gram 锚定**解决了这一问题，这是一种让图像块特征的 Gram 矩阵保持接近较早检查点的损失。其结果在稠密任务上部分平息了该时代关于"最佳自监督目标"的争论：一个冻结的、无标签训练的主干网络在分割和深度估计上也能像在分类上一样具有竞争力。

## 遗产：作为通用视觉算子的 Transformer（2025–2026）

在 2025–2026 年，这个时代的架构统一论点被扩展到了几何领域。**VGGT** [Wang2025]（CVPR 2025 最佳论文）使用一个交替进行帧内注意力与全局注意力的大型 Transformer，在一次前馈传播中为多个视图预测相机、深度和点图；而 CVPR 2026 最佳论文（*D4RT*，Google DeepMind、UCL、牛津大学）将前馈重建扩展到动态视频。再加上概念可提示分割（SAM 3）和基于 Transformer 的 VLA 策略，2026 年几乎每一项主要视觉任务——识别、分割、深度、多视图几何、生成和控制——都运行在 ViT 于 2020 年引入视觉领域的注意力算子之上。

```mermaid
graph LR
    A["2017 Transformer NLP"] --> B["2020/21 ViT<br/>注意力进入视觉"]
    B --> C["2021 CLIP / DINO<br/>多模态 + 自监督"]
    C --> D["2022 MAE / Swin<br/>可扩展主干"]
    D --> E["基础模型<br/>SAM, DINOv2"]
    E --> F["2025-26 DINOv3 / VGGT / D4RT<br/>7B 自监督 + 前馈三维/四维"]
    style C fill:#2d6a4f,color:#fff
```

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| Attention Is All You Need | Vaswani, Shazeer, Parmar, et al. | 2017 | NeurIPS | Transformer / 自注意力 |
| ViT | Dosovitskiy, Beyer, Kolesnikov, et al. | 2021 | ICLR | 用于图像识别的纯 Transformer |
| CLIP | Radford, Kim, Hallacy, et al. | 2021 | ICML | 对比式视觉-语言预训练 |
| MAE | He, Chen, Xie, et al. | 2022 | CVPR | 掩码自编码器；可扩展的自监督学习 |
| DINO | Caron, Touvron, Misra, et al. | 2021 | ICCV | 自蒸馏；涌现的分割能力 |
| Swin Transformer | Liu, Lin, Cao, et al. | 2021 | ICCV | 层次化移位窗口注意力 |
| DINOv3 | Siméoni et al. | 2025 | arXiv | 70 亿参数自蒸馏 ViT；用于稠密特征的 Gram 锚定 |

---

## 影响与局限

| 方面 | 影响 | 局限 |
|--------|--------|------------|
| 注意力主干 | 统一视觉与语言；可扩展 | 二次方代价；数据饥渴 |
| CLIP | 零样本迁移；基础模型前端 | 空间/组合推理能力弱 |
| 自监督 | 无标签、具有涌现特性的特征 | 计算量大；目标设计仍是开放问题 |
| 层次化（Swin） | 适用于稠密预测 | 重新引入了 ViT 所去除的归纳偏置 |

---

## 开放问题与研究空白（这个时代遗留的未解问题）

- 高分辨率下的**二次方注意力代价**，推动了状态空间模型的发展（参见[状态空间模型](../03_architectures/05_state_space_models.md)）。
- **空间推理。** 以 CLIP 为基础的模型继承了较差的空间/关系理解能力（参见[空间智能](../12_research_frontier_2024_2026/06_spatial_intelligence.md)）。
- **数据效率。** ViT 需要大规模预训练；小数据情形下仍偏向混合架构/CNN。
- **最佳自监督目标。** 对比式与掩码建模之间的取舍仍未确定，尽管 DINOv3（2025）加强了在稠密任务上大规模自蒸馏的论据。
- 注意力与涌现特征的**可解释性**仍不完整。
- **从表示到智能体。** 将强大的感知与动作衔接起来定义了下一个时代（参见[基础模型时代](./06_foundation_model_era.md)）。

---

## 延伸阅读

- [ViT (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 一张图像值 16×16 个词
- [CLIP (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — 从自然语言监督中学习
- [MAE (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — 掩码自编码器
- [DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 鲁棒的自监督特征
- [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — Transformer
- [DINOv3 (arXiv:2508.10104)](https://arxiv.org/abs/2508.10104) — 采用 Gram 锚定的 7B 规模自蒸馏
- [VGGT (arXiv:2503.11651)](https://arxiv.org/abs/2503.11651) — 用于前馈多视图几何的 Transformer
