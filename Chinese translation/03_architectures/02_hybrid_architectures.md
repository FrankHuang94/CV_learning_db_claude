# CNN-Transformer 混合架构

> **英文原文：** [cv-knowledge-base/03_architectures/02_hybrid_architectures.md](../../cv-knowledge-base/03_architectures/02_hybrid_architectures.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 中级
> **相关章节：**
> - [视觉 Transformer](./01_vision_transformers.md)
> - [CNN 架构](./00_cnn_architectures.md)
> - [状态空间模型](./05_state_space_models.md)
> - [高效架构](../09_efficiency_and_deployment/01_efficient_architectures.md)

> **🆕 2026年10月更新：**
> - 新增章节 **基础模型时代的混合架构（2023–2026）**：面向 VLM 的混合编码器（FastViT → FastVLM 中的 FastViTHD，CVPR 2025）、层级式 Hiera，以及卷积 + Mamba + 注意力骨干网络（MambaVision，CVPR 2025）。
> - **聚合式混合模型**：AM-RADIO → C-RADIOv3（2025年6月）→ C-RADIOv4（2026年1月），将 SigLIP 2、DINOv3 和 SAM 3 蒸馏进同一个学生模型。C-RADIOv4-H 是 NVIDIA Nemotron 3 Nano Omni 的视觉编码器。
> - 更新了关键论文、基准性能（使用 IN-21K 时 MambaVision 最高达 88.1%）与开放问题。

---

## 概述

CNN-Transformer 混合架构在 2021 年后的骨干网络格局中占据了务实的中间地带：它们将**卷积的局部归纳偏置与硬件效率**和**自注意力的全局感受野**融合在一起。其动机来自经验。纯 ViT 缺乏局部性与平移等变性，因此在中小规模数据集和高分辨率下训练时不如 CNN；纯 CNN 只能通过堆叠层缓慢地聚合长程上下文。混合架构的解决方式是：在卷积最有帮助的地方使用卷积——即低层特征局部且丰富的早期层；在注意力最有帮助的地方使用注意力——即全局语义上下文重要的后期层。由此产生了一个模型家族（CvT、CoAtNet、LeViT、MobileViT、MaxViT），它们在给定计算预算下持续匹敌或超越两种纯范式，在小数据上尤其如此。

这一路线最深刻的教训是 **CNN 与 Transformer 设计的趋同**。ConvNeXt [Liu2022] 是一个*纯 CNN*，它通过逐步用受 Transformer 启发的选择（patchify 主干、深度 7×7 卷积核、倒置瓶颈、GELU、LayerNorm、与 Swin 匹配的阶段比例）现代化 ResNet，直到追平 Swin 的精度——这表明 Transformer 的大部分优势来自*训练配方和宏观设计*，而非注意力本身。本文综述主要的混合架构、它们组合两种算子的机制，以及它们占优的小数据和高效率场景。本文还介绍了混合思想在 2023–2026 年的演变：降低 VLM 延迟的高分辨率混合编码器（FastViTHD）、混合卷积、Mamba 与注意力的骨干网络（MambaVision），以及将多个基础模型（SigLIP 2、DINOv3、SAM 3）融合进同一网络的"聚合式"（agglomerative）学生模型，如 C-RADIOv4。

---

## 组合机制

混合架构的区别在于*如何*以及*在何处*注入卷积：

- **卷积式 token 化/主干** —— 用重叠卷积替代 ViT 的线性 patch 投影（CvT、LeViT 的 4 层卷积主干），以获得更丰富的低层特征和隐式位置信息。
- **投影中的卷积** —— 对 Q/K/V 投影使用深度可分离卷积（CvT 的卷积投影），从而去掉显式位置编码。
- **分阶段的卷积→注意力** —— 早期阶段使用卷积（MBConv），后期阶段使用注意力（CoAtNet：S0 卷积主干，S1–S2 MBConv，S3–S4 Transformer）。
- **注意力即卷积** —— 将全局注意力折叠进卷积式模块中（MobileViT 展开 patch，在对应位置间施加注意力，再折叠回去）。
- **多轴注意力** —— 结合分块局部注意力与空洞网格注意力，以线性复杂度实现全局覆盖（MaxViT）。

```mermaid
graph LR
    A["图像"] --> B["卷积主干 / MBConv<br/>局部特征，低成本"]
    B --> C["中间阶段<br/>卷积或局部注意力"]
    C --> D["后期阶段<br/>全局自注意力"]
    D --> E["预测头"]
    style B fill:#7b2d8b,color:#fff
    style D fill:#2d6a4f,color:#fff
```

---

## 主要架构

### CvT —— 卷积视觉 Transformer
**CvT** [Wu2021]（ICCV 2021，Microsoft）提出了**卷积 token 嵌入**（在 3 阶段层级结构中使用重叠卷积）和**卷积投影**（对 Q/K/V 使用深度可分离卷积），并且由于卷积隐式编码了位置，*去掉了位置编码*。CvT-13（20M）达到 81.6%；CvT-W24（277M，IN-22K）达到 87.7%。

### CoAtNet —— 卷积 + 注意力
**CoAtNet** [Dai2021]（NeurIPS 2021，Google）通过**相对注意力**统一了深度卷积与自注意力，早期堆叠 MBConv，后期堆叠 Transformer。CoAtNet-2 达到 84.1%（IN-1K）/ 87.1%（IN-21K）；最大的变体在 IN-21K 预训练下达到 88.56%，在 JFT-3B 预训练下达到 90.88%——属于已报道的最高结果之一。它明确提出了"卷积提供泛化 / 注意力提供容量"的权衡框架。

### LeViT —— 披着卷积网络外衣的 ViT
**LeViT** [Graham2021]（ICCV 2021，Meta）以**推理速度**为目标：4 层卷积主干、基于注意力的下采样、取代位置嵌入的逐头可学习**注意力偏置**，以及逐级减少的头数。LeViT-128S 以约 1/4 的 FLOPs 追平 DeiT-Tiny 的 76.6%。

### MobileViT —— 移动端友好的混合模型
**MobileViT** [Mehta2022]（ICLR 2022，Apple）把 Transformer 当作卷积来用：展开不重叠的 patch，在 patch 间的对应位置上施加全局注意力，再折叠回去，并与 MBConv 模块交错排列。MobileViT-S（5.6M）达到 78.4%，在相近规模下超过 MobileNetV3-Large（约 75.2%），并以约 1/4 的参数量超过 DeiT-S（79.8%）。

### MaxViT —— 多轴视觉 Transformer
**MaxViT** [Tu2022]（ECCV 2022，Google + UT Austin）交替使用**分块局部注意力**（P×P 窗口内）和**空洞网格注意力**（覆盖稀疏全局网格），二者均为线性复杂度，前面接 MBConv。MaxViT-Base 达到 85.0%（224²）/ 86.7%（512²）；MaxViT-XLarge 在 IN-21K 下达到 88.7%。

### ConvNeXt —— 趋同的 CNN
**ConvNeXt** [Liu2022]（CVPR 2022）是用 Transformer 设计选择现代化的纯 CNN；ConvNeXt-XL 达到 87.8%（IN-22K，384²），在相近精度下比 Swin-B 快约 35%——这证明架构上的二分在很大程度上是训练配方上的二分。

---

## 基础模型时代的混合架构（2023–2026）

2022 年之后，"早期卷积、后期注意力"的模板从 ImageNet 分类器迁移到三种新角色中：对延迟敏感的 **VLM 视觉编码器**、混合卷积、状态空间层和注意力的**三算子骨干网络**，以及将多个基础模型蒸馏进同一网络的**聚合式学生模型**。

### 面向 VLM 的混合编码器：FastViT → FastViTHD
**FastViT** [Vasu2023]（ICCV 2023，Apple）在早期阶段使用结构重参数化的卷积 token 混合器（RepMixer），在后期阶段使用注意力，以实现移动端低延迟。**FastVLM** [Vasu2025]（CVPR 2025）将其扩展为 **FastViTHD**——一种在高分辨率下输出*更少 token* 的混合编码器，使 VLM 的首 token 延迟（time-to-first-token, TTFT）不再由视觉编码器主导。作者报告称，0.5B 的 FastVLM 优于 LLaVA-OneVision-0.5B，TTFT 快 85 倍，视觉编码器小 3.4 倍。其背后的教训是：在编码器和 LLM 的二次复杂度层*之前*用卷积下采样阶段削减视觉 token 数，是成本最低的方式。

### 学习偏置的层级式 ViT：Hiera
**Hiera** [Ryali2023]（ICML 2023 Oral）是手工设计混合模型的一个反例。它没有添加卷积或专用模块，而是将 MViT 式的层级结构精简为普通注意力 + 池化，并依靠 MAE 预训练来*教会*模型原本由卷积硬编码的空间偏置。它是 SAM 2 的图像编码器（参见[视觉 Transformer](./01_vision_transformers.md)）。

### 卷积 + Mamba + 注意力：MambaVision
**MambaVision** [Hatamizadeh2025]（CVPR 2025，NVIDIA）是一种层级式混合模型：高分辨率处为卷积残差阶段，随后的阶段将重新设计的 Mamba 混合器（增加了一个不含 SSM 的对称分支以捕获全局上下文）与**放在低分辨率阶段最后几层的自注意力模块**结合。它在 ImageNet 上定义了新的 top-1/吞吐量帕累托前沿：MambaVision-B 以 3,670 img/s 达到 84.2%，经 IN-21K 预训练的 MambaVision-L3 在 512² 下达到 **88.1%**。它还能迁移到 COCO（Cascade Mask R-CNN，B 型 52.8 box AP）和 ADE20K（最高 53.2 mIoU）。该设计中 SSM 一侧的内容参见[状态空间模型](./05_state_space_models.md)。

### 聚合式混合模型：RADIO
**AM-RADIO** [Ranzinger2024]（CVPR 2024，NVIDIA）将多个异构教师模型（CLIP 变体、DINOv2、SAM）蒸馏进一个带有教师专属适配头的学生模型，并报告在零样本、k-NN 和分割探测上超过其教师。这一路线持续发展：

| 版本 | 日期 | 教师 / 适配头 | 备注 |
|---|---|---|---|
| RADIOv2.5 | 2024（CVPR 2025） | CLIP、SigLIP、DINOv2、SAM | PHI-S 分布平衡；B/L/H/g |
| C-RADIOv3 | 2025年6月 | SigLIP2-g、DINOv2、SAM、CLIP | 允许商用的许可证；FeatSharp（ICML 2025）上采样 |
| C-RADIOv4 | 2026年1月 | SigLIP2-g-384、**DINOv3-7B**、**SAM 3** | SO400M 和 H 变体；迄今最强的 RADIO |

聚合式模型在*监督*意义上是"混合"的：一个骨干网络同时具备 SigLIP 的语言对齐语义、DINO 的密集几何特征以及 SAM 的边界精度。它们正越来越多地被用作 VLM 前端。NVIDIA 的 Nemotron 3 Nano Omni 使用 **C-RADIOv4-H** 作为视觉编码器，后接混合 Mamba-Transformer LLM；Nemotron 3.5 Super 也使用 C-RADIO 编码器（参见[多模态架构](./04_multimodal_architectures.md)）。

### 再论趋同
ConvNeXt 的教训（"配方比算子更重要"）在 2025 年从另一个方向得到了强化。**DINOv3** 将 7B ViT 蒸馏为 ConvNeXt 学生模型，**MambaOut** 则表明不含任何 SSM 的门控 CNN 模块在 ImageNet 上胜过视觉 Mamba 模型（参见 [CNN 架构](./00_cnn_architectures.md)）。在 2026 年的实践中，算子的选择主要由 token 数、延迟和显存等部署约束决定，而表征质量则来自教师模型和数据。

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| CvT | Wu, Xiao, Codella, et al. | 2021 | ICCV | 卷积 token 嵌入与投影；无位置编码 |
| CoAtNet | Dai, Liu, Le, Tan | 2021 | NeurIPS | 统一 MBConv 与相对注意力；分阶段堆叠 |
| LeViT | Graham, El-Nouby, Touvron, et al. | 2021 | ICCV | 快速推理；卷积主干 + 注意力偏置 |
| MobileViT | Mehta, Rastegari | 2022 | ICLR | 面向移动端的"注意力即卷积" |
| MaxViT | Tu, Talebi, Zhang, et al. | 2022 | ECCV | 多轴（分块 + 网格）线性复杂度注意力 |
| ConvNeXt | Liu, Mao, Wu, Feichtenhofer, et al. | 2022 | CVPR | 通过现代化设计追平 Swin 的纯 CNN |
| FastViT | Vasu, Gabriel, Zhu, Tuzel, Ranjan | 2023 | ICCV | 重参数化卷积混合器 + 后期注意力，实现移动端低延迟 |
| Hiera | Ryali, Hu, Bolya, et al. | 2023 | ICML (Oral) | 简洁的层级式 ViT；由 MAE 教会空间偏置 |
| AM-RADIO | Ranzinger, Heinrich, Kautz, Molchanov | 2024 | CVPR | 多教师聚合式蒸馏进单一骨干网络 |
| FastVLM (FastViTHD) | Vasu, Faghri, Li, et al. | 2025 | CVPR | 输出更少 token 的高分辨率混合 VLM 编码器 |
| MambaVision | Hatamizadeh, Kautz | 2025 | CVPR | 卷积阶段 + Mamba 混合器 + 末层自注意力 |

---

## 基准性能

| 模型 | 参数量 | FLOPs | ImageNet top-1 | 备注 |
|-------|--------|-------|----------------|-------|
| CvT-13 | 20M | 4.5G | 81.6% | IN-1K，224² |
| CoAtNet-2 | 75M | 15.7G | 84.1% / 87.1% | IN-1K / IN-21K |
| LeViT-128S | 7.8M | 0.3G | 76.6% | FLOPs 约为 DeiT-Ti 的 1/4 |
| MobileViT-S | 5.6M | 1.1G | 78.4% | 超过 MobileNetV3-L |
| MaxViT-Base | 119M | 24.2G | 85.0% / 86.7% | 224² / 512² |
| ConvNeXt-XL | 350M | 60.9G | 87.8% | IN-22K，384² |
| MambaVision-T | 31.8M | 4.4G | 82.3% | IN-1K，224²；6,298 img/s |
| MambaVision-B | 97.7M | 15.0G | 84.2% | IN-1K，224²；3,670 img/s |
| MambaVision-L3-512-21K | 739.6M | 489.1G | 88.1% | IN-21K，512² |

---

## 优缺点

| 方面 | 优点 | 缺点 |
|--------|------|------|
| 卷积归纳偏置 | 小数据性能强；样本效率高 | 相比普通 ViT 增加了架构复杂度 |
| 全局注意力（后期） | 在关键处提供长程上下文 | 不使用窗口时在高分辨率下仍然代价高 |
| 硬件效率 | 卷积算子高度优化；MaxViT/LeViT 为线性代价 | FLOPs ≠ 延迟；收益依赖于硬件 |
| 设计趋同（ConvNeXt） | CNN 的简洁 + ViT 的精度 | 表明注意力可能并非必需——削弱了"混合"的理由 |
| 混合 VLM 编码器（FastViTHD） | 高分辨率下视觉 token 更少；TTFT 大幅降低 | 社区工具支持不如普通 ViT/SigLIP 编码器 |
| 聚合式学生模型（RADIO） | 单一骨干网络兼具类 CLIP、类 DINO 和类 SAM 特征 | 继承教师的许可证/偏差；平衡各教师较为微妙 |

---

## 开放问题与研究空白

- **最优卷积/注意力比例。** 对于在何种深度注入多少卷积，尚无有原则的规则；现有设计都是经验性的。
- **小数据理论。** 卷积偏置的样本效率优势有经验记录，但缺乏可预测的理论。
- **延迟 vs. FLOPs。** 报告的 FLOPs 节省往往无法在多样的硬件上转化为实际耗时的加速（参见[高效架构](../09_efficiency_and_deployment/01_efficient_architectures.md)）。
- **缩放行为。** 混合模型在基础模型规模下能否保持优势，抑或会收敛到普通 ViT/ConvNeXt，尚无定论。2025 年最大的编码器（DINOv3、PE、SigLIP 2）都是普通 ViT，而混合模型主要在延迟（FastViTHD）或作为蒸馏学生模型（RADIO）方面胜出。
- **与状态空间模型的统一。** MambaVision 表明三算子（卷积 + SSM + 注意力）层级结构可以确立帕累托前沿，但 MambaOut 表明对分类而言 SSM 可能是可有可无的。在高分辨率密集任务上，究竟是哪个组件带来了收益，仍是开放问题（参见[状态空间模型](./05_state_space_models.md)）。
- **多教师蒸馏。** 如何对特征几何相互冲突的教师（SigLIP vs. DINO vs. SAM）进行加权和调和而不冲淡各自的优势，以及学生模型能在多大程度上超越教师，都缺乏理论。
- **密集预测迁移。** 将混合模型适配为检测/分割骨干网络的最佳实践不如 Swin 成熟。

---

## 延伸阅读

- [CoAtNet (arXiv:2106.04803)](https://arxiv.org/abs/2106.04803) —— 卷积与注意力的结合
- [MaxViT (arXiv:2204.01697)](https://arxiv.org/abs/2204.01697) —— 多轴线性复杂度注意力
- [MobileViT (arXiv:2110.02178)](https://arxiv.org/abs/2110.02178) —— 移动端友好的混合模型
- [ConvNeXt (arXiv:2201.03545)](https://arxiv.org/abs/2201.03545) —— 面向 2020 年代的卷积网络
- [CvT (arXiv:2103.15808)](https://arxiv.org/abs/2103.15808) —— 视觉 Transformer 中的卷积
- [MambaVision (arXiv:2407.08083)](https://github.com/NVlabs/MambaVision) —— 混合 Mamba-Transformer 骨干网络，CVPR 2025
- [FastVLM / FastViTHD (arXiv:2412.13303)](https://github.com/apple/ml-fastvlm) —— 面向 VLM 的高效混合视觉编码
- [Hiera (arXiv:2306.00989)](https://github.com/facebookresearch/hiera) —— 去除繁复组件的层级式 ViT
- [RADIO 家族（AM-RADIO、RADIOv2.5、C-RADIOv3/v4）](https://github.com/NVlabs/RADIO) —— 聚合式视觉基础模型
