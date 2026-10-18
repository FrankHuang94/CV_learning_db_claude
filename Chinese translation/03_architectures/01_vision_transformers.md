# 视觉 Transformer

> **英文原文：** [cv-knowledge-base/03_architectures/01_vision_transformers.md](../../cv-knowledge-base/03_architectures/01_vision_transformers.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 中级
> **相关章节：**
> - [CNN 架构](./00_cnn_architectures.md)
> - [自监督学习](./03_self_supervised_learning.md)
> - [状态空间模型](./05_state_space_models.md)
> - [视觉中的缩放定律](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)

> **🆕 2026年10月更新：**
> - 新增章节 **现代 ViT 配方（2023–2026）**：寄存器（register，训练得到的与测试时的）、2D RoPE、SwiGLU，以及简化的层级式 ViT（Hiera）。
> - **原生分辨率 ViT**：NaViT 的 patch 打包、Qwen2-VL/2.5-VL/3-VL 的动态分辨率编码器、SigLIP 2 NaFlex，以及 AIMv2 的原生分辨率变体。
> - **7B 规模视觉编码器**：DINOv3 ViT-7B（2025年8月）、Web-DINO 7B、Perception Encoder G（中间层特征），并新增一张参考架构对比表。
> - 更新了关键论文、基准性能（AIMv2-3B 冻结主干 89.5%、PE-Core G 零样本 85.4%、SigLIP 2 g 零样本 85.0%）、开放问题与延伸阅读。

---

## 概述

视觉 Transformer（Vision Transformer, ViT）[Dosovitskiy2021] 证明了卷积并不是实现最先进图像识别的必要归纳偏置：将纯 Transformer 编码器应用于图像 patch 序列，*只要预训练规模足够大*，即可匹敌甚至超越 CNN。这是一次观念上的断裂。CNN 在每一层都硬编码了局部性与平移等变性；ViT 则抛弃了它们，把图像视为一组无序的 16×16 patch token，从数据和可学习的位置嵌入中学习空间结构。代价是样本效率——仅在 ImageNet-1K 上训练时 ViT 不如 CNN；收益则是*可扩展性*：ViT 的性能随数据和参数增长沿着比 CNN 更平滑的曲线持续攀升，正是这一特性使其成为基础模型的底座（参见[视觉中的缩放定律](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)）。

自 2020 年以来，架构演进有两条主线。第一条是为提升效率和小数据性能而**重新引入归纳偏置**：DeiT 通过蒸馏恢复了仅用 ImageNet-1K 的训练能力；Swin 通过移位窗口注意力重新引入了层级结构和局部性，使 Transformer 成为可用的密集预测骨干网络。第二条是**规模与自监督**：BEiT 和 EVA 采用了掩码图像建模；ViT-22B 将参数推到 220 亿，并为在该规模上稳定训练做了必要的架构修改（并行注意力/MLP、QK 归一化）。本文梳理这两条主线，从数学上阐明注意力机制，并界定 ViT 在哪些方面胜过 CNN、卷积又在哪些方面仍具竞争力。

第三条主线主导了 2023–2026 年：**ViT 作为基础模型和视觉-语言模型（VLM）的通用编码器**。架构本身在宏观层面变化不大，但参考配方现在包含寄存器 token、2D 旋转位置嵌入、SwiGLU 前馈层，以及在 VLM 前端中的**原生分辨率**可变长度 patch 序列。2025 年，公开发布的编码器在三种不同的监督信号下达到了数十亿参数规模：自蒸馏（DINOv3 ViT-7B）、无语言的网络规模自监督学习（Web-DINO 7B）以及对比式图文训练（Perception Encoder、SigLIP 2）。

---

## 核心机制

ViT 将 `H×W` 图像切分为 `N = HW/P²` 个不重叠的 `P×P` patch（默认 P=16 → 224×224 下为 196 个 patch），将每个展平的 patch 线性投影为 `D` 维 token，在序列前添加一个可学习的 `[CLS]` token（其最终状态作为分类特征，类似 BERT），并加上**可学习的一维位置嵌入**（[Dosovitskiy2021] 的消融实验发现 2D/相对编码没有带来有意义的提升）。token 序列随后经过标准 Transformer 编码器。其核心是缩放点积自注意力 [Vaswani2017]：

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
  Q = X W_Q,  K = X W_K,  V = X W_V    （每个头）
  √d_k 缩放使 logits 保持在 softmax 梯度较大的区域
```

多头注意力并行运行 `h` 个这样的操作并将结果拼接。关键在于，自注意力是**全局的**——每个 token 都关注所有其他 token——从而带来 O(1) 跳的感受野，但其复杂度相对 token 数为 O(N²)，这正是后续变体所要攻克的瓶颈。

```mermaid
graph LR
    A["图像 224x224"] --> B["切分为 16x16 patch<br/>→ 196 个 token"]
    B --> C["线性投影 → D=768"]
    C --> D["+ [CLS] token<br/>+ 可学习位置嵌入"]
    D --> E["Transformer 编码器<br/>L× MHSA + MLP"]
    E --> F["[CLS] 状态 → MLP 头"]
    style E fill:#2d6a4f,color:#fff
```

ViT-B/16 使用 12 层、D=768、12 个头，约 86M 参数；ViT-L/16 有 24 层、D=1024、16 个头，约 307M；ViT-H/14 达到约 632M。

---

## 演进历程

### 数据效率：DeiT

**DeiT** [Touvron2021]（ICML 2021）表明，ViT 规模的模型*可以*仅在 ImageNet-1K（120 万张图像）上进行有竞争力的训练，方法是使用激进的数据增强（RandAugment、Mixup、CutMix）以及一种新颖的**蒸馏 token**：第二个可学习 token，通过来自 CNN 教师（RegNet）的硬标签蒸馏进行训练。DeiT-B 达到 81.8% top-1（无蒸馏），使用 token 蒸馏时达到 85.2%——消除了 ViT 对 JFT-300M 的依赖。

### 层级与局部性：Swin

**Swin Transformer** [Liu2021]（ICCV 2021，Marr 奖）在局部不重叠窗口（如 7×7）内计算自注意力，并在相邻层交替将窗口移位 `⌊M/2⌋` 以实现跨窗口信息流动。这使复杂度**与图像尺寸呈线性关系**，并产生**层级式**多尺度特征金字塔（4×/8×/16×/32×），使 Swin 成为检测和分割的即插即用骨干网络——而普通 ViT 的单一尺度在这些任务中较为别扭。Swin-L 在 384² 下达到 87.3%。**PVT** 和 **MViT** 各自独立地提出了带空间缩减/池化注意力的金字塔式 Transformer，用于密集预测和视频。

### 自监督与规模：BEiT、EVA、ViT-22B

**BEiT** [Bao2022]（ICLR 2022）借鉴了 BERT 的掩码 token 目标：掩盖约 40% 的 patch 并预测其离散视觉 token（来自 dVAE），经 22k 微调后在 512² 下达到 88.6%。**EVA** [Fang2023]（CVPR 2023）通过重建被掩盖的 *CLIP 特征*，将普通 ViT 扩展到约 10 亿参数，在 336² 下达到 89.6%。**ViT-22B** [Dehghani2023]（ICML 2023）借助**并行注意力+MLP**、**QK 归一化**（对 query/key 做 LayerNorm 以稳定训练）、去除偏置项以及多头注意力池化，扩展到 220 亿参数——取得 89.5% 的线性探测精度，并在鲁棒性/公平性上大幅提升，但每 FLOP 的回报明显呈次线性（参见[缩放定律](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)）。

---

## 现代 ViT 配方（2023–2026）

到 2025 年，基础模型内部使用的"普通"ViT 已在几个地方悄然改变。这些改变都没有触及核心注意力机制，但它们共同定义了 DINOv3 / SigLIP 2 / Perception Encoder 这一代的参考骨干网络。

### 寄存器与高范数伪影 token

**Vision Transformers Need Registers** [Darcet2024]（ICLR 2024）发现，大型、训练充分的 ViT（DINOv2、CLIP、DeiT-III）会把少数低信息量的背景 patch token 挪用为全局"草稿"存储。这些 token 的范数异常高，会在注意力图和密集特征中造成伪影。追加少量可学习的**寄存器 token**（在输出端丢弃）为模型提供专用存储，从而消除伪影，并改善密集预测和目标发现。带寄存器的 DINOv2 达到 87.1% 的 ImageNet 线性探测精度（ViT-g/14）。**测试时寄存器**（test-time registers）[Jiang2025]（NeurIPS 2025 Spotlight）将伪影追溯到一小组稀疏的"寄存器神经元"，并在推理时把它们的激活转移到一个额外的未训练 token 中，**无需重新训练**即可恢复大部分收益。同样的现象也出现在视觉 SSM 中（Mamba®，CVPR 2025；参见[状态空间模型](./05_state_space_models.md)）。DINOv3 默认使用 4 个寄存器（"存储"）token。

### 位置编码：从可学习绝对编码到 2D RoPE

可学习的一维绝对嵌入在分辨率变化时必须插值，这会损害外推能力。**旋转位置嵌入（Rotary Position Embedding, RoPE）**通过对 query/key 的特征对按行坐标和列坐标分别使用不同频率进行旋转而扩展到二维（RoPE-ViT [Heo2024]，ECCV 2024），使注意力仅依赖于*相对*偏移：

```
q'_(x,y) = R(θ_x · x) R(θ_y · y) q_(x,y)      # 对特征对进行轴向 2D 旋转
<q'_(x1,y1), k'_(x2,y2)> 仅依赖于 (x1 − x2, y1 − y2)
```

2D/轴向 RoPE 如今已是大型编码器的标准配置：DINOv3 使用归一化并重缩放的 patch 坐标的 RoPE；Qwen2-VL 提出了 **M-RoPE**（将位置分解为时间、高度和宽度分量的多模态 RoPE），并在 Qwen3-VL 中改进为 **Interleaved-MRoPE**。

### 其他模块级变化

- **SwiGLU 前馈层**（来自大语言模型）在最大的编码器中取代了 GELU MLP（如 DINOv3 ViT-7B；Qwen2.5-VL 的 ViT 也使用 SwiGLU 和 RMSNorm）。
- **LayerScale** 和 **QK 归一化**（分别来自 CaiT 和 ViT-22B）可稳定约 10 亿参数以上的训练。
- 在最新的编码器中，**patch 大小 16** 配合更高的输入分辨率已基本取代 /14（DINOv3 使用 /16；Qwen3-VL 将 patch 从 14 改为 16）。

### 简化的层级结构：Hiera

**Hiera** [Ryali2023]（ICML 2023 Oral）认为，层级式 ViT（Swin、MViT）中的专用模块增加了延迟，因此改为通过 MAE 预训练来*教会*模型空间偏置，只保留带池化的普通多阶段 Transformer。据报道，Hiera 在相近精度下比以往的层级式 ViT 更快，并作为 SAM 2 的图像编码器，使其成为可提示分割（promptable segmentation）中部署最广泛的层级式 ViT。

---

## 原生分辨率 ViT

经典 ViT 把每张图像缩放到固定的正方形（如 224² 或 384²），这会扭曲长宽比并浪费或破坏细节，对文档、图表、OCR 以及高分辨率 VLM 输入尤其有害。原生分辨率 ViT 将 patch 序列视为**可变长度**：

- **NaViT** [Dehghani2023b]（NeurIPS 2023，"Patch n' Pack"）将任意分辨率和长宽比的多张图像的 patch 打包进同一个序列，使用防止跨图像注意力的注意力掩码以及分解式位置嵌入。这提高了训练效率，并使同一个模型可以在多种分辨率下部署。
- **Qwen2-VL**（2024）提出了*朴素动态分辨率*（Naive Dynamic Resolution）：图像以原生分辨率处理为数量可变的 token，在送入 LLM 前进行 2×2 合并，并与 M-RoPE 结合。**Qwen2.5-VL**（2025年1月）重新设计了 ViT，在**大多数层中使用窗口注意力**（只有少数层使用全注意力），使计算代价大致随像素数线性增长。**Qwen3-VL**（2025年9–10月，技术报告于2025年11月发布）使用 patch 16，以 32 倍空间压缩映射为 LLM token，并加入 **DeepStack**，将多层级 ViT 特征注入 LLM，而不仅仅使用最后一层。
- **SigLIP 2 NaFlex** [Tschannen2025]（2025年2月）在单个对比式编码器中支持原生长宽比下的多种序列长度（提供 B/16 和 So400m/16 检查点）。
- **AIMv2** [Fini2025]（CVPR 2025）提供了一个原生分辨率的 AIMv2-L 变体，经微调后可接受 112–4096 个 patch，使用 2D 正弦位置嵌入。

```mermaid
graph LR
    A["任意尺寸 / 长宽比的图像"] --> B["按原生分辨率切分 patch<br/>每张图像 token 数可变"]
    B --> C["序列打包 + 块对角掩码<br/>(NaViT / NaFlex)"]
    C --> D["2D RoPE 或分解式位置嵌入"]
    D --> E["ViT（窗口注意力 + 少数全局层）"]
    E --> F["2x2 token 合并 → LLM / 任务头"]
    style C fill:#1d3557,color:#fff
    style E fill:#2d6a4f,color:#fff
```

---

## 将视觉编码器扩展到数十亿参数（2025）

2025 年出现了三个独立训练、公开发布的 2–7B 规模视觉编码器，各自采用不同的监督信号：

| 编码器 | 监督方式 | 规模 | 值得注意的架构细节 |
|---|---|---|---|
| **DINOv3 ViT-7B/16** [Simeoni2025]（Meta，2025年8月） | 自蒸馏（DINO + iBOT）+ **Gram 锚定**（Gram anchoring） | 6.7B 参数；40 层，宽度 4096，32 个头；LVD-1689M 图像 | Patch 16、SwiGLU FFN、4 个寄存器 token、2D RoPE；蒸馏为 ViT-S/B/L/H+ 和 ConvNeXt 学生模型 |
| **Web-DINO 7B** [Fan2025]（Meta/NYU，2025年4月） | DINOv2 式自监督学习，不使用语言 | 在 MC-2B（20 亿张网络图像）上从 0.3B 扩展到 7B | 表明纯视觉自监督学习随规模持续提升，并在 VQA 上追平 CLIP |
| **Perception Encoder G/14** [Bolya2025]（Meta，2025年4月） | 对比式视觉-语言 | 最大的"G"规模，patch 14，448²；家族从 T/16 到 G/14 | 最佳通用特征位于**中间层**；对齐微调产生 PE-Lang（用于 LLM）和 PE-Spatial（用于密集任务） |

Perception Encoder 的发现在架构上很重要：对比训练的 ViT 的*输出*层专门服务于图文目标，而中间层承载着更强的 OCR、VQA、检测和深度特征。因此，"读取哪一层"与"选择哪个骨干网络"一样成为一项设计选择，这也是 VLM 中多层特征融合（如 Qwen3-VL 的 DeepStack）的动机。DINOv3 的 **Gram 锚定**解决了一个互补的失效模式：在非常长的训练中，全局（分类）指标持续提升，而密集的 patch 特征却在退化；将 patch 相似度（Gram）矩阵正则化到较早的检查点即可解决这一问题（参见[自监督学习](./03_self_supervised_learning.md)）。

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| An Image is Worth 16×16 Words (ViT) | Dosovitskiy, Beyer, Kolesnikov, et al. | 2021 | ICLR | 大规模图像识别的纯 Transformer |
| Training data-efficient image transformers (DeiT) | Touvron, Cord, Douze, et al. | 2021 | ICML | 蒸馏 token；仅用 ImageNet-1K 训练 |
| Swin Transformer | Liu, Lin, Cao, Hu, et al. | 2021 | ICCV (Marr Prize) | 移位窗口注意力；层级式骨干网络 |
| BEiT | Bao, Dong, Piao, Wei | 2022 | ICLR | 基于离散视觉 token 的掩码图像建模 |
| ViT-22B | Dehghani, Djolonga, Mustafa, et al. | 2023 | ICML | 22B 参数 ViT；并行注意力+MLP、QK 归一化 |
| EVA | Fang, Wang, Xie, et al. | 2023 | CVPR | 通过掩码 CLIP 特征重建得到的 1B 参数 ViT |
| Patch n' Pack (NaViT) | Dehghani, Mustafa, Djolonga, et al. | 2023 | NeurIPS | 支持任意分辨率/长宽比的序列打包 |
| Hiera | Ryali, Hu, Bolya, et al. | 2023 | ICML (Oral) | 去除繁复组件的层级式 ViT；由 MAE 教会空间偏置 |
| Vision Transformers Need Registers | Darcet, Oquab, Mairal, Bojanowski | 2024 | ICLR | 寄存器 token 消除高范数伪影 token |
| Rotary Position Embedding for ViT (RoPE-ViT) | Heo, Park, Han, Yun | 2024 | ECCV | 用于分辨率外推的 2D RoPE |
| SigLIP 2 | Tschannen, Gritsenko, Wang, et al. | 2025 | arXiv | 多语言 sigmoid 编码器 + NaFlex 原生分辨率变体 |
| Perception Encoder | Bolya, Huang, Sun, et al. | 2025 | arXiv | 最佳特征位于中间层；PE-Core/Lang/Spatial |
| Vision Transformers Don't Need *Trained* Registers | Jiang, Dravid, Efros, Gandelsman | 2025 | NeurIPS (Spotlight) | 基于寄存器神经元的免训练测试时寄存器 |
| DINOv3 | Siméoni, Vo, Seitzer, et al. | 2025 | arXiv | 采用 Gram 锚定的 7B 自监督 ViT；RoPE + 寄存器 |

---

## 基准性能

| 模型 | 预训练 | 分辨率 | ImageNet top-1 | 备注 |
|-------|-------------|-----------|----------------|-------|
| ViT-H/14 | JFT-300M | 224 | 88.55% | 原论文中最大的 ViT |
| DeiT-B（蒸馏） | IN-1K | 224 | 85.2% | 无外部数据 |
| Swin-L | IN-22K→1K | 384 | 87.3% | 层级式骨干网络 |
| BEiT-L | 自监督 + 22K 微调 | 512 | 88.6% | 掩码图像建模 |
| EVA | 掩码 CLIP 特征 | 336 | 89.6% | 约 1B 参数 |
| ViT-22B | JFT | 224 | 89.5%（线性探测） | 冻结特征 |
| DINOv2 ViT-g/14 + 寄存器 | LVD-142M（自监督） | 224 | 87.1%（线性探测） | 寄存器变体 [Darcet2024] |
| AIMv2-3B | 多模态自回归 | 448 | 89.5%（冻结主干） | [Fini2025]，CVPR 2025 |
| AIMv2-L 原生分辨率 | 多模态自回归 | 112–4096 个 patch | 87.3%（冻结主干） | 可变分辨率/长宽比 |
| PE-Core G/14 | 对比式图文 | 448 | 85.4%（零样本） | [Bolya2025] |
| SigLIP 2 g-opt/16 | Sigmoid 图文 + 解码器/自监督损失 | 384 | 85.0%（零样本） | [Tschannen2025] |
| SigLIP 2 So400m/16 NaFlex | 同上，原生长宽比 | 序列长度 256 | 83.5%（零样本） | NaFlex 变体 |

---

## 优缺点

| 方面 | 优点 | 缺点 |
|--------|------|------|
| 全局注意力 | O(1) 跳感受野；长程上下文 | O(N²) 代价；不使用窗口时高分辨率扩展性差 |
| 可扩展性 | 随数据/参数平滑提升；基础模型的底座 | 数据饥渴；在小数据集上从头训练表现较弱 |
| 层级结构（Swin/PVT） | 多尺度；适合密集预测；线性代价 | 工程化程度更高；失去 ViT 的简洁性 |
| 自监督（BEiT/EVA） | 利用无标注数据；迁移能力强 | 计算量大；依赖 tokenizer/教师模型 |
| 原生分辨率（NaViT/NaFlex/Qwen-VL） | 无长宽比失真；对 OCR/文档输入表现强；计算量灵活 | 可变长度批处理和掩码使内核与推理部署更复杂 |
| 寄存器 + RoPE（现代配方） | 干净的密集特征；更好的分辨率外推 | 额外的 token；RoPE 频率/缩放选择增加超参数 |

---

## 开放问题与研究空白

- **高分辨率下的二次注意力。** 窗口/线性注意力和状态空间模型（参见[状态空间模型](./05_state_space_models.md)）能缓解但不能完全解决这一代价。
- **小数据训练。** 没有大规模预训练时，ViT 仍落后于 CNN/混合模型（参见[混合架构](./02_hybrid_architectures.md)）。
- **面向可变分辨率的位置编码。** 2D RoPE 和原生分辨率打包（NaViT、NaFlex、Qwen-VL）已基本取代固定网格的可学习嵌入，但在远超训练分辨率/长宽比时的稳健外推，以及 RoPE 频率在各轴之间的合理分配（参见 Interleaved-MRoPE），仍不完善。
- **伪影 token 与密集特征退化。** 寄存器（训练得到的或测试时的）和 DINOv3 的 Gram 锚定治疗的是大型 ViT 将全局信息分配给 patch token 这一现象的症状；对于其为何在大规模下发生，仍缺乏机理层面的解释。
- **读取哪一层。** Perception Encoder 表明，对比式 ViT 的最佳通用特征位于中间层；面向 VLM 和密集预测头的有原则的多层读出（DeepStack 式融合 vs. 对齐微调）尚无定论。
- **可解释性。** 注意力图并不是忠实的解释；对 ViT 特征的机理理解仍不成熟。
- **计算最优缩放。** 视觉领域尚无公认的 Chinchilla 式定律；其指数与语言领域不同。
- **归纳偏置的平衡。** 应注入多少卷积先验仍未解决——ConvNeXt 表明 CNN 可以匹敌 ViT，模糊了二者的界限。

---

## 延伸阅读

- [ViT (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) —— 最初的视觉 Transformer
- [DeiT (arXiv:2012.12877)](https://arxiv.org/abs/2012.12877) —— 数据高效训练与蒸馏
- [Swin Transformer (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030) —— 移位窗口层级注意力
- [ViT-22B (arXiv:2302.05442)](https://arxiv.org/abs/2302.05442) —— 扩展到 220 亿参数
- [EVA (arXiv:2211.07636)](https://arxiv.org/abs/2211.07636) —— 十亿级掩码视觉表征学习
- [Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) —— 寄存器 token 与伪影消除
- [测试时寄存器 (arXiv:2506.08010)](https://github.com/nickjiang2378/test-time-registers) —— 免训练寄存器，NeurIPS 2025
- [NaViT / Patch n' Pack (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304) —— 通过序列打包实现任意分辨率 ViT
- [Hiera (arXiv:2306.00989)](https://github.com/facebookresearch/hiera) —— 去除繁复组件的层级式 ViT
- [SigLIP 2 (arXiv:2502.14786)](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) —— NaFlex 与多语言编码器
- [Perception Encoder (arXiv:2504.13181)](https://github.com/facebookresearch/perception_models) —— 中间层特征，PE-Core/Lang/Spatial
- [DINOv3 (arXiv:2508.10104)](https://github.com/facebookresearch/dinov3) —— 7B ViT、Gram 锚定、蒸馏模型家族
- [Qwen3-VL (arXiv:2511.21631)](https://github.com/QwenLM/Qwen3-VL) —— 带 DeepStack 和 Interleaved-MRoPE 的原生分辨率 VLM 编码器
