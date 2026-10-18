# 模型压缩：剪枝、量化、蒸馏与低秩分解

> **英文原文：** [cv-knowledge-base/09_efficiency_and_deployment/00_model_compression.md](../../cv-knowledge-base/09_efficiency_and_deployment/00_model_compression.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：**
> - [高效架构](./01_efficient_architectures.md)
> - [硬件加速](./02_hardware_acceleration.md)
> - [边缘部署](./03_edge_deployment.md)
> - [核心架构](../03_architectures/00_cnns_and_convolutions.md)

> **🆕 2026年10月更新：**
> - 新增**块缩放低精度格式**小节：FP8（E4M3/E5M2）、OCP **MXFP4** 和 NVIDIA **NVFP4**（E2M1 数值 + FP8 微块缩放因子），如今在 Blackwell 级 GPU 和 Jetson Thor 上获得原生支持，并配有 W4A4 NVFP4 + 量化感知蒸馏（quantization-aware distillation, QAD）方案。
> - 新增**面向 VLM 的视觉 token 剪枝与合并**一节（FastV、SparseVLM/SparseVLM+、VisionZip、VisionThink、InternVL3.5 的视觉分辨率路由器），这是多模态模型特有的主要压缩手段。
> - 补充了面向 LLM/VLM 骨干的“剪枝 + 蒸馏”流程（NVIDIA Model Optimizer 中的 Minitron、Puzzletron）以及 FP8 KV 缓存量化。
> - 相应更新了关键论文、基准性能和开放问题。

---

## 概述

模型压缩是一组技术，旨在降低已训练神经网络的内存占用、计算成本和/或推理延迟，同时尽可能保留其预测精度。动机很直接：最先进的视觉模型（ViT-L、CLIP、Swin-L）每幅图像需要数十亿次浮点运算和数 GB 的参数存储，使其难以部署在嵌入式设备、手机以及对延迟敏感的生产服务中。压缩方法从四个互补的角度应对这一差距：**剪枝**（移除参数）、**量化**（降低数值精度）、**知识蒸馏**（将知识迁移到更小的架构）以及**低秩分解**（用更小矩阵的乘积近似权重矩阵）。

这些技术并不互斥。现代生产流水线通常将量化感知训练（QAT）与结构化剪枝以及来自更大教师模型的蒸馏结合使用。对于应用于视觉-语言任务的大语言模型（LLM），训练后量化方法（GPTQ、AWQ）已成为主流部署路径，可将 7B–70B 参数模型压缩到 4 比特精度，其中 W8A8 的平均任务精度损失约为 1–3%。自 2025 年以来，另外两种手段主导了多模态部署：**硬件原生的块缩放浮点格式**（FP8、MXFP4、NVFP4），使 4 比特权重*和*激活都能在 Tensor Core 上运行；以及**视觉 token 削减**，用于应对单幅高分辨率图像或视频就可能向 VLM 序列贡献数千个 token 的问题。

一个核心理论基础是**过参数化假说**：现代网络相对于任务复杂度而言严重过参数化，这意味着可以移除（剪枝）大部分参数或降低其精度，而不会造成灾难性的功能损失。**彩票假说（Lottery Ticket Hypothesis）** [Frankle2019] 对此给出了精确表述：一个大型网络包含一个小型稀疏子网络（“中奖彩票”），当它从原始随机初始化开始、以相同训练预算训练时，可以达到完整网络的精度。

---

## 剪枝

### 概念与分类

剪枝从已训练网络中移除权重（或整个结构组件）。根本性的权衡是**压缩率**与**精度下降**之间的关系，并通过**微调**（迭代幅值剪枝 + 重新训练）进行调节。

**非结构化剪枝**根据显著性准则（通常为幅值 $|w_{ij}|$）将单个权重元素置零。得到的稀疏权重张量形式上表示为：

$$W_{\text{pruned}} = W \odot M, \quad M_{ij} = \mathbb{1}[|W_{ij}| \geq \tau]$$

其中 $\tau$ 为稀疏阈值，$M$ 为二值掩码。非结构化剪枝能以最小的精度损失实现最高的稀疏度，但需要稀疏张量运算库（cuSPARSE、SpAtten）才能获得硬件加速——稠密 GPU 内核本身并不利用稀疏性。

**结构化剪枝**移除整个滤波器（通道）、注意力头或 Transformer 层，生成更小的稠密张量，无需专用内核即可在标准硬件上加速推理：

```
原始：W ∈ R^{C_out × C_in × k × k}
滤波器剪枝后：W' ∈ R^{C'_out × C_in × k × k},  C'_out < C_out
```

常见的结构化剪枝显著性分数：
- 滤波器权重的 **L1 范数**：$s_j = \sum_{ij} |W_{ij}|$ [Li2017]
- 损失的**泰勒展开**：$\Delta \mathcal{L} \approx -g_j^T w_j$ [Molchanov2017]
- **激活统计**：剪除输出激活均值或方差较低的滤波器

### 彩票假说 [Frankle2019]

彩票假说（LTH）指出：随机初始化的稠密网络 $f(x; \theta_0)$ 包含一个稀疏子网络 $f(x; m \odot \theta_0)$（“中奖彩票”），当它从 $\theta_0$ 开始单独训练时，能在相同的训练迭代次数内达到完整网络的精度。其发现过程是**迭代幅值剪枝（IMP）**：

```
1. 初始化 θ₀ ~ 随机
2. 训练至收敛 → θ_T
3. 剪除 p% 幅值最小的权重 → 掩码 m
4. 将剩余权重重置为 θ₀（而非 θ_T）
5. 重复 2-4 直至达到目标稀疏度
```

关键且违反直觉的一步是将权重**回卷（rewinding）**到 $\theta_0$，而不是置零——重置为 $\theta_0$ 保留了使这张彩票可训练的“幸运初始化”。后续工作 [Frankle2020] 表明，对于大型网络，需要回卷到训练早期的迭代（而非初始化）才能保持稳定。在 CIFAR-10 上，50–90% 稀疏度的中奖彩票可以达到完整网络的精度；在大规模 ImageNet 模型上，结果不太一致，需要较晚的重置点。

**结构化 vs. 非结构化 LTH**：彩票假说在非结构化剪枝中可靠成立；对于结构化剪枝（移除整个滤波器），中奖彩票并不能始终优于基于随机初始化的结构化稀疏网络 [Liu2019]，这表明结构化剪枝更适合被理解为神经架构搜索，而非寻找特殊的初始化点。

---

## 量化

### 基础

量化将浮点权重和/或激活映射到更低精度的定点表示。对于缩放因子为 $s$、零点为 $z$ 的均匀 $b$ 比特量化：

$$x_Q = \text{round}\!\left(\frac{x}{s}\right) + z, \quad x \in [\alpha, \beta]$$
$$s = \frac{\beta - \alpha}{2^b - 1}, \quad z = \text{round}\!\left(-\frac{\alpha}{s}\right)$$

量化值可以用 $b$ 比特存储；反量化得到 $\hat{x} = s(x_Q - z) \approx x$。

| 精度 | 每参数内存 | 典型用途 |
|-----------|-----------------|-------------|
| FP32 | 4 字节 | 训练、参考基准 |
| FP16 / BF16 | 2 字节 | 混合精度训练 |
| INT8 | 1 字节 | 推理（CNN、ViT） |
| INT4 | 0.5 字节 | LLM 推理（GPTQ、AWQ） |
| INT2 | 0.25 字节 | 极限压缩；有精度损失 |
| FP8 (E4M3 / E5M2) | 1 字节 | Hopper/Blackwell 上的推理与训练；FP8 KV 缓存 |
| MXFP4 (OCP) | 约 0.53 字节 | E2M1 数值、32 元素块、2 的幂次（E8M0）缩放因子；例如 gpt-oss 的 MoE 权重 |
| NVFP4 | 约 0.56 字节 | E2M1 数值、16 元素块、FP8（E4M3）缩放因子 + 逐张量 FP32 缩放因子；Blackwell / Jetson Thor |

### 训练后量化（PTQ）

PTQ 使用少量校准数据集（通常为 128–1024 个代表性样本）对已训练的 FP32 模型进行量化，无需重新训练。它速度快，并且不需要访问原始训练流程。

**GPTQ [Frantar2022]** 在量化过程中逐层应用近似二阶更新以最小化重建误差。对于每层权重矩阵 $W$，GPTQ 求解：

$$\underset{\hat{W}}{\arg\min} \|WX - \hat{W}X\|_F^2, \quad \hat{W} \text{ quantized to INT4}$$

它使用源自 Optimal Brain Surgeon 的 Optimal Brain Quantization（OBQ）框架。GPTQ 在 ICLR 2023 上为 LLM 开启了每参数 3–4 比特压缩的“激进区间”。

**AWQ [Lin2023]**（Activation-aware Weight Quantization，MIT/NVIDIA）观察到并非所有权重贡献相同：一小部分（约 1%）“显著”权重——对应高激活输入通道的权重——对输出质量的影响不成比例。AWQ 通过搜索使量化误差最小的逐通道缩放因子 $s_c$ 来保护这些显著权重：

$$\min_{s_c > 0} \|Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X - WX\|$$

在标准学术基准上，AWQ 通常比 GPTQ 高约 0.23 分（0-100 分制），但在真实任务评测中 GPTQ 比 AWQ 高约 2.9 分 [Arxiv2411.02355]——排名取决于任务。

### 量化感知训练（QAT）

QAT 在前向传播中模拟量化，并对不可微的取整操作使用**直通估计器（straight-through estimator, STE）**：

```python
# 前向：量化
x_q = round(x / scale) * scale
# 反向：直通（梯度直接穿过 round）
∂L/∂x ≈ ∂L/∂x_q  # STE
```

在 INT8 下，QAT 通常比 PTQ 多恢复 0.5–2% 的精度，代价是需要完整的再训练（原始训练算力的 10–20%）。对于 INT4，QAT 往往是在困难基准上保持精度的必要手段。

### W4A8 / W8A8 混合精度

近期关于 LLM 服务的工作（LiquidGEMM、SmoothQuant）使用 **W4A8**：权重为 INT4（节省内存带宽），激活为 INT8（以利用 INT8 Tensor Core）。这将内存受限瓶颈（从 DRAM 加载权重）与计算瓶颈（矩阵乘法）解耦。W8A8-INT 的精度损失在可接受的范围内，平均每任务约 1–3%。

### 块缩放浮点格式：FP8、MXFP4、NVFP4（2025–2026）

每个通道只用一个缩放因子的整数量化在 4 比特下表现不佳，因为通道内的离群值会拉伸取值范围。**微缩放（Microscaling）**格式则为每个由连续数值组成的小块附加一个缩放因子，并由硬件在 Tensor Core 的 MMA 指令内部应用。实践中有两种重要的 4 比特变体：

```
元素类型（两者相同）：FP4 E2M1  ->  可表示的幅值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}

MXFP4（OCP 微缩放规范）：
  块大小 = 32 个数值，共享缩放因子 = E8M0（2 的幂）

NVFP4（NVIDIA，Blackwell Tensor Core）：
  块大小 = 16 个数值，块缩放因子 = FP8 E4M3（可为分数，粒度更细）
  + 一个 FP32 逐张量缩放因子（"scale_2"）

块 b 中第 i 个元素的反量化：
  x_hat_i = s_tensor * s_b * q_i,   q_i in E2M1
```

更小的块和非 2 的幂次缩放因子，使 NVFP4 在相同位宽下明显比 MXFP4 更精确，代价是每个数值约 4.5 比特的存储开销。由于权重*和*激活都可以是 FP4，NVFP4 能在 Blackwell 数据中心 GPU、RTX 50 系列显卡以及 Jetson Thor（SM110）边缘模组上实现真正的 **W4A4** 推理（参见[硬件加速](./02_hardware_acceleration.md)）。NVIDIA Model Optimizer（2025年12月由 TensorRT Model Optimizer 更名而来）中的方案将 NVFP4 PTQ 与**量化感知蒸馏（QAD）**结合，即用 BF16 教师模型的输出训练 FP4 学生模型；2026年9月针对 Qwen3.6-35B-A3B 的教程报告，相比 BF16，vLLM 吞吐量最高提升 1.30×、检查点缩小 3.1×，同时恢复了 W4A4 带来的精度损失。模型提供方越来越多地直接*发布*低精度检查点：OpenAI 的 gpt-oss MoE 权重以 MXFP4 发布，Qwen3-VL 提供官方 FP8 检查点，NVIDIA 的 Nemotron 3 系列（包括 2026年10月的视觉-语言模型 Nemotron 3.5 Super VL）提供 NVFP4 版本。**FP8 KV 缓存**量化可将长多模态上下文的 KV 内存大致减半。

---

## 知识蒸馏

### Hinton 等人 2015 [Hinton2015]

知识蒸馏 [Hinton2015] 训练一个紧凑的**学生**网络 $S$ 去模仿一个更大的预训练**教师**网络 $T$，所最小化的损失结合了：
1. **硬目标损失**：与真实标签 $y$ 的标准交叉熵。
2. **软目标损失**：软化后的教师与学生 softmax 输出之间的 KL 散度。

$$\mathcal{L}_{KD} = (1-\alpha) \mathcal{L}_{CE}(y, \sigma(z_S)) + \alpha \tau^2 \cdot D_{KL}\!\left(\sigma\!\left(\frac{z_T}{\tau}\right) \| \sigma\!\left(\frac{z_S}{\tau}\right)\right)$$

**温度** $\tau > 1$ 使概率分布变得平滑，揭示出**暗知识（dark knowledge）**：教师赋予错误类别的相对概率。例如，当图像是“狗”时，教师可能给“猫”赋予 0.01 的概率——这种关系信息对语义相似性的编码远比 one-hot 标签丰富。Hinton 发现，根据学生容量的不同，$\tau \in [2, 8]$ 范围内的温度效果较好。

### 扩展

| 变体 | 核心思想 | 参考文献 |
|---------|----------|-----------|
| FitNets | 匹配中间特征图（hints） | Romero2015 |
| Attention Transfer | 匹配空间注意力图 | Zagoruyko2017 |
| CRD（对比表示蒸馏） | 学生/教师嵌入之间的对比损失 | Tian2020 |
| DeiT | 使用蒸馏 token 从 CNN 教师训练 ViT | Touvron2021 |
| DIST | 用于保持关系的 Pearson 相关性损失 | Huang2022 |

DeiT [Touvron2021] 表明，以 RegNetY-16GF 为教师进行蒸馏训练的 ViT-S/16 仅用 2200 万参数即可在 **ImageNet 上达到 79.8% 的 top-1**——说明在数据受限的情况下（仅用 ImageNet-1K，没有 JFT-300M），蒸馏对训练 ViT 不可或缺。

**基础模型蒸馏（2025–2026）。** 蒸馏如今已成为视觉基础模型的*交付*方式：DINOv3 的 7B 自监督教师被蒸馏为 ViT-S/B/L/H+ 和 ConvNeXt 学生模型；MobileCLIP2（TMLR 2025）利用多模态强化训练（教师集成加合成描述），以约一半的参数达到 SigLIP-SO400M 级别的零样本精度（参见[高效架构](./01_efficient_architectures.md)）。对于 LLM/VLM 骨干，**结构化剪枝后再蒸馏**（NVIDIA 的 Minitron 方案，以及 Model Optimizer 中 2026 年面向 LLM 和 VLM 的异构剪枝/NAS 算法 *Puzzletron*）被用于从已训练的母模型派生出更小的模型，而不是从零开始训练。

---

## 低秩分解

矩阵分解将权重矩阵 $W \in \mathbb{R}^{m \times n}$ 近似为两个低秩矩阵的乘积：

$$W \approx UV^T, \quad U \in \mathbb{R}^{m \times r}, \quad V \in \mathbb{R}^{n \times r}, \quad r \ll \min(m, n)$$

参数量从 $mn$ 降为 $r(m+n)$。对于 $m=n=1024$、$r=64$ 的线性层：$1024^2 \approx 1$M → $64 \times 2048 = 131$K（8× 压缩）。

**Tucker 分解**将其扩展到卷积权重张量 $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$：

$$W \approx G \times_1 U_1 \times_2 U_2$$

其中 $G$ 是较小的核心张量，$U_1, U_2$ 是沿输出和输入通道维度的因子矩阵。

**LoRA [Hu2022]** 最初为 LLM 微调而设计，它使用低秩分解并非为了压缩预训练模型，而是将微调更新参数化为 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$。微调期间只更新 $A$ 和 $B$；原始 $W_0$ 保持冻结。LoRA 以不到全参数更新 1% 的成本取得有竞争力的微调效果，且其适配器可在推理时与基础权重合并（零额外延迟）。

---

## 面向视觉-语言模型的视觉 token 削减

在 VLM 中，LLM 的开销主要由序列长度决定，而视觉 token 往往占据其中大部分（每个图像切块数百个，每段视频数千个）。由于在 LLM 的最初几层之后，对视觉 token 的注意力会变得非常稀疏，许多 token 可以被**剪枝**（丢弃）或**合并**而几乎不损失精度。各方法的区别在于作用的*位置*以及用于对 token 排序的*信号*：

```
在 LLM 第 K 层的通用注意力引导剪枝：
  a_j = 对文本/查询位置 i 取平均的 Attn_K(i -> 视觉 token j)
  按 a_j 保留前 r 比例的视觉 token；在所有 > K 的层中
  丢弃其余 token（节省的 FLOPs 随深度和 1 - r 增长）
```

| 方法 | 作用位置 | 排序信号 | 备注 |
|--------|-------|----------------|-------|
| FastV [Chen2024] | LLM 内部（第 2 层之后） | 来自其他 token 的注意力 | 免训练；理论 FLOPs 减少约 45% 而不损失性能（ECCV 2024 oral） |
| SparseVLM [Zhang2025] | LLM 内部，渐进式 | 文本引导：来自相关文本“评分者”的注意力 | 免训练；回收被剪除的 token；ICML 2025；SparseVLM+ 发表于 TPAMI 2026 |
| VisionZip [Yang2025] | LLM 之前（编码器输出） | 通过 [CLS]/注意力选出主导 token + 合并的上下文 token | 与文本无关，因此兼容 KV 缓存；CVPR 2025；支持 Qwen2.5-VL |
| VisionThink (2025) | 输入分辨率 | 强化学习策略决定何时请求高分辨率图像 | 保留了固定剪枝会丢失的 OCR 密集型精度 |
| InternVL3.5 中的 ViR | 逐 patch，LLM 之前 | 可学习路由器选择压缩率 | 通过视觉一致性学习训练；是所报告的最高 4.05× 加速的一部分 |
| FastVLM / FastViTHD | 视觉编码器设计 | 混合编码器输出更少的 token | CVPR 2025；最小变体的首 token 时间（TTFT）比 LLaVA-OneVision-0.5B 快 85× |

由此形成了两点经验：（1）LLM 内部的*文本引导*剪枝最能保持精度，但与前缀/KV 缓存以及不显式生成注意力图的融合注意力内核相冲突；（2）激进的固定比例剪枝会损害密集文本任务（OCR、图表、文档），因此在这些场景中更倾向于使用自适应方案（VisionThink、ViR）和紧凑型文档编码器（参见[文档与 OCR](../05_multimodal_vision_language/04_document_and_ocr.md)）。

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| Distilling the Knowledge in a Neural Network | Hinton, Vinyals, Dean | 2015 | NeurIPS workshop | 温度软化的软目标；暗知识；奠基性的蒸馏框架 |
| The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | Frankle, Carlin | 2019 | ICLR | 通过带权重回卷的 IMP 找到中奖彩票；非结构化剪枝的理论基础 |
| Rethinking the Value of Network Pruning | Liu et al. | 2019 | ICLR | 结构化剪枝：训练得到的稀疏架构比保留的权重更重要 |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | 2022 | ICLR 2023 | 基于 OBQ 的逐层 INT4 LLM PTQ；首个实用的 4 比特 LLM 推理 |
| AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration | Lin et al. | 2023 | arXiv/MLSys 2024 | 通过逐通道缩放保护显著权重；在学术基准上优于 GPTQ |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2022 | ICLR 2022 | 可训练参数不足 1% 的低秩微调；推理时可合并 |
| Training Data-Efficient Image Transformers & Distillation (DeiT) | Touvron et al. | 2021 | ICML | 用于 ViT 的蒸馏 token；仅用 ImageNet-1K 达到 79.8% ImageNet top-1 |
| An Image is Worth 1/2 Tokens After Layer 2 (FastV) | Chen et al. | 2024 | ECCV (oral) | LLM 内部的免训练视觉 token 剪枝 |
| SparseVLM: Visual Token Sparsification for Efficient VLM Inference | Zhang et al. | 2025 | ICML | 文本引导、免训练的渐进式 token 稀疏化 |
| VisionZip: Longer is Better but Not Necessary in VLMs | Yang et al. | 2025 | CVPR | 在 LLM 之前选择主导 + 上下文视觉 token |
| OCP Microscaling (MX) Formats v1.0 | Open Compute Project | 2023 | 规范 | 块缩放的 FP8/FP6/FP4 格式（MXFP4） |
| NVFP4 (Introducing NVFP4 for Efficient and Accurate Low-Precision Inference) | NVIDIA | 2025 | 技术博客 | 带 E4M3 缩放因子的 16 元素块；Blackwell 上的 W4A4 |

---

## 基准性能

| 方法 | 模型 | 任务 | 指标 | 分数 | 备注 |
|--------|-------|------|--------|-------|-------|
| GPTQ (INT4) | LLaMA-2 7B | 困惑度（WikiText-2） | PPL | ~5.8 | FP16 基线约 5.5；困惑度上升约 0.3 |
| AWQ (INT4) | LLaMA-2 7B | OpenLLM 平均 | 分数 | FP16 − 0.23 分 | 6 个任务的平均（0–100 分制） |
| W8A8-INT | ResNet-50 | ImageNet top-1 | 准确率 | ~75.5% | 对比 FP32 基线 76.1%；损失约 0.6% |
| 结构化剪枝（50%） | ResNet-50 | ImageNet top-1 | 准确率 | ~74.9% | 含微调；滤波器 L1 准则 |
| DeiT-S + 蒸馏 | ViT-S/16 | ImageNet top-1 | 准确率 | 79.8% | 教师：RegNetY-16GF；2200 万参数 |
| 知识蒸馏（ResNet-50→MobileNetV2） | MobileNetV2 | ImageNet top-1 | 准确率 | 72.0% | 无蒸馏时为 68.8% |
| FastV（token 剪枝） | LLaVA-1.5 | 多模态基准 | 理论 FLOPs | −45% | 报告称无性能损失 [Chen2024] |
| NVFP4 W4A4 + QAD | Qwen3.6-35B-A3B | vLLM 服务 | 吞吐量 / 体积 | 1.30× / 缩小 3.1× | 相对 BF16；NVIDIA Model Optimizer 教程（2026年9月） |
| 剪枝 + 蒸馏 + FP8 | Nemotron-3-Nano-30B-A3B | vLLM 服务 | 吞吐量 / 内存 | 2.6× / 降低 2.6× | NVIDIA Model Optimizer 教程（2026年5月） |
| FP8 KV 缓存 | LLM/VLM 解码器 | 上下文注意力 | KV 内存 | 约 −50% | TensorRT Edge-LLM 文档；Blackwell 上注意力快 9–17% |

---

## 优缺点

| 技术 | 优点 | 缺点 |
|-----------|------|------|
| 非结构化剪枝 | 理论压缩率最高；保持架构不变 | 需要稀疏内核才能获得硬件加速；难以在稠密 GPU 上利用 |
| 结构化剪枝 | 硬件友好；在标准硬件上直接加速 | 最大压缩率较低；显著性分数为启发式；可能移除有用容量 |
| PTQ（GPTQ / AWQ） | 无需重新训练；部署快；适用于任何预训练模型 | 小模型在 INT4 下有精度损失；对校准数据敏感 |
| QAT | 低位宽下精度最佳；精确模拟硬件量化 | 需要完整再训练；成本为原始训练的 10–20% |
| 知识蒸馏 | 与任务无关；小学生模型可达到大模型质量 | 需要教师；学生架构需预先选定；教师过大时速度慢 |
| 低秩分解 | 有严谨的线性代数基础；无需特殊训练 | 假设有效秩较低（未必对所有层成立）；分解后需要微调 |
| NVFP4 / MXFP4 | 在 Tensor Core 上实现真正的 W4A4；比 BF16 小约 3.5× | 需要 Blackwell 级硬件；精度恢复常需 QAT/QAD |
| 视觉 token 剪枝 | 为 VLM 大幅节省 FLOPs/延迟，常常免训练 | 高剪枝率下损害 OCR/密集任务；可能与 KV 缓存和融合注意力冲突 |

---

## 开放问题与研究空白

- **对量化误差累积的理论理解**：逐层 PTQ 方法（GPTQ、AWQ）贪心地最小化每层重建损失，但跨层的联合误差累积并未被优化；端到端的 PTQ 目标仍是开放问题。
- **自动化联合压缩**：在无需人工配置的情况下，同时搜索剪枝率、量化位宽和蒸馏教师架构（使用类 NAS 的多目标优化），在大规模上尚未解决。
- **稀疏训练 vs. 事后剪枝**：彩票假说推动了从零开始训练稀疏网络；近期工作（Sparse Transformers、RigL）表明动态稀疏训练在 80% 以上稀疏度时可达到稠密训练的精度，但适用于 ViT 和大模型的可靠方案仍在形成中。
- **激活量化（W4A4）**：W4A8（4 比特权重、8 比特激活）已被充分研究，但 W4A4 由于激活离群值会导致严重精度下降；基于旋转的量化（QuaRot、SpinQuant）部分缓解了这一问题，但并不通用。块缩放的 NVFP4 配合 QAD 在 Blackwell 硬件上缩小了差距，但其在视觉编码器和小型边缘模型上的表现远不如在 LLM 上那样有充分记录。
- **依查询而定的视觉 token 预算**：token 剪枝比例大多由人工固定；如何在不破坏缓存的前提下，针对每幅图像和每个问题决定需要多少视觉 token（以及何种分辨率），仍是开放问题。
- **压缩与定位**：对 VLM 的剪枝或低比特量化主要以问答精度评估；其对定位、OCR 精确性和幻觉率的影响鲜有报告。
- **分布偏移下的压缩**：在分布内校准数据上压缩的模型，在领域偏移下可能比其全精度版本失效得更严重；压缩的鲁棒性研究不足。
- **跨任务蒸馏**：将专精于单一任务的教师蒸馏到处理多任务的学生（多任务蒸馏），同时避免负迁移或容量瓶颈，是一个开放的设计挑战。
- **能耗感知压缩**：大多数压缩方法以 FLOPs 或模型大小为代理指标进行优化；直接针对目标硬件上实测的能耗（取决于内存访问模式和内核利用率）进行优化在很大程度上尚未被探索。

---

## 延伸阅读

- [Hinton 2015 蒸馏论文（arXiv）](https://arxiv.org/abs/1503.02531)
- [Lottery Ticket Hypothesis (Frankle & Carlin, ICLR 2019)](https://arxiv.org/abs/1803.03635)
- [GPTQ 论文（arXiv / ICLR 2023）](https://arxiv.org/abs/2210.17323)
- [AWQ 论文（arXiv 2023）](https://arxiv.org/pdf/2306.00978)
- [SurveyQ: Benchmarking Post-Training Quantization in LLMs (arXiv 2025)](https://arxiv.org/pdf/2502.13178)
- [Neural Network Intelligence (NNI) 工具包 — Microsoft 剪枝/量化框架](https://github.com/microsoft/nni)
- [NVIDIA Model Optimizer（原 TensorRT Model Optimizer）](https://github.com/NVIDIA/Model-Optimizer) — NVFP4/FP8 PTQ、QAT/QAD、剪枝、蒸馏
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference（NVIDIA 博客，2025）](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling Formats (MX) v1.0 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [FastV (arXiv:2403.06764)](https://arxiv.org/abs/2403.06764) — LLM 内部的视觉 token 剪枝
- [SparseVLM (arXiv:2410.04417)](https://arxiv.org/abs/2410.04417) — 文本引导的视觉 token 稀疏化
- [VisionZip (arXiv:2412.04467)](https://arxiv.org/abs/2412.04467) — LLM 之前的视觉 token 选择与合并
