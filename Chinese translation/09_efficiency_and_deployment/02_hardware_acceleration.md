# 硬件加速

> **英文原文：** [cv-knowledge-base/09_efficiency_and_deployment/02_hardware_acceleration.md](../../cv-knowledge-base/09_efficiency_and_deployment/02_hardware_acceleration.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：**
> - [边缘部署](./03_edge_deployment.md)
> - [模型压缩](./00_model_compression.md)
> - [高效架构](./01_efficient_architectures.md)
> - [机器人硬件平台](../06_robotics_and_embodied_ai/12_hardware_robot_platforms.md)

> **🆕 2026年10月更新：**
> - **FP4 成为一等推理精度**：NVFP4 可在 Blackwell / Blackwell Ultra 数据中心 GPU、RTX 50 系列以及 Jetson AGX Thor（SM110）边缘模组上使用。
> - 新增 2025–2026 年的硬件浪潮：Jetson AGX Thor 正式上市（2025年8月，JetPack 7 / CUDA 13）、DGX Spark 桌面级 Grace-Blackwell（2025年10月）、Google TPU v7 Ironwood（以推理为先），以及 Apple M5 的 GPU 神经加速器。
> - 软件栈新增内容：TensorRT Edge-LLM（面向 Jetson、DRIVE 和 DGX Spark 的 VLM/VLA 运行时）、NVIDIA Model Optimizer，以及在 Thor 与 Orin 上统一的 Arm SBSA CUDA。

---

## 概述

硬件加速是整个现代计算机视觉赖以运行的基础：过去十年能力的巨大提升，与使大型神经网络的训练和推理在经济上可行的并行计算硬件（先是 GPU，后是 TPU 和 NPU）密不可分。对研究者而言，具备硬件素养并非可有可无——它决定了哪些模型可以训练、哪些批大小和分辨率可行，以及至关重要的：在目标延迟和功耗范围内能部署什么。该领域涵盖从数据中心训练加速器（NVIDIA H100/B200、Google TPU）到嵌入相机、汽车和机器人中的边缘推理芯片（Jetson、Coral、Hailo）的整个层级（参见[机器人硬件平台](../06_robotics_and_embodied_ai/12_hardware_robot_platforms.md)）。

核心性能原则是**屋顶线模型（roofline model）**：一个内核要么是*计算受限*（受算术吞吐量 FLOPS 限制），要么是*内存受限*（受内存带宽 GB/s 限制），它处于哪种区间决定了优化方式。现代加速器提供了巨大的算力（Tensor Core 以低精度执行融合的矩阵乘加），但内存带宽相对稀缺，因此许多视觉负载——尤其是深度卷积和注意力——都是内存受限的，而主流优化技术（算子融合、量化、混合精度）针对的既是算术，也同样是*数据搬运*。本文梳理加速器格局、将模型映射到硬件上的软件栈（TensorRT、ONNX、XLA），以及决定实际性能的瓶颈。

---

## 加速器格局

- **GPU** —— 主力。NVIDIA GPU 依靠 CUDA 和 **Tensor Core**（支持 FP16/BF16/FP8/INT8 的混合精度矩阵单元）占据主导。数据中心：A100/H100/B200（Hopper/Blackwell）；工作站：RTX 4090/5090。高吞吐 + 高灵活性。
- **TPU**（Google）—— 针对大矩阵乘法优化的脉动阵列 ASIC；通过 JAX/XLA 非常适合大规模训练（ViT-22B）；具备 pod 级互连。
- **NPU / AI 加速器** —— Apple Neural Engine、Qualcomm Hexagon、移动端 NPU；集成于 SoC 中用于端侧推理。
- **边缘加速器** —— NVIDIA Jetson（Orin：最高 275 INT8 TOPS；Thor：Blackwell，128 GB）、Google Coral（Edge TPU）、Hailo-8/15、Intel Movidius。面向机器人/相机的功耗受限推理（参见[边缘部署](./03_edge_deployment.md)）。
- **FPGA** —— 可重构；为专用流水线提供低延迟、确定性推理；编程难度较高。

### 2025–2026 年硬件浪潮

- **Blackwell / Blackwell Ultra（B200 → B300 / GB300 NVL72，2025）。** 第五代 Tensor Core 在 FP8 之外新增原生 **FP4**（NVFP4 和 MXFP4），使 4 比特权重*和*激活都能在 Tensor Core 上运行（参见[模型压缩](./00_model_compression.md)）。NVIDIA 已宣布 Rubin 一代作为其 2026 年的后继平台。
- **DGX Spark（GB10 Grace-Blackwell，2025年10月出货）。** 一台桌面系统，具备 128 GB CPU-GPU 统一内存和最高约 1 PFLOP 的稀疏 FP4 算力，适合在本地对最多约 2000 亿参数的 VLM 进行推理和微调；两台设备可以互联（TensorRT Edge-LLM 于 2026 年增加了在双 DGX Spark 上的实验性张量并行推理）。
- **Jetson AGX Thor（2025年8月正式上市）。** Blackwell 架构 GPU（计算能力 **SM110**），最高约 2,070 稀疏 FP4 TFLOPS、128 GB 内存、40–130 W 功耗范围；NVIDIA 称其 AI 算力约为 Jetson AGX Orin 的 7.5 倍。它运行 JetPack 7（CUDA 13）；自 CUDA 13.2 / JetPack 7.2 起，同一套 Arm SBSA CUDA 工具包覆盖 Thor、Orin 以及服务器端 Grace 系统。其目标是在机器人本体上运行 GR00T、π0.5 等 VLM 和 VLA（参见[边缘部署](./03_edge_deployment.md)）。
- **Google TPU v7 “Ironwood”（2025年4月发布，2025年末正式上市）。** Google 首款主要定位于推理的 TPU，每颗芯片 192 GB HBM，pod 规模最多 9,216 颗芯片。
- **终端 SoC。** Apple 的 M5（2025年10月）在每个 GPU 核心中加入了神经加速器，Apple 称其 AI 峰值 GPU 算力超过 M4 的 4 倍；手机 NPU（Qualcomm Hexagon、Apple Neural Engine、MediaTek APU）如今已能常规运行 10–40 亿参数的多模态模型。

## 屋顶线模型与精度

```
# 屋顶线：可达到的性能
P = min( Peak_FLOPS,  Bandwidth × Arithmetic_Intensity )
# 算术强度 = FLOPs / 搬运的字节数
# 低算术强度（深度卷积、注意力）→ 内存受限
```

**降低精度**是提升吞吐的首要手段：FP32 → FP16/BF16（训练），→ INT8/FP8/INT4（推理），自 Blackwell 起硬件支持 → **FP4**（NVFP4/MXFP4）。Tensor Core 在更低精度下能带来成倍的加速，而量化（参见[模型压缩](./00_model_compression.md)）既减少了内存流量，又能启用这些计算单元。块缩放格式将部分反量化操作移入 MMA 指令本身，因此缩放因子几乎不增加额外的内存流量：

```
每个权重的有效比特数（存储含缩放因子）：
  BF16 = 16      FP8 = 8      MXFP4 = 4 + 8/32 = 4.25      NVFP4 = 4 + 8/16（+ 极小的逐张量 FP32）≈ 4.5
带宽受限解码相对 BF16 的加速比（理想情况）≈ 16 / 比特数  ->  NVFP4 约 3.5x
```

## 软件栈

将模型高效映射到硬件上需要一套编译/运行时栈：

- **TensorRT**（NVIDIA）—— 图优化、层/算子融合、精度校准、内核自动调优，用于 GPU/Jetson 推理。
- **ONNX / ONNX Runtime** —— 交换格式 + 带执行提供程序（execution provider）的跨平台运行时。
- **XLA**（TensorFlow/JAX）和 **torch.compile / TorchInductor** —— 融合算子并生成优化内核的图编译器。
- **TVM**、**OpenVINO**（Intel）、厂商 SDK —— 面向特定硬件的编译。
- **NVIDIA Model Optimizer**（2025年12月由 TensorRT Model Optimizer 更名）—— 生成 FP8/NVFP4/INT4 检查点（PTQ、QAT、QAD），支持剪枝与蒸馏，可导出到 TensorRT、TensorRT-LLM、vLLM 或 SGLang。
- **TensorRT Edge-LLM** —— NVIDIA 面向 Jetson、DRIVE 和 DGX Spark 的 C++ 运行时，可运行文本、视觉、音频和动作模型，具备 Thor 专用 NVFP4 内核、FP8 KV 缓存、推测解码，并支持 VLM（Qwen3-VL/3.5、Gemma 4）和实验性的 VLA 策略（π0.5）。
- **vLLM / SGLang** —— 数据中心中事实上的开源 VLM 服务引擎，支持多模态前缀缓存以及 FP8/FP4 内核。

**算子融合**（将逐元素/归一化操作合并为一个内核，避免往返 HBM）和**内核自动调优**是收益最高的优化，直接针对内存受限瓶颈。

```mermaid
graph LR
    A["已训练模型<br/>PyTorch/JAX"] --> B["导出 ONNX / 计算图"]
    B --> C["编译：TensorRT / XLA / TVM<br/>融合 + 精度"]
    C --> D["加速器上的运行时<br/>GPU/TPU/NPU/Jetson"]
    style C fill:#1d3557,color:#fff
    style D fill:#2d6a4f,color:#fff
```

---

## 关键参考文献

| 参考文献 | 机构/作者 | 年份 | 类型 | 主要贡献 |
|-----------|-------------|------|------|-----------------|
| Roofline Model | Williams, Waterman, Patterson | 2009 | CACM | 计算受限 vs 内存受限分析 |
| Tensor Cores / Mixed Precision | Micikevicius et al. (NVIDIA) | 2018 | ICLR | 大规模 FP16 训练 |
| TPU | Jouppi et al. (Google) | 2017 | ISCA | 数据中心内的脉动阵列加速器 |
| FlashAttention | Dao et al. | 2022 | NeurIPS | IO 感知的精确注意力；解决内存受限问题 |
| Jetson Thor | NVIDIA | 2025 | 产品 | 面向机器人的 Blackwell 边缘算力；2025年8月正式上市；SM110；FP4 |
| OCP Microscaling (MX) Formats | Open Compute Project (AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm) | 2023 | 规范 | 块缩放的 FP8/FP6/FP4 格式 |
| NVFP4 | NVIDIA | 2025 | 格式 / 博客 | 以 E4M3 缩放的 16 元素 FP4 块，用于 W4A4 推理 |
| DGX Spark | NVIDIA | 2025 | 产品 | 桌面级 Grace-Blackwell，128 GB 统一内存 |
| TPU v7 Ironwood | Google | 2025 | 产品 | 以推理为先的 TPU 世代 |
| TensorRT Edge-LLM | NVIDIA | 2025–2026 | 运行时 | 面向 Jetson/DRIVE/Spark 的端侧 LLM/VLM/VLA 推理 |

---

## 优缺点（加速器选择）

| 方面 | GPU | TPU | 边缘 NPU/Jetson |
|--------|-----|-----|-----------------|
| 灵活性 | 高（CUDA 生态） | 较低（受限于 XLA） | 算子支持有限 |
| 吞吐量 | 非常高 | 非常高（大矩阵乘） | 受功耗限制 |
| 部署 | 数据中心/工作站 | 数据中心（Google） | 端侧、机器人 |
| 精度支持 | FP16/FP8/INT8；FP4（Blackwell 及以后） | BF16/INT8；FP8（较新世代） | 侧重 INT8/INT4；Jetson Thor 支持 FP8/FP4 |

---

## 开放问题与研究空白

- **内存墙。** 带宽的增长慢于算力；许多视觉/注意力内核受内存限制（FlashAttention 式的 IO 感知设计只解决了部分问题）。
- **边缘端的算子覆盖。** 边缘加速器支持的算子集合有限，迫使模型重新设计（参见[边缘部署](./03_edge_deployment.md)）。
- **编译器成熟度。** 与手工调优内核相比，自动调优/融合（TVM、Inductor）仍有性能未被挖掘。
- **精度鲁棒性。** INT4/FP8 推理可能会不可预测地降低某些视觉模型的精度。
- **能效。** 越来越多地限制部署的是功耗而非 FLOPS（机器人续航；数据中心成本）。
- **异构调度。** 在 SoC 上高效地将模型划分到 CPU/GPU/NPU 之间仍未解决。
- **视觉领域的 FP4。** NVFP4 的精度在 LLM 解码器上已被充分刻画，但在视觉编码器、稠密预测头以及扩散/世界模型上远未如此；逐层精度分配（例如 AutoQuantize 式搜索）仍是启发式的。
- **格式碎片化。** MXFP4、NVFP4、各厂商的 INT4 方案以及 FP8 变体在 GPU、NPU 和 TPU 之间不可互换，增加了可移植部署的难度。

---

## 延伸阅读

- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) — IO 感知注意力
- [Mixed Precision Training (arXiv:1710.03740)](https://arxiv.org/abs/1710.03740) — FP16 训练
- [TensorRT 文档](https://developer.nvidia.com/tensorrt) — 推理优化
- [In-Datacenter TPU (arXiv:1704.04760)](https://arxiv.org/abs/1704.04760) — TPU 架构
- [Roofline Model (CACM 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785) — 性能分析
- [Introducing NVFP4（NVIDIA 博客，2025）](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — Blackwell 上的块缩放 FP4
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — MXFP8/MXFP6/MXFP4 规范
- [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) — 面向 LLM、VLM 和 VLA 的边缘运行时
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) — 量化/剪枝/蒸馏工具包
