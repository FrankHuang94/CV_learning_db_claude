# Model Compression: Pruning, Quantization, Distillation, and Low-Rank Factorization

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:**
> - [Efficient Architectures](./01_efficient_architectures.md)
> - [Hardware Acceleration](./02_hardware_acceleration.md)
> - [Edge Deployment](./03_edge_deployment.md)
> - [Core Architectures](../03_architectures/00_cnns_and_convolutions.md)

> **🆕 October 2026 update:**
> - New subsection on **block-scaled low-precision formats**: FP8 (E4M3/E5M2), OCP **MXFP4** and NVIDIA **NVFP4** (E2M1 values + FP8 micro-block scales), now native on Blackwell-class GPUs and Jetson Thor, with W4A4 NVFP4 + quantization-aware distillation (QAD) recipes.
> - New section on **visual-token pruning and merging for VLMs** (FastV, SparseVLM/SparseVLM+, VisionZip, VisionThink, InternVL3.5's Visual Resolution Router), the main compression lever specific to multimodal models.
> - Added pruning-plus-distillation pipelines for LLM/VLM backbones (Minitron, Puzzletron in NVIDIA Model Optimizer) and FP8 KV-cache quantisation.
> - Updated Key Papers, Benchmark Performance and Open Problems accordingly.

---

## Overview

Model compression is the set of techniques that reduce the memory footprint, computational cost, and/or inference latency of a trained neural network while preserving as much of its predictive accuracy as possible. The motivation is straightforward: state-of-the-art vision models (ViT-L, CLIP, Swin-L) require billions of floating-point operations per image and gigabytes of parameter storage, placing them out of reach for embedded devices, mobile phones, and latency-sensitive production services. Compression methods attack this gap from four complementary angles: **pruning** (removing parameters), **quantization** (reducing numerical precision), **knowledge distillation** (transferring knowledge to a smaller architecture), and **low-rank factorization** (approximating weight matrices with products of smaller matrices).

These techniques are not mutually exclusive. Modern production pipelines routinely combine quantization-aware training (QAT) with structured pruning and distillation from a larger teacher model. For large language models (LLMs) applied to vision-language tasks, post-training quantization methods (GPTQ, AWQ) have become the dominant deployment path, compressing 7B–70B parameter models to 4-bit precision with W8A8 showing ~1–3% average task accuracy loss. Since 2025 two further levers dominate multimodal deployment: **hardware-native block-scaled FP formats** (FP8, MXFP4, NVFP4), which let 4-bit weights *and* activations run on tensor cores, and **visual-token reduction**, which attacks the fact that a single high-resolution image or video can contribute thousands of tokens to a VLM's sequence.

A central theoretical underpinning is the **overparameterization hypothesis**: modern networks are vastly overparameterized relative to the task complexity, meaning large fractions of parameters can be removed (pruned) or their precision reduced without catastrophic loss of function. The **Lottery Ticket Hypothesis** [Frankle2019] makes this precise: a large network contains a small, sparse subnetwork (a "winning ticket") that, when trained from its original random initialization with the same training budget, matches the full network's accuracy.

---

## Pruning

### Concepts and Taxonomy

Pruning removes weights (or entire structural components) from a trained network. The fundamental trade-off is **compression ratio** vs. **accuracy degradation**, modulated by **fine-tuning** (iterative magnitude pruning + retraining).

**Unstructured pruning** sets individual weight elements to zero based on a saliency criterion (typically magnitude $|w_{ij}|$). The resulting sparse weight tensors are formally represented as:

$$W_{\text{pruned}} = W \odot M, \quad M_{ij} = \mathbb{1}[|W_{ij}| \geq \tau]$$

where $\tau$ is a sparsity threshold and $M$ is a binary mask. Unstructured pruning achieves the highest sparsity at minimal accuracy loss but requires sparse tensor arithmetic libraries (cuSPARSE, SpAtten) to realize hardware speedups—dense GPU kernels do not exploit sparsity natively.

**Structured pruning** removes entire filters (channels), attention heads, or transformer layers, producing dense smaller tensors that accelerate inference on standard hardware without specialized kernels:

```
Original: W ∈ R^{C_out × C_in × k × k}
After filter pruning: W' ∈ R^{C'_out × C_in × k × k},  C'_out < C_out
```

Common structured pruning saliency scores:
- **L1 norm** of filter weights: $s_j = \sum_{ij} |W_{ij}|$ [Li2017]
- **Taylor expansion** of the loss: $\Delta \mathcal{L} \approx -g_j^T w_j$ [Molchanov2017]
- **Activation statistics**: prune filters with low mean or variance of output activations

### Lottery Ticket Hypothesis [Frankle2019]

The Lottery Ticket Hypothesis (LTH) states: a randomly initialized dense network $f(x; \theta_0)$ contains a sparse subnetwork $f(x; m \odot \theta_0)$ (the "winning ticket") such that when trained in isolation from $\theta_0$, it matches the full network's accuracy in the same number of training iterations. The discovery procedure is **iterative magnitude pruning (IMP)**:

```
1. Initialize θ₀ ~ random
2. Train to convergence → θ_T
3. Prune p% of lowest-magnitude weights → mask m
4. Reset remaining weights to θ₀ (not θ_T)
5. Repeat 2-4 until target sparsity
```

The critical and counter-intuitive step is **weight rewinding** to $\theta_0$, not to zero—resetting to $\theta_0$ preserves the "lucky initialization" that made the ticket trainable. Later work [Frankle2020] showed that for large networks, rewinding to early training iterations (rather than initialization) is necessary for stability. On CIFAR-10, winning tickets at 50–90% sparsity match full network accuracy; on large-scale ImageNet models, results are less consistent and require late resetting.

**Structured vs. unstructured LTH**: the lottery ticket hypothesis holds reliably for unstructured pruning; for structured pruning (removing entire filters), winning tickets do not consistently outperform random initialization-based structured sparse networks [Liu2019], suggesting structured pruning is better understood as neural architecture search than as identifying special initialization points.

---

## Quantization

### Fundamentals

Quantization maps floating-point weights and/or activations to a lower-precision fixed-point representation. For uniform $b$-bit quantization with scale $s$ and zero-point $z$:

$$x_Q = \text{round}\!\left(\frac{x}{s}\right) + z, \quad x \in [\alpha, \beta]$$
$$s = \frac{\beta - \alpha}{2^b - 1}, \quad z = \text{round}\!\left(-\frac{\alpha}{s}\right)$$

The quantized value can be stored in $b$ bits; dequantization recovers $\hat{x} = s(x_Q - z) \approx x$.

| Precision | Memory per param | Typical use |
|-----------|-----------------|-------------|
| FP32 | 4 bytes | Training, reference |
| FP16 / BF16 | 2 bytes | Mixed-precision training |
| INT8 | 1 byte | Inference (CNN, ViT) |
| INT4 | 0.5 bytes | LLM inference (GPTQ, AWQ) |
| INT2 | 0.25 bytes | Extreme compression; accuracy loss |
| FP8 (E4M3 / E5M2) | 1 byte | Inference and training on Hopper/Blackwell; FP8 KV cache |
| MXFP4 (OCP) | ~0.53 bytes | E2M1 values, 32-element blocks, power-of-two (E8M0) scale; e.g. gpt-oss MoE weights |
| NVFP4 | ~0.56 bytes | E2M1 values, 16-element blocks, FP8 (E4M3) scale + per-tensor FP32 scale; Blackwell / Jetson Thor |

### Post-Training Quantization (PTQ)

PTQ applies quantization to an already-trained FP32 model using a small calibration dataset (typically 128–1024 representative samples) without retraining. It is fast and requires no access to the original training pipeline.

**GPTQ [Frantar2022]** applies an approximate second-order update to minimize the reconstruction error layer-by-layer during quantization. For each layer weight matrix $W$, GPTQ solves:

$$\underset{\hat{W}}{\arg\min} \|WX - \hat{W}X\|_F^2, \quad \hat{W} \text{ quantized to INT4}$$

using the Optimal Brain Quantization (OBQ) framework derived from Optimal Brain Surgeon. GPTQ opened the "aggressive zone" of 3–4 bit per parameter compression for LLMs at ICLR 2023.

**AWQ [Lin2023]** (Activation-aware Weight Quantization, MIT/NVIDIA) observed that not all weights contribute equally: a small fraction (~1%) of "salient" weights—those corresponding to high-activation input channels—disproportionately affect output quality. AWQ protects these salient weights by searching for a per-channel scaling factor $s_c$ that minimizes quantization error:

$$\min_{s_c > 0} \|Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X - WX\|$$

AWQ generally outperforms GPTQ on standard academic benchmarks by ~0.23 points (0-100 scale), though GPTQ outperforms AWQ on real-world task evaluations by ~2.9 points [Arxiv2411.02355] — the ranking is task-dependent.

### Quantization-Aware Training (QAT)

QAT simulates quantization during forward passes using **straight-through estimators (STE)** for the non-differentiable rounding operation:

```python
# Forward: quantize
x_q = round(x / scale) * scale
# Backward: straight-through (gradient passes through round)
∂L/∂x ≈ ∂L/∂x_q  # STE
```

QAT typically recovers 0.5–2% accuracy compared to PTQ at INT8, at the cost of requiring full retraining (10–20% of original training compute). For INT4, QAT is often essential to maintain accuracy on difficult benchmarks.

### W4A8 / W8A8 Mixed Precision

Recent work on LLM serving (LiquidGEMM, SmoothQuant) uses **W4A8**: weights in INT4 (saving memory bandwidth), activations in INT8 (enabling INT8 tensor core utilization). This decouples the memory-bound bottleneck (weight loading from DRAM) from the compute bottleneck (matrix multiply). W8A8-INT shows tolerable accuracy loss in the 1–3% per-task average range.

### Block-Scaled Floating-Point Formats: FP8, MXFP4, NVFP4 (2025–2026)

Integer quantization with one scale per channel struggles at 4 bits because outliers inside a channel stretch the range. **Microscaling** formats instead attach a scale to every small block of contiguous values, and the hardware applies it inside the tensor-core MMA. Two 4-bit variants matter in practice:

```
Element type (both): FP4 E2M1  ->  representable magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}

MXFP4 (OCP Microscaling spec):
  block size = 32 values, shared scale = E8M0 (a power of two)

NVFP4 (NVIDIA, Blackwell tensor cores):
  block size = 16 values, block scale = FP8 E4M3 (fractional, finer-grained)
  + one FP32 per-tensor scale ("scale_2")

Dequantisation of element i in block b:
  x_hat_i = s_tensor * s_b * q_i,   q_i in E2M1
```

The smaller block and non-power-of-two scale make NVFP4 noticeably more accurate than MXFP4 at the same bit width, at a storage cost of ~4.5 bits per value. Because both weights *and* activations can be FP4, NVFP4 enables genuine **W4A4** inference on Blackwell data-center GPUs, RTX 50-series cards and the Jetson Thor (SM110) edge module (see [Hardware Acceleration](./02_hardware_acceleration.md)). Recipes in NVIDIA Model Optimizer (renamed from TensorRT Model Optimizer in Dec 2025) combine NVFP4 PTQ with **quantization-aware distillation (QAD)**, where the FP4 student is trained against the BF16 teacher's outputs; a Sept 2026 tutorial for Qwen3.6-35B-A3B reports up to 1.30× vLLM throughput over BF16 and 3.1× smaller checkpoints with the W4A4 accuracy loss recovered. Model producers increasingly *release* low-precision checkpoints directly: OpenAI's gpt-oss MoE weights ship in MXFP4, Qwen3-VL has official FP8 checkpoints, and NVIDIA's Nemotron 3 family (including the Nemotron 3.5 Super VL vision-language model, Oct 2026) ships NVFP4 versions. **FP8 KV-cache** quantization roughly halves KV memory for long multimodal contexts.

---

## Knowledge Distillation

### Hinton et al. 2015 [Hinton2015]

Knowledge distillation [Hinton2015] trains a compact **student** network $S$ to mimic a larger pre-trained **teacher** network $T$ by minimizing a loss that combines:
1. **Hard target loss**: standard cross-entropy with ground-truth labels $y$.
2. **Soft target loss**: KL divergence between softened teacher and student softmax outputs.

$$\mathcal{L}_{KD} = (1-\alpha) \mathcal{L}_{CE}(y, \sigma(z_S)) + \alpha \tau^2 \cdot D_{KL}\!\left(\sigma\!\left(\frac{z_T}{\tau}\right) \| \sigma\!\left(\frac{z_S}{\tau}\right)\right)$$

The **temperature** $\tau > 1$ softens the probability distribution, revealing **dark knowledge**: the relative probabilities the teacher assigns to incorrect classes. For example, a teacher may assign probability 0.01 to "cat" when the image is a "dog"—this relational information encodes semantic similarity far more richly than the one-hot label. Hinton found that temperatures in the range $\tau \in [2, 8]$ work well depending on the student capacity.

### Extensions

| Variant | Key Idea | Reference |
|---------|----------|-----------|
| FitNets | Match intermediate feature maps (hints) | Romero2015 |
| Attention Transfer | Match spatial attention maps | Zagoruyko2017 |
| CRD (Contrastive Representation Distillation) | Contrastive loss between student/teacher embeddings | Tian2020 |
| DeiT | Distillation token for ViT training from CNN teacher | Touvron2021 |
| DIST | Pearson correlation loss for relation-preserving distillation | Huang2022 |

DeiT [Touvron2021] showed that a ViT-S/16 trained with distillation from a RegNetY-16GF teacher achieves **79.8% top-1 on ImageNet** with only 22M parameters—demonstrating that distillation is indispensable for training ViTs in the data-limited setting (ImageNet-1K without JFT-300M).

**Foundation-model distillation (2025–2026).** Distillation is now how vision foundation models are *shipped*: DINOv3's 7B self-supervised teacher is distilled into ViT-S/B/L/H+ and ConvNeXt students, and MobileCLIP2 (TMLR 2025) uses multi-modal reinforced training (teacher ensembles plus synthetic captions) to reach SigLIP-SO400M-level zero-shot accuracy with roughly half the parameters (see [Efficient Architectures](./01_efficient_architectures.md)). For LLM/VLM backbones, **structured pruning followed by distillation** (NVIDIA's Minitron recipe, and the 2026 *Puzzletron* heterogeneous pruning/NAS algorithm for LLMs and VLMs in Model Optimizer) is used to derive smaller models from a trained parent instead of training them from scratch.

---

## Low-Rank Factorization

Matrix factorization approximates a weight matrix $W \in \mathbb{R}^{m \times n}$ as a product of two low-rank matrices:

$$W \approx UV^T, \quad U \in \mathbb{R}^{m \times r}, \quad V \in \mathbb{R}^{n \times r}, \quad r \ll \min(m, n)$$

Parameter reduction: from $mn$ to $r(m+n)$. For a linear layer with $m=n=1024$ and $r=64$: $1024^2 \approx 1$M → $64 \times 2048 = 131$K (8× compression).

**Tucker decomposition** extends this to convolution weight tensors $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$:

$$W \approx G \times_1 U_1 \times_2 U_2$$

where $G$ is a smaller core tensor and $U_1, U_2$ are factor matrices along output and input channel dimensions.

---

## Visual-Token Reduction for Vision-Language Models

In a VLM the LLM's cost is dominated by sequence length, and visual tokens often make up most of it (hundreds per image tile, thousands per video). Since attention to visual tokens becomes very sparse after the first few LLM layers, many tokens can be **pruned** (dropped) or **merged** with little accuracy loss. Methods differ in *where* they act and *what signal* ranks tokens:

```
Generic attention-guided pruning at LLM layer K:
  a_j = mean over text/query positions i of Attn_K(i -> visual token j)
  keep the top-r fraction of visual tokens by a_j; drop the rest
  for all layers > K  (FLOPs saved grow with depth and with 1 - r)
```

| Method | Where | Ranking signal | Notes |
|--------|-------|----------------|-------|
| FastV [Chen2024] | Inside the LLM (after layer 2) | Attention received from other tokens | Training-free; ~45% theoretical FLOPs reduction without performance loss (ECCV 2024 oral) |
| SparseVLM [Zhang2025] | Inside the LLM, progressive | Text-guided: attention from relevant text "raters" | Training-free; recycles pruned tokens; ICML 2025; SparseVLM+ in TPAMI 2026 |
| VisionZip [Yang2025] | Before the LLM (encoder output) | Dominant tokens via [CLS]/attention + merged contextual tokens | Text-agnostic, so compatible with KV caching; CVPR 2025; Qwen2.5-VL support |
| VisionThink (2025) | Input resolution | RL policy decides when to request a high-resolution image | Keeps OCR-heavy accuracy that fixed pruning loses |
| ViR in InternVL3.5 | Per patch, before the LLM | Learned router chooses compression rate | Trained via visual consistency learning; part of a reported up-to-4.05× speedup |
| FastVLM / FastViTHD | Vision encoder design | Fewer tokens emitted by a hybrid encoder | CVPR 2025; 85× faster TTFT than LLaVA-OneVision-0.5B for the smallest variant |

Two lessons have emerged: (1) *text-guided* pruning inside the LLM preserves accuracy best but conflicts with prefix/KV caching and with fused attention kernels that never materialise attention maps; (2) aggressive fixed-ratio pruning hurts dense-text tasks (OCR, charts, documents), which is why adaptive schemes (VisionThink, ViR) and compact document encoders (see [Document & OCR](../05_multimodal_vision_language/04_document_and_ocr.md)) are preferred there.

---

**LoRA [Hu2022]**, originally designed for LLM fine-tuning, applies low-rank decomposition not to compress a pretrained model but to parameterize fine-tuning updates as $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. During fine-tuning, only $A$ and $B$ are updated; the original $W_0$ is frozen. LoRA achieves competitive fine-tuning performance at <1% of the full-parameter update cost, and its adapters are mergeable with the base weights at inference (zero added latency).

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Distilling the Knowledge in a Neural Network | Hinton, Vinyals, Dean | 2015 | NeurIPS workshop | Temperature-softened soft targets; dark knowledge; foundational distillation framework |
| The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | Frankle, Carlin | 2019 | ICLR | Winning tickets via IMP with weight rewinding; theoretical basis for unstructured pruning |
| Rethinking the Value of Network Pruning | Liu et al. | 2019 | ICLR | Structured pruning: trained sparse architecture matters more than preserved weights |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | 2022 | ICLR 2023 | OBQ-based layer-wise INT4 PTQ for LLMs; first practical 4-bit LLM inference |
| AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration | Lin et al. | 2023 | arXiv/MLSys 2024 | Salient weight protection via per-channel scale; outperforms GPTQ on academic benchmarks |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2022 | ICLR 2022 | Low-rank fine-tuning at <1% trainable params; merge-at-inference |
| Training Data-Efficient Image Transformers & Distillation (DeiT) | Touvron et al. | 2021 | ICML | Distillation token for ViT; 79.8% ImageNet top-1 from ImageNet-1K only |
| An Image is Worth 1/2 Tokens After Layer 2 (FastV) | Chen et al. | 2024 | ECCV (oral) | Training-free visual-token pruning inside the LLM |
| SparseVLM: Visual Token Sparsification for Efficient VLM Inference | Zhang et al. | 2025 | ICML | Text-guided, training-free progressive token sparsification |
| VisionZip: Longer is Better but Not Necessary in VLMs | Yang et al. | 2025 | CVPR | Select dominant + contextual visual tokens before the LLM |
| OCP Microscaling (MX) Formats v1.0 | Open Compute Project | 2023 | Specification | Block-scaled FP8/FP6/FP4 formats (MXFP4) |
| NVFP4 (Introducing NVFP4 for Efficient and Accurate Low-Precision Inference) | NVIDIA | 2025 | Technical blog | 16-element blocks with E4M3 scales; W4A4 on Blackwell |

---

## Benchmark Performance

| Method | Model | Task | Metric | Score | Notes |
|--------|-------|------|--------|-------|-------|
| GPTQ (INT4) | LLaMA-2 7B | Perplexity (WikiText-2) | PPL | ~5.8 | FP16 baseline ~5.5; ~0.3 PPL degradation |
| AWQ (INT4) | LLaMA-2 7B | OpenLLM avg | Score | FP16 − 0.23 pts | Avg over 6 tasks (0–100 scale) |
| W8A8-INT | ResNet-50 | ImageNet top-1 | Accuracy | ~75.5% | vs. FP32 baseline 76.1%; ~0.6% loss |
| Structured pruning (50%) | ResNet-50 | ImageNet top-1 | Accuracy | ~74.9% | With fine-tuning; filter L1 criterion |
| DeiT-S + distillation | ViT-S/16 | ImageNet top-1 | Accuracy | 79.8% | Teacher: RegNetY-16GF; 22M params |
| Knowledge distillation (ResNet-50→MobileNetV2) | MobileNetV2 | ImageNet top-1 | Accuracy | 72.0% | vs. 68.8% without distillation |
| FastV (token pruning) | LLaVA-1.5 | Multimodal benchmarks | Theoretical FLOPs | −45% | Reported without performance loss [Chen2024] |
| NVFP4 W4A4 + QAD | Qwen3.6-35B-A3B | vLLM serving | Throughput / size | 1.30× / 3.1× smaller | vs. BF16; NVIDIA Model Optimizer tutorial (Sept 2026) |
| Pruning + distillation + FP8 | Nemotron-3-Nano-30B-A3B | vLLM serving | Throughput / memory | 2.6× / 2.6× lower | NVIDIA Model Optimizer tutorial (May 2026) |
| FP8 KV cache | LLM/VLM decoders | Context attention | KV memory | ~−50% | TensorRT Edge-LLM docs; 9–17% faster attention on Blackwell |

---

## Pros & Cons

| Technique | Pros | Cons |
|-----------|------|------|
| Unstructured pruning | Highest theoretical compression ratio; preserves architecture | Requires sparse kernels for hardware speedup; difficult to exploit on dense GPUs |
| Structured pruning | Hardware-friendly; direct speedup on standard hardware | Lower maximum compression; heuristic saliency scores; may remove useful capacity |
| PTQ (GPTQ / AWQ) | No retraining; fast deployment; works on any pretrained model | Accuracy loss at INT4 for small models; calibration data sensitivity |
| QAT | Best accuracy at low bit-width; models hardware quantization exactly | Requires full retraining; 10–20% of original training cost |
| Knowledge distillation | Task-agnostic; can match larger model quality with small student | Teacher required; student architecture must be pre-selected; slow if teacher is huge |
| Low-rank factorization | Principled linear algebra basis; no special training required | Assumes low effective rank (may not hold for all layers); fine-tuning needed after factorization |
| NVFP4 / MXFP4 | True W4A4 on tensor cores; ~3.5× smaller than BF16 | Needs Blackwell-class hardware; accuracy recovery often requires QAT/QAD |
| Visual-token pruning | Large FLOP/latency savings for VLMs, often training-free | Hurts OCR/dense tasks at high ratios; can conflict with KV caching and fused attention |

---

## Open Problems & Research Gaps

- **Theoretical understanding of quantization error accumulation**: layer-wise PTQ methods (GPTQ, AWQ) greedily minimize per-layer reconstruction loss, but joint error accumulation across layers is not optimized; end-to-end PTQ objectives remain an open problem.
- **Automated joint compression**: simultaneously searching over pruning ratios, quantization bit-widths, and distillation teacher architectures (using NAS-like multi-objective optimization) without manual configuration is unsolved at scale.
- **Sparse training vs. post-hoc pruning**: the lottery ticket hypothesis motivated training sparse networks from scratch; recent work (Sparse Transformers, RigL) shows dynamic sparse training can match dense training accuracy at 80%+ sparsity, but reliable protocols for ViTs and large models are still emerging.
- **Quantization for activations (W4A4)**: while W4A8 (4-bit weights, 8-bit activations) is well-studied, W4A4 causes severe accuracy degradation due to activation outlier values; rotation-based quantization (QuaRot, SpinQuant) partially addresses this but is not general. Block-scaled NVFP4 with QAD narrows the gap on Blackwell hardware, but behaviour on vision encoders and small edge models is far less documented than for LLMs.
- **Query-aware visual-token budgets**: token pruning ratios are mostly fixed by hand; deciding per image and per question how many visual tokens (and what resolution) are needed, without breaking caching, remains open.
- **Compression and grounding**: pruning or low-bit quantization of VLMs is evaluated mainly on QA accuracy; effects on grounding, OCR exactness and hallucination rates are rarely reported.
- **Compression under distribution shift**: models compressed on in-distribution calibration data may fail more severely under domain shift than their full-precision counterparts; compression-robustness is understudied.
- **Cross-task distillation**: distilling from a single teacher specializing in one task to a student handling multiple tasks (multi-task distillation) without negative transfer or capacity bottlenecks is an open design challenge.
- **Energy-aware compression**: most compression methods optimize for FLOPs or model size as proxies; directly optimizing for measured energy consumption on target hardware (which depends on memory access patterns and kernel utilization) is largely unexplored.

---

## Further Reading

- [Hinton 2015 distillation arXiv](https://arxiv.org/abs/1503.02531)
- [Lottery Ticket Hypothesis (Frankle & Carlin, ICLR 2019)](https://arxiv.org/abs/1803.03635)
- [GPTQ paper (arXiv / ICLR 2023)](https://arxiv.org/abs/2210.17323)
- [AWQ paper (arXiv 2023)](https://arxiv.org/pdf/2306.00978)
- [SurveyQ: Benchmarking Post-Training Quantization in LLMs (arXiv 2025)](https://arxiv.org/pdf/2502.13178)
- [Neural Network Intelligence (NNI) toolkit — Microsoft pruning/quantization framework](https://github.com/microsoft/nni)
- [NVIDIA Model Optimizer (formerly TensorRT Model Optimizer)](https://github.com/NVIDIA/Model-Optimizer) — NVFP4/FP8 PTQ, QAT/QAD, pruning, distillation
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA blog, 2025)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling Formats (MX) v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [FastV (arXiv:2403.06764)](https://arxiv.org/abs/2403.06764) — visual-token pruning inside the LLM
- [SparseVLM (arXiv:2410.04417)](https://arxiv.org/abs/2410.04417) — text-guided visual-token sparsification
- [VisionZip (arXiv:2412.04467)](https://arxiv.org/abs/2412.04467) — pre-LLM visual-token selection and merging
