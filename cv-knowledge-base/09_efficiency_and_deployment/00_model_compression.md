# Model Compression: Pruning, Quantization, Distillation, and Low-Rank Factorization

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:**
> - [Efficient Architectures](./01_efficient_architectures.md)
> - [Hardware Acceleration](./02_hardware_acceleration.md)
> - [Edge Deployment](./03_edge_deployment.md)
> - [Core Architectures](../03_architectures/00_cnns_and_convolutions.md)

---

## Overview

Model compression is the set of techniques that reduce the memory footprint, computational cost, and/or inference latency of a trained neural network while preserving as much of its predictive accuracy as possible. The motivation is straightforward: state-of-the-art vision models (ViT-L, CLIP, Swin-L) require billions of floating-point operations per image and gigabytes of parameter storage, placing them out of reach for embedded devices, mobile phones, and latency-sensitive production services. Compression methods attack this gap from four complementary angles: **pruning** (removing parameters), **quantization** (reducing numerical precision), **knowledge distillation** (transferring knowledge to a smaller architecture), and **low-rank factorization** (approximating weight matrices with products of smaller matrices).

These techniques are not mutually exclusive. Modern production pipelines routinely combine quantization-aware training (QAT) with structured pruning and distillation from a larger teacher model. For large language models (LLMs) applied to vision-language tasks, post-training quantization methods (GPTQ, AWQ) have become the dominant deployment path, compressing 7B–70B parameter models to 4-bit precision with W8A8 showing ~1–3% average task accuracy loss.

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

---

## Low-Rank Factorization

Matrix factorization approximates a weight matrix $W \in \mathbb{R}^{m \times n}$ as a product of two low-rank matrices:

$$W \approx UV^T, \quad U \in \mathbb{R}^{m \times r}, \quad V \in \mathbb{R}^{n \times r}, \quad r \ll \min(m, n)$$

Parameter reduction: from $mn$ to $r(m+n)$. For a linear layer with $m=n=1024$ and $r=64$: $1024^2 \approx 1$M → $64 \times 2048 = 131$K (8× compression).

**Tucker decomposition** extends this to convolution weight tensors $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$:

$$W \approx G \times_1 U_1 \times_2 U_2$$

where $G$ is a smaller core tensor and $U_1, U_2$ are factor matrices along output and input channel dimensions.

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

---

## Open Problems & Research Gaps

- **Theoretical understanding of quantization error accumulation**: layer-wise PTQ methods (GPTQ, AWQ) greedily minimize per-layer reconstruction loss, but joint error accumulation across layers is not optimized; end-to-end PTQ objectives remain an open problem.
- **Automated joint compression**: simultaneously searching over pruning ratios, quantization bit-widths, and distillation teacher architectures (using NAS-like multi-objective optimization) without manual configuration is unsolved at scale.
- **Sparse training vs. post-hoc pruning**: the lottery ticket hypothesis motivated training sparse networks from scratch; recent work (Sparse Transformers, RigL) shows dynamic sparse training can match dense training accuracy at 80%+ sparsity, but reliable protocols for ViTs and large models are still emerging.
- **Quantization for activations (W4A4)**: while W4A8 (4-bit weights, 8-bit activations) is well-studied, W4A4 causes severe accuracy degradation due to activation outlier values; rotation-based quantization (QuaRot, SpinQuant) partially addresses this but is not general.
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
