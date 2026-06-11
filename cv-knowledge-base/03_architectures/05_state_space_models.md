# State Space Models for Vision

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:**
> - [Vision Transformers](./01_vision_transformers.md)
> - [Hybrid Architectures](./02_hybrid_architectures.md)
> - [Efficient Architectures](../09_efficiency_and_deployment/01_efficient_architectures.md)

---

## Overview

State space models (SSMs) are the principal challenger to self-attention for long-sequence and high-resolution vision, offering **linear-time sequence modeling** in place of attention's quadratic cost. An SSM maps an input sequence to an output through a latent recurrent state governed by continuous-time linear dynamics, discretized for digital computation. The breakthrough that made SSMs competitive was **S4** [Gu2022], which combined a principled HiPPO initialization with an efficient diagonal-plus-low-rank parameterization, and **Mamba** [Gu2023], which added *input-dependent (selective)* parameters and a hardware-aware parallel scan—closing the long-standing quality gap to Transformers on language while retaining O(N) complexity and O(1) per-token inference memory.

For vision, the appeal is acute: self-attention's O(N²) cost in token count is the dominant bottleneck at high resolution, exactly where dense-prediction and robotics perception need to operate. Vision SSMs (Vim, VMamba) adapt Mamba's inherently 1D, causal scan to the 2D, non-causal structure of images via bidirectional and multi-directional scanning, achieving accuracy comparable to Swin at substantially lower memory and faster high-resolution throughput. This file develops the SSM mathematics, the selective-scan innovation, the vision adaptations, and the regime where linear-time models beat attention.

---

## The Mathematics

An SSM models a continuous system with latent state `h(t)`:

```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)        (D often = 0, the direct feedthrough)
```

To apply it to discrete tokens with step size Δ, **zero-order-hold (ZOH) discretization** gives:

```
Ā = exp(ΔA)
B̄ = (ΔA)⁻¹ (exp(ΔA) − I) (ΔB)
hₜ = Ā hₜ₋₁ + B̄ xₜ
yₜ = C hₜ
```

This admits a **dual view**: a *recurrence* (O(1) memory per step, ideal for inference) and an equivalent *global convolution* with a structured kernel (parallelizable for training, like a Transformer). S4 [Gu2022] (ICLR 2022, Outstanding Paper Honorable Mention) made the convolution efficient by initializing `A` with the **HiPPO-LegS** matrix (principled long-range memory via orthogonal-polynomial projection) and decomposing it as **diagonal-plus-low-rank (DPLR)**, reducing the kernel cost from O(N²) to O(N log N).

### Selectivity: Mamba

S4 is **linear time-invariant (LTI)**—`A, B, C` are fixed, so the model cannot condition its memory on content. **Mamba** [Gu2023] makes `B`, `C`, and `Δ` functions of the input (only `A` stays fixed):

```
Bₜ = Linear_B(xₜ);  Cₜ = Linear_C(xₜ);  Δₜ = softplus(Linear_Δ(xₜ))
```

Selectivity in `Δ` lets the model *ignore* irrelevant tokens (Δ→0, no state update) or *memorize* important ones. But input-dependent parameters break the time-invariance the convolution relied on, forcing a recurrent scan. Mamba keeps this fast with a **hardware-aware parallel scan**: an associative prefix scan (O(log N) depth), **kernel fusion** (discretization + scan + projection in one CUDA kernel, avoiding materializing the expanded state in HBM), and **recomputation** during backprop. The result is ~5× higher inference throughput than comparable Transformers with O(1) memory per token.

```mermaid
graph LR
    A[Input tokens] --> B[Selective params<br/>B,C,Δ = f(x)]
    B --> C[Hardware-aware parallel scan<br/>associative prefix scan]
    C --> D[Linear-time output<br/>O(N) train, O(1)/token infer]
    style C fill:#1d3557,color:#fff
    style D fill:#2d6a4f,color:#fff
```

---

## Vision Adaptations

Images are 2D and non-causal, but Mamba's scan is 1D and causal. The two leading vision SSMs solve this differently:

**Vision Mamba (Vim)** [Zhu2024] (ICML 2024) patchifies the image (ViT-style), adds position embeddings, and processes tokens through **bidirectional** Vim blocks (forward + backward scans combined), so no token has asymmetric context. Vim is **2.8× faster than DeiT** and uses **86.8% less GPU memory** at 1248×1248 resolution. Vim-S (26M) reaches 80.5% top-1, edging DeiT-S (79.8%) at similar size.

**VMamba** [Liu2024] (NeurIPS 2024 Spotlight) introduces **2D Selective Scan (SS2D)** with a **Cross-Scan** module: the feature map is unfolded into *four* 1D sequences (the two diagonal directions and their reverses), each run through a selective SSM, then merged—so every patch receives context from all four spatial directions, analogous to a 2D convolution. VMamba-T (~31M) reaches 82.5%, beating Swin-T (81.2%) at comparable FLOPs, with consistent +0.4–1.3 pp margins across scales.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Efficiently Modeling Long Sequences (S4) | Gu, Goel, Ré | 2022 | ICLR (Outstanding HM) | HiPPO init + DPLR; efficient long-range SSM |
| Mamba | Gu, Dao | 2023 | arXiv→COLM 2024 | Selective SSM + hardware-aware parallel scan |
| Vision Mamba (Vim) | Zhu, Liao, Zhang, et al. | 2024 | ICML | Bidirectional SSM for non-causal vision |
| VMamba | Liu, Tian, Zhao, et al. | 2024 | NeurIPS (Spotlight) | 2D Selective Scan (Cross-Scan, SS2D) |
| HiPPO | Gu, Dao, Ermon, Rudra, Ré | 2020 | NeurIPS | Polynomial-projection memory theory underlying S4 |

---

## Benchmark Performance

| Model | Params | ImageNet top-1 | Efficiency note | Reference |
|-------|--------|----------------|-----------------|-----------|
| Vim-S | 26M | 80.5% | 2.8× faster, −86.8% memory vs DeiT @1248² | [Zhu2024] |
| Vim-B | 98M | 81.9% | — | [Zhu2024] |
| VMamba-T | 31M | 82.5% | +1.3 pp over Swin-T @ comparable FLOPs | [Liu2024] |
| VMamba-S | 50M | 83.6% | — | [Liu2024] |
| VMamba-B | 89M | 83.9% | +0.4 pp over Swin-B | [Liu2024] |

*Complexity: self-attention is O(N²); SSMs are O(N) in sequence length (O(N·D²) training with the parallel scan, D≪N).*

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Linear complexity | O(N) vs attention O(N²); scales to high resolution | Quadratic in state dim D (usually small) |
| Inference memory | O(1) per token (recurrent state) | Selective scan needs custom CUDA kernels |
| Long-range modeling | HiPPO-grounded memory; strong on long sequences | Less mature tooling/ecosystem than Transformers |
| Vision accuracy | Beats Swin at comparable FLOPs (VMamba) | Multi-directional scan adds design complexity |

---

## Open Problems & Research Gaps

- **Canonical 2D scan.** No consensus on the best way to impose 1D ordering on 2D images; Cross-Scan is effective but ad hoc.
- **Pretraining at scale.** Whether vision SSMs match Transformers under large-scale self-supervised pretraining (MAE/DINO-style, see [Self-Supervised Learning](./03_self_supervised_learning.md)) is under-explored.
- **Hardware support.** SSM kernels are less optimized across accelerators than attention; portability and on-device support lag.
- **Hybrid SSM-attention.** The optimal combination of selective scan and local attention for vision is open (cf. [Hybrid Architectures](./02_hybrid_architectures.md)).
- **Dense prediction.** SSM backbones for detection/segmentation are promising but less validated than Swin.
- **Theoretical understanding** of what selectivity learns, and its expressivity relative to attention, remains incomplete.
- **Multimodal SSMs.** Extending selective state spaces to vision-language and video at scale is early.

---

## Further Reading

- [Mamba (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — selective state spaces, linear-time sequence modeling
- [S4 (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — structured state spaces for long sequences
- [Vision Mamba / Vim (arXiv:2401.09417)](https://arxiv.org/abs/2401.09417) — bidirectional SSM for vision
- [VMamba (arXiv:2401.10166)](https://arxiv.org/abs/2401.10166) — 2D selective scan visual state space model
- [HiPPO (arXiv:2008.07669)](https://arxiv.org/abs/2008.07669) — the memory theory underlying S4/Mamba
