# Vision Transformers

> **Last Updated:** June 2026
> **Level:** Intermediate
> **Related Sections:**
> - [CNN Architectures](./00_cnn_architectures.md)
> - [Self-Supervised Learning](./03_self_supervised_learning.md)
> - [State Space Models](./05_state_space_models.md)
> - [Scaling Laws in Vision](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)

---

## Overview

The Vision Transformer (ViT) [Dosovitskiy2021] demonstrated that the convolution is not a necessary inductive bias for state-of-the-art image recognition: a pure Transformer encoder applied to a sequence of image patches matches or exceeds CNNs *when pretrained at sufficient scale*. This was a conceptual rupture. CNNs hard-code locality and translation equivariance into every layer; ViT discards them, treating an image as an unordered set of 16×16 patch tokens and learning spatial structure from data and learned positional embeddings. The cost is sample efficiency—ViT underperforms CNNs when trained on ImageNet-1K alone—and the benefit is *scalability*: ViT's performance keeps climbing with data and parameters along a smoother curve than CNNs, the property that made it the substrate of foundation models (see [Scaling Laws in Vision](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)).

The architectural story since 2020 has two threads. The first is **reintroducing inductive bias** for efficiency and small-data performance: DeiT recovered ImageNet-1K-only training via distillation; Swin reintroduced hierarchy and locality via shifted-window attention, making Transformers viable dense-prediction backbones. The second is **scale and self-supervision**: BEiT and EVA applied masked image modeling; ViT-22B pushed to 22 billion parameters with architectural modifications (parallel attention/MLP, QK-normalization) needed for stable training at that scale. This file traces both threads, grounds the attention mechanism mathematically, and delineates where ViTs win over CNNs and where the convolution still competes.

---

## The Core Mechanism

ViT splits a `H×W` image into `N = HW/P²` non-overlapping `P×P` patches (P=16 by default → 196 patches at 224×224), linearly projects each flattened patch to a `D`-dimensional token, prepends a learnable `[CLS]` token (whose final state is the classification feature, à la BERT), and adds **learned 1D positional embeddings** (an ablation in [Dosovitskiy2021] found 2D/relative encodings gave no meaningful gain). The token sequence passes through a standard Transformer encoder. The heart is scaled dot-product self-attention [Vaswani2017]:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
  Q = X W_Q,  K = X W_K,  V = X W_V    (per head)
  √d_k scaling keeps logits in the high-gradient region of softmax
```

Multi-head attention runs `h` such operations in parallel and concatenates. Crucially, self-attention is **global**—every token attends to every other—giving an O(1)-hop receptive field but O(N²) complexity in token count, the bottleneck that later variants attack.

```mermaid
graph LR
    A[Image 224x224] --> B[Patchify 16x16<br/>→ 196 tokens]
    B --> C[Linear projection → D=768]
    C --> D[+ [CLS] token<br/>+ learned pos embeddings]
    D --> E[Transformer encoder<br/>L× MHSA + MLP]
    E --> F[[CLS] state → MLP head]
    style E fill:#2d6a4f,color:#fff
```

ViT-B/16 uses 12 layers, D=768, 12 heads, ~86M params; ViT-L/16 has 24 layers, D=1024, 16 heads, ~307M; ViT-H/14 reaches ~632M.

---

## The Evolution

### Data Efficiency: DeiT

**DeiT** [Touvron2021] (ICML 2021) showed ViT-scale models *can* train competitively on ImageNet-1K alone (1.2M images) with aggressive augmentation (RandAugment, Mixup, CutMix) and a novel **distillation token**: a second learnable token trained via hard-label distillation from a CNN teacher (RegNet). DeiT-B reaches 81.8% top-1 (no distillation) and 85.2% with token distillation—erasing ViT's dependence on JFT-300M.

### Hierarchy & Locality: Swin

**Swin Transformer** [Liu2021] (ICCV 2021, Marr Prize) computes self-attention within local non-overlapping windows (e.g., 7×7), shifting windows by `⌊M/2⌋` in alternating layers to allow cross-window flow. This makes complexity **linear in image size** and produces a **hierarchical** multi-scale feature pyramid (4×/8×/16×/32×), making Swin a drop-in backbone for detection and segmentation where plain ViT's single scale is awkward. Swin-L reaches 87.3% at 384². **PVT** and **MViT** independently introduced pyramidal Transformers with spatial-reduction/pooling attention for dense prediction and video.

### Self-Supervision & Scale: BEiT, EVA, ViT-22B

**BEiT** [Bao2022] (ICLR 2022) adapted BERT's masked-token objective: mask ~40% of patches and predict their discrete visual tokens (from a dVAE), reaching 88.6% with 22k fine-tuning at 512². **EVA** [Fang2023] (CVPR 2023) scaled a vanilla ViT to ~1B parameters by reconstructing masked *CLIP features*, hitting 89.6% at 336². **ViT-22B** [Dehghani2023] (ICML 2023) scaled to 22 billion parameters using **parallel attention+MLP**, **QK-normalization** (LayerNorm on queries/keys to stabilize training), no biases, and multi-head attention pooling—achieving 89.5% linear-probe and large gains in robustness/fairness, though with clearly sublinear per-FLOP returns (see [Scaling Laws](../12_research_frontier_2024_2026/01_scaling_laws_vision.md)).

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| An Image is Worth 16×16 Words (ViT) | Dosovitskiy, Beyer, Kolesnikov, et al. | 2021 | ICLR | Pure Transformer for image recognition at scale |
| Training data-efficient image transformers (DeiT) | Touvron, Cord, Douze, et al. | 2021 | ICML | Distillation token; ImageNet-1K-only training |
| Swin Transformer | Liu, Lin, Cao, Hu, et al. | 2021 | ICCV (Marr Prize) | Shifted-window attention; hierarchical backbone |
| BEiT | Bao, Dong, Piao, Wei | 2022 | ICLR | Masked image modeling with discrete visual tokens |
| ViT-22B | Dehghani, Djolonga, Mustafa, et al. | 2023 | ICML | 22B-param ViT; parallel attn+MLP, QK-norm |
| EVA | Fang, Wang, Xie, et al. | 2023 | CVPR | 1B-param ViT via masked CLIP-feature reconstruction |

---

## Benchmark Performance

| Model | Pretraining | Resolution | ImageNet top-1 | Notes |
|-------|-------------|-----------|----------------|-------|
| ViT-H/14 | JFT-300M | 224 | 88.55% | Largest ViT in original paper |
| DeiT-B (distil.) | IN-1K | 224 | 85.2% | No external data |
| Swin-L | IN-22K→1K | 384 | 87.3% | Hierarchical backbone |
| BEiT-L | self-sup + 22K FT | 512 | 88.6% | Masked image modeling |
| EVA | masked CLIP feat. | 336 | 89.6% | ~1B params |
| ViT-22B | JFT | 224 | 89.5% (linear probe) | Frozen features |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Global attention | O(1)-hop receptive field; long-range context | O(N²) cost; poor high-res scaling without windowing |
| Scalability | Smooth gains with data/params; foundation-model substrate | Data-hungry; weak on small datasets from scratch |
| Hierarchy (Swin/PVT) | Multi-scale; dense-prediction-ready; linear cost | More engineered; loses ViT's simplicity |
| Self-supervision (BEiT/EVA) | Leverages unlabeled data; strong transfer | Heavy compute; tokenizer/teacher dependence |

---

## Open Problems & Research Gaps

- **Quadratic attention at high resolution.** Window/linear-attention and state-space models (see [State Space Models](./05_state_space_models.md)) mitigate but don't fully resolve the cost.
- **Small-data training.** ViTs still trail CNNs/hybrids without large-scale pretraining (see [Hybrid Architectures](./02_hybrid_architectures.md)).
- **Positional encoding for variable resolution.** Robust extrapolation to unseen resolutions/aspect ratios remains imperfect.
- **Interpretability.** Attention maps are not faithful explanations; mechanistic understanding of ViT features is immature.
- **Compute-optimal scaling.** No agreed vision Chinchilla law; exponents differ from language.
- **Inductive-bias balance.** The optimal amount of convolutional prior to inject is unresolved—ConvNeXt shows CNNs can match ViTs, blurring the dichotomy.

---

## Further Reading

- [ViT (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — the original Vision Transformer
- [DeiT (arXiv:2012.12877)](https://arxiv.org/abs/2012.12877) — data-efficient training & distillation
- [Swin Transformer (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030) — shifted-window hierarchical attention
- [ViT-22B (arXiv:2302.05442)](https://arxiv.org/abs/2302.05442) — scaling to 22 billion parameters
- [EVA (arXiv:2211.07636)](https://arxiv.org/abs/2211.07636) — billion-scale masked visual representation learning
