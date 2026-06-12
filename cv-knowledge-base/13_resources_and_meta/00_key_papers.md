# 50 Must-Read Computer Vision Papers

> **Last Updated:** June 2026
> **Level:** Foundational
> **Related Sections:**
> - [Timeline Overview](../01_history/00_timeline_overview.md)
> - [Key Venues](./01_key_venues.md)
> - [PhD Study Guide](./04_phd_study_guide.md)
> - [Latest Developments](../12_research_frontier_2024_2026/00_overview_latest.md)

---

## Overview

This is a curated reading list of fifty papers that a computer-vision PhD student should know—not necessarily the fifty *most cited*, but the fifty that best convey the field's conceptual structure and trajectory from the deep-learning revolution to the embodied-AI frontier. They are organized by topic. For each: title, authors, year, venue, and a one-line statement of why it matters. The list is deliberately weighted toward the architectural and paradigm-shifting works that recur as building blocks throughout this database; consult the linked section files for technical depth on each.

Reading strategy: a newcomer should start with the **classification backbones** (1–6) and **the Transformer/CLIP cluster** (16–20) to understand the modern substrate, then branch into the task and frontier areas matching their interests. The 2024–2026 entries are recent and partly non-archival (company reports); treat their reported results as preliminary pending replication (see [Latest Developments](../12_research_frontier_2024_2026/00_overview_latest.md)).

---

## Classification & Backbones

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 1 | ImageNet | Deng et al. | 2009 | CVPR | The dataset that enabled deep learning in vision |
| 2 | AlexNet | Krizhevsky, Sutskever, Hinton | 2012 | NeurIPS | Started the deep-learning revolution |
| 3 | VGGNet | Simonyan, Zisserman | 2015 | ICLR | Depth via uniform 3×3 convolutions |
| 4 | GoogLeNet/Inception | Szegedy et al. | 2015 | CVPR | Multi-branch efficiency |
| 5 | ResNet | He et al. | 2016 | CVPR | Residual connections; trainable very-deep nets |
| 6 | EfficientNet | Tan, Le | 2019 | ICML | Compound scaling of depth/width/resolution |
| 7 | ConvNeXt | Liu et al. | 2022 | CVPR | Modernized CNN matching Transformers |

## Detection & Segmentation

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 8 | R-CNN | Girshick et al. | 2014 | CVPR | CNN features for detection |
| 9 | Faster R-CNN | Ren et al. | 2015 | NeurIPS | Learned region proposals; two-stage template |
| 10 | YOLO | Redmon et al. | 2016 | CVPR | Real-time single-stage detection |
| 11 | Focal Loss / RetinaNet | Lin et al. | 2017 | ICCV | Solved foreground-background imbalance |
| 12 | DETR | Carion et al. | 2020 | ECCV | End-to-end detection as set prediction |
| 13 | FCN | Long et al. | 2015 | CVPR | Fully convolutional segmentation |
| 14 | U-Net | Ronneberger et al. | 2015 | MICCAI | Encoder-decoder; medical-imaging standard |
| 15 | Mask R-CNN | He et al. | 2017 | ICCV | Unified detection + instance segmentation |

## Transformers, CLIP & Self-Supervision

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 16 | Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | The Transformer |
| 17 | ViT | Dosovitskiy et al. | 2021 | ICLR | Pure Transformer for vision at scale |
| 18 | Swin Transformer | Liu et al. | 2021 | ICCV | Hierarchical shifted-window attention |
| 19 | CLIP | Radford et al. | 2021 | ICML | Contrastive vision-language; zero-shot transfer |
| 20 | MAE | He et al. | 2022 | CVPR | Masked autoencoders; scalable SSL |
| 21 | DINO | Caron et al. | 2021 | ICCV | Self-distillation; emergent segmentation |
| 22 | DINOv2 | Oquab et al. | 2023 | TMLR | General-purpose self-supervised features |
| 23 | MoCo | He et al. | 2020 | CVPR | Momentum contrast; SSL at scale |
| 24 | SimCLR | Chen et al. | 2020 | ICML | Simple contrastive framework |
| 25 | Mamba | Gu, Dao | 2023 | COLM 2024 | Linear-time selective state spaces |

## 3D Vision & Scene

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 26 | NeRF | Mildenhall et al. | 2020 | ECCV | Neural radiance fields; novel-view synthesis |
| 27 | Instant-NGP | Müller et al. | 2022 | SIGGRAPH | Hash encoding; ~5-min NeRF training |
| 28 | 3D Gaussian Splatting | Kerbl et al. | 2023 | SIGGRAPH | Real-time explicit 3D; superseded NeRF practically |
| 29 | COLMAP | Schönberger, Frahm | 2016 | CVPR | Standard structure-from-motion |
| 30 | DUSt3R | Wang et al. | 2024 | CVPR | Pointmap regression; learned MVS |
| 31 | PointNet | Qi et al. | 2017 | CVPR | Deep learning on raw point clouds |

## Generative Vision

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 32 | GAN | Goodfellow et al. | 2014 | NeurIPS | Adversarial generation |
| 33 | StyleGAN | Karras et al. | 2019 | CVPR | Photorealistic, controllable synthesis |
| 34 | DDPM | Ho et al. | 2020 | NeurIPS | Denoising diffusion |
| 35 | Latent Diffusion (SD) | Rombach et al. | 2022 | CVPR | Diffusion in latent space; Stable Diffusion |
| 36 | Classifier-Free Guidance | Ho, Salimans | 2022 | NeurIPS WS | Core conditioning technique for diffusion |
| 37 | Sora (tech report) | OpenAI | 2024 | — | Spacetime-patch DiT video generation |

## Multimodal & VLMs

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 38 | Flamingo | Alayrac et al. | 2022 | NeurIPS | Few-shot interleaved vision-language |
| 39 | BLIP-2 | Li et al. | 2023 | ICML | Q-Former connector to frozen LLMs |
| 40 | LLaVA | Liu et al. | 2023 | NeurIPS | Visual instruction tuning |
| 41 | GLIP | Li et al. | 2022 | CVPR | Grounded language-image pretraining |

## Foundation, VLA & World Models (2023–2026)

| # | Paper | Authors | Year | Venue | Why it matters |
|---|-------|---------|------|-------|----------------|
| 42 | Segment Anything (SAM) | Kirillov et al. | 2023 | ICCV | Promptable segmentation + billion-mask engine |
| 43 | Depth Anything V2 | Yang et al. | 2024 | NeurIPS | Foundation monocular depth |
| 44 | RT-1 | Brohan et al. | 2022 | RSS 2023 | Robotics transformer at scale |
| 45 | RT-2 | Brohan et al. | 2023 | CoRL | VLA: web knowledge → robot control |
| 46 | Open X-Embodiment | OXE Collaboration | 2023 | ICRA 2024 | Cross-embodiment dataset & RT-X |
| 47 | π₀ | Black et al. | 2024 | arXiv (PI) | Flow-matching VLA; high dexterity |
| 48 | Diffusion Policy | Chi et al. | 2023 | RSS | Diffusion over action trajectories |
| 49 | DreamerV3 | Hafner et al. | 2023 | ICLR 2024 | World-model RL; Minecraft diamonds |
| 50 | Genie | Bruce et al. | 2024 | ICML | Generative interactive environments from video |

---

## Pros & Cons (using a canonical list)

| Aspect | Pros | Cons |
|--------|------|------|
| Curated canon | Efficient on-ramp; shared vocabulary | Risk of canon bias; omits niche gems |
| Topic organization | Easy to branch by interest | Cross-cutting works fit awkwardly |
| Recency inclusion | Captures live frontier | 2024–26 entries may not replicate |

---

## Open Problems & Research Gaps (meta)

- **Canon bias.** Highly cited backbone papers crowd out important applied/dataset/theory work.
- **Recency vs. durability.** Which 2024–2026 works will prove foundational is unknown.
- **Underrepresented areas.** Fairness, robustness, video, and low-resource vision are under-weighted in most canons.
- **Reproducibility.** Several landmark results (closed VLMs/VLAs) cannot be independently verified.
- **Geographic/lab concentration.** The canon over-represents a few labs.
- **Survey gap.** Fast-moving subfields (VLA, world models) lack settled authoritative surveys.

---

## Further Reading

- [CVF Open Access](https://openaccess.thecvf.com/) — free proceedings for most listed papers
- [Semantic Scholar](https://www.semanticscholar.org/) — citation graphs and influence
- [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent) — preprint firehose
- [Timeline Overview](../01_history/00_timeline_overview.md) — historical context for this list
- [Latest Developments](../12_research_frontier_2024_2026/00_overview_latest.md) — the 2024–2026 frontier
