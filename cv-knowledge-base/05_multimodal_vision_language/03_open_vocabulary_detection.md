# Open-Vocabulary Detection & Segmentation

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:**
> - [Object Detection](../02_core_tasks/01_object_detection.md)
> - [The "Anything" Model Paradigm](../12_research_frontier_2024_2026/02_any_model_paradigm.md)
> - [VLP Models](./01_vlp_models.md)
> - [Large Vision-Language Models](./02_large_vision_language_models.md)

---

## Overview

Open-vocabulary detection (OVD) and segmentation aim to localize and recognize object categories *not seen during training*, specified at inference time by arbitrary text. This breaks the fundamental constraint of classical detectors (R-CNN, YOLO, DETR—see [Object Detection](../02_core_tasks/01_object_detection.md)), which can only predict from a fixed label set defined at training time. The enabling idea is to replace the closed-set classification head with **alignment to a language embedding space**: a region's visual feature is matched against text embeddings (typically from CLIP), so any concept expressible in language becomes a candidate class. This reframes detection as a vision-language grounding problem and is the localization counterpart to the universal segmentation models in [The "Anything" Model Paradigm](../12_research_frontier_2024_2026/02_any_model_paradigm.md).

The field splits along two strategies. **Embedding-alignment methods** (ViLD, RegionCLIP, OWL-ViT) distill or align region features to CLIP's text space, inheriting CLIP's open-vocabulary knowledge. **Grounding-pretraining methods** (GLIP, Grounding DINO) reformulate detection as phrase grounding and pretrain on large-scale detection + grounding + caption data, unifying detection and referring expression comprehension. A third practical line (Detic, YOLO-World) emphasizes scaling vocabulary via image-level labels or real-time efficiency. This file surveys these approaches, their benchmarks (LVIS, ODinW), and the open challenges of localization quality versus semantic coverage.

---

## Embedding-Alignment Methods

**ViLD** [Gu2021] (ICLR 2022) distills CLIP image embeddings into a two-stage detector: a class-agnostic region proposal network generates boxes, and region features are trained to match CLIP's image embeddings (ViLD-image) and text embeddings of category names (ViLD-text). At inference, novel categories are detected by embedding their names. **RegionCLIP** extends CLIP to region-level alignment via region-text pretraining. **OWL-ViT** [Minderer2022] (ECCV 2022) and **OWLv2** take a ViT-based one-stage approach: a CLIP-pretrained ViT with detection heads, fine-tuned on detection data, enabling efficient open-vocabulary and one-shot (image-conditioned) detection. OWLv2 scales via self-training on pseudo-annotations.

```mermaid
graph LR
    I[Image] --> R[Region proposals /<br/>per-patch features]
    R --> V[Region visual embeddings]
    T[Text prompts<br/>'a photo of a {category}'] --> TE[CLIP text encoder]
    V --> M[Cosine similarity]
    TE --> M
    M --> O[Open-vocab boxes + labels]
    style M fill:#1d3557,color:#fff
```

---

## Grounding-Pretraining Methods

**GLIP** [Li2022b] (CVPR 2022) unifies object detection and phrase grounding: it reformulates detection as aligning region features with words in a text prompt, pretraining on 27M grounding pairs (including image-text with self-trained boxes). This gives strong zero-shot transfer and a single model for detection and referring. **Grounding DINO** [Liu2023b] marries the DINO DETR-style detector with grounded pretraining, achieving strong open-set detection from arbitrary text and becoming the detection half of **Grounded-SAM** (text → boxes → masks via SAM). These models excel at language-driven localization but are heavier than embedding-alignment detectors.

---

## Scaling and Efficiency

**Detic** [Zhou2022] (ECCV 2022) expands the detectable vocabulary to ~21k classes by training the *classifier* on image-level labels (ImageNet-21k), decoupling localization from classification and sidestepping the need for box annotations on rare classes. **YOLO-World** [Cheng2024] (CVPR 2024) brings open-vocabulary detection to **real-time** by re-parameterizing CLIP text embeddings into a YOLO detector with a vision-language path aggregation network, achieving high LVIS zero-shot AP at YOLO speeds—important for robotics and edge deployment.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| ViLD | Gu, Lin, Kuo, Cui | 2021 | ICLR 2022 | Distill CLIP embeddings into a detector |
| GLIP | Li, Zhang, Zhang, et al. | 2022 | CVPR | Unify detection & phrase grounding pretraining |
| OWL-ViT | Minderer, Gritsenko, Stone, et al. | 2022 | ECCV | ViT-based open-vocab & one-shot detection |
| Detic | Zhou, Girdhar, Joulin, et al. | 2022 | ECCV | 21k-class detection via image-level labels |
| Grounding DINO | Liu, Zeng, Ren, et al. | 2023 | arXiv→ECCV 2024 | Grounded DETR; basis of Grounded-SAM |
| YOLO-World | Cheng, Song, Ge, et al. | 2024 | CVPR | Real-time open-vocabulary detection |

---

## Benchmark Performance

| Model | Benchmark | Metric | Score | Notes |
|-------|-----------|--------|-------|-------|
| ViLD | LVIS | AP_rare (novel) | ~16.1 | Two-stage distillation [Gu2021] |
| GLIP-L | LVIS / ODinW | zero-shot AP | strong | Grounding pretraining [Li2022b] |
| OWLv2 | LVIS | zero-shot AP | SOTA (2023) | Self-training scaling |
| Detic | LVIS | AP_rare | ~24+ | Image-level supervision [Zhou2022] |
| YOLO-World-L | LVIS | zero-shot AP | ~35 AP @ real-time | Edge-friendly [Cheng2024] |

*Numbers approximate; LVIS rare-class AP is the standard open-vocab metric.*

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Embedding alignment (ViLD/OWL) | Inherits CLIP knowledge; simple | Localization quality bounded by proposals/CLIP |
| Grounding pretraining (GLIP/G-DINO) | Strong language-driven localization; referring | Heavy; slower; data-intensive |
| Image-level scaling (Detic) | Huge vocabulary cheaply | Weaker localization on rare classes |
| Real-time (YOLO-World) | Edge/robotics deployable | Lower ceiling than heavy grounders |

---

## Open Problems & Research Gaps

- **Localization vs. recognition gap.** Open-vocab recognition outpaces open-vocab *localization*; novel-class boxes are still imprecise.
- **CLIP's limitations propagate.** Spatial and compositional weaknesses of CLIP (see [Spatial Intelligence](../12_research_frontier_2024_2026/06_spatial_intelligence.md)) bound detector performance.
- **Long-tail and rare classes.** LVIS rare-class AP remains far below common-class AP.
- **Segmentation parity.** Open-vocabulary *segmentation* (OV-Seg, OpenSeeD) lags detection in maturity.
- **Prompt sensitivity.** Performance varies with prompt phrasing; robust prompting is unsolved.
- **Real-time heavy models.** Bringing GLIP/Grounding-DINO accuracy to real-time edge budgets is open (relevant to [Perception for Robotics](../06_robotics_and_embodied_ai/08_perception_for_robotics.md)).

---

## Further Reading

- [ViLD (arXiv:2104.13921)](https://arxiv.org/abs/2104.13921) — open-vocab detection via vision-language distillation
- [GLIP (arXiv:2112.03857)](https://arxiv.org/abs/2112.03857) — grounded language-image pretraining
- [Grounding DINO (arXiv:2303.05499)](https://arxiv.org/abs/2303.05499) — open-set detection transformer
- [Detic (arXiv:2201.02605)](https://arxiv.org/abs/2201.02605) — detecting twenty-thousand classes
- [YOLO-World (arXiv:2401.17270)](https://arxiv.org/abs/2401.17270) — real-time open-vocabulary detection
