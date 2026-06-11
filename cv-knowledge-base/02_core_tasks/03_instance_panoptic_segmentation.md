# Instance & Panoptic Segmentation

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [Semantic Segmentation](./02_semantic_segmentation.md) · [Object Detection](./01_object_detection.md) · [Vision Transformers](../03_architectures/01_vision_transformers.md) · [Any-Model Paradigm](../12_research_frontier_2024_2026/02_any_model_paradigm.md)

---

## Overview

**Instance segmentation** extends object detection by predicting a pixel-accurate binary mask for each detected object instance, in addition to the bounding box and class label. Two cars produce two separate masks, each covering only that specific car's pixels. **Panoptic segmentation** [Kirillov2019pan] unifies instance segmentation of **things** (countable objects: people, cars, animals) with semantic segmentation of **stuff** (amorphous regions: sky, road, vegetation), producing a single coherent scene labelling where every pixel belongs to exactly one segment with a semantic class and, for things, a unique instance ID.

The dominant paradigm shift in this field parallels that in detection: early approaches decomposed the problem into detect-then-segment pipelines (Mask R-CNN [He2017], YOLACT [Bolya2019]); later work moved towards unified query-based transformer architectures that produce both thing and stuff segments from shared learnable queries (Panoptic-DeepLab [Cheng2020], Panoptic FPN [Kirillov2019fpn], Mask2Former [Cheng2022], OneFormer [Jain2023]). The advantages of unification are significant: a single model trained jointly on all segmentation objectives achieves better accuracy per parameter than three specialised models, and a consistent interface simplifies downstream system design.

The **Panoptic Quality (PQ)** metric [Kirillov2019pan] provides a unified score that reflects both recognition (are the classes predicted correctly?) and segmentation quality (are the mask shapes accurate?). Its multiplicative decomposition into **Segmentation Quality (SQ)** and **Recognition Quality (RQ)** cleanly separates mask boundary precision from detection recall, making it interpretable and actionable for model analysis.

---

## Task Definitions

**Instance Segmentation Output:**
- A set of (mask_i, class_i, score_i) tuples, one per detected instance
- mask_i ∈ {0,1}^(H×W), class_i ∈ {1,…,K_things}, score_i ∈ [0,1]
- Evaluated with AP@[0.50:0.95] on COCO (mask IoU threshold instead of box IoU)

**Panoptic Segmentation Output:**
- A single map (S, I) ∈ ({1,…,K} × ℤ_≥0)^(H×W)
- S[h,w] = semantic class; I[h,w] = instance ID (0 for stuff)
- Every pixel assigned; no overlapping segments allowed

---

## Panoptic Quality Metric

```latex
% Panoptic Quality decomposes into SQ × RQ
\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in \mathrm{TP}} \mathrm{IoU}(p,g)}{|\mathrm{TP}|}}_{\text{Segmentation Quality (SQ)}}
\times
\underbrace{\frac{|\mathrm{TP}|}{|\mathrm{TP}| + \frac{1}{2}|\mathrm{FP}| + \frac{1}{2}|\mathrm{FN}|}}_{\text{Recognition Quality (RQ)}}

% Matching rule: (p,g) \in TP iff IoU(p,g) > 0.5
% Note: IoU > 0.5 uniquely determines the match (no two predictions
%        can both exceed 0.5 IoU with the same ground-truth segment)
% PQ_things and PQ_stuff are computed separately and averaged
```

In practice:
- **SQ** measures the average mask overlap for matched segments; typically 70–80 for strong models
- **RQ** is equivalent to F1 score over segments; penalises missing and spurious detections
- **PQ** = SQ × RQ; COCO SOTA models achieve ~58–60 PQ

---

## Architectural Milestones

### Mask R-CNN

**Mask R-CNN** [He2017] (ICCV 2017): the canonical top-down instance segmentation model. Extends Faster R-CNN by adding a parallel **mask branch**—a small FCN attached to each RoI feature—that predicts a K×28×28 binary mask for each class. Introduces **RoI Align** to replace RoI Pooling: instead of quantising to integer grid positions, it uses bilinear interpolation at exactly aligned floating-point coordinates, eliminating misalignment artefacts that degrade mask accuracy.

```latex
% RoI Align bilinear interpolation at fractional coordinates:
% For a query point (x, y) in the feature map:
% f(x,y) = \sum_{i,j \in \{0,1\}} (1 - |x - x_i|)(1 - |y - y_j|) \cdot v_{ij}
% where (x_i, y_i) are the four nearest integer grid points
% This ensures that mask branch gradients flow through precise spatial locations
```

Mask R-CNN with ResNet-101-FPN achieves 35.7 AP_mask on COCO test-dev (2017). Adding Deformable Convolutions pushes to 37.5. Authors: Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick.

### YOLACT and Real-Time Instance Segmentation

**YOLACT** [Bolya2019] (ICCV 2019): reformulates instance segmentation as a linear combination of a small set of **prototype masks**: a backbone produces K=32 global prototype masks; simultaneously, a per-instance head predicts K linear combination coefficients. Final mask = sigmoid(P · c^T) where P ∈ ℝ^(H×W×K) is the prototype bank and c ∈ ℝ^(1×K) are the coefficients for that instance. Achieves 29.8 AP_mask on COCO test-dev at real-time (33.5 fps on a Titan Xp). **YOLACT++** adds mask re-scoring and deformable convolutions; 34.1 AP.

**SOLOv2** [Wang2020] (NeurIPS 2020): Segmenting Objects by Locations. Divides image into S×S grid cells; each cell is responsible for instances whose centre falls within it. A dynamic convolution mechanism generates instance-specific mask kernels from a compact feature tensor, applying them on a mask feature map. Decouples mask learning from feature representation, enabling a dynamic filtering mechanism with O(1) mask prediction per instance. SOLOv2 achieves 39.7 AP_mask on COCO test-dev with ResNet-101. Authors: Xinlong Wang et al.

### Panoptic Segmentation Architectures

**Panoptic FPN** [Kirillov2019fpn] (CVPR 2019): adds a semantic segmentation branch to the FPN neck of Mask R-CNN. The stuff branch upsamples FPN features to 1/4 resolution with additive fusion; the thing branch retains the Mask R-CNN instance head. Combined into a panoptic output by merging instance masks (higher priority) with stuff predictions. R101-FPN achieves 40.9 PQ on COCO val.

**Panoptic-DeepLab** [Cheng2020] (CVPR 2020): bottom-up approach using dual-decoder with instance-centre prediction and semantic segmentation head; instance centres are detected as local maxima, and pixels are assigned to nearest centre. Xception-71 backbone achieves 39.7 PQ COCO val at real-time speeds; key advantage is no NMS and no RoI operations.

**MaskFormer** [Cheng2021] (NeurIPS 2021): precursor to Mask2Former. Per-pixel embeddings from a FPN-like pixel decoder; N learnable queries decoded by a Transformer decoder into (class, mask) pairs; mask predicted as dot product of query embedding with per-pixel embeddings. Unifies semantic and panoptic segmentation under one framework. Mask R-CNN surpassed for semantic segmentation at high mIoU.

**Mask2Former** [Cheng2022] (CVPR 2022): the current gold standard for all segmentation tasks. Key improvement over MaskFormer is **masked cross-attention**: instead of attending globally, each query's cross-attention is restricted to its predicted mask region:

```latex
% Masked Cross-Attention (Mask2Former Eq. 1)
X_l = \text{softmax}(\mathbf{M}_{l-1} + Q_l K_l^\top / \sqrt{d}) V_l + X_{l-1}

% M_{l-1}[h,w] = 0  if pixel (h,w) is in the predicted foreground mask
%              = -inf otherwise (masked out from attention)
% This confines each query's context to its predicted object region
```

Results with ResNet-50: 51.9 PQ panoptic, 44.2 AP instance, 57.7 mIoU semantic on COCO/ADE20K respectively. With Swin-L: 57.8 PQ panoptic, 50.1 AP instance, 57.7 mIoU semantic on COCO/ADE20K.

**OneFormer** [Jain2023] (CVPR 2023): trains a single Mask2Former-style model with a task-conditioning token ("the task is {semantic/instance/panoptic}") that modulates both the text queries and the loss function. During inference, selecting the appropriate task token routes the model. Achieves Mask2Former-level SOTA on all three tasks with a single set of weights, outperforming task-specific models in some settings. Authors: Jitesh Jain, Jiachen Li, MangTik Chiu et al.

```mermaid
flowchart TD
    A[Input Image] --> B[Backbone + Pixel Decoder\nMulti-scale features P2-P5]
    B --> C[Per-pixel\nEmbedding Map]
    B --> D[Transformer Decoder\nN queries + masked cross-attn]
    D --> E_class[Class Logits per query]
    D --> F_mask[Mask Embedding per query]
    F_mask --> G["Mask = sigmoid(Q_embed · pixel_embed^T)"]
    C --> G
    E_class --> H[Panoptic Merging\nThings: argmax scores\nStuff: semantic label]
    G --> H
    H --> I[Panoptic Map\n(class, instance_id) per pixel]
    style D fill:#dfd,stroke:#5a5
    style B fill:#ddf,stroke:#55a
```

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Mask R-CNN | He, Gkioxari, Dollár, Girshick | 2017 | ICCV | RoI Align + mask branch; instance seg baseline |
| Panoptic Segmentation (metric) | Kirillov, He, Girshick et al. | 2019 | CVPR | Panoptic Quality (PQ) metric definition |
| Panoptic FPN | Kirillov, Girshick et al. | 2019 | CVPR | Unified FPN for things + stuff |
| YOLACT | Bolya, Zhou et al. | 2019 | ICCV | Real-time prototype-based instance seg |
| Panoptic-DeepLab | Cheng, Collins et al. | 2020 | CVPR | Bottom-up; instance centre estimation |
| SOLOv2 | Wang, Kong et al. | 2020 | NeurIPS | Dynamic kernels; 39.7 AP COCO test-dev |
| MaskFormer | Cheng, Schwing, Kirillov | 2021 | NeurIPS | Mask-classification unification |
| Masked-Attention Mask Transformer (Mask2Former) | Cheng, Misra et al. | 2022 | CVPR | Masked cross-attention; 57.8 PQ COCO |
| OneFormer | Jain, Li et al. | 2023 | CVPR | Single multi-task model; task-conditioned |

---

## Benchmark Performance

### COCO val2017 — Instance Segmentation (AP_mask @[0.50:0.95])

| Model | Backbone | AP_mask | AP_S | AP_M | AP_L | Notes |
|-------|----------|---------|------|------|------|-------|
| Mask R-CNN | ResNet-50-FPN | 35.7 | 15.5 | 38.1 | 52.4 | Baseline |
| Mask R-CNN | ResNet-101-FPN | 36.1 | 16.4 | 38.8 | 53.1 | — |
| YOLACT-550 | ResNet-101 | 29.8 | 8.6 | 30.8 | 50.0 | Real-time 33 fps |
| SOLOv2 | ResNet-101 | 39.7 | 17.3 | 42.9 | 57.4 | Dynamic kernels |
| Mask2Former | ResNet-50 | 44.2 | 22.8 | 47.3 | 66.1 | Universal |
| Mask2Former | Swin-L | 50.1 | 29.9 | 53.7 | 72.1 | SOTA single model |
| OneFormer | Swin-L | 50.0 | — | — | — | Multi-task trained |

### COCO val2017 — Panoptic Segmentation (PQ)

| Model | Backbone | PQ | PQ_things | PQ_stuff | Notes |
|-------|----------|----|-----------|---------|----|
| Panoptic FPN | ResNet-101 | 40.9 | 48.3 | 29.7 | Mask R-CNN + stuff branch |
| Panoptic-DeepLab | Xception-71 | 39.7 | 43.9 | 33.2 | Bottom-up; real-time |
| MaskFormer | Swin-L | 52.7 | 58.5 | 44.0 | Mask-classification |
| Mask2Former | ResNet-50 | 51.9 | 57.7 | 43.0 | Universal; 50 epochs |
| Mask2Former | Swin-L | 57.8 | 64.2 | 48.1 | COCO val SOTA |
| OneFormer | Swin-L | 57.9 | 64.4 | 48.0 | Single multi-task model |
| Panoptic SegFormer | Swin-L | 56.2 | — | — | Test-dev |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Top-down (Mask R-CNN) | High instance-level accuracy; well-calibrated confidence; mature ecosystem | Two-stage; slow for high-res; separate stuff head needed for panoptic |
| Bottom-up (Panoptic-DeepLab / SOLO) | Fast inference; no RoI operations or NMS; handles overlapping gracefully | Instance grouping errors; under-performs top-down on COCO AP |
| Universal transformer (Mask2Former/OneFormer) | Single model for all tasks; SOTA accuracy; unified training | High memory; complex masked-attention; slow training; large backbone needed |
| Prototype-based (YOLACT) | Real-time (33+ fps); simple linear decoding | Limited mask quality; struggles with small, thin, or occluded objects |
| Panoptic quality metric (PQ) | Unified; interpretable SQ/RQ decomposition; standard | Sensitive to over-segmentation; hard objects' PQ dominates aggregate score |

---

## Open Problems & Research Gaps

- **Open-vocabulary panoptic segmentation:** Current models segment a fixed set of COCO things/stuff categories; grounding panoptic segmentation to arbitrary text descriptions (analogous to Grounding DINO for detection) is an active frontier, with early work in FC-CLIP and OpenSeg.
- **Video instance and panoptic segmentation:** Extending panoptic quality to video (VPQ, STQ) adds temporal consistency requirements; current methods struggle under occlusion, appearance change, and re-identification across long sequences.
- **Efficient universal models:** Mask2Former achieves SOTA but is computationally expensive (>300M parameters with Swin-L); lightweight universal segmentors competitive with task-specific models at real-time speeds do not yet exist.
- **Crowded scene handling:** Panoptic models confuse instances when objects are severely occluded or touching; contrastive instance discrimination losses and depth-aware grouping are promising but unresolved.
- **Boundary quality metrics:** PQ uses IoU > 0.5 threshold, which is insensitive to boundary precision; boundary-aware metrics (BoundaryIoU [Cheng2021b]) are not yet standardly reported, obscuring progress on edge localisation.
- **Annotation efficiency:** Pixel-level panoptic annotation is ~5× more expensive than bounding-box annotation; interactive and semi-automatic annotation pipelines combining SAM with automatic class assignment are nascent.
- **Generalisation of instance IDs:** Panoptic models assign local instance IDs per image but do not maintain consistent cross-image identities; this limits use in multi-view 3D understanding and long-form video analysis.

---

## Further Reading

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870)
- [Panoptic Segmentation — metric paper (Kirillov et al., 2019)](https://arxiv.org/abs/1801.00868)
- [YOLACT: Real-time Instance Segmentation (Bolya et al., 2019)](https://arxiv.org/abs/1904.02689)
- [SOLOv2: Dynamic and Fast Instance Segmentation (Wang et al., 2020)](https://arxiv.org/abs/2003.10152)
- [Masked-attention Mask Transformer — Mask2Former (Cheng et al., 2021)](https://arxiv.org/abs/2112.01527)
- [OneFormer: One Transformer to Rule Universal Image Segmentation (Jain et al., 2022)](https://arxiv.org/abs/2211.06220)
