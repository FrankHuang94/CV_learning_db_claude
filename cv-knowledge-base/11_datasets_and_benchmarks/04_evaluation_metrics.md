# Evaluation Metrics

> **Last Updated:** June 2026
> **Level:** Intermediate
> **Related Sections:**
> - [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md)
> - [Core Datasets](./00_core_datasets.md)
> - [Multimodal Datasets](./03_multimodal_datasets.md)
> - [Object Detection](../02_core_tasks/01_object_detection.md)

---

## Overview

Evaluation metrics are the quantitative instruments by which computer vision measures progress, and their precise definitions matter more than newcomers often appreciate: subtle differences in IoU thresholds, averaging conventions, or matching rules can change rankings and mislead comparisons. A working researcher must understand not just *which* metric a benchmark uses but *exactly how* it is computed, because reported numbers are only comparable under identical protocols. This file catalogs the standard metrics across the major task families—classification, detection, segmentation, generation, retrieval, and depth—with their mathematical definitions and the pitfalls that make naive comparison hazardous.

A cross-cutting theme is that **every metric is a proxy** that captures some aspects of quality while ignoring others, and optimizing the metric is not the same as solving the task. mAP rewards ranking but not calibration; FID measures distributional similarity in a feature space but not semantic correctness or compositionality; mIoU averages over classes but hides per-class failures. The most important metrics are also frequently *gamed* through test-time tricks, ensembling, or benchmark overfitting, so a critical reader treats leaderboard numbers as necessary but insufficient evidence (see [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md) for the especially fraught generative case).

---

## Classification & Retrieval

**Top-1 / Top-5 accuracy** — fraction of images whose true label is the top (or among top-5) prediction. **Precision/Recall/F1** for imbalanced settings; **Average Precision (AP)** = area under the precision-recall curve. For retrieval, **Recall@K** (R@1, R@5, R@10) measures whether the correct item appears in the top-K results; **mAP** aggregates AP across queries.

## Detection: mAP

Detection's core metric is **mean Average Precision** at an Intersection-over-Union threshold:

```
IoU(A, B) = |A ∩ B| / |A ∪ B|
# A prediction is a true positive if IoU(pred, gt) ≥ threshold and class matches
AP = ∫₀¹ precision(recall) d(recall)     # area under PR curve, per class
mAP = mean over classes of AP
# COCO mAP averages over IoU thresholds 0.50:0.05:0.95 (i.e., AP@[.5:.95])
```

COCO's primary metric averages AP over ten IoU thresholds (0.50–0.95), making it stricter than the PASCAL VOC AP@0.5 (see [Object Detection](../02_core_tasks/01_object_detection.md)).

## Segmentation: mIoU, Dice, PQ

**Mean Intersection-over-Union (mIoU)** averages per-class IoU—the semantic-segmentation standard. **Dice coefficient** (= F1 over pixels) is favored in medical imaging. **Panoptic Quality (PQ)** unifies semantic and instance evaluation:

```
PQ = (Σ_{(p,g)∈TP} IoU(p,g)) / (|TP| + ½|FP| + ½|FN|)
   = SQ × RQ      # Segmentation Quality × Recognition Quality
```

## Generation: FID, IS, CLIPScore, FVD

**Fréchet Inception Distance (FID)** compares the distributions of Inception features for real and generated images:

```
FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2})
# lower is better; assumes Gaussian feature distributions
```

**Inception Score (IS)** measures quality+diversity via label entropy; **CLIPScore** measures image-text alignment for text-to-image; **FVD** extends FID to video via a video-feature network. These are critiqued for poor correlation with human judgment, motivating learned/human-preference metrics (GenEval, ImageReward; see [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md)).

## Depth, Flow, Pose

**Depth**: AbsRel, RMSE, and threshold accuracy δ<1.25. **Optical flow**: End-Point Error (EPE). **Pose**: PCK, OKS-based AP (COCO keypoints), MPJPE (3D, mm). **Tracking**: MOTA, IDF1, HOTA.

---

## Key References

| Metric | Introduced/standardized by | Year | Domain | Note |
|--------|---------------------------|------|--------|------|
| mAP (COCO) | Lin et al. (MS-COCO) | 2014 | Detection | AP@[.5:.95] |
| mIoU | PASCAL VOC / Cityscapes | 2008+ | Segmentation | Per-class IoU mean |
| Panoptic Quality | Kirillov et al. | 2019 | Panoptic | SQ × RQ |
| FID | Heusel et al. | 2017 | Generation | Inception-feature Fréchet distance |
| FVD | Unterthiner et al. | 2018 | Video gen | Video-feature FID |
| HOTA | Luiten et al. | 2021 | Tracking | Balanced detection/association |

---

## Pros & Cons (metric design)

| Aspect | Pros | Cons |
|--------|------|------|
| mAP | Rank-aware, threshold-robust (COCO) | Ignores calibration; complex to interpret |
| mIoU/PQ | Standard, per-class insight | Hides rare-class failure; boundary-sensitive |
| FID/FVD | Distribution-level quality | Poor human correlation; feature-extractor bias |
| Human-preference | Aligns with perceived quality | Costly, noisy, hard to reproduce |

---

## Open Problems & Research Gaps

- **Human-correlation gap.** FID/IS/CLIPScore correlate weakly with human judgment; better automatic generative metrics are unsolved (see [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md)).
- **Benchmark gaming.** Test-time augmentation, ensembling, and overfitting inflate metrics without real progress.
- **Compositional evaluation.** Metrics rarely test attribute binding, counting, or relational correctness.
- **Saturation.** Many metrics are near-ceiling on standard datasets, losing discriminative power.
- **Protocol inconsistency.** Subtle implementation differences (crops, thresholds) break comparability across papers.
- **Holistic/agentic evaluation.** No standard metrics for VLA/world-model utility (physical plausibility, task success) (see [Pros, Cons & Roadmaps](../06_robotics_and_embodied_ai/07_pros_cons_roadmaps.md)).

---

## Further Reading

- [MS-COCO (arXiv:1405.0312)](https://arxiv.org/abs/1405.0312) — detection/segmentation metrics
- [Panoptic Segmentation (arXiv:1801.00868)](https://arxiv.org/abs/1801.00868) — the PQ metric
- [FID / GANs Trained by TTUR (arXiv:1706.08500)](https://arxiv.org/abs/1706.08500) — Fréchet Inception Distance
- [HOTA (arXiv:2009.07736)](https://arxiv.org/abs/2009.07736) — higher-order tracking accuracy
- [GenEval (arXiv:2310.11513)](https://arxiv.org/abs/2310.11513) — compositional text-to-image evaluation
