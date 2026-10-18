# Evaluation Metrics

> **Last Updated:** October 2026
> **Level:** Intermediate
> **Related Sections:**
> - [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md)
> - [Core Datasets](./00_core_datasets.md)
> - [Multimodal Datasets](./03_multimodal_datasets.md)
> - [Object Detection](../02_core_tasks/01_object_detection.md)

> **🆕 October 2026 update:**
> - Added **cgF1** (classification-gated F1), the open-vocabulary concept-segmentation metric introduced with SAM 3 / SA-Co (Nov 2025). It multiplies image-level presence calibration (IL_MCC) by positive micro-F1.
> - Added **end-to-end driving metrics**: NAVSIM's PDMS → **EPDMS** (NAVSIM v2, CoRL 2025) and Waymo's **Rater Feedback Score** (WOD-E2E, 2025).
> - Added **LMM evaluation protocols**: circular evaluation, **pass@k vs pass^k** (ZeroBench), MMStar's multi-modal gain/leakage, and MRA for numeric spatial answers (VSI-Bench).
> - Updated the generative metrics with **FD<sub>DINOv2</sub>** as the recommended replacement for Inception features, and with **VBench++ / VBench-2.0** for video generation.
> - Added **embodied-policy evaluation**: success rate under perturbation (LIBERO-Plus) and distributed real-world arenas (RoboArena).

---

## Overview

Evaluation metrics are the quantitative instruments by which computer vision measures progress, and their precise definitions matter more than newcomers often appreciate: subtle differences in IoU thresholds, averaging conventions, or matching rules can change rankings and mislead comparisons. A working researcher must understand not just *which* metric a benchmark uses but *exactly how* it is computed, because reported numbers are only comparable under identical protocols. This file catalogs the standard metrics across the major task families—classification, detection, segmentation, generation, retrieval, and depth—with their mathematical definitions and the pitfalls that make naive comparison hazardous.

A cross-cutting theme is that **every metric is a proxy** that captures some aspects of quality while ignoring others, and optimizing the metric is not the same as solving the task. mAP rewards ranking but not calibration; FID measures distributional similarity in a feature space but not semantic correctness or compositionality; mIoU averages over classes but hides per-class failures. The most important metrics are also frequently *gamed* through test-time tricks, ensembling, or benchmark overfitting, so a critical reader treats leaderboard numbers as necessary but insufficient evidence (see [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md) for the especially fraught generative case).

Two developments in 2025–2026 changed the metric landscape. First, as open-vocabulary and embodied systems matured, metrics began to **penalise miscalibration and reward acceptable alternatives** rather than a single ground truth. cgF1 gates localisation quality by whether the model correctly decides a concept is present at all. Waymo's Rater Feedback Score and NAVSIM's EPDMS score a planned trajectory against human-rated or rule-based admissibility rather than against the single logged path. Second, for generative multimodal models, **sampling and protocol become part of the metric**. pass@k vs pass^k, circular option permutation, frame budgets and "thinking" budgets can move headline numbers by more than a model generation. Reports that omit them are not comparable.

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

## Open-Vocabulary Concept Segmentation: cgF1

AP is ill-suited to *promptable concept segmentation*, where a text prompt may have zero, one or many matching instances in an image. AP averages over confidence thresholds and ranks predictions across images, so it does not penalise a model that confidently segments something for a concept that is absent. SAM 3 / SA-Co [Carion2025] therefore introduced the **classification-gated F1 (cgF1)**:

```
pmF1   = positive micro-F1 of instance masks, computed only on (image, phrase) pairs
         whose concept is present (localisation quality at a fixed operating point)
IL_MCC = Matthews correlation coefficient of the image-level binary decision
         "is the concept present?" over all positive and negative pairs (calibration)
cgF1   = 100 × pmF1 × IL_MCC
# On SA-Co/Gold, ground truth comes from 3 independent annotators;
# the most favourable annotation per datapoint is used (oracle setting).
```

Reference values on SA-Co/Gold: human 72.8, SAM 3 54.1, OWLv2* 24.6, Gemini 2.5 13.0. For video, SA-Co reports cgF1 alongside a HOTA variant (**pHOTA**).

## Generation: FID, IS, CLIPScore, FVD

**Fréchet Inception Distance (FID)** compares the distributions of Inception features for real and generated images:

```
FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2})
# lower is better; assumes Gaussian feature distributions
```

**Inception Score (IS)** measures quality+diversity via label entropy; **CLIPScore** measures image-text alignment for text-to-image; **FVD** extends FID to video via a video-feature network. These are critiqued for poor correlation with human judgment, motivating learned/human-preference metrics (GenEval, ImageReward; see [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md)).

**Feature-extractor choice (FD<sub>DINOv2</sub>):** A study of 41 generative models [Stein2023] found that Inception-V3 features do not reflect the human-judged realism of diffusion models. It recommended replacing Inception with **DINOv2** features in Fréchet distance and related metrics (precision/recall, density/coverage, kernel distance). **FD<sub>DINOv2</sub>** is now commonly reported next to FID on ImageNet generation (e.g. in EDM2-era papers). The same study found that current memorisation-detection metrics fail on complex datasets.

**Video generation suites:** VBench decomposes video quality into hierarchical dimensions with per-dimension human-alignment checks. **VBench++** (TPAMI 2025) adds image-to-video and trustworthiness evaluation. **VBench-2.0** (2025) targets *intrinsic faithfulness* (physics, commonsense, human fidelity, controllability, creativity) rather than superficial pixel quality. Video arenas based on human pairwise votes (including the VBench-2.0 and VBench-I2V arenas) complement these automatic scores.

## Depth, Flow, Pose

**Depth**: AbsRel, RMSE, and threshold accuracy δ<1.25. **Optical flow**: End-Point Error (EPE). **Pose**: PCK, OKS-based AP (COCO keypoints), MPJPE (3D, mm). **Tracking**: MOTA, IDF1, HOTA.

## Driving and Embodied Policy Metrics

**PDMS / EPDMS (NAVSIM).** NAVSIM's PDM Score combines multiplicative safety gates with weighted comfort and progress terms. NAVSIM v2's **Extended PDMS** [Cao2025] adds driving-direction and traffic-light compliance gates, lane-keeping and extended-comfort terms, and a *human filter* that disables a penalty when the human driver commits the same violation:

```
EPDMS = Π_{m ∈ {NC, DAC, DDC, TLC}} filter_m(agent, human)
        × ( Σ_{m ∈ {TTC, EP, HC, LK, EC}} w_m · filter_m(agent, human) ) / Σ w_m
# weights: EP = 5, TTC = 5, LK = 2, HC = 2, EC = 2
# filter_m(agent, human) = 1.0 if m(human) = 0, else m(agent)
# NC: no at-fault collision, DAC: drivable-area compliance, DDC: driving-direction
# compliance, TLC: traffic-light compliance, EP: ego progress, TTC: time-to-collision
```

In v2, scores are aggregated in two stages over the original scene and synthetic perturbed follow-up scenes (*pseudo-simulation*). This approximates closed-loop evaluation while remaining open-loop.

**Rater Feedback Score (Waymo WOD-E2E).** For rated frames, human raters assign 0–10 scores to up to three candidate future trajectories. A predicted trajectory is scored according to how it matches these rated alternatives, so a planner is not penalised for choosing a different but acceptable manoeuvre. Displacement errors (ADE/FDE) remain secondary diagnostics.

**Robot policies.** The dominant metric is still **task success rate** (often with partial-progress credit for long-horizon tasks). In 2025–2026 two qualifiers became standard. The first is *success under controlled perturbation*: LIBERO-Plus reports success across seven perturbation axes and exposes drops from ~95% to under 30%. The second is *distributed real-world comparison*: RoboArena evaluates remotely served policies on a shared real-robot platform across institutions instead of on a single lab's setup.

## Multimodal LLM Evaluation Protocols

- **Circular evaluation (MMBench, OpenCompass spatial leaderboards):** a multiple-choice item counts as correct only if the model answers correctly under every cyclic permutation of the options. This removes position bias and reduces lucky guesses.
- **pass@k vs pass^k (ZeroBench):** pass@k counts a question as solved if *any* of k samples is correct. pass^k requires *all* k samples to be correct. On ZeroBench the best 2026 model scores 30.0 pass@5 but 13.0 pass^5, so the two measure capability and reliability respectively.
- **Multi-modal gain / leakage (MMStar):** MG = an LVLM's score with images − the same LVLM's score without images. ML = the amount by which the LVLM *without* images outperforms its base LLM, which indicates that benchmark samples leaked into multimodal training. Together they quantify how much a "vision" score actually depends on vision.
- **Mean Relative Accuracy (MRA, VSI-Bench):** for numeric answers (distances, sizes, counts), accuracy is averaged over a range of relative-error tolerance thresholds instead of exact match, giving partial credit for approximately correct spatial estimates.
- **Protocol variables that must be reported:** number of frames (or fps) and resolution for video, subtitle/audio access, reasoning/"thinking" budget, tool use, number of samples, and the answer-extraction method. Third-party re-evaluations frequently differ from self-reported model-card numbers by several points because of these choices.

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
| FD<sub>DINOv2</sub> | Stein et al. | 2023 | Generation | DINOv2 features replace Inception |
| MG / ML | Chen et al. (MMStar) | 2024 | LMM eval | Multi-modal gain / leakage |
| PDMS → EPDMS | Dauner et al. → Cao et al. (NAVSIM) | 2024 → 2025 | AV planning | Gated, human-filtered pseudo-closed-loop score |
| Rater Feedback Score | Waymo (WOD-E2E) | 2025 | AV planning | Human-rated trajectory alternatives (0–10) |
| VBench-2.0 | Zheng et al. | 2025 | Video gen | Intrinsic-faithfulness dimensions |
| cgF1 / pHOTA | Carion et al. (SAM 3 / SA-Co) | 2025 | Concept seg | IL_MCC × positive micro-F1 |
| pass^k | Roberts et al. (ZeroBench) and others | 2025 | LMM eval | All-k reliability vs pass@k capability |

---

## Pros & Cons (metric design)

| Aspect | Pros | Cons |
|--------|------|------|
| mAP | Rank-aware, threshold-robust (COCO) | Ignores calibration; complex to interpret |
| mIoU/PQ | Standard, per-class insight | Hides rare-class failure; boundary-sensitive |
| FID/FVD | Distribution-level quality | Poor human correlation; feature-extractor bias |
| Human-preference | Aligns with perceived quality | Costly, noisy, hard to reproduce |
| cgF1 | Penalises hallucinated masks for absent concepts; single operating point | Not comparable with AP; depends on the chosen threshold; new and not yet widely adopted |
| EPDMS / RFS | Reward admissible behaviour rather than imitation of one log | Rule weights (EPDMS) and rater subjectivity (RFS) embed design choices; correlation with real-world safety unproven |
| pass@k / pass^k | Separate capability from reliability | Requires k× inference; k must be reported for comparability |

---

## Open Problems & Research Gaps

- **Human-correlation gap.** FID/IS/CLIPScore correlate weakly with human judgment; better automatic generative metrics are unsolved (see [Generative Evaluation & Safety](../10_generative_vision/04_evaluation_and_safety.md)). Swapping in DINOv2 features helps but does not address compositional or physical correctness. VBench-2.0-style decomposed suites trade a single scalar for many partially validated dimensions.
- **Benchmark gaming.** Test-time augmentation, ensembling, and overfitting inflate metrics without real progress. In the LMM era, the corresponding levers are thinking budgets, tool use, best-of-k sampling and benchmark-specific prompting.
- **Compositional evaluation.** Metrics rarely test attribute binding, counting, or relational correctness.
- **Saturation.** Many metrics are near-ceiling on standard datasets, losing discriminative power.
- **Protocol inconsistency.** Subtle implementation differences (crops, thresholds) break comparability across papers. For video-LLMs, frame sampling and subtitle access play the same role.
- **Calibration-aware metrics.** cgF1 shows how to fold presence calibration into localisation metrics for open-vocabulary perception. Analogous calibration-aware metrics for VLM grounding, VQA abstention and embodied "I can't do this" decisions are largely missing.
- **Holistic/agentic evaluation.** No standard metrics for VLA/world-model utility (physical plausibility, task success) (see [Pros, Cons & Roadmaps](../06_robotics_and_embodied_ai/07_pros_cons_roadmaps.md)). Progress so far: success-under-perturbation (LIBERO-Plus), distributed real-world arenas (RoboArena), and human-preference planning scores (RFS). A metric that predicts real-world deployment reliability from offline evaluation does not yet exist.

---

## Further Reading

- [MS-COCO (arXiv:1405.0312)](https://arxiv.org/abs/1405.0312) — detection/segmentation metrics
- [Panoptic Segmentation (arXiv:1801.00868)](https://arxiv.org/abs/1801.00868) — the PQ metric
- [FID / GANs Trained by TTUR (arXiv:1706.08500)](https://arxiv.org/abs/1706.08500) — Fréchet Inception Distance
- [HOTA (arXiv:2009.07736)](https://arxiv.org/abs/2009.07736) — higher-order tracking accuracy
- [GenEval (arXiv:2310.11513)](https://arxiv.org/abs/2310.11513) — compositional text-to-image evaluation
- [Stein et al. (2023) — Exposing flaws of generative model evaluation metrics (arXiv:2306.04675)](https://github.com/layer6ai-labs/dgm-eval) — FD<sub>DINOv2</sub> and 15 metrics × 8 encoders
- [VBench / VBench++ / VBench-2.0 (arXiv:2503.21755)](https://github.com/Vchitect/VBench) — video generation evaluation suites
- [SAM 3 SA-Co/Gold evaluator (cgF1)](https://github.com/facebookresearch/sam3/tree/main/scripts/eval/gold)
- [NAVSIM EPDMS definition](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md) — and [Pseudo-Simulation (arXiv:2506.04218)](https://arxiv.org/abs/2506.04218)
- [Waymo Open Dataset — E2E driving and Rater Feedback Score code](https://github.com/waymo-research/waymo-open-dataset)
- [MMStar (arXiv:2403.20330)](https://github.com/MMStar-Benchmark/MMStar) — multi-modal gain/leakage
- [ZeroBench (arXiv:2502.09696)](https://github.com/jonathan-roberts1/zerobench) — pass@k vs pass^k reporting
