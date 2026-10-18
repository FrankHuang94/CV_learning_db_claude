# Object Detection

> **Last Updated:** October 2026
> **Level:** Intermediate → Advanced
> **Related Sections:** [Open-Vocabulary Detection](../05_multimodal_vision_language/03_open_vocabulary_detection.md) · [Semantic Segmentation](./02_semantic_segmentation.md) · [CNN Architectures](../03_architectures/00_cnn_architectures.md) · [Vision Transformers](../03_architectures/01_vision_transformers.md)

> **🆕 October 2026 update:**
> - New subsection on **2024–2026 real-time detectors**: D-FINE, DEIM (CVPR 2025), DEIMv2 (DINOv3 features), RT-DETRv4 (ECCV 2026), **RF-DETR** (ICLR 2026; first real-time model above 60 AP on COCO) and the YOLOv12 → YOLOv13 → **YOLO26** line (NMS-free, DFL-free).
> - Corrected the earlier RF-DETR entry (it builds on LW-DETR with a DINOv2 backbone and weight-sharing NAS, not RT-DETR; best real-time result 60.1 AP for RF-DETR-2XL at 17.2 ms on a T4).
> - Added **SAM 3** (Meta, Nov 2025) as a promptable concept detector, plus YOLOE (ICCV 2025) and Rex-Omni (MLLM-based detection) to the open-vocabulary section.
> - Extended Key Papers and COCO tables; added pros/cons rows and open problems on benchmark protocol drift, COCO saturation and concept-level detection; new Further Reading.

---

## Overview

Object detection is the task of localising and classifying all instances of objects in an image, producing a set of bounding boxes {(**b**_i, c_i, s_i)} where **b**_i ∈ ℝ^4 is a box (typically in [x1, y1, x2, y2] or [cx, cy, w, h] format), c_i ∈ {1, …, K} is the class label, and s_i ∈ [0,1] is a confidence score. The COCO benchmark [Lin2014] has been the dominant evaluation protocol since 2014, using mean Average Precision averaged over ten IoU thresholds from 0.50 to 0.95 (AP, also written mAP@[0.50:0.95]) as well as AP50 and AP75.

Two architectural paradigms dominated the first half of the 2010s: **two-stage** detectors generate candidate region proposals in a first stage and classify/refine them in a second stage (R-CNN family [Girshick2014, Girshick2015, Ren2015]); **one-stage** detectors predict boxes and classes directly from a regular grid of anchors without a proposal stage (SSD [Liu2016], YOLO [Redmon2016], RetinaNet [Lin2017]). The central tension—two-stage detectors are more accurate but slower; one-stage are faster but have lower accuracy—was progressively eroded by design advances on both sides, and eventually dissolved by anchor-free detectors (FCOS [Tian2019]) and end-to-end Transformer-based detectors (DETR [Carion2020] and successors).

A third wave of **open-vocabulary** and **grounded** detection [Li2022, Liu2023] shifted focus from closed-set recognition to language-guided detection, enabling detectors to locate arbitrary objects described by free-text phrases. Grounding DINO [Liu2023] achieves 52.5 AP on COCO zero-shot while simultaneously solving referring expression comprehension, collapsing the boundary between detection and vision–language alignment.

Between 2024 and 2026 two further shifts occurred. First, **real-time DETRs overtook YOLO-style CNNs** on the COCO accuracy–latency Pareto front: D-FINE [Peng2024], DEIM [Huang2025] and RF-DETR [Robinson2026] reach 55–60 AP at T4 latencies of roughly 10–17 ms, largely by inheriting strong pre-trained ViT features (DINOv2, DINOv3) and by refining box regression and one-to-one matching. The YOLO line answered by going natively end-to-end (YOLOv10, YOLO26). Second, promptable **concept detection** — SAM 3 [Carion2025] — merges open-vocabulary detection, instance segmentation and tracking into one model that returns *all* instances of a noun phrase or exemplar.

---

## Two-Stage Detectors

### R-CNN Family

**R-CNN** [Girshick2014] (CVPR 2014): selective search generates ~2000 candidate regions; each is warped and passed through a CNN for feature extraction; a per-class SVM classifies features; a linear regressor refines box coordinates. Training and inference are disjoint; 47 seconds per image on a GPU.

**Fast R-CNN** [Girshick2015] (ICCV 2015): shares the convolutional backbone across all proposals via **RoI Pooling**—a fixed-size pooling operation applied to each proposal on the shared feature map. A single softmax classifier and box regressor share the FC layers. End-to-end training (except proposal generation); ~200 ms/image.

**Faster R-CNN** [Ren2015] (NeurIPS 2015): replaces selective search with a **Region Proposal Network (RPN)**—a lightweight fully convolutional network that slides over the feature map and predicts objectness scores and offsets for a set of anchor boxes at each location. RPN shares the backbone with the detection head (four-step alternating training or approximate joint training). Achieves 21.9 mAP@[0.50:0.95] on COCO at ~17 fps with ZF-net, 27.2 with ResNet-101.

**Feature Pyramid Network (FPN)** [Lin2017a] (CVPR 2017): addresses scale variation by constructing a top-down pathway with lateral connections that merges coarse, semantically rich deep features with fine, spatially precise shallow features. Produces multi-scale feature maps {P2, P3, P4, P5, P6} where P_k has stride 2^k. Combined with Faster R-CNN and ResNet-101, FPN achieves 36.2 mAP on COCO minival with no increase in inference time relative to single-scale.

---

## One-Stage Detectors

### SSD and YOLO Family

**SSD** [Liu2016] (ECCV 2016): places default anchor boxes of different scales and aspect ratios on multiple feature maps (conv4_3 through conv11_2), predicting offsets and class scores for each anchor in a single forward pass. VGG-16 backbone; 74.3 mAP@0.50 (VOC 2007), 25.1 mAP@[0.50:0.95] on COCO at 46 fps.

**YOLOv1** [Redmon2016] (CVPR 2016): divides image into S×S grid; each cell predicts B bounding boxes and C class probabilities, treating detection as a single regression problem. GoogLeNet-inspired backbone; 63.4 mAP@0.50 VOC 2007, 45 fps.

**YOLOv2 / YOLO9000** [Redmon2017] adds batch normalisation, anchor boxes, high-resolution classification pre-training, and multi-scale training. **YOLOv3** [Redmon2018] uses Darknet-53 backbone and FPN-like multi-scale predictions; 33.0 mAP@[0.50:0.95] COCO test.

**YOLOv4** [Bochkovskiy2020] (arXiv 2020): systematically integrates bag-of-freebies (MixUp, CutMix, mosaic, self-adversarial training) and bag-of-specials (SPP, PAN, SAM). CSPDarknet53 backbone; 43.5 mAP@[0.50:0.95] at 65 fps (V100).

**YOLOv5** (Jocher, Ultralytics, 2020): unofficial but widely adopted; PyTorch-native, anchor-based, CSP backbone; ~43.3 mAP (YOLOv5x).

**YOLOv8** (Jocher, Ultralytics, 2023): anchor-free head, C2f bottleneck, decoupled classification and regression; YOLOv8x achieves 53.9 mAP@[0.50:0.95] COCO val.

**YOLOv10** [Wang2024] (NeurIPS 2024): dual-label assignment NMS-free training; YOLOv10-X achieves 54.4 mAP at 29 ms on CPU.

**YOLO11** (Ultralytics, 2024): reduces parameters by 22 % vs YOLOv8m while +1.3 % mAP; YOLO11x 54.7 mAP COCO val.

**YOLOv12** [Tian2025] (arXiv 2502.12524, Feb 2025): an *attention-centric* YOLO using area attention and residual efficient layer aggregation (R-ELAN) while preserving YOLO-level latency; YOLOv12-X reaches 54.4 AP (59.1 M params). **YOLOv13** (arXiv 2506.17733, Jun 2025) adds hypergraph-based adaptive correlation enhancement to model high-order feature correlations; YOLOv13-X reports 54.8 AP at 64.0 M params.

**YOLO26** [Jocher2026] (Ultralytics; technical report arXiv 2606.03748, Jun 2026): a unified family for detection, instance/semantic segmentation, pose, classification and oriented boxes, built around four changes: (1) **native end-to-end NMS-free inference** via an optional one-to-one head; (2) **removal of Distribution Focal Loss (DFL)** for a lighter, export-friendly regression head; (3) a new training recipe combining **MuSGD** (a hybrid of the Muon optimiser from LLM training and SGD), **Progressive Loss** and **STAL** (small-target-aware label assignment); and (4) task-specific heads (e.g. RLE for pose). YOLO26n–x report **40.9–57.5 mAP** on COCO val2017 (40.1–56.9 in NMS-free end-to-end mode) at **1.7–11.8 ms** on a T4 with TensorRT, and up to 43 % faster CPU ONNX inference for YOLO26n than YOLO11n. An open-vocabulary variant, YOLOE-26, accepts text and visual prompts.

### RetinaNet and Focal Loss

**RetinaNet** [Lin2017b] (ICCV 2017): identifies class imbalance between foreground and background anchors as the root cause of one-stage detector accuracy gaps. Introduces **focal loss**—a dynamically scaled cross-entropy that down-weights easy negatives:

```latex
% Standard cross-entropy for binary classification
\text{CE}(p_t) = -\log(p_t)

% Focal loss with focusing parameter \gamma and balancing factor \alpha
\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)

% p_t = p if y=1, else 1-p
% \gamma = 2, \alpha = 0.25 are default values
% (1-p_t)^\gamma down-weights easy examples exponentially
% Hard examples (p_t << 1) receive weight ~1; easy (p_t -> 1) receive ~0
```

RetinaNet with ResNeXt-101-FPN achieves 40.8 mAP@[0.50:0.95] on COCO test-dev, surpassing all prior one-stage detectors and matching the best two-stage models of its era.

### Anchor-Free Detectors

**FCOS** [Tian2019] (ICCV 2019): reformulates detection as dense per-pixel prediction. For each foreground pixel (x, y), the network predicts (l, t, r, b)—distances to the left, top, right, and bottom of the ground-truth box—along with a centerness score to suppress low-quality boxes. Eliminates all anchor hyperparameters. FCOS achieves 44.7 mAP with ResNeXt-101-FPN, 49.0 with multi-scale testing.

---

## Transformer-Based Detectors

**DETR** [Carion2020] (ECCV 2020): treats detection as a **direct set prediction** problem. A CNN backbone extracts feature maps; a Transformer encoder processes flattened spatial features; a Transformer decoder produces N learnable **object queries** in parallel, each decoded to a box and class via FFN. Hungarian algorithm finds a bipartite matching between predictions and ground truth, eliminating NMS. DETR-R50 achieves 42.0 mAP on COCO val after 500 epochs.

```mermaid
flowchart LR
    A[Input Image] --> B[CNN Backbone\nResNet-50]
    B --> C[Flatten + PE\nH'×W' tokens]
    C --> D[Transformer Encoder\n6 layers]
    D --> E[Transformer Decoder\n6 layers\nN=100 object queries]
    E --> F[FFN per query]
    F --> G["(box, class) per query\n100 predictions"]
    G --> H[Hungarian Matching\nvs. GT set]
    H --> I[Set Loss\nL1 + GIoU + CE]
    style D fill:#ddf,stroke:#99a
    style E fill:#dfd,stroke:#9a9
```

**Deformable DETR** [Zhu2020] (ICLR 2021): replaces global attention with **deformable attention**—each query attends to a small set of learned reference points per feature level, reducing encoder complexity from O(N²) to O(N·K) (K=4 sampled points). Converges 10× faster (50 epochs vs. 500); 46.2 mAP with ResNet-50.

**DAB-DETR** [Liu2022b] reformulates object queries as dynamic anchor boxes (x, y, w, h), providing spatial priors to the decoder cross-attention. **DN-DETR** [Li2022b] adds denoising training to stabilise the bipartite matching.

**DINO (Detection Transformer with Improved Denoising)** [Zhang2022] (ICLR 2023): combines DAB-DETR, DN-DETR, and deformable attention with a mixed query selection strategy; DINO-5scale-R50 achieves 49.0 mAP in 12 epochs; DINO-Swin-L achieves 58.5 mAP on COCO val.

**RT-DETR** [Zhao2024] (CVPR 2024): decouples intra-scale feature interaction (efficient encoder with IoU-aware query selection) from cross-scale fusion (PAN-like neck), enabling real-time inference. RT-DETR-R50 achieves 53.1 mAP at 108 fps on T4 GPU; RT-DETR-R101 achieves 54.3 mAP at 74 fps—the first DETR-family model competitive with YOLO on both accuracy and speed.

### Real-Time DETRs Overtake YOLO (2024–2026)

**D-FINE** [Peng2024] (arXiv 2410.13842) redefines DETR box regression as **Fine-grained Distribution Refinement (FDR)**: each decoder layer iteratively refines per-edge probability distributions over offsets rather than regressing coordinates directly, and **Global Optimal Localization Self-Distillation (GO-LSD)** distils the final layer's refined distributions into earlier layers at no extra inference cost. D-FINE-X reaches 55.8 AP (COCO only) and 59.3 AP with Objects365 pre-training at 12.9 ms on a T4.

**DEIM** [Huang2025] (CVPR 2025) attacks the slow convergence of one-to-one matching with **Dense O2O** (more targets per image via mosaic-style composition so that one-to-one matching yields more positives) and a matchability-aware loss; DEIM-D-FINE-X reaches 56.5 AP on COCO (59.5 AP with Objects365 pre-training) with substantially shorter schedules. **DEIMv2** (arXiv 2509.20787, Sep 2025; "Real-Time Object Detection Meets DINOv3") plugs DINOv3-derived ViT features into DEIM through a lightweight spatial tuning adapter (STA) for the S–X sizes and uses HGNetv2 for ultra-light sizes: DEIMv2-X reaches 57.8 AP with 50.3 M params, and DEIMv2-Atto 23.8 AP with only 0.5 M params for mobile devices.

**RT-DETRv4** (arXiv 2510.25257; accepted to ECCV 2026) keeps the lightweight RT-DETR/D-FINE detector at inference but adds a training-only distillation framework that injects vision-foundation-model representations (a DINOv3 ViT-B/16 teacher) into it; RT-DETRv4-X reports 57.0 AP at 78 fps (T4).

**RF-DETR** [Robinson2026] (Roboflow; arXiv 2511.09554; ICLR 2026) combines the LW-DETR detector with a **DINOv2 ViT backbone** (PE-Core-T for the smallest sizes) and **weight-sharing neural architecture search** over resolution, patch size, decoder depth and query count, so a single training run yields a whole Pareto family. In Roboflow's reproducible benchmark (all models measured on COCO val2017 with pycocotools; T4, TensorRT FP16, batch 1), **RF-DETR-2XL is the first real-time detector above 60 AP (60.1 AP at 17.2 ms)**, RF-DETR-N reaches 48.4 AP at 2.3 ms, and the family also leads on the RF100-VL transfer benchmark (100 diverse real-world datasets). Segmentation and (preview) keypoint heads share the same API.

```mermaid
flowchart LR
    A[Input] --> B[Pre-trained ViT\nDINOv2 / DINOv3]
    B --> C[Multi-scale projector\nor adapter STA]
    C --> D[Deformable DETR decoder\nfew layers, few queries]
    D --> E[Box head\nFDR distributions]
    D --> F[Class head]
    E --> G[One-to-one outputs\nno NMS]
    F --> G
    style B fill:#ddf,stroke:#99a
    style D fill:#dfd,stroke:#9a9
```

The common recipe of 2025–26 real-time detectors is therefore: *foundation-model features + light deformable decoder + better box parameterisation/matching + NMS-free outputs*, tuned for TensorRT latency rather than FLOPs.

---

## Open-Vocabulary and Grounded Detection

**OWL-ViT** [Minderer2022]: uses CLIP-style image–text alignment; bipartite matching loss on image patches; zero-shot detection by matching text queries to patch features.

**Grounding DINO** [Liu2023] (arXiv 2303.05499): marries DINO detector with grounded pre-training from GLIP/CLIP. A feature enhancer fuses text and image features at multiple scales; language-guided query selection initialises object queries from text-relevant image regions; a cross-modality decoder performs tight fusion. Grounding DINO achieves 52.5 mAP (COCO zero-shot, no COCO training data) and 63.0 mAP (COCO fine-tuned). Authors: Shilong Liu, Zhaoyang Zeng, Tianhe Ren et al. (2023).

**SAM 3** [Carion2025] (Meta, arXiv 2511.16719, Nov 2025) introduces **Promptable Concept Segmentation (PCS)**: given a short noun phrase and/or image exemplars, it detects, segments and (in video) tracks *every* matching instance. A DETR-style detector conditioned on text, geometry and exemplars shares a vision encoder with a SAM 2-style tracker (848 M params total); a learned **presence token** decouples "is the concept present?" from localisation, improving discrimination between closely related phrases. Trained with a data engine covering over 4 M unique concepts, SAM 3 reports 53.6 box AP on LVIS and 56.4 box AP on COCO, versus 52.4 / 56.0 for DINO-X, and reaches 75–80 % of human performance on the new SA-Co benchmark (270 K unique concepts). SAM 3.1 (Mar 2026) adds "Object Multiplex" shared-memory multi-object tracking (~7× faster at 128 objects on an H100).

**YOLOE** [Wang2025] (ICCV 2025) brings text-prompt, visual-prompt and prompt-free open-vocabulary detection and segmentation to a YOLO-speed model with lower training cost than YOLO-Worldv2. **Rex-Omni** (IDEA Research, arXiv 2510.12798, Oct 2025) instead casts detection as next-point/next-token prediction in a 3B-parameter MLLM built on Qwen2.5-VL, illustrating the convergence of detection with multimodal LLMs.

> *Correction (Oct 2026):* earlier versions of this file stated that RF-DETR is RT-DETR with a DINOv2 backbone reaching 60.5 AP at 25 fps. It is in fact built on LW-DETR, and its published best real-time result is 60.1 AP (RF-DETR-2XL, 17.2 ms on T4); see the Real-Time DETRs subsection above.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Rich Feature Hierarchies… (R-CNN) | Girshick et al. | 2014 | CVPR | Region-based CNN; proposal + classify pipeline |
| Fast R-CNN | Girshick | 2015 | ICCV | RoI Pooling; end-to-end joint training |
| Faster R-CNN | Ren, He, Girshick, Sun | 2015 | NeurIPS | Region Proposal Network; ~17 fps |
| Feature Pyramid Networks (FPN) | Lin, Dollár, Girshick et al. | 2017 | CVPR | Multi-scale top-down feature pyramid |
| Focal Loss / RetinaNet | Lin, Goyal, Girshick et al. | 2017 | ICCV | Focal loss; one-stage matches two-stage accuracy |
| FCOS | Tian, Shen, Chen, He | 2019 | ICCV | Anchor-free per-pixel prediction; centerness |
| YOLOv4 | Bochkovskiy, Wang, Liao | 2020 | arXiv | Systematic bag-of-freebies integration |
| End-to-End Object Detection (DETR) | Carion, Massa et al. | 2020 | ECCV | Transformer set prediction; eliminates NMS |
| Deformable DETR | Zhu, Su et al. | 2020 | ICLR 2021 | Deformable attention; 10× faster convergence |
| DINO (Detection Transformer) | Zhang et al. | 2022 | ICLR 2023 | DAB+DN+deformable; 58.5 mAP Swin-L |
| RT-DETR | Zhao, Lv et al. | 2023 | CVPR 2024 | Real-time DETR; 53.1 mAP @ 108 fps |
| Grounding DINO | Liu, Zeng, Ren et al. | 2023 | arXiv | Open-set detection; 52.5 mAP zero-shot COCO |
| D-FINE | Peng, Li, Wu et al. | 2024 | arXiv | Fine-grained distribution refinement + GO-LSD self-distillation |
| DEIM: DETR with Improved Matching | Huang, Lu, Cun et al. | 2024 | CVPR 2025 | Dense O2O matching; fast real-time DETR convergence |
| YOLOv12: Attention-Centric Real-Time Object Detectors | Tian, Ye, Doermann | 2025 | arXiv | Area attention + R-ELAN at YOLO latency |
| YOLOE: Real-Time Seeing Anything | Wang, Liu, Chen et al. | 2025 | ICCV | Text/visual/prompt-free open-vocabulary YOLO |
| Real-Time Object Detection Meets DINOv3 (DEIMv2) | Huang, Hou, Liu et al. | 2025 | arXiv | DINOv3 features in a real-time DETR; 0.5 M–50 M params |
| RT-DETRv4 | Liao, Zhao, Shan et al. | 2025 | ECCV 2026 | Training-time DINOv3 distillation into lightweight DETRs |
| SAM 3: Segment Anything with Concepts | Carion, Gustafson et al. | 2025 | arXiv | Promptable concept detection/segmentation/tracking; SA-Co benchmark |
| RF-DETR: Real-Time Detection Transformer | Robinson, Robicheaux, Popov, Ramanan, Peri | 2025 | ICLR 2026 | DINOv2 + LW-DETR + weight-sharing NAS; first real-time > 60 AP |
| Ultralytics YOLO26 | Jocher, Qiu et al. | 2026 | arXiv | NMS-free, DFL-free YOLO; MuSGD; multi-task family |

---

## Benchmark Performance: COCO mAP Progression

| Model | Backbone | COCO AP | AP50 | FPS (GPU) | Notes |
|-------|----------|---------|------|-----------|-------|
| Fast R-CNN | VGG-16 | 19.7 | 35.9 | ~1 | COCO 2015 test |
| Faster R-CNN | ResNet-101 | 27.2 | 48.4 | 5 | 2015; single scale |
| SSD-300 | VGG-16 | 25.1 | 43.1 | 46 | 300² input |
| YOLOv3 | Darknet-53 | 33.0 | 57.9 | 35 | COCO test-dev |
| RetinaNet | ResNeXt-101-FPN | 40.8 | 61.1 | 7 | COCO test-dev |
| YOLOv4 | CSPDarknet53 | 43.5 | 65.7 | 65 | V100 |
| Faster R-CNN + FPN | ResNet-101 | 36.2 | 59.1 | 6 | COCO minival |
| FCOS | ResNeXt-101-FPN | 44.7 | 64.1 | 14 | Multi-scale test |
| DETR | ResNet-50 | 42.0 | 62.4 | 28 | 500 epochs |
| Deformable DETR | ResNet-50 | 46.2 | 65.2 | 19 | 50 epochs |
| DINO | Swin-L | 58.5 | 77.0 | — | COCO val |
| YOLOv8x | CSP+C2f | 53.9 | 71.0 | 52 | COCO val |
| RT-DETR-R101 | ResNet-101 | 54.3 | 72.7 | 74 | T4; real-time |
| Grounding DINO | Swin-L | 63.0 | 81.4 | — | COCO fine-tuned |
| D-FINE-X | HGNetv2 | 55.8 | — | ~78 | T4 12.9 ms; 59.3 AP with Objects365 pre-training |
| DEIM-D-FINE-X | HGNetv2 | 56.5 | — | ~78 | T4 12.9 ms; CVPR 2025 |
| YOLOv12-X | R-ELAN + area attn | 54.4 | 71.1 | ~80 | T4 12.5 ms (as reported by YOLOv13 authors) |
| DEIMv2-X | DINOv3-derived ViT | 57.8 | — | ~73 | T4 13.75 ms; 50.3 M params |
| RT-DETRv4-X | HGNetv2 (+VFM distill.) | 57.0 | 74.6 | 78 | T4 |
| YOLO26x | YOLO26 | 57.5 | — | ~85 | T4 11.8 ms; 56.9 AP in NMS-free mode |
| RF-DETR-2XL | DINOv2 ViT (NAS) | 60.1 | 78.5 | ~58 | T4 17.2 ms; Roboflow in-house protocol |
| SAM 3 | Shared ViT encoder | 56.4 | — | — | Concept-prompted (text/exemplar); 848 M params; not real-time |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Two-stage (Faster R-CNN / FPN) | High accuracy; flexible proposal stage; strong baseline | Slower; two-stage complexity; not end-to-end NMS-free |
| One-stage anchor-based (RetinaNet/SSD) | Fast; simple architecture; well-calibrated with focal loss | Manual anchor design; struggles with extreme aspect ratios |
| Anchor-free (FCOS/CenterNet) | No anchor hyperparameters; per-pixel predictions; easy transfer | Centerness ambiguity for overlapping objects; slightly lower AP on crowded scenes |
| Transformer (DETR/RT-DETR) | Fully end-to-end (no NMS); global context; open-vocabulary extension | Slow convergence; high memory; O(N²) attention in base DETR |
| Open-vocabulary (Grounding DINO) | Detects arbitrary text-described objects; zero-shot generalisation | Slower inference; requires large-scale grounding pre-training; closed-form text precision |
| Real-time DETR on foundation backbones (D-FINE/DEIMv2/RF-DETR) | NMS-free; best accuracy–latency Pareto front in 2025–26; strong fine-tuning transfer (RF100-VL) | Depends on large pre-trained ViTs (licensing, memory); latency claims sensitive to TensorRT version and protocol |
| End-to-end YOLO (YOLOv10/YOLO26) | Very low CPU/edge latency; simple export; one family for many tasks | NMS-free mode trails NMS mode by ~0.6–0.8 AP (YOLO26); AGPL licensing for Ultralytics models |
| Concept-prompted foundation detectors (SAM 3) | Exhaustive detection of any noun phrase; unified with segmentation and tracking | ~850 M params; far from real-time on edge; reaches only 75–80 % of human performance on SA-Co |

---

## Open Problems & Research Gaps

- **NMS-free convergence speed:** Despite Deformable DETR and DN-DETR reducing DETR's 500-epoch requirement to 12–24 epochs, end-to-end transformer detectors still converge slower than anchor-based methods; theoretical understanding of set matching dynamics remains incomplete.
- **Small object detection:** AP_S (objects < 32² pixels) lags significantly behind AP_M and AP_L across all architectures; high-resolution backbone features at scale remain computationally prohibitive.
- **Domain generalisation:** COCO-trained detectors degrade sharply on sketch, art, or domain-shifted imagery; bridging photorealistic training to open-world deployment is unsolved.
- **Efficient open-vocabulary detection:** Grounding DINO and OWL-ViT require large language encoders that dominate inference cost; distilling open-vocabulary capabilities into compact detectors is an active challenge.
- **Dense prediction under occlusion:** Panoptic and instance segmentation resolve some crowded-scene issues, but 3D-aware detection (accounting for depth order and occlusion reasoning) at scale is still lacking robust solutions.
- **Temporal object detection:** Extending single-frame detection to video with consistent track identities under appearance change, occlusion, and motion blur remains dependent on post-hoc tracking heuristics rather than joint end-to-end learning.
- **Calibration of confidence scores:** High-AP models are often poorly calibrated—reported confidence does not match true precision; calibrated uncertainty estimates are necessary for safety-critical deployment.
- **Benchmark protocol drift:** 2025–26 real-time detectors report COCO AP under different pre-training (COCO-only vs Objects365), resolutions, TensorRT versions and end-to-end vs NMS modes; vendor tables can differ by several AP for the same checkpoint. Standardised, reproducible latency–accuracy protocols (and transfer suites such as RF100-VL) are needed.
- **COCO saturation vs. real-world transfer:** With real-time models at ~60 AP, further COCO gains are small and may partly reflect fitting annotation idiosyncrasies; fine-tuning efficiency on small, domain-specific datasets is becoming the more meaningful target.
- **Concept-level open-vocabulary detection:** SAM 3 shows that exhaustive concept detection is feasible, but performance on fine-grained or compositional phrases (attributes, relations, negatives) remains well below human agreement, and distilling it into real-time models is open.

---

## Further Reading

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)
- [Feature Pyramid Networks for Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1612.03144)
- [Focal Loss for Dense Object Detection — RetinaNet (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [End-to-End Object Detection with Transformers — DETR (Carion et al., 2020)](https://arxiv.org/abs/2005.12872)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection (Zhao et al., 2024)](https://arxiv.org/abs/2304.08069)
- [Grounding DINO: Marrying DINO with Grounded Pre-Training (Liu et al., 2023)](https://arxiv.org/abs/2303.05499)
- [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement (Peng et al., 2024)](https://arxiv.org/abs/2410.13842)
- [DEIM: DETR with Improved Matching for Fast Convergence (Huang et al., CVPR 2025)](https://arxiv.org/abs/2412.04234) · [DEIMv2 (2025)](https://arxiv.org/abs/2509.20787)
- [RF-DETR: Real-Time Detection Transformer (Robinson et al., ICLR 2026)](https://arxiv.org/abs/2511.09554) · [code & benchmarks](https://github.com/roboflow/rf-detr)
- [RT-DETRv4 (Liao et al., ECCV 2026)](https://arxiv.org/abs/2510.25257)
- [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models (Jocher et al., 2026)](https://arxiv.org/abs/2606.03748)
- [SAM 3: Segment Anything with Concepts (Carion et al., 2025)](https://arxiv.org/abs/2511.16719)
