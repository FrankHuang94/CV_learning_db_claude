# Core Vision Datasets: Classification, Detection, Segmentation, and Scene Understanding

> **Last Updated:** October 2026
> **Level:** Intermediate / Advanced
> **Related Sections:** [Image Classification](../02_core_tasks/00_image_classification.md) · [Object Detection](../02_core_tasks/01_object_detection.md) · [Semantic Segmentation](../02_core_tasks/02_semantic_segmentation.md) · [Evaluation Metrics](./04_evaluation_metrics.md)

> **🆕 October 2026 update:**
> - Added **SA-1B** and Meta's **SA-Co** (Segment Anything with Concepts, Nov 2025): an open-vocabulary concept-segmentation benchmark with ~270K unique concepts, built by the SAM 3 data engine. SAM 3 reaches 54.1 cgF1 on SA-Co/Gold against a 72.8 human reference.
> - Added the DINOv3 web-scale curated pretraining corpora **LVD-1689M** and **SAT-493M** (Aug 2025). These continue the shift from labelled ImageNet-21K pretraining to curated, label-free image pools.
> - Added **RF100-VL** (Roboflow + CMU, 2025), a 100-dataset multi-domain detection benchmark now reported alongside COCO by real-time detectors (e.g. RF-DETR-2XL reports 60.1 COCO AP at 17.2 ms).
> - Refreshed the SOTA claims for COCO and LVIS: Co-DETR reaches 66.0 COCO test-dev AP, and SAM 3 reaches 48.5 zero-shot LVIS mask AP vs 38.5 for DINO-X. Also noted that Papers With Code leaderboards are now archived.

---

## Overview

The modern computer vision research ecosystem is anchored by a small number of canonical datasets that simultaneously serve as training corpora, evaluation benchmarks, and implicit definitions of what tasks "matter." Understanding their design choices—annotation methodology, class taxonomy, data splits, and evaluation protocols—is essential for any PhD-level researcher, because dataset idiosyncrasies often explain more about published numbers than the model architectures themselves.

The datasets surveyed here span nearly two decades of dataset design evolution: from ImageNet's WordNet-synset taxonomy (Deng et al., 2009) through COCO's instance-level annotations (Lin et al., 2014), the large-vocabulary long-tail distribution of LVIS (Gupta et al., 2019), and the panoptic unification in ADE20K (Zhou et al., 2017). Each design decision—what to label, at what granularity, with what consensus mechanism, and over what image distribution—embeds assumptions that propagate into model biases. A researcher relying on ImageNet top-1 accuracy alone, for instance, will systematically miss failure modes on rare object classes, texture-shifted images, and out-of-distribution scenes.

A second theme is the shift from single-task to multi-task annotation. COCO was groundbreaking in providing bounding boxes, segmentation masks, keypoints, and captions on the same images. Open Images V7 and LVIS push this further with relationship annotations and dense long-tail coverage. Modern pretraining work leverages these multi-annotation datasets to jointly supervise detection, segmentation, and captioning heads—a paradigm central to systems like Florence-2 and InternImage. Understanding which datasets support which annotation types is prerequisite knowledge for designing any multi-task learning pipeline.

A third theme, which dominated 2025–2026, is the rise of **model-in-the-loop data engines** and **label-free curated pools**. SA-1B (2023) and SA-Co (2025) were annotated mostly by successive SAM generations with human verification, and their scale and concept coverage now exceed anything produced by manual crowdsourcing. DINOv3's LVD-1689M likewise shows that a carefully curated, unlabelled image pool can replace ImageNet-21K as the default backbone pretraining set. The classical benchmarks (ImageNet, COCO, ADE20K) have increasingly become *evaluation-only* artefacts, often run with frozen backbones, while the training data behind frontier models is internal or machine-annotated.

---

## ImageNet

**Full name:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
**Citation:** [Deng2009] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*. [Russakovsky2015] Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. *IJCV*.

The full ImageNet database contains over 14 million images organized under the WordNet noun hierarchy, covering more than 20,000 synsets. The ILSVRC benchmark uses a curated 1,000-class, 1.28M training / 50K validation / 100K test-image subset. Each training class contains approximately 1,300 images; validation and test sets have 50 images per class.

**Annotation:** Single image-level labels per image in ILSVRC-CLS; ILSVRC-DET adds bounding boxes for 200 object classes. ImageNet-21K (the full superset) is commonly used for pretraining.

**Key splits:** train (~1.28M), val (50K), test (100K) for ILSVRC-1K. ImageNet-21K has ~14.2M images across 21,841 classes.

**Benchmark significance:** The AlexNet 2012 result (top-5 error 15.3% vs. 26.2% for the prior best) catalysed the deep learning revolution. As of 2024, ViT-based models exceed 90% top-1 on ImageNet-1K, and the dataset is now considered "solved" for coarse classification, driving the community toward harder variants (ImageNet-A, -R, -Sketch, ObjectNet) that test robustness and out-of-distribution generalization.

**2025–2026 status:** Fine-tuned ImageNet-1K top-1 has barely moved since 2022 (≈91%). Its role has shifted. It is now mainly a *diagnostic* for frozen representations (linear or k-NN probes on backbones such as DINOv3 [Simeoni2025]) and a source of robustness variants. It is no longer used to rank architectures. Label errors in the validation set (documented by [Beyer2020]) cap the meaningful ceiling well below 100%.

---

## MS-COCO

**Full name:** Microsoft COCO: Common Objects in Context
**Citation:** [Lin2014] Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*.

COCO contains 328K images annotated with 80 object categories, 2.5 million labeled instances, and 5 captions per image. The canonical COCO 2017 split uses 118K train / 5K val / 41K test images. Object instances are annotated with both bounding boxes and high-quality polygon segmentation masks (instance segmentation). The dataset also includes a 17-keypoint skeleton annotation for person instances (COCO-Keypoints) and panoptic segmentation annotations covering 80 "thing" categories and 53 "stuff" categories.

**Evaluation protocol:** Detection is evaluated with COCO AP (mean over IoU thresholds 0.5:0.05:0.95), a stricter standard than the Pascal VOC mAP@0.5. The COCO leaderboard has driven the majority of detection and instance segmentation research from 2015 onward.

**Crowdsourcing design:** Annotations were collected via a staged Amazon Mechanical Turk pipeline with instance segmentation via workers clicking boundary polygons. Inter-annotator agreement and hierarchical quality control are described in [Lin2014].

**Caption annotations:** Each of the 328K images has 5 independently written captions, yielding a total of ~1.5M reference captions used for image captioning evaluation (CIDEr, BLEU-4, METEOR, SPICE).

**Current state of the leaderboard (2026):** Closed-vocabulary COCO detection is close to ceiling. Co-DETR with a 304M-parameter ViT-L reported 66.0 box AP on test-dev in 2023, and later systems have added only marginal gains. Research attention has moved to two regimes. The first is (i) **real-time detection**, where DETR-family detectors now beat YOLO-family detectors. For example, RF-DETR-2XL reports 60.1 AP<sub>50:95</sub> at 17.2 ms, and RF-DETR-M reports 54.7 AP at 4.4 ms, with all numbers re-measured in-house by Roboflow. The second is (ii) **zero-shot / open-vocabulary transfer**. Here SAM 3 reports 53.6 COCO box AP when prompted with category names, without COCO-specific training [Carion2025].

---

## PASCAL VOC

**Full name:** PASCAL Visual Object Classes Challenge
**Citation:** [Everingham2010] Everingham, M. et al. (2010). The Pascal Visual Object Classes (VOC) Challenge. *IJCV*.

PASCAL VOC ran from 2005–2012 and defined early object detection benchmarks. VOC 2007 contains 9,963 images (5,011 train/val + 4,952 test) with 24,640 annotated objects across 20 categories. VOC 2012 contains 22,531 images (11,540 train/val + 10,991 test). The combined VOC07+12 training set (≈16.5K images) was the standard detection pretraining set before COCO.

**Annotation:** Bounding boxes + class labels; segmentation masks added in later editions. 20 categories cover person, animals (bird, cat, cow, dog, horse, sheep), vehicles, and indoor objects.

**Legacy:** VOC mAP@0.5 remains reported for historical comparison. The 20-class setting is intentionally small enough that rare-class failure modes are not visible—a limitation directly addressed by LVIS.

---

## Open Images

**Full name:** Open Images V7 (Google)
**Citation:** [Kuznetsova2020] Kuznetsova, A. et al. (2020). The Open Images Dataset V4. *IJCV*. V7 released 2022.

Open Images V7 covers approximately 9 million images with multi-level annotations:
- 16M bounding boxes for 600 object classes on 1.9M images
- 2.7M instance segmentation masks across 350 classes on 944K images
- 3.3M visual relationship annotations across 1,466 relationship triplets
- 66.4M point-level labels on 1.4M images spanning 5,827 classes
- 61.4M image-level labels across 20,638 classes

**Hierarchical taxonomy:** Classes are organized in a hierarchy allowing evaluation at different levels of specificity. Open Images also provides the "Localized Narratives" annotation—spoken descriptions synchronized with mouse traces—enabling dense referring expression and dense captioning research.

**Train/val/test split:** 9M total images, with the annotated subset split across train (1.74M), validation (41K), and test (125K) for the detection task.

---

## ADE20K

**Full name:** ADE20K Scene Parsing Dataset (MIT CSAIL)
**Citation:** [Zhou2017] Zhou, B. et al. (2017). Scene Parsing through ADE20K Dataset. *CVPR 2017*.

ADE20K provides dense pixel-level semantic annotations across 150 categories (stuffs and things unified) on 25,562 images: 20,210 train, 2,000 val, and 3,352 test. Images were sourced from SUN and Places databases. On average each image contains 19.5 object instances and 10.5 distinct semantic classes. The benchmark was the primary evaluation venue for semantic segmentation until ~2021; models like Swin-T, SegFormer, and Mask2Former all report ADE20K mIoU as a primary metric.

**Annotation depth:** Unlike COCO (binary foreground mask), ADE20K uses a hierarchical object-part annotation: each object can have annotated sub-parts (e.g., "wheel" as a part of "car"). This enables scene parsing research that goes beyond flat segmentation.

---

## Cityscapes

**Full name:** The Cityscapes Dataset for Semantic Urban Scene Understanding
**Citation:** [Cordts2016] Cordts, M. et al. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. *CVPR 2016*.

Cityscapes provides 5,000 finely annotated images (2,975 train / 500 val / 1,525 test) and 20,000 coarsely annotated images captured from moving vehicles in 50 cities. Images are 2048×1024 pixels with 30-class annotations (19 evaluated). The fine annotation used polygon tools and averages ~90 minutes per image. Evaluates semantic segmentation (mIoU), instance segmentation (AP), and panoptic segmentation.

**Domain specificity:** As an urban driving dataset, Cityscapes is the canonical benchmark for street scene understanding. Models must handle scale variability (distant pedestrians vs. nearby buses), motion blur, and weather/lighting variation. Video sequences are available for temporal methods.

**Coarse annotations:** The additional 20K coarsely annotated images were designed to support semi-supervised and self-supervised pretraining studies.

---

## LVIS

**Full name:** Large Vocabulary Instance Segmentation
**Citation:** [Gupta2019] Gupta, A., Dollar, P., & Girshick, R. (2019). LVIS: A Dataset for Large Vocabulary Instance Segmentation. *CVPR 2019*.

LVIS annotates 164K images (from COCO) with high-quality instance segmentation masks for 1,203 categories. It deliberately preserves the long-tail frequency distribution of objects in natural images: 337 "rare" categories (1–10 training images), 461 "common" (11–100), and 405 "frequent" (>100). Total annotations: 2.2M instances. Training split: 100K images / 1.2M annotations.

**Significance for long-tail learning:** Standard detectors trained on COCO collapse on rare LVIS categories. LVIS drove a large body of work on repeat factor sampling [Gupta2019], equalization loss [Tan2020], and federated loss, directly relevant to any production system handling open-vocabulary detection.

**Evaluation:** AP_r, AP_c, AP_f stratify performance by frequency, exposing model behavior on underrepresented classes.

**Open-vocabulary use (2025–2026):** LVIS has become the standard *zero-shot* testbed for concept-prompted detectors and segmenters. SAM 3 reports 48.5 LVIS mask AP without LVIS training, against 38.5 for DINO-X. OWLv2 reaches 43.4, but it was partially trained on LVIS. Supervised Co-DETR fine-tuned on LVIS reached 68.0 box AP and 56.0 mask AP on LVIS val (2023).

---

## Places

**Full name:** Places365 / Places205
**Citation:** [Zhou2018] Zhou, B. et al. (2018). Places: A 10 Million Image Database for Scene Recognition. *TPAMI*.

Places365-Standard: 1.8M training images × 365 scene categories + 50 val / 900 test images per category. The Places365-Challenge extends training to ~8M images. The full Places database exceeds 10M images. The database is organized by semantic scene categories (e.g., "abbey", "kitchen", "highway") rather than object categories, making it complementary to ImageNet for transfer learning studies on scene-level tasks.

**Cross-dataset transfer:** Models pretrained on Places transfer better to satellite image classification and medical scene labeling than ImageNet-pretrained models in some studies, motivating dataset-selection analysis in transfer learning.

---

## SA-1B and SA-Co (Segment Anything Data Engines)

**Citation:** [Kirillov2023] Kirillov, A. et al. (2023). Segment Anything. *ICCV 2023*. [Carion2025] Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). SAM 3: Segment Anything with Concepts. Meta Superintelligence Labs.

**SA-1B** (2023) contains ~11M licensed, privacy-protected images with ~1.1B class-agnostic masks, produced by a three-stage model-in-the-loop data engine (assisted-manual → semi-automatic → fully automatic). It defined the *promptable segmentation* task. Its masks carry no semantic labels.

**SA-Co** (Segment Anything with Concepts, Nov 2025) is the data and benchmark release behind SAM 3. It adds *semantics*: each image or video is paired with short noun phrases, and every instance matching a phrase is exhaustively masked with a unique object ID. Phrases with no matching objects serve as hard *negative prompts*. Key facts from the official release:
- The benchmark spans **~270K unique concepts**, which Meta describes as over 50× more than earlier open-vocabulary segmentation benchmarks. It has two image splits (**SA-Co/Gold**, triple-annotated; **SA-Co/Silver**) and a video split (**SA-Co/VEval**).
- The training data engine automatically annotated **over 4M unique concepts**, combining SAM/MLLM proposals with AI and human verifiers.
- The primary metric is **cgF1**, a classification-gated F1 that multiplies image-level presence calibration (IL_MCC) by positive micro-F1 on localisation (see [Evaluation Metrics](./04_evaluation_metrics.md)).
- Reference results on SA-Co/Gold (instance segmentation, cgF1): human 72.8; **SAM 3 54.1**; OWLv2* 24.6; DINO-X 21.3; Gemini 2.5 13.0. SAM 3 is an 848M-parameter detector plus tracker sharing one vision encoder. SAM 3.1 checkpoints with faster multi-object tracking followed in March 2026.

**Why it matters:** SA-Co turns open-vocabulary segmentation from a 1,203-class LVIS problem into a quarter-million-concept problem with explicit negatives. Under that change, VLM-based detectors that look strong on COCO-style benchmarks lose most of their accuracy. The approach of a model-in-the-loop engine with human verification only on hard cases has become the template for new large annotation efforts. SAM 3D (Nov 2025) extended it to 3D (see [3D and Robotics Datasets](./02_3d_and_robotics_datasets.md)).

---

## LVD-1689M and SAT-493M (Curated Pretraining Pools)

**Citation:** [Simeoni2025] Siméoni, O., Vo, H. V., Seitzer, M., et al. (2025). DINOv3. arXiv:2508.10104 (Meta FAIR).

DINOv3 (Aug 2025) was trained without labels on **LVD-1689M**, a curated pool of roughly 1.7B web images, and in a separate variant on **SAT-493M**, ~493M satellite images. Its 7B-parameter ViT teacher was distilled into ViT-S/B/L/H+ and ConvNeXt students. The datasets themselves are not released. They matter as evidence that *curation by retrieval/clustering around a seed distribution* scales label-free pretraining beyond ImageNet-21K. Frozen DINOv3 features are reported to outperform specialised state-of-the-art models across many dense tasks without fine-tuning. That makes ADE20K, COCO and NYUv2 increasingly *probe* benchmarks for frozen backbones. Downstream releases include a COCO2017 detection head, an ADE20K segmentation head, and the Canopy Height Maps v2 model (Mar 2026), all built on the frozen 7B backbone.

---

## RF100-VL (Multi-Domain Detection for VLMs)

**Citation:** [Robicheaux2025] Robicheaux, P., Popov, M., Madan, A., Robinson, I., Nelson, J., Ramanan, D., & Peri, N. (2025). Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models. Roboflow / CMU.

RF100-VL is a curated set of 100 real-world detection datasets hosted on Roboflow Universe and distributed in COCO-JSON format. Most of its domains are under-represented in web pretraining data. It has a few-shot variant (**RF100-VL-FSOD**) and a 20-dataset subset (**RF20-VL**), so it can evaluate both conventional fine-tuned detectors and VLM-style zero-shot and few-shot detection. It has quickly become the *second axis* reported by real-time detectors next to COCO. RF-DETR, for example, reports both COCO AP and RF100-VL AP, because COCO gains increasingly fail to transfer to out-of-distribution domains.

---

## Key Papers / Datasets

| Name | Authors/Org | Year | Venue | Key Stats / Contribution |
|---|---|---|---|---|
| ImageNet ILSVRC | Deng, Dong, Socher, Li, Fei-Fei (Stanford) | 2009 | CVPR | 14.2M images, 21K synsets (ILSVRC: 1.28M / 1K classes); triggered deep learning revolution |
| MS-COCO | Lin, Maire, et al. (Microsoft Research) | 2014 | ECCV | 328K images, 2.5M instances, 80 categories; introduced AP@0.5:0.95 |
| PASCAL VOC 2012 | Everingham et al. | 2012 | IJCV | 22.5K images, 20 categories; standard pre-COCO detection benchmark |
| Open Images V7 | Kuznetsova et al. (Google) | 2022 | IJCV | 9M images, 600 bbox classes, 350 mask classes, visual relationships |
| ADE20K | Zhou et al. (MIT CSAIL) | 2017 | CVPR | 25.5K images, 150 semantic categories, hierarchical part annotations |
| Cityscapes | Cordts et al. (Daimler et al.) | 2016 | CVPR | 5K fine + 20K coarse images; 2048×1024 urban, 30 classes |
| LVIS | Gupta, Dollar, Girshick (FAIR) | 2019 | CVPR | 164K images, 1,203 categories, 2.2M instances; long-tail distribution |
| Places365 | Zhou, Lapedriza et al. (MIT) | 2018 | TPAMI | 1.8M train images, 365 scene categories; largest scene recognition DB |
| ImageNet-21K | Deng et al. / Ridnik et al. (2021) | 2009/2021 | CVPR/arXiv | 14.2M images, 21.8K classes; preferred pretraining for ViT-based models |
| SA-1B | Kirillov et al. (Meta AI) | 2023 | ICCV | ~11M images, ~1.1B class-agnostic masks; model-in-the-loop data engine |
| SA-Co (SAM 3) | Carion, Gustafson, Hu et al. (Meta) | 2025 | Meta tech report | ~270K-concept open-vocabulary segmentation benchmark (Gold/Silver/VEval); 4M+ auto-annotated concepts; cgF1 metric |
| LVD-1689M / SAT-493M (DINOv3) | Siméoni et al. (Meta FAIR) | 2025 | arXiv | ~1.7B curated web images / ~493M satellite images for label-free 7B ViT pretraining (not released) |
| RF100-VL | Robicheaux, Popov et al. (Roboflow/CMU) | 2025 | arXiv | 100 multi-domain detection datasets + few-shot (FSOD) variant; OOD complement to COCO |

---

## Benchmark Comparison Table

| Dataset | Images (train) | Categories | Annotation Type | Primary Task | Eval Metric |
|---|---|---|---|---|---|
| ImageNet-1K | 1.28M | 1,000 | Image-level class | Classification | Top-1 / Top-5 Acc |
| MS-COCO 2017 | 118K | 80 things + 53 stuff | BBox + mask + caption | Det / Seg / Cap | AP@0.5:0.95 |
| PASCAL VOC 2012 | 11.5K | 20 | BBox + seg mask | Detection | mAP@0.5 |
| Open Images V7 | 1.74M (annotated) | 600 (det) | BBox + mask + relations | Det / Seg / VRD | mAP |
| ADE20K | 20.2K | 150 | Dense pixel | Semantic seg | mIoU |
| Cityscapes | 2.975K fine | 19 (eval) | Dense pixel + instance | Sem / Inst seg | mIoU / AP |
| LVIS v1 | 100K | 1,203 | Instance mask | Instance seg | AP, AP_r/c/f |
| Places365 | 1.8M | 365 | Image-level class | Scene classif. | Top-1 / Top-5 |
| SA-1B | ~11M | none (class-agnostic) | Mask | Promptable seg | mIoU (prompted) |
| SA-Co/Gold | eval only | ~270K concepts (full benchmark) | Phrase → all instance masks + negatives | Concept seg / det | cgF1 (IL_MCC × pmF1) |
| RF100-VL | 100 datasets (varied) | per-dataset | BBox | Multi-domain det | AP<sub>50:95</sub> (avg) |

---

## Pros & Cons

| Dataset | Strengths | Limitations |
|---|---|---|
| ImageNet-1K | Canonical; enormous ecosystem of pretrained models; well-understood biases | Texture bias; single label per image; not representative of real-world distribution; near-saturated |
| MS-COCO | Multi-task annotations; rich ecosystem; strict AP metric exposes localization quality | 80 categories miss rare objects; COCO images are curated, not naturalistic |
| LVIS | Realistic long-tail distribution; high-quality masks | Small per-class train sets for rare classes; harder to train from scratch |
| Open Images | Massive scale; multi-type annotations including relations | Crowd-sourced labels noisier; verification methodology differs from COCO |
| ADE20K | Deep hierarchical annotation; scene + object + part | Small absolute size (20K train); domain covers interior/exterior scenes non-uniformly |
| Cityscapes | High-resolution, high-quality fine annotations; real-world urban | Single domain (European cities); 5K fine images limit model capacity |
| Places365 | Largest scene database; 365 diverse categories | Image-level only; no spatial annotations |
| SA-Co | Massive concept vocabulary; explicit negative prompts; image + video splits | Machine-in-the-loop labels inherit SAM/MLLM biases; noun-phrase prompts only (no complex referring expressions) |
| RF100-VL | Domain diversity; exposes COCO-overfitting; few-shot protocol | Heterogeneous label quality across 100 community datasets; averaged AP hides per-domain failure |

---

## Open Problems & Research Gaps

- **Long-tail and open-vocabulary evaluation:** LVIS exposes failure at 1,203 categories; real-world systems encounter millions. SA-Co (2025) raises the vocabulary to ~270K concepts with negative prompts. Even so, the best model (SAM 3, 54.1 cgF1 on SA-Co/Gold) remains well below human agreement (72.8), and general-purpose VLMs score far lower (Gemini 2.5: 13.0). Long referring expressions, compositional attributes and relations are still outside the benchmark's noun-phrase scope.
- **Machine-annotated ground truth:** As data engines (SA-1B, SA-Co) replace manual labelling, benchmarks risk measuring agreement with the *previous model generation*. Protocols for auditing model-in-the-loop labels, such as triple human annotation in SA-Co/Gold and oracle-over-annotators scoring, are still ad hoc.
- **Dataset bias and spurious correlations:** ImageNet models rely on texture cues [Geirhos2019]. Systematic tools for measuring and correcting distribution shift remain an open area, especially as synthetic data increasingly enters training pipelines.
- **Annotation noise quantification:** Ground-truth labels in crowdsourced datasets contain systematic errors (e.g., COCO instance boundaries). Benchmarks for measuring annotation noise impact on model performance are underdeveloped.
- **Multi-label and compositional ground truth:** Most benchmarks assign one dominant category; real scenes have compositional semantics. GQA and Visual Genome point toward richer ground truth, but scalable annotation for compositional scene graphs remains unsolved.
- **Temporal and video extensions:** Static image benchmarks do not measure temporal consistency. Video Object Segmentation benchmarks (DAVIS, YouTube-VOS) exist but lack the scale of image-domain equivalents.
- **Geographical and demographic diversity:** ImageNet, COCO, and Places are disproportionately drawn from Western contexts. Recent work (GeoDE, Dollar-Sign datasets) quantifies this; fixing it at scale is unresolved.
- **Benchmark saturation and Goodhart's Law:** Once a benchmark like ImageNet-1K is effectively saturated (>90% top-1), continued optimization risks overfitting evaluation rather than improving real-world capability. New evaluation paradigms (e.g., model soups, diverse test suites) are active research areas. Multi-domain suites (RF100-VL) and frozen-backbone probing protocols (DINOv3-style) are the current responses.
- **Leaderboard infrastructure:** Papers With Code, the de facto public SOTA tracker for these datasets, is no longer maintained. Its evaluation tables survive only as a static archive (`pwc-archive` on Hugging Face). Leaderboards are now fragmented across benchmark-owner sites, Hugging Face Spaces and self-reported model cards, so "current SOTA" claims are harder to verify and protocol differences are less visible.

---

## Further Reading

- [ImageNet website and ILSVRC results archive](https://www.image-net.org/)
- [COCO benchmark leaderboard](https://cocodataset.org/#detection-leaderboard)
- [Papers With Code — ImageNet benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet) (archived; data dumps at [paperswithcode-data](https://github.com/paperswithcode/paperswithcode-data))
- [Beyer et al. (2020) — Are we done with ImageNet? arXiv:2006.07159](https://arxiv.org/abs/2006.07159)
- [Gupta et al. (2019) — LVIS paper arXiv:1908.03195](https://arxiv.org/abs/1908.03195)
- [Open Images V7 facts & figures](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [SAM 3 and the SA-Co benchmark — official repository](https://github.com/facebookresearch/sam3)
- [Kirillov et al. (2023) — Segment Anything (SA-1B) arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- [DINOv3 (LVD-1689M / SAT-493M) — arXiv:2508.10104 and code](https://github.com/facebookresearch/dinov3)
- [RF100-VL benchmark repository](https://github.com/roboflow/rf100-vl)
- [RF-DETR — real-time detection benchmarked on COCO and RF100-VL](https://github.com/roboflow/rf-detr)
