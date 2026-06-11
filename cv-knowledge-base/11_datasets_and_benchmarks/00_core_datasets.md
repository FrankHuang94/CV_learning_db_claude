# Core Vision Datasets: Classification, Detection, Segmentation, and Scene Understanding

> **Last Updated:** June 2026
> **Level:** Intermediate / Advanced
> **Related Sections:** [Image Classification](../02_core_tasks/00_image_classification.md) · [Object Detection](../02_core_tasks/01_object_detection.md) · [Semantic Segmentation](../02_core_tasks/02_semantic_segmentation.md) · [Evaluation Metrics](./04_evaluation_metrics.md)

---

## Overview

The modern computer vision research ecosystem is anchored by a small number of canonical datasets that simultaneously serve as training corpora, evaluation benchmarks, and implicit definitions of what tasks "matter." Understanding their design choices—annotation methodology, class taxonomy, data splits, and evaluation protocols—is essential for any PhD-level researcher, because dataset idiosyncrasies often explain more about published numbers than the model architectures themselves.

The datasets surveyed here span nearly two decades of dataset design evolution: from ImageNet's WordNet-synset taxonomy (Deng et al., 2009) through COCO's instance-level annotations (Lin et al., 2014), the large-vocabulary long-tail distribution of LVIS (Gupta et al., 2019), and the panoptic unification in ADE20K (Zhou et al., 2017). Each design decision—what to label, at what granularity, with what consensus mechanism, and over what image distribution—embeds assumptions that propagate into model biases. A researcher relying on ImageNet top-1 accuracy alone, for instance, will systematically miss failure modes on rare object classes, texture-shifted images, and out-of-distribution scenes.

A second theme is the shift from single-task to multi-task annotation. COCO was groundbreaking in providing bounding boxes, segmentation masks, keypoints, and captions on the same images. Open Images V7 and LVIS push this further with relationship annotations and dense long-tail coverage. Modern pretraining work leverages these multi-annotation datasets to jointly supervise detection, segmentation, and captioning heads—a paradigm central to systems like Florence-2 and InternImage. Understanding which datasets support which annotation types is prerequisite knowledge for designing any multi-task learning pipeline.

---

## ImageNet

**Full name:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
**Citation:** [Deng2009] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*. [Russakovsky2015] Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. *IJCV*.

The full ImageNet database contains over 14 million images organized under the WordNet noun hierarchy, covering more than 20,000 synsets. The ILSVRC benchmark uses a curated 1,000-class, 1.28M training / 50K validation / 100K test-image subset. Each training class contains approximately 1,300 images; validation and test sets have 50 images per class.

**Annotation:** Single image-level labels per image in ILSVRC-CLS; ILSVRC-DET adds bounding boxes for 200 object classes. ImageNet-21K (the full superset) is commonly used for pretraining.

**Key splits:** train (~1.28M), val (50K), test (100K) for ILSVRC-1K. ImageNet-21K has ~14.2M images across 21,841 classes.

**Benchmark significance:** The AlexNet 2012 result (top-5 error 15.3% vs. 26.2% for the prior best) catalysed the deep learning revolution. As of 2024, ViT-based models exceed 90% top-1 on ImageNet-1K, and the dataset is now considered "solved" for coarse classification, driving the community toward harder variants (ImageNet-A, -R, -Sketch, ObjectNet) that test robustness and out-of-distribution generalization.

---

## MS-COCO

**Full name:** Microsoft COCO: Common Objects in Context
**Citation:** [Lin2014] Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*.

COCO contains 328K images annotated with 80 object categories, 2.5 million labeled instances, and 5 captions per image. The canonical COCO 2017 split uses 118K train / 5K val / 41K test images. Object instances are annotated with both bounding boxes and high-quality polygon segmentation masks (instance segmentation). The dataset also includes a 17-keypoint skeleton annotation for person instances (COCO-Keypoints) and panoptic segmentation annotations covering 80 "thing" categories and 53 "stuff" categories.

**Evaluation protocol:** Detection is evaluated with COCO AP (mean over IoU thresholds 0.5:0.05:0.95), a stricter standard than the Pascal VOC mAP@0.5. The COCO leaderboard has driven the majority of detection and instance segmentation research from 2015 onward.

**Crowdsourcing design:** Annotations were collected via a staged Amazon Mechanical Turk pipeline with instance segmentation via workers clicking boundary polygons. Inter-annotator agreement and hierarchical quality control are described in [Lin2014].

**Caption annotations:** Each of the 328K images has 5 independently written captions, yielding a total of ~1.5M reference captions used for image captioning evaluation (CIDEr, BLEU-4, METEOR, SPICE).

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

---

## Places

**Full name:** Places365 / Places205
**Citation:** [Zhou2018] Zhou, B. et al. (2018). Places: A 10 Million Image Database for Scene Recognition. *TPAMI*.

Places365-Standard: 1.8M training images × 365 scene categories + 50 val / 900 test images per category. The Places365-Challenge extends training to ~8M images. The full Places database exceeds 10M images. The database is organized by semantic scene categories (e.g., "abbey", "kitchen", "highway") rather than object categories, making it complementary to ImageNet for transfer learning studies on scene-level tasks.

**Cross-dataset transfer:** Models pretrained on Places transfer better to satellite image classification and medical scene labeling than ImageNet-pretrained models in some studies, motivating dataset-selection analysis in transfer learning.

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

---

## Open Problems & Research Gaps

- **Long-tail and open-vocabulary evaluation:** LVIS exposes failure at 1,203 categories; real-world systems encounter millions. Open-vocabulary detection benchmarks (OWLv2, GLIP evaluations) are still being standardized.
- **Dataset bias and spurious correlations:** ImageNet models rely on texture cues [Geirhos2019]. Systematic tools for measuring and correcting distribution shift remain an open area, especially as synthetic data increasingly enters training pipelines.
- **Annotation noise quantification:** Ground-truth labels in crowdsourced datasets contain systematic errors (e.g., COCO instance boundaries). Benchmarks for measuring annotation noise impact on model performance are underdeveloped.
- **Multi-label and compositional ground truth:** Most benchmarks assign one dominant category; real scenes have compositional semantics. GQA and Visual Genome point toward richer ground truth, but scalable annotation for compositional scene graphs remains unsolved.
- **Temporal and video extensions:** Static image benchmarks do not measure temporal consistency. Video Object Segmentation benchmarks (DAVIS, YouTube-VOS) exist but lack the scale of image-domain equivalents.
- **Geographical and demographic diversity:** ImageNet, COCO, and Places are disproportionately drawn from Western contexts. Recent work (GeoDE, Dollar-Sign datasets) quantifies this; fixing it at scale is unresolved.
- **Benchmark saturation and Goodhart's Law:** Once a benchmark like ImageNet-1K is effectively saturated (>90% top-1), continued optimization risks overfitting evaluation rather than improving real-world capability. New evaluation paradigms (e.g., model soups, diverse test suites) are active research areas.

---

## Further Reading

- [ImageNet website and ILSVRC results archive](https://www.image-net.org/)
- [COCO benchmark leaderboard](https://cocodataset.org/#detection-leaderboard)
- [Papers With Code — ImageNet benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet)
- [Beyer et al. (2020) — Are we done with ImageNet? arXiv:2006.07159](https://arxiv.org/abs/2006.07159)
- [Gupta et al. (2019) — LVIS paper arXiv:1908.03195](https://arxiv.org/abs/1908.03195)
- [Open Images V7 facts & figures](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
