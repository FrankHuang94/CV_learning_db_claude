# Medical Imaging: Modalities, Segmentation, and Clinical CV

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:**
> - [Semantic Segmentation](../02_core_tasks/02_semantic_segmentation.md)
> - [Foundation Models in Medical Imaging](./01_foundation_models_medical.md)
> - [Scientific Imaging](./02_scientific_imaging.md)
> - [Datasets and Benchmarks](../11_datasets_and_benchmarks/00_image_classification_benchmarks.md)

---

## Overview

Medical imaging represents one of the most consequential application domains for computer vision, spanning a heterogeneous collection of physical acquisition modalities each governed by distinct signal physics, spatial resolutions, and noise characteristics. The core challenge is that deep learning systems trained on one institution's scanner configurations, patient demographics, or staining protocols frequently fail to generalize to external sites—a phenomenon known as **domain shift**—while simultaneously being challenged by severe class imbalance (lesions are small relative to background tissue) and the scarcity of expert-annotated data. Unlike natural image benchmarks, ground-truth annotations in medicine require specialized clinical expertise, creating a fundamental bottleneck that shapes the entire field.

From a signal-processing perspective, the five major modalities—X-ray (projection radiography), computed tomography (CT), magnetic resonance imaging (MRI), ultrasound (US), and histopathology (whole-slide images, WSI)—present fundamentally different learning problems. CT encodes Hounsfield Units on a calibrated linear scale and is well-suited to 3D convolutional processing; MRI requires multi-sequence (T1, T2, FLAIR, DWI) fusion where each channel encodes different tissue contrast; ultrasound is plagued by speckle noise and probe-dependent acquisition geometry; WSI operates at gigapixel scale and demands hierarchical or patch-based processing. Histopathology images, typically stained with hematoxylin and eosin (H&E), additionally suffer from stain variation across preparation protocols, motivating stain normalization as a preprocessing step.

The seminal architectural contribution to medical image segmentation is the **U-Net** [Ronneberger2015], which introduced a fully convolutional encoder–decoder with skip connections that preserve fine-grained spatial detail lost during downsampling. This design philosophy—encode global context, decode with preserved local detail—has proven remarkably durable. Its principled extension, **nnU-Net** [Isensee2021], automated the configuration of preprocessing, patch size, normalization, network topology, and post-processing via a dataset fingerprinting mechanism, achieving top performance on 33 of 55 segmentation tasks across 23 public benchmarks without task-specific manual tuning. As of 2026, nnU-Net remains a dominant baseline that specialized methods must beat.

---

## Imaging Modalities

### X-ray and Computed Radiography

Projection radiography collapses 3D anatomy onto a 2D detector plane, creating overlapping structures and limiting depth discrimination. CNNs adapted from ImageNet pretraining (DenseNet-121 via CheXNet [Rajpurkar2017]) demonstrated radiologist-competitive pneumonia detection on the NIH ChestX-ray14 dataset. Key tasks include: multi-label pathology classification (14-class NIH), pneumothorax segmentation, and bone suppression.

### Computed Tomography

CT volumes are natural inputs for 3D ConvNets and nnU-Net's 3D full-resolution configuration. The Hounsfield scale provides modality-consistent intensity normalization, enabling straightforward cross-site calibration. Landmark papers include V-Net [Milletari2016] (Dice loss for volumetric segmentation) and the 3D U-Net [Ccieck2016]. Multi-organ segmentation on abdominal CT (liver, spleen, pancreas, kidneys) is a canonical task.

### Magnetic Resonance Imaging

MRI's multi-modal sequences are intrinsically multi-channel inputs. Brain tumor segmentation (BraTS) requires joint processing of T1, T1ce, T2, and FLAIR volumes. Key challenges include: intensity non-standardness across scanners (MRI lacks the absolute calibration of CT), motion artifacts, and the need for 3D context. Transformer-based methods (Swin-UNETR [Tang2022]) have surpassed pure-CNN baselines on BraTS by capturing long-range spatial dependencies.

### Ultrasound

Real-time ultrasound applications demand both accuracy and inference speed under ~30 FPS constraints. Speckle noise and anisotropic point-spread functions motivate domain-specific augmentation strategies. Echocardiography segmentation (EchoNet-Dynamic [Ouyang2020]) and thyroid nodule detection are established benchmarks.

### Histopathology / Whole-Slide Imaging

WSI at 40× magnification produces gigapixel images ($\sim 100{,}000 \times 100{,}000$ pixels) that cannot fit in GPU memory. Standard pipelines apply patch extraction at multiple magnifications, followed by aggregation via multiple-instance learning (MIL) or attention-pooling. The **CAMELYON** challenge (lymph node metastasis detection) and **TCGA** pan-cancer cohorts are primary benchmarks. Digital pathology foundation models (CONCH, UNI, PLIP) exploit self-supervised pretraining on millions of patches.

---

## Core Architectures

### U-Net [Ronneberger2015]

The U-Net encoder–decoder introduces **long skip connections** concatenating encoder feature maps to corresponding decoder layers:

```
Encoder path:  Conv → MaxPool → ... (contracting)
Skip:          encoder[i] → concat → decoder[i]
Decoder path:  UpConv → Conv → ... (expanding)
Final:         1×1 Conv → Softmax/Sigmoid
```

Loss function combines binary cross-entropy with a Dice overlap term:

$$\mathcal{L} = -\frac{1}{N}\sum_i y_i \log \hat{y}_i + \left(1 - \frac{2\sum_i y_i \hat{y}_i}{\sum_i y_i + \sum_i \hat{y}_i}\right)$$

### nnU-Net [Isensee2021]

nnU-Net's self-configuration pipeline performs:
1. **Dataset fingerprinting**: median image size, spacing, intensity statistics, class frequencies.
2. **Architecture selection**: 2D, 3D full-resolution, or 3D low-resolution (cascade) based on patch-to-image ratio.
3. **Hyperparameter derivation**: patch size, batch size, pooling kernel sizes derived from fingerprint rules.
4. **Post-processing**: connected-component analysis applied if it improves single-fold performance on validation.

On the Medical Segmentation Decathlon, nnU-Net won first place in the original MICCAI 2018 challenge and maintained first rank on the open rolling leaderboard for nearly a year after its December 2019 submission.

### Attention U-Net and Swin-UNETR

Attention-gating [Oktay2018] introduces soft spatial attention maps to suppress irrelevant background activations. Swin-UNETR [Tang2022] replaces the CNN encoder with a hierarchical Swin Transformer, enabling global self-attention at multiple resolutions and achieving top performance on the BraTS 2021 challenge.

---

## Medical Image Registration

Deformable image registration—finding a dense displacement field $\phi$ that maps a moving image $I_M$ to a fixed image $I_F$—is critical for longitudinal monitoring and multi-modal fusion. Classical methods (ANTs, Elastix) are iterative and slow. **VoxelMorph** [Balakrishnan2019] reformulated registration as a learning problem by training a U-Net to predict $\phi$ end-to-end, achieving competitive accuracy at inference speeds orders of magnitude faster than ANTs:

$$\phi^* = \arg\min_\phi \mathcal{L}_{sim}(I_F, I_M \circ \phi) + \lambda \mathcal{L}_{reg}(\phi)$$

where $\mathcal{L}_{sim}$ is local normalized cross-correlation and $\mathcal{L}_{reg}$ penalizes displacement field smoothness via $\|\nabla \phi\|^2$.

---

## Key Challenges

### Data Scarcity and Label Efficiency

Expert annotation is expensive: a single abdominal CT segmentation takes ~1.5 hours per case [not publicly reported for all tasks]. Semi-supervised approaches (pseudo-labeling, mean-teacher, FixMatch adapted for medical imaging), and few-shot methods are active research areas. Federated learning enables multi-site training without centralizing protected health information.

### Class Imbalance

Small lesions (polyps, microaneurysms, pulmonary nodules) occupy <1% of image voxels, causing naive cross-entropy training to collapse to background prediction. Remedies include: Dice loss, focal loss [Lin2017], asymmetric losses, and hard-example mining. For the Pancreas-CT task in the Decathlon, the pancreas occupies ~0.5% of abdominal volume, making it among the hardest segmentation targets.

### Domain Shift

Scanner manufacturer, magnetic field strength (1.5T vs 3T), acquisition protocol, and site all introduce covariate shift. Adaptation strategies include: domain adversarial training [Ganin2015], instance normalization to remove scanner-specific style, and test-time adaptation with entropy minimization.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| U-Net: Convolutional Networks for Biomedical Image Segmentation | Ronneberger, Fischer, Brox | 2015 | MICCAI | Encoder–decoder with skip connections; foundational segmentation architecture |
| nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation | Isensee, Jaeger, Kohl et al. | 2021 | Nature Methods | Automated pipeline configuration; first place on 33/55 Decathlon/competition tasks |
| V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | Milletari, Navab, Ahmadi | 2016 | 3DV | 3D volumetric segmentation; Dice loss formulation |
| Attention U-Net: Learning Where to Look for the Pancreas | Oktay et al. | 2018 | MIDL | Soft attention gating in skip connections; pancreas segmentation |
| VoxelMorph: A Learning Framework for Deformable Medical Image Registration | Balakrishnan et al. | 2019 | IEEE TMI | Learned deformable registration; ~100× faster than classical methods |
| Swin-UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors | Tang et al. | 2022 | arXiv/MICCAI | Swin Transformer encoder in U-Net; state-of-art BraTS 2021 |
| CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays | Rajpurkar et al. | 2017 | arXiv | DenseNet-121 outperforms mean radiologist F1; multi-label chest X-ray |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| nnU-Net (3D) | BraTS 2021 — Whole Tumor | Dice | 0.9265 | Ensemble, top-performing team |
| nnU-Net (3D) | BraTS 2021 — Tumor Core | Dice | 0.8868 | Self-configured, no manual tuning |
| nnU-Net (3D) | BraTS 2021 — Enhancing Tumor | Dice | 0.8600 | Public validation split |
| Swin-UNETR | BraTS 2021 — Whole Tumor | Dice | 0.9328 | Pre-trained on 5,050 CT/MRI scans |
| nnU-Net | MSD Decathlon (avg, 10 tasks) | Dice | First rank | Open leaderboard; held first place Dec 2019 – Oct 2020 |
| CheXNet (DenseNet-121) | NIH ChestX-ray14 | F1 (pneumonia) | 0.435 | Exceeds mean radiologist F1 of 0.387 |
| VoxelMorph | ABIDE brain MRI registration | Dice (35 regions) | 0.729 | ~100× faster than ANTs |

---

## Pros & Cons

| Approach | Pros | Cons |
|----------|------|------|
| U-Net / nnU-Net | Proven baseline; self-configuring; strong on 3D volumetric data | Limited global context; scales poorly to gigapixel WSI |
| Transformer-based (Swin-UNETR) | Long-range dependencies; strong on multi-sequence MRI | High memory; requires large pretraining datasets |
| Self-supervised pretraining (SimCLR, MAE adapted) | Exploits large unlabeled datasets; label-efficient fine-tuning | Pretraining domain must match target; compute-intensive |
| Semi-supervised (pseudo-label, mean-teacher) | Reduces annotation burden | Confirmation bias; sensitive to hyperparameters |
| Domain adaptation (adversarial, normalization) | Improves cross-site generalization | Adversarial training instability; requires target-domain samples |

---

## Open Problems & Research Gaps

- **Annotation efficiency at scale**: despite advances in active learning and semi-supervision, large-scale 3D annotation (e.g., 1,000+ whole-body CT volumes with 100+ structures) remains prohibitively expensive, and no method closes the gap to full supervision under <10 labeled cases per rare structure.
- **Reliable uncertainty quantification**: clinical deployment demands calibrated uncertainty (epistemic + aleatoric) rather than point estimates; Bayesian deep learning and conformal prediction for medical segmentation remain underexplored at scale.
- **Causal generalization vs. spurious correlations**: models trained on data from Siemens scanners may exploit scanner-specific artifacts; disentangling clinically meaningful features from acquisition confounders is an open theoretical and practical challenge.
- **Multi-modal fusion across heterogeneous availability**: patients may have CT but not MRI; models that gracefully handle missing modalities at inference via learned imputation or modality-agnostic latent spaces are nascent.
- **Pathology at full resolution**: processing gigapixel WSI end-to-end (rather than patch-based with aggressive downsampling) at clinically relevant resolution remains computationally intractable without new hierarchical architectures or approximate attention mechanisms.
- **Regulatory and prospective validation**: very few AI-assisted diagnostic systems have undergone prospective randomized validation; there is a growing gap between benchmark performance and real-world clinical impact.
- **Continual learning in deployed systems**: as patient populations, scanners, and disease prevalence shift over time, deployed models require continual adaptation without catastrophic forgetting of previously learned anatomy.

---

## Further Reading

- [nnU-Net paper (Nature Methods 2021)](https://www.nature.com/articles/s41592-020-01008-z)
- [Medical Segmentation Decathlon (Nature Communications 2022)](https://www.nature.com/articles/s41467-022-30695-9)
- [BraTS 2023 challenge overview (arXiv)](https://arxiv.org/abs/2305.09011)
- [MedSAM: Segment Anything in Medical Images (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-44824-z)
- [MONAI framework documentation](https://monai.io/)
- [Grand Challenge benchmarking platform](https://grand-challenge.org/)
