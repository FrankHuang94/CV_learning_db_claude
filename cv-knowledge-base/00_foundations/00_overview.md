# Computer Vision: Foundations and Overview

> **Last Updated:** October 2026
> **Level:** Foundational
> **Related Sections:** [Image Formation](./01_image_formation.md) · [CV Timeline](../01_history/00_timeline_overview.md) · [Feature Engineering](./03_feature_engineering.md) · [CNN Architectures](../03_architectures/00_cnn_architectures.md)

> **🆕 October 2026 update:**
> - The hand-crafted → learned narrative now continues into 2025–2026 foundation models: DINOv3 (Aug 2025) for self-supervised dense features, SAM 3 (Nov 2025; SAM 3.1 Mar 2026) for open-vocabulary "promptable concept segmentation", and feed-forward 3D geometry networks (VGGT, CVPR 2025 Best Paper; MapAnything; Depth Anything 3).
> - Key Papers table and the Levels-of-Analysis taxonomy gain DINOv3, SAM 3 and VGGT.
> - Marr's framework is revisited for feed-forward geometry (classical computational goals, learned algorithmic level), and a new open problem on guarantees for learned geometry is added.

---

## Overview

Computer vision is the scientific discipline concerned with enabling machines to interpret and understand visual information from the world — images, video, and other sensor modalities that encode spatial and photometric structure. At its core, CV solves an inherently ill-posed inverse problem: given a 2D projection (the image), recover properties of the underlying 3D scene — its geometry, material composition, lighting, and semantics. This inversion is ill-posed because infinitely many 3D configurations can produce the same image; resolving the ambiguity requires priors, constraints, and — in modern systems — learned statistical regularities extracted from massive data.

The scope of computer vision spans low-level signal processing (noise reduction, deblurring), mid-level scene analysis (edge detection, segmentation, depth estimation), and high-level semantic inference (object recognition, action understanding, visual question answering). It draws from projective geometry, linear algebra, probability theory, optimization, and — increasingly since the late 2000s — deep learning. The field intersects robotics, autonomous driving, medical imaging, remote sensing, and augmented reality, making it one of the most practically consequential branches of artificial intelligence.

Understanding CV rigorously requires grasping both the mathematical models that describe image formation and the statistical/computational machinery that inverts them. The distinction between these levels of description — *what* needs to be computed versus *how* it is computed — was famously codified by David Marr, and remains a useful conceptual scaffold even in an era dominated by end-to-end learned systems.

---

## The Vision Pipeline

Modern CV systems can be understood as a cascade of transformations from raw sensor data to high-level inference. Although deep networks blur many of these boundaries, the conceptual decomposition remains instructive:

```
World (3D scene)
      │  Illumination, geometry, materials
      ▼
Image Formation        ← optics, sensor, digitization
      │  I(x,y) : pixel intensities
      ▼
Pre-processing         ← noise removal, color correction, normalization
      │
      ▼
Feature Extraction     ← edges, corners, blobs, or learned embeddings
      │
      ▼
Mid-level Analysis     ← segmentation, optical flow, stereo, depth
      │
      ▼
High-level Inference   ← recognition, detection, 3D reconstruction, captioning
      │
      ▼
Task Output            ← bounding boxes, poses, captions, point clouds
```

Each stage introduces assumptions and approximations. The **image formation** stage (see [01_image_formation.md](./01_image_formation.md)) converts 3D radiance into a 2D discrete array via the camera's optical and electronic system. **Pre-processing** normalises the raw signal for downstream algorithms. **Feature extraction** historically involved hand-crafted operators (Sobel, SIFT, HOG); modern systems replace or augment these with convolutional or transformer-based learned features. **Mid-level analysis** recovers geometric and motion cues that inform **high-level inference**, where semantic labels, 3D structure, or language descriptions are produced.

---

## Marr's Three Levels of Analysis

The most influential theoretical framework for thinking about visual computation was articulated by David Marr in his posthumously published monograph *Vision* (MIT Press, 1982). Marr proposed that any information-processing system — biological or artificial — should be understood at three distinct and largely independent levels:

1. **Computational level** — *What* is the goal of the computation and *why* is it appropriate? This level asks about the function being computed, the problem being solved, and the constraints that make a solution possible. For stereopsis, the computational theory states: given two images of a scene from slightly different viewpoints, recover the depth map by exploiting binocular disparity. The logic of this computation follows from the geometry of projection and the assumption of scene rigidity.

2. **Algorithmic level** — *How* is the computation performed? What representations are used for input and output, and what procedure transforms one into the other? For stereopsis, the algorithm must specify a matching criterion (e.g., normalised cross-correlation, mutual information, or feature-based matching), a search strategy (e.g., constrained along epipolar lines), and a regularisation scheme for filling in occluded regions.

3. **Implementational level** — *How is the algorithm physically realised?* In biological vision this concerns neural circuits; in artificial systems it concerns hardware, memory layout, and numerical precision. A GPU-optimised block-matching implementation of stereo differs vastly in implementation from a neuromorphic spike-based approach, yet both can instantiate the same algorithm.

Marr's key insight — that these levels are *relatively independent* — implies that one can understand the computational goal without knowing the algorithm, and understand the algorithm without knowing the implementation. In practice, the levels interact: hardware constraints influence which algorithms are practical, which in turn influences how the problem is formulated. Nonetheless, the framework clarifies why seemingly different systems (human visual cortex, traditional CV pipelines, and deep networks) can solve the same perceptual problems while differing radically in their mechanisms.

---

## From Hand-Crafted to Learned Representations

For most of the field's history (roughly 1960–2010), CV pipelines were built from hand-engineered components. Researchers manually designed feature detectors (Canny, 1986; Harris & Stephens, 1988), descriptors (SIFT, Lowe 2004; HOG, Dalal & Triggs 2005), and classifiers (SVMs, boosted cascades). This approach required deep domain expertise and produced brittle systems that generalised poorly across domains.

The **deep learning revolution** — catalysed by AlexNet's decisive win on ImageNet in 2012 — shifted the paradigm to **learned representations**. Convolutional neural networks (CNNs) learn hierarchical features directly from labelled data, with low-level layers detecting edges and textures and higher layers encoding semantic concepts. This data-driven approach eliminated manual feature engineering and produced dramatic accuracy gains across nearly every CV benchmark.

The subsequent decade saw transformers (originally from NLP) enter CV via Vision Transformers (ViT, Dosovitskiy et al., 2020), enabling global self-attention over image patches and further performance improvements, particularly with large-scale pre-training. Contrastive and self-supervised methods (SimCLR, MoCo, DINO, MAE) then demonstrated that powerful visual representations can be learned without manual labels, and vision-language models (CLIP, Flamingo, GPT-4V) learned to align visual and textual semantics at scale.

**2025–2026: foundation-scale vision reaches every stage of the pipeline.** Three releases illustrate how far learned representations have now reached into territory once reserved for hand-designed modules:

- **Self-supervised dense features.** DINOv3 [Simeoni2025] scales DINO-style self-distillation to a 7B-parameter ViT trained on a curated 1.7B-image corpus (LVD-1689M), and introduces *Gram anchoring* — a regulariser that keeps patch-level feature similarity structure stable during long training — to stop the degradation of dense feature maps seen in earlier large-scale runs. The 7B teacher is distilled into ViT-S/B/L/H+ and ConvNeXt students, and a variant trained on satellite imagery (SAT-493M) is also released. These frozen features serve as general-purpose low/mid-level descriptors (see [Feature Engineering](./03_feature_engineering.md)).
- **Open-vocabulary segmentation.** SAM 3 [Carion2025] extends the promptable-segmentation paradigm from "segment *this* object" (points/boxes) to *Promptable Concept Segmentation*: given a short noun phrase or image exemplars, detect, segment and track *every* instance of the concept in images and video. It pairs a detector and a memory-based tracker on a shared backbone (≈848M parameters) and is evaluated on the new SA-Co benchmark (≈270K unique concepts). SAM 3.1 (March 2026) added faster joint multi-object tracking.
- **Feed-forward geometry.** VGGT [Wang2025] (CVPR 2025 Best Paper) predicts camera intrinsics/extrinsics, depth maps, point maps and point tracks for one to hundreds of views in a single forward pass (the authors report typical scenes reconstructed in under a second). Follow-ups such as MapAnything [Keetha2025] (metric-scale, with optional intrinsics/pose/depth inputs) and Depth Anything 3 (ByteDance Seed, Nov 2025; a single depth-ray representation on a plain DINO-style transformer) push the same idea. Notably, VGGT's outputs can still be exported to COLMAP and refined by bundle adjustment — the classical machinery of [Classical Geometry](./04_classical_geometry.md) becomes a refinement and verification layer rather than the whole pipeline.

In Marr's terms these systems keep the *computational* specification of classical vision (recover depth, pose, and object extent from images) but replace the hand-designed *algorithmic* level with a learned, amortised inference procedure.

Understanding the *classical* pipeline — hand-crafted features, explicit geometric models, optimization-based inference — remains essential for several reasons: (1) it provides interpretable, theoretically grounded models whose failure modes are understood; (2) classical geometry underpins modern 3D reconstruction, SLAM, and camera calibration pipelines; (3) many deployment-critical systems (autonomous driving sensors, medical devices) use hybrid pipelines; and (4) the classical methods often serve as targets or priors for learned systems. The shift from hand-crafted to learned is a change of *algorithmic level* in Marr's sense; the *computational* goals and *physical* constraints remain the same.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information* | D. Marr | 1982 | MIT Press (book) | Three-level framework for visual computation; primal sketch, 2.5D sketch, 3D model |
| "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet) | A. Krizhevsky, I. Sutskever, G. Hinton | 2012 | NeurIPS | Proved deep CNNs decisively outperform hand-crafted pipelines on large-scale recognition |
| "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (ViT) | A. Dosovitskiy et al. | 2020 | ICLR 2021 | Applied transformer self-attention directly to image patches; global context without convolutions |
| "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) | A. Radford et al. | 2021 | ICML | Contrastive vision-language pre-training enabling zero-shot visual recognition |
| "Masked Autoencoders Are Scalable Vision Learners" (MAE) | K. He et al. | 2022 | CVPR | Self-supervised ViT pre-training by reconstructing masked image patches |
| "Rich Feature Hierarchies for Accurate Object Detection" (R-CNN) | R. Girshick et al. | 2014 | CVPR | Bridged classical region proposals with CNN features; launched modern object detection |
| "Fully Convolutional Networks for Semantic Segmentation" | J. Long, E. Shelhamer, T. Darrell | 2015 | CVPR | End-to-end pixel-wise classification via fully convolutional architectures |
| "VGGT: Visual Geometry Grounded Transformer" | J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny | 2025 | CVPR (Best Paper) | Single feed-forward network predicting cameras, depth, point maps and tracks from many views |
| "DINOv3" | O. Siméoni et al. | 2025 | arXiv 2508.10104 | 7B self-supervised ViT with Gram anchoring; high-quality dense features; distilled model family |
| "SAM 3: Segment Anything with Concepts" | N. Carion et al. | 2025 | arXiv 2511.16719 | Promptable Concept Segmentation in images and video; SA-Co benchmark |

---

## Key Formulas: The General Inversion Problem

Computer vision is fundamentally the inversion of image formation. Let $\mathbf{x} \in \mathbb{R}^3$ be a 3D scene point, $\mathbf{I}$ the observed image, and $\mathcal{F}$ the forward imaging model:

```latex
\mathbf{I} = \mathcal{F}(\mathbf{x}, \boldsymbol{\theta}) + \boldsymbol{\eta}
```

where $\boldsymbol{\theta}$ encodes camera parameters, lighting, and material properties, and $\boldsymbol{\eta}$ is observation noise. The goal is to recover $\mathbf{x}$ (and possibly $\boldsymbol{\theta}$) from $\mathbf{I}$. This inverse problem is ill-posed (Hadamard, 1902): the solution may not exist, may not be unique, or may not depend continuously on the data. Regularization — encoding prior knowledge about the scene — is essential:

```latex
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \; \underbrace{\|\mathbf{I} - \mathcal{F}(\mathbf{x}, \boldsymbol{\theta})\|^2}_{\text{data fidelity}} + \lambda \underbrace{\mathcal{R}(\mathbf{x})}_{\text{regularizer}}
```

In a Bayesian formulation, the regularizer corresponds to a log-prior $-\log p(\mathbf{x})$, making maximum-a-posteriori (MAP) estimation equivalent to regularized least squares under Gaussian likelihood.

---

## Levels of Visual Analysis (Taxonomy)

| Level | Goal | Classic Operators | Learned Analogues |
|-------|------|-------------------|-------------------|
| Low-level | Signal cleaning, basic structure | Gaussian filter, Sobel, Canny | First conv layers |
| Mid-level | Geometric/structural grouping | Hough transform, RANSAC, optical flow | FPN, deformable convolutions |
| High-level | Semantic understanding | SVM + HoG, Bag-of-words | ResNet, ViT, DINO/DINOv3, SAM 3 |
| 3D / Scene | Geometry, depth, motion | Epipolar geometry, SfM | NeRF, 3DGS, DUSt3R, VGGT, MapAnything |

---

## Open Problems & Research Gaps

- **Robustness and distribution shift.** Deep models trained on curated benchmarks fail catastrophically on out-of-distribution inputs (weather changes, adversarial perturbations, sensor artifacts). Developing provably robust CV systems remains open.
- **Data efficiency and few-shot generalisation.** Biological vision achieves remarkable generalisation from very few examples; current learned systems require orders of magnitude more data.
- **Causal and compositional understanding.** Models excel at correlation-based pattern matching but struggle with causal reasoning — predicting what happens if an object is moved, removed, or occluded.
- **Unifying geometry and semantics.** Bridging classical geometric 3D reconstruction (SfM, SLAM) with semantic scene understanding in a principled, end-to-end framework remains an active research challenge.
- **Interpretability and failure-mode understanding.** Even state-of-the-art networks remain largely opaque; understanding *why* a model succeeds or fails is essential for safety-critical deployment.
- **Efficient inference at scale.** Foundation models for vision (SAM/SAM 3, CLIP, DINOv3-7B, GPT-4V-class VLMs) are computationally expensive; compressing them for edge deployment without capability loss is an unsolved engineering and algorithmic challenge. Distilled families (e.g., DINOv3's ViT-S to ViT-H+ students) are a partial answer.
- **Guarantees for learned geometry.** Feed-forward 3D models (VGGT, MapAnything, Depth Anything 3) now rival optimisation-based pipelines on many benchmarks, but offer no analogue of the identifiability and degeneracy results of multi-view geometry; knowing *when* their camera and depth predictions can be trusted, and how best to combine them with bundle adjustment, is open.
- **Embodied and active perception.** Most CV research addresses passive recognition; extending to *active* agents that control their sensors and move through environments to reduce uncertainty is nascent.

---

## Further Reading

- [Marr, D. (1982). *Vision*. MIT Press.](https://direct.mit.edu/books/monograph/3299/VisionA-Computational-Investigation-into-the-Human) — The canonical theoretical foundation.
- [Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press.](https://www.robots.ox.ac.uk/~vgg/hzbook/) — Definitive reference on geometric CV.
- [Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed. Springer.](https://szeliski.org/Book/) — Comprehensive modern textbook (freely available online).
- [LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.](https://www.nature.com/articles/nature14539) — Overview of the deep learning revolution.
- [Dosovitskiy, A. et al. (2021). An image is worth 16×16 words. *ICLR 2021*.](https://arxiv.org/abs/2010.11929) — Vision Transformers.
- [McClamrock, R. Marr's Three Levels: A Re-evaluation.](https://www.albany.edu/~ron/papers/marrlevl.html) — Critical philosophical analysis of Marr's framework.
- [Siméoni, O. et al. (2025). DINOv3. arXiv:2508.10104.](https://arxiv.org/abs/2508.10104) — Self-supervised foundation features with Gram anchoring; code and weights at [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
- [Carion, N. et al. (2025). SAM 3: Segment Anything with Concepts. arXiv:2511.16719.](https://arxiv.org/abs/2511.16719) — Promptable Concept Segmentation; code at [facebookresearch/sam3](https://github.com/facebookresearch/sam3).
- [Wang, J. et al. (2025). VGGT: Visual Geometry Grounded Transformer. *CVPR 2025*.](https://arxiv.org/abs/2503.11651) — Feed-forward multi-view geometry; CVPR 2025 Best Paper.
