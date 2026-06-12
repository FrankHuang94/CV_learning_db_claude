# Feature Engineering: Classical Descriptors, Detectors, and Their Legacy

> **Last Updated:** June 2026
> **Level:** Intermediate
> **Related Sections:** [CNN Architectures](../03_architectures/00_cnn_architectures.md) · [Classical Geometry](./04_classical_geometry.md) · [Image Processing](./02_image_processing.md) · [CV Overview](./00_overview.md)

---

## Overview

Feature engineering in computer vision refers to the design of compact, discriminative representations of image content — from single pixel neighborhoods to global image statistics — that enable matching, recognition, and retrieval. Before the deep learning era, the quality of a CV system was almost entirely determined by the quality of its hand-crafted features; a better descriptor meant better performance, and enormous research effort went into designing features that were invariant to the nuisances present in real images: viewpoint change, illumination variation, scale change, partial occlusion, and image noise.

The canonical classical pipeline can be decomposed into two stages: **detection** (finding which image locations are worth describing) and **description** (characterising the local image appearance at those locations with a compact vector). Good detectors must be **repeatable** — the same physical location should be detected in multiple views of the scene. Good descriptors must be **distinctive** (two points with different appearances should have dissimilar descriptors) and **invariant** (the same physical point should have similar descriptors despite nuisance transformations). The tension between invariance and distinctiveness is fundamental: perfect invariance to all transformations would make the descriptor uninformative.

The progression from Harris corners (1988) through SIFT (2004) to ORB (2011) reflects an optimisation over this invariance-distinctiveness-efficiency tradeoff. SIFT was the gold standard for a decade — invariant to scale and rotation, robust to affine changes and illumination — but computationally expensive and patented (US Patent 6,711,293, expired). SURF traded descriptor quality for speed via Haar wavelets and integral images. ORB (2011) achieved similar matching quality to SIFT while being roughly 100× faster and patent-free, enabling real-time applications. Understanding why learned features (from CNNs, SuperPoint, DINO) eventually superseded all of these requires understanding not just what they do, but *why* the hand-crafted invariances are insufficient in practice.

---

## Interest Point Detection

### Harris Corner Detector

Harris & Stephens (Alvey Vision Conference, 1988) defined corners as locations where the image intensity varies significantly in *all* directions. The **structure tensor** (second-moment matrix) over a local window $W$ is:

```latex
\mathbf{M} = \sum_{(x,y) \in W} w(x,y)
\begin{pmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{pmatrix}
```

where $w(x,y)$ is a Gaussian weighting, and $I_x, I_y$ are spatial image gradients. The eigenvalues $\lambda_1, \lambda_2$ of $\mathbf{M}$ characterise the local geometry:
- Both small: flat region
- One large, one small: edge
- Both large: corner

The **Harris response function** avoids explicit eigenvalue computation:

```latex
R = \det(\mathbf{M}) - k \cdot \text{tr}^2(\mathbf{M}) = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2
```

with $k \in [0.04, 0.06]$ empirically. Corners are local maxima in $R$ above a threshold. Harris corners are **rotation-invariant** (structure tensor is symmetric under rotation) but **not scale-invariant** — a critical limitation.

The **Shi-Tomasi** detector (Shi & Tomasi, CVPR 1994) modifies the response to $R = \min(\lambda_1, \lambda_2)$, producing corners that are better conditioned for tracking.

### Scale Selection: LoG and DoG

Lindeberg (1998) showed that the normalized Laplacian of Gaussian (LoG) achieves maximum response at scale $\sigma$ matching the feature size:

```latex
\sigma^2 \nabla^2 G_\sigma * I = \sigma^2 (I_{xx} + I_{yy})
```

The SIFT detector approximates the LoG with a Difference of Gaussians (DoG):

```latex
\text{DoG}(x, y, \sigma) = G_{k\sigma} * I - G_\sigma * I \approx (k-1)\sigma^2 \nabla^2 G
```

Keypoints are detected as extrema (minima or maxima) in the DoG pyramid across both space and scale, yielding **scale-invariant** detections.

---

## SIFT: Scale-Invariant Feature Transform

Lowe (IJCV, 2004) is the canonical reference. The complete SIFT pipeline:

### 1. Scale-Space Extrema Detection

Construct a DoG pyramid with $S$ levels per octave (Lowe used $S=3$, $k = 2^{1/S}$). Each candidate keypoint is a local extremum in a $3 \times 3 \times 3$ neighbourhood (spatial + scale).

### 2. Keypoint Localisation and Filtering

Sub-pixel localisation via Taylor expansion of the DoG $D(\mathbf{x})$:

```latex
\hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}}
```

Keypoints with low contrast ($|D(\hat{\mathbf{x}})| < 0.03$) or lying on edges (Harris ratio $> 10$) are discarded.

### 3. Orientation Assignment

A dominant orientation is assigned by building a weighted gradient orientation histogram in the keypoint's neighbourhood. The canonical orientation aligns the descriptor with the gradient peak, conferring **rotation invariance**.

### 4. Descriptor Computation

A $16 \times 16$ pixel region around the keypoint (aligned to the dominant orientation, scaled by $\sigma$) is divided into a $4 \times 4$ grid of cells. Within each cell, an 8-bin gradient orientation histogram is computed (with magnitude weighting and trilinear interpolation). This produces a $4 \times 4 \times 8 = 128$-dimensional vector. The descriptor is normalised to unit $L_2$ norm, then components clipped at $0.2$, then renormalised — making it robust to nonlinear illumination changes.

### SIFT Properties

| Property | Status |
|----------|--------|
| Scale invariance | Yes (DoG pyramid) |
| Rotation invariance | Yes (dominant orientation) |
| Affine invariance | Approximately (within ≈ 50° tilt) |
| Illumination invariance | Yes (gradient normalisation) |
| Descriptor dimension | 128 floats |
| Computational cost | High ($O(N \log N)$ feature extraction) |

---

## SURF: Speeded-Up Robust Features

Bay, Tuytelaars & Van Gool (ECCV, 2006) accelerated SIFT-like descriptors using integral images and Haar wavelets:

- **Detector**: Hessian matrix determinant approximated with box filters (fast via integral images):

```latex
\det(\mathbf{H}_{\text{approx}}) = D_{xx} D_{yy} - (0.9 D_{xy})^2
```

- **Descriptor**: Haar wavelet responses in a $4 \times 4$ subregion grid, yielding a 64-dimensional vector (basic SURF) or 128-dimensional (extended). Roughly 3× faster than SIFT at publication.

Note: SURF is patented (US Patent 7,630,564) and was removed from OpenCV's main module in 2015 over licensing concerns.

---

## HOG: Histograms of Oriented Gradients

Dalal & Triggs (CVPR, 2005) introduced HOG for pedestrian detection, achieving state-of-the-art results on the INRIA pedestrian dataset. HOG represents local shape by the distribution of gradient orientations:

### Algorithm

```
1. Optionally apply Gamma/power-law normalisation (or square root)
2. Compute gradient magnitudes and orientations (no smoothing, or 1px Gaussian)
3. Partition image into cells (e.g., 8×8 pixels)
4. For each cell: build a weighted orientation histogram with B bins (9 or 18)
5. Group cells into overlapping blocks (e.g., 2×2 cells = 16×16 px)
6. L2-normalise each block descriptor (or L2-Hys)
7. Concatenate all block descriptors into the final HOG vector
```

For a $64 \times 128$ pedestrian window with 8×8 cells and 2×2 block overlap with 9 bins: $(7 \times 15) \times 4 \times 9 = 3780$-dimensional descriptor. SVM classifier over this descriptor achieved 89% detection rate at 10⁻⁴ FPPI on INRIA (2005), far above prior methods.

### Why HOG Works

HOG captures **local shape** information implicitly: gradients at object boundaries are the dominant signal, and their orientation distribution characterises the local part geometry regardless of exact magnitude. The block normalisation provides robustness to illumination gradients. Unlike SIFT, HOG is dense (computed over a regular grid) rather than sparse (only at detected keypoints), which is more appropriate for sliding-window detection.

---

## ORB: Oriented FAST and Rotated BRIEF

Rublee, Rabaud, Konolige & Bradski (ICCV, 2011) designed ORB as a free, fast alternative to SIFT and SURF:

**Keypoint detection** uses the FAST (Features from Accelerated Segment Test, Rosten & Drummond 2006) detector, extended with Harris scoring to select the top $N$ keypoints, and a scale pyramid for multi-scale detection.

**Orientation assignment** computes the **intensity centroid** of the patch:

```latex
m_{pq} = \sum_{x,y} x^p y^q I(x,y), \quad
\theta = \text{atan2}(m_{01}, m_{10})
```

**Descriptor**: Steered BRIEF (Binary Robust Independent Elementary Features) — a binary string of pairwise intensity comparisons, rotated according to $\theta$:

```latex
\tau(p; x, y) = \begin{cases} 1 & p(x) < p(y) \\ 0 & \text{otherwise} \end{cases}
```

A 256-bit binary descriptor is computed from 256 learned pairs. Matching is done via **Hamming distance** (XOR + popcount) — extremely fast on modern CPUs (1 instruction for 64 bits).

ORB achieves within ~1–2% of SIFT matching quality on standard benchmarks while being approximately 100× faster, and binary descriptors use 16–32× less memory than float descriptors.

---

## Bag of Visual Words (BoVW)

Sivic & Zisserman (ICCV, 2003, "Video Google") adapted information retrieval methods (TF-IDF, inverted file indexes) to visual descriptors:

### Pipeline

```
1. Extract local descriptors (SIFT) from a large training corpus
2. Cluster descriptors with k-means → vocabulary of k "visual words"
3. For each image: assign each descriptor to its nearest visual word (quantize)
4. Represent image as a histogram of visual word frequencies
5. Apply TF-IDF weighting: downweight frequent words, upweight rare ones
6. Index with inverted file for scalable retrieval
```

The TF-IDF weight for visual word $i$ in image $j$:

```latex
w_{ij} = \text{tf}_{ij} \cdot \text{idf}_i = \frac{n_{ij}}{n_j} \cdot \log\frac{N}{d_i}
```

where $n_{ij}$ = occurrences of word $i$ in image $j$, $n_j$ = total words in image $j$, $N$ = total images, $d_i$ = images containing word $i$.

BoVW scaled to databases of millions of images; the Fisher Vector (Perronnin & Dance, 2007) and VLAD (Jégou et al., 2010) extensions improved discriminability by encoding first- and second-order statistics relative to a GMM vocabulary.

---

## Why Learned Features Replaced Hand-Crafted Descriptors

The shift from SIFT/HOG to CNN features was not simply one of accuracy — it was structural:

1. **Domain-specific optimality**: SIFT's invariances (scale, rotation, affine) are engineered for generic texture-rich scenes. Deep features learn task-specific invariances from data — rotation invariance when needed, texture sensitivity when useful.

2. **Semantic abstraction**: CNN layers build progressively abstract representations — edges → textures → parts → objects. SIFT is stuck at the level of local gradient statistics; it cannot represent semantic concepts like "wheel" or "eye."

3. **End-to-end training**: A CNN trained for classification learns features optimised for the *task loss*, not a surrogate like repeatability or descriptor distance. This global optimisation is unavailable to hand-crafted features.

4. **Self-supervised large-scale pre-training**: DINO (Caron et al., 2021), DINOv2 (Oquab et al., 2023), and others learn universal features from unlabelled images that outperform SIFT on both semantic and geometric matching tasks.

5. **Robustness to extreme appearance change**: SIFT fails under heavy blur, large illumination changes, or non-rigid deformation. Learned features, when trained with appropriate augmentations, handle these gracefully.

However, classical features retain roles in: (a) systems with strict computational budgets (ORB is real-time on embedded devices); (b) domains with limited training data (satellite imagery, medical imaging); (c) interpretable pipelines where explicit geometric reasoning (via homographies or fundamental matrices) is required; and (d) hybrid systems like SuperPoint + SuperGlue which marry learned feature extraction with classical geometric verification.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| "Distinctive Image Features from Scale-Invariant Keypoints" | D. G. Lowe | 2004 | IJCV 60(2), 91–110 | SIFT: scale/rotation-invariant detector+descriptor; 128-d float vector |
| "Histograms of Oriented Gradients for Human Detection" | N. Dalal, B. Triggs | 2005 | CVPR | HOG dense descriptor + SVM; SOTA pedestrian detection |
| "A Combined Corner and Edge Detector" | C. Harris, M. Stephens | 1988 | Alvey Vision Conf. | Second-moment matrix corner detector |
| "SURF: Speeded Up Robust Features" | H. Bay, T. Tuytelaars, L. Van Gool | 2006 | ECCV | Fast SIFT-like features via integral images and Haar wavelets |
| "ORB: An Efficient Alternative to SIFT or SURF" | E. Rublee et al. | 2011 | ICCV | Binary descriptor, FAST+Harris detection, ~100× faster than SIFT |
| "Video Google: A Text Retrieval Approach to Object Matching in Videos" | J. Sivic, A. Zisserman | 2003 | ICCV | Bag of visual words; TF-IDF for image retrieval |
| "SuperPoint: Self-Supervised Interest Point Detection and Description" | D. DeTone, T. Malisiewicz, A. Rabinovich | 2018 | CVPR Workshops | Self-supervised learned keypoints competitive with SIFT |
| "SuperGlue: Learning Feature Matching with Graph Neural Networks" | P. Sarlin et al. | 2020 | CVPR | GNN-based descriptor matching; attention-based assignment |

---

## Benchmark Performance

Matching performance on HPatches benchmark (108 sequences, photometric + viewpoint changes), measured as Mean Matching Accuracy (MMA) at 3-pixel threshold:

| Method | MMA@3px (illumination) | MMA@3px (viewpoint) | Descriptor dim |
|--------|----------------------|-------------------|----------------|
| SIFT | ~65% | ~52% | 128 float |
| ORB | ~52% | ~38% | 256 bit |
| SuperPoint (2018) | ~73% | ~63% | 256 float |
| D2-Net (2019) | ~71% | ~67% | 512 float |
| SuperPoint + SuperGlue (2020) | ~88% | ~78% | 256 float + matching |

Numbers are approximate; exact values depend on detector parameters and evaluation protocol.

---

## Pros & Cons

| Descriptor | Pros | Cons |
|-----------|------|------|
| SIFT | Robust; well-studied; scale+rotation invariant | Slow; patent expired 2020; float storage |
| SURF | Faster than SIFT; good invariance | Patented (US 7,630,564); less accurate than SIFT |
| HOG | Dense; great for rigid object detection | Not keypoint-based; high dimensionality |
| ORB | Real-time; free; binary (compact + fast match) | Less invariant than SIFT; worse on viewpoint extremes |
| BoVW | Scalable retrieval; interpretable | Quantisation loss; sensitive to vocabulary size $k$ |
| Learned (SuperPoint) | Task-optimised; robust to extreme changes | Requires GPU; less interpretable; training data needed |

---

## Open Problems & Research Gaps

- **Cross-modal feature matching.** Matching features across image modalities (RGB–thermal, RGB–depth, RGB–SAR) requires learned representations that classical descriptors entirely lack.
- **Deformable/dynamic scene matching.** SIFT and ORB assume local rigidity; matching features in scenes with non-rigid deformation (cloth, tissue, faces) remains unsolved without task-specific training.
- **Descriptor learning with weak supervision.** Self-supervised descriptor learning (DINO, DINOv2) is powerful but requires large compute; few-shot or zero-shot transfer of descriptors to new domains is an open challenge.
- **Long-term place recognition.** For robot localisation, features must be consistent across day/night, seasons, and weather — a regime where both classical and basic learned descriptors fail without explicit appearance modelling.
- **Theoretical foundations of learned features.** Unlike SIFT, which has known invariance properties and failure modes, learned descriptors lack formal characterisation; understanding what invariances they capture and when they fail is a largely open theoretical question.
- **Efficiency at scale for learned matchers.** SuperGlue's attention mechanism scales quadratically in the number of keypoints; efficient approximate matching for large-vocabulary retrieval is an active engineering problem.
- **Unifying detection and description.** Detection and description are jointly trained in SuperPoint but still conceptually separate; fully unified detection-description-matching architectures (e.g., LoFTR, which abandons detection entirely) are an evolving frontier.

---

## Further Reading

- [Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV*, 60(2), 91–110.](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) — SIFT original paper.
- [Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) — HOG paper.
- [Rublee, E. et al. (2011). ORB: An Efficient Alternative to SIFT or SURF. *ICCV*.](https://doi.org/10.1109/ICCV.2011.6126544) — ORB paper.
- [Tuytelaars, T. & Mikolajczyk, K. (2008). Local Invariant Feature Detectors: A Survey. *Foundations and Trends in Computer Graphics and Vision*, 3(3).](https://doi.org/10.1561/0600000017) — Comprehensive survey of classical detectors.
- [Sarlin, P. E. et al. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. *CVPR*.](https://arxiv.org/abs/1911.11763) — State-of-the-art learned matching.
- [Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *TMLR*.](https://arxiv.org/abs/2304.07193) — Self-supervised foundation features.
