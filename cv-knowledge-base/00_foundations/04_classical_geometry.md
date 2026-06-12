# Classical Multi-View Geometry

> **Last Updated:** June 2026
> **Level:** Foundational
> **Related Sections:**
> - [Image Formation](./01_image_formation.md)
> - [SfM and SLAM](../04_3d_vision_and_scene/03_sfm_and_slam.md)
> - [3D Reconstruction](../02_core_tasks/09_3d_reconstruction.md)
> - [Feature Engineering](./03_feature_engineering.md)

---

## Overview

Classical multi-view geometry is the mathematical foundation that relates 2D image observations to 3D scene structure and camera motion. It is the rigorous, model-based core on which learned 3D vision is built: even in the era of NeRF and Gaussian Splatting, the geometric relationships—epipolar constraints, homographies, triangulation, perspective-n-point—remain indispensable, both as the scaffolding of structure-from-motion and SLAM pipelines (see [SfM and SLAM](../04_3d_vision_and_scene/03_sfm_and_slam.md)) and as the inductive structure that learned systems must respect. A PhD student in 3D or robotics vision must command this material because it defines what is geometrically *possible* to recover from images and under what conditions reconstruction is well-posed.

The subject is organized around projective geometry: cameras are projective devices (see [Image Formation](./01_image_formation.md)), and the relationships between multiple views of a rigid scene are captured by a small set of algebraic objects—the **fundamental** and **essential matrices** (two-view geometry), the **homography** (planar scenes or pure rotation), and the machinery of **triangulation**, **PnP**, and **bundle adjustment** that turns noisy correspondences into metric 3D. Robust estimation (RANSAC) threads through all of it, because real correspondences (from SIFT/ORB or learned matchers; see [Feature Engineering](./03_feature_engineering.md)) contain outliers. This file develops these core results with their defining equations.

---

## Epipolar Geometry

Given two views of a 3D point, the **epipolar constraint** relates corresponding image points `x` and `x'` (homogeneous coordinates). The **fundamental matrix** `F` (uncalibrated) and **essential matrix** `E` (calibrated) encode this:

```
xᵀ F x' = 0            # fundamental matrix constraint (pixel coords)
x̂ᵀ E x̂' = 0           # essential matrix (normalized coords), x̂ = K⁻¹ x
E = [t]_× R            # essential = skew-translation × rotation
F = K⁻ᵀ E K'⁻¹         # relation via intrinsics K, K'
```

`F` (rank 2, 7 DoF) is estimated from ≥7 (or 8, the normalized 8-point algorithm) correspondences; `E` (5 DoF) needs ≥5 points and decomposes into relative rotation `R` and translation `t` (up to scale), giving camera motion. The **epipolar line** `Fx'` constrains the search for a match in the other image to 1D—the foundation of stereo matching.

## Homography

For points on a **plane**, or under **pure camera rotation**, two views are related by a `3×3` **homography** `H` (8 DoF):

```
x = H x'      # planar / pure-rotation mapping
```

Estimated from ≥4 correspondences (DLT). Homographies underpin image stitching/panoramas, planar AR, and rotation estimation.

## Triangulation, PnP, and Calibration

- **Triangulation**: given known camera matrices and a correspondence, recover the 3D point by intersecting back-projected rays (linear DLT, or optimal midpoint/least-squares).
- **Perspective-n-Point (PnP)**: given ≥3 (P3P) known 3D-2D correspondences, recover camera pose `(R, t)`—central to localization and AR.
- **Camera calibration** (Zhang's method [Zhang2000]): recover intrinsics `K` and distortion from multiple views of a planar checkerboard via homographies.

## Robust Estimation: RANSAC

Real correspondences contain outliers, so geometric models are fit with **RANSAC**: repeatedly sample a minimal set, fit the model, count inliers (points consistent within a threshold), and keep the best-supported hypothesis.

```mermaid
graph LR
    A[Feature matches<br/>SIFT/ORB/LightGlue] --> B[RANSAC<br/>minimal-sample + inlier count]
    B --> C[Estimate F/E/H]
    C --> D[Decompose → R, t]
    D --> E[Triangulate → 3D points]
    E --> F[Bundle adjustment refine]
    style B fill:#1d3557,color:#fff
    style F fill:#2d6a4f,color:#fff
```

## Bundle Adjustment

The gold-standard refinement minimizes **reprojection error** over all cameras and 3D points jointly:

```
min_{Cᵢ, Xⱼ}  Σᵢⱼ  ρ( || π(Cᵢ, Xⱼ) − xᵢⱼ ||² )
# Cᵢ: camera params, Xⱼ: 3D points, π: projection, ρ: robust loss
```

Solved with Levenberg-Marquardt exploiting sparsity—the computational heart of SfM/SLAM (see [SfM and SLAM](../04_3d_vision_and_scene/03_sfm_and_slam.md)).

---

## Key References

| Work | Authors | Year | Type | Key Contribution |
|------|---------|------|------|-----------------|
| Multiple View Geometry | Hartley, Zisserman | 2004 | Book (2nd ed.) | The definitive reference text |
| Normalized 8-point algorithm | Hartley | 1997 | TPAMI | Stable fundamental-matrix estimation |
| RANSAC | Fischler, Bolles | 1981 | CACM | Robust model fitting with outliers |
| Five-point algorithm | Nistér | 2004 | TPAMI | Minimal essential-matrix estimation |
| Camera calibration | Zhang | 2000 | TPAMI | Flexible planar calibration |

---

## Pros & Cons (classical vs. learned 3D)

| Aspect | Classical geometry | Learned 3D |
|--------|--------------------|-----------|
| Guarantees | Provable, interpretable | Data-dependent, opaque |
| Texture/feature reliance | Needs distinctive features | Can use priors in textureless regions |
| Generalization | Works on any rigid scene | Bounded by training distribution |
| Robustness | RANSAC handles outliers | May hallucinate plausible-but-wrong structure |

---

## Open Problems & Research Gaps

- **Textureless / repetitive scenes.** Correspondence fails where features are absent or ambiguous; learned matchers help but don't fully solve it.
- **Dynamic scenes.** Multi-view geometry assumes rigidity; non-rigid/dynamic reconstruction is much harder (links to [4D scenes](../12_research_frontier_2024_2026/05_4d_and_dynamic_scenes.md)).
- **Scale ambiguity.** Monocular reconstruction recovers structure only up to scale without extra constraints.
- **Robust-estimation efficiency.** RANSAC degrades with high outlier ratios; better samplers remain active research.
- **Classical–learned integration.** Optimally combining geometric constraints with learned priors (DUSt3R-style) is an open direction (see [3D Reconstruction](../02_core_tasks/09_3d_reconstruction.md)).
- **Global consistency.** Drift and loop closure in large-scale reconstruction remain challenging.

---

## Further Reading

- [Hartley & Zisserman, *Multiple View Geometry*](https://www.robots.ox.ac.uk/~vgg/hzbook/) — the canonical textbook
- [Zhang's calibration (TPAMI 2000)](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/) — flexible calibration
- [RANSAC (CACM 1981)](https://dl.acm.org/doi/10.1145/358669.358692) — robust fitting
- [COLMAP](https://colmap.github.io/) — practical SfM implementing this theory
- [Five-Point Algorithm (Nistér 2004)](https://ieeexplore.ieee.org/document/1288525) — minimal relative pose
