# Structure from Motion and SLAM

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [3D Reconstruction (Core Tasks)](../02_core_tasks/09_3d_reconstruction.md) · [Gaussian Splatting](./02_gaussian_splatting.md) · [3D Representations](./00_3d_representations.md) · [4D and Dynamic Scenes](../12_research_frontier_2024_2026/05_4d_and_dynamic_scenes.md)

> **🆕 October 2026 update:**
> - New section on **feed-forward multi-view reconstruction**: VGGT (CVPR 2025 Best Paper) and its successors, namely π³ (ICLR 2026), MapAnything (3DV 2026), Depth Anything 3 (ICLR 2026), WorldMirror (ICML 2026) and VGGT-Ω (CVPR 2026 Oral). Also covers streaming/memory models (CUT3R, MUSt3R, StreamVGGT).
> - **Dynamic scenes**: the CVPR 2026 Best Paper D4RT (Google DeepMind/UCL/Oxford) jointly infers depth, correspondence and cameras from dynamic video.
> - New section on **foundation-model SLAM**: MASt3R-SLAM (CVPR 2025, real-time dense, calibration-free) and VGGT-SLAM (NeurIPS 2025, SL(4) submap alignment).
> - **Classical tooling**: GLOMAP global SfM (ECCV 2024) merged into **COLMAP 4.0** (2026), along with ALIKED + LightGlue support. COLMAP 4.1–4.2 add a GPU bundle-adjustment backend and LoMa learned features.
> - Key Papers, Benchmark, Pros & Cons, Open Problems and Further Reading extended.

---

## Overview

Structure from Motion (SfM) and Simultaneous Localization and Mapping (SLAM) are the foundational technologies for recovering 3D scene geometry and camera pose from visual input. SfM operates offline on unordered image collections, performing joint estimation of camera intrinsics, extrinsics, and sparse 3D structure via bundle adjustment. SLAM operates online (or pseudo-online) on video streams, maintaining a live map and localizing the camera in real time. Both fields have undergone profound transformation over the past decade — from purely classical handcrafted pipelines (SIFT, RANSAC, bundle adjustment) to deep-learning-augmented systems, and most recently to end-to-end learned approaches that subsume the entire pipeline within a neural network.

The classical SfM pipeline, epitomized by COLMAP [Schönberger2016], achieves high accuracy by carefully solving each sub-problem (feature detection, description, matching, geometric verification, incremental reconstruction, bundle adjustment) with hand-engineered algorithms optimized over decades. This robustness and accuracy remains the gold standard for offline 3D reconstruction pipelines used by NeRF and 3DGS as pre-processing. However, classical pipelines are slow (hours for large datasets) and brittle with low-texture, repetitive patterns, or wide baselines.

Deep learning has attacked this from multiple angles: learned feature detectors/descriptors (SuperPoint [DeTone2018], D2-Net), learned matching (SuperGlue [Sarlin2020, CVPR], LightGlue [Lindenberger2023, ICCV]), and end-to-end SLAM systems (DROID-SLAM [Teed2021, NeurIPS]). The most radical departure is DUSt3R [Wang2024, CVPR] and its successor MASt3R [Leroy2024, ECCV], which eliminate the traditional pipeline entirely — a transformer directly regresses dense 3D pointmaps from image pairs without requiring known camera parameters. In 2025–2026 this line scaled from pairs to arbitrary view sets. VGGT [Wang2025] won the CVPR 2025 Best Paper Award for predicting cameras, depth, pointmaps and tracks for up to hundreds of views in a single forward pass. A wave of successors followed, and SLAM systems (MASt3R-SLAM, VGGT-SLAM) are now built on these reconstruction priors. Classical SfM evolved in parallel: global SfM (GLOMAP) became a first-class mapper in COLMAP 4.

---

## Classical SfM: COLMAP

Johannes L. Schönberger and Jan-Michael Frahm (UNC Chapel Hill) published **"Structure-from-Motion Revisited"** at CVPR 2016, which is the foundational reference for COLMAP — the most widely used open-source SfM system. COLMAP provides an end-to-end pipeline from unordered images to calibrated camera poses and dense 3D reconstruction.

### Pipeline Overview

```mermaid
flowchart TD
    A[Unordered Image Collection] --> B[Feature Extraction\nSIFT or learned features]
    B --> C[Feature Matching\nExhaustive / Vocab Tree]
    C --> D[Geometric Verification\nTwo-view RANSAC + Essential Matrix]
    D --> E[Scene Graph Construction\nImages as nodes, matches as edges]
    E --> F[Incremental Reconstruction\n1. Find initial pair\n2. Register images one-by-one\n3. Triangulate new 3D points\n4. Bundle adjustment every N images]
    F --> G[Sparse Point Cloud\n+ Calibrated Camera Poses]
    G --> H[Multi-View Stereo\nPatched Match MVS]
    H --> I[Dense Point Cloud / Depth Maps]
```

### Bundle Adjustment

The core optimization in SfM is **bundle adjustment** (BA) — jointly minimizing the reprojection error of all observed 3D points across all cameras:

```
# Bundle Adjustment objective:
min_{P_j, X_i}  Σ_{(i,j) ∈ visibility}  ρ( || π(P_j, X_i) - x_{ij} ||^2 )

where:
  P_j ∈ SE(3)        — camera pose (rotation + translation) for image j
  X_i ∈ R^3         — 3D point position
  x_{ij} ∈ R^2      — observed 2D keypoint location
  π(P_j, X_i)       — projection function (camera model)
  ρ(·)              — robust loss (Cauchy/Huber to downweight outliers)

Solved via Levenberg-Marquardt with sparse linear solvers (Schur complement)
exploiting the block-sparse structure of the Jacobian.
```

COLMAP's incremental approach registers cameras one by one, triangulating new points and running local BA frequently. Global BA runs at the end. Loop closure is not built in (unlike SLAM systems).

### Global SfM: GLOMAP and COLMAP 4 [Pan2024, ECCV 2024]

Linfei Pan, Dániel Baráth, Marc Pollefeys and Johannes L. Schönberger revisited **global SfM** in *"Global Structure-from-Motion Revisited"* (ECCV 2024). GLOMAP estimates all camera rotations and positions simultaneously from the view graph. Its key change is a **global positioning** step that jointly optimizes camera centres and 3D points, replacing the fragile translation-averaging stage of earlier global pipelines; global BA follows. GLOMAP is reported to be 1–2 orders of magnitude faster than incremental COLMAP at on-par or better accuracy. In 2026 the standalone repository was archived after being fully merged into **COLMAP 4.0**. COLMAP 4.0 exposes incremental, hierarchical and **global** mappers and adds ALIKED features and LightGlue matching for SIFT/ALIKED. Later releases add a GPU-accelerated bundle-adjustment backend ("Caspar", 4.1), spherical camera models for 360° imagery, and LoMa learned features via ONNX (4.2).

### COLMAP MVS

After SfM, COLMAP's Multi-View Stereo (MVS) step uses **PatchMatch** applied in a depth-map fusion framework to produce dense reconstruction. Each reference image estimates a depth map by evaluating photometric consistency of small patches across neighboring views, then per-image depths are fused into a consistent point cloud.

---

## Feature Detection and Matching Evolution

### Classical: SIFT [Lowe2004]

SIFT (Scale-Invariant Feature Transform) has been the dominant handcrafted detector-descriptor for 20 years. It detects keypoints at scale-space extrema (DoG), computes 128-dim histogram-of-gradient descriptors, and achieves invariance to scale, rotation, and partial illumination changes.

### Learned: SuperPoint [DeTone2018]

Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich (Magic Leap) presented **SuperPoint** at the CVPR 2018 Deep Vision workshop. SuperPoint is a self-supervised homographic adaptation approach: a base **MagicPoint** detector is trained on synthetic geometrics, then domain-adapted to real images using random homographic warpings. The network jointly detects interest points and computes 256-dim descriptors in a single forward pass.

The training uses **homographic adaptation**: for each training image, apply many random homographies $H_1, \ldots, H_K$, detect keypoints in each warped image, warp detections back to original coordinates, and aggregate to create pseudo-ground-truth keypoint positions for self-supervised training.

### Learned Matching: SuperGlue [Sarlin2020, CVPR 2020]

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich (ETH Zürich + Magic Leap) introduced **SuperGlue** as a feature **matching** network at CVPR 2020 (Oral). Given keypoints and descriptors from two images, SuperGlue formulates matching as optimal transport, solved differentiably via the Sinkhorn algorithm. The keypoints are first embedded with their positions and descriptors, then processed by a **Graph Neural Network (GNN)** with cross-attention between images to exchange information about co-visible structure.

```
# SuperGlue matching pipeline:
1. Encode: positional encoding (position + score) → initial descriptor
2. GNN: L layers of self-attention (within one image) + cross-attention (between images)
3. Final MLP head → matching descriptor
4. Compute score matrix S_{ij} = <d_i, d_j>
5. Sinkhorn algorithm → soft assignment matrix P (dustbin for unmatchable points)
6. Mutual nearest neighbor: select matches from P
```

### Efficient Matching: LightGlue [Lindenberger2023, ICCV 2023]

Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys (ETH Zürich) published **LightGlue** at ICCV 2023. LightGlue is a drop-in replacement for SuperGlue with adaptive computation: it predicts per-point confidence and prunes both unmatchable points (early in the network) and network depth (stopping early for easy pairs). The result is 4–10× faster than SuperGlue on easy pairs (large visual overlap) while maintaining accuracy on hard pairs. LightGlue also offers improved training stability and lower memory usage.

---

## Classical SLAM: ORB-SLAM2 and ORB-SLAM3

### ORB-SLAM2 [Mur-Artal2017]

Raúl Mur-Artal and Juan D. Tardós (Universidad de Zaragoza) published **ORB-SLAM2** in IEEE Transactions on Robotics (TRO) 2017. It is the most widely cited classical SLAM system, supporting monocular, stereo, and RGB-D cameras with real-time loop closure and relocalization.

Architecture:
- **Tracking thread**: estimates camera pose from ORB features via PnP with RANSAC
- **Local mapping thread**: triangulates new map points, performs local BA
- **Loop closing thread**: detects loops via DBoW2 vocabulary tree, corrects accumulated drift with Sim(3) pose-graph optimization

### ORB-SLAM3 [Campos2021]

Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel, and Juan D. Tardós extended ORB-SLAM2 to ORB-SLAM3 (IEEE TRO, 2021, DOI: 10.1109/TRO.2021.3075644). Key additions:
- **Visual-Inertial (VI-SLAM)**: tightly-coupled IMU integration via Maximum-a-Posteriori estimation through the IMU initialization phase, rather than treating IMU as a separate system
- **Multi-map SLAM**: maintains an atlas of maps for recovering from tracking failures and multi-session mapping
- **Fisheye lens models**: supports wide field-of-view cameras common in robotics and automotive

---

## Deep Visual SLAM: DROID-SLAM [Teed2021, NeurIPS 2021]

Zachary Teed and Jia Deng (Princeton University) introduced **DROID-SLAM** at NeurIPS 2021 (Oral), receiving significant recognition. DROID-SLAM is a deep end-to-end SLAM system with a key architectural insight: a **Dense Bundle Adjustment (DBA) layer** — a differentiable version of bundle adjustment integrated into the deep network, enabling gradients to flow from the rendering loss back through the BA solver.

```
# DROID-SLAM pipeline:
Input: video frames {I_t}
1. Feature extraction: context + appearance networks (ResNet-based)
2. Correlation volume: dense L2 similarity between feature maps across frames
3. Update operator (GRU-based): 
   hidden state h_t → predicts per-pixel flow revisions (Δu) 
   and confidence weights (w_t)
4. Dense Bundle Adjustment layer:
   min_{G_t, D_t}  Σ_{(i,j) ∈ frame_pairs}  Σ_p  w_{ij,p} · ||u_{ij,p} - Π(G_i, G_j, D_i, p)||^2
   where G_t ∈ SE(3), D_t = inverse depth map
   Solved as a sparse least-squares system (Gauss-Newton)
5. Output: camera poses {G_t} + depth maps {D_t}
```

DROID-SLAM trained on monocular video generalizes to stereo and RGB-D at test time by modifying the measurement model. It achieves large improvements over classical methods (ORB-SLAM2/3) and prior deep SLAM (DeepV2D), suffering fewer catastrophic failures.

---

## Neural SLAM with Gaussian Representations

### SplaTAM [Keetha2024, CVPR 2024]

Nikhil Keetha, Jay Karhade et al. (Carnegie Mellon) introduced **SplaTAM** for dense RGB-D SLAM using 3D Gaussians as the map representation. Camera tracking minimizes the photometric + depth rendering loss w.r.t. camera pose (with Gaussian parameters fixed); mapping optimizes Gaussian parameters (with pose fixed). New Gaussians are added at unobserved pixels using the known depth. The 3DGS representation enables dense color map extraction and high-quality rendering, unlike the sparse maps of ORB-SLAM.

### MonoGS [Matsuki2024, CVPR 2024 Highlight + Best Demo Award]

Hidenobu Matsuki, Riku Murai et al. (Imperial College London) extended Gaussian SLAM to monocular cameras (no depth sensor) in **MonoGS**, requiring simultaneous tracking, Gaussian initialization from depth estimates, and isotropic-to-anisotropic Gaussian refinement. MonoGS received both a CVPR 2024 Highlight and the Best Demo Award.

---

## DUSt3R and MASt3R: End-to-End 3D Reconstruction

### DUSt3R [Wang2024, CVPR 2024]

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud (NAVER LABS Europe) published **DUSt3R** (Dense and Unconstrained Stereo 3D Reconstruction) at CVPR 2024. DUSt3R is a radically different approach — it formulates all of geometric 3D vision as a **pointmap regression** problem:

```
# DUSt3R formulation:
Input: two images I_1, I_2 (arbitrary, uncalibrated, no known poses)
Output: 
  X^{1→1} ∈ R^{H×W×3}  (pointmap of I_1 in frame of I_1)
  X^{2→1} ∈ R^{H×W×3}  (pointmap of I_2 in frame of I_1)
  conf_1, conf_2         (confidence maps)

Architecture: ViT encoder → cross-attention transformer → two DPT decoders

From pointmaps, recover:
  - Camera intrinsics (from pointmap geometry)
  - Relative pose (from comparing X^{1→1} and X^{2→1})
  - Depth maps (||X^{i→1}[u,v]||_2)
  - Dense correspondences (argmin ||X^{1→1}[u,v] - X^{2→1}[u',v']||)

Multi-image: connect pairwise graphs → global alignment via least-squares
```

DUSt3R requires no camera calibration or pose initialization — it works from any image pair in any order. It sets state-of-the-art on monocular and multi-view depth estimation and relative pose estimation benchmarks.

### MASt3R [Leroy2024, ECCV 2024]

Vincent Leroy, Yohann Cabon, and Jerome Revaud (NAVER LABS Europe) extended DUSt3R to **MASt3R** (Matching And Stereo 3D Reconstruction) at ECCV 2024. MASt3R adds a dense local feature matching head to DUSt3R, enabling pixel-aligned matching between images. This is achieved by adding an extra decoder head and a fast matching algorithm operating in the 3D pointmap space. MASt3R significantly outperforms DUSt3R on matching-centric tasks (map-free relocalization) while retaining reconstruction quality, and also introduces MASt3R-SfM — a complete SfM pipeline built on MASt3R pairwise predictions.

---

## Feed-Forward Multi-View Reconstruction (2025–2026)

DUSt3R/MASt3R are pairwise models, so scenes with many images still need a global alignment step. The 2025–2026 generation removes it: one network ingests *all* views jointly.

```mermaid
flowchart LR
    A["DUSt3R (CVPR 2024)<br/>pairwise pointmaps"] --> B["MASt3R (ECCV 2024)<br/>+ matching head"]
    A --> C["MUSt3R / CUT3R (CVPR 2025)<br/>multi-view memory / recurrent state"]
    A --> D["VGGT (CVPR 2025 Best Paper)<br/>N views, one forward pass"]
    D --> E["π³ (ICLR 2026)<br/>reference-free, permutation-equivariant"]
    D --> F["MapAnything (3DV 2026)<br/>depth + ray + pose + metric scale"]
    D --> G["Depth Anything 3 (ICLR 2026)<br/>plain ViT, depth-ray target"]
    D --> H["StreamVGGT (ICLR 2026)<br/>causal streaming"]
    D --> I["VGGT-Ω (CVPR 2026 Oral)"]
    D --> J["D4RT (CVPR 2026 Best Paper)<br/>dynamic 4D"]
    B --> K["MASt3R-SLAM (CVPR 2025)"]
    D --> L["VGGT-SLAM (NeurIPS 2025)"]
```

### VGGT: Visual Geometry Grounded Transformer [Wang2025, CVPR 2025 Best Paper]

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht (Oxford VGG) and David Novotny (Meta AI) introduced **VGGT**, a ~1B-parameter feed-forward transformer. Given one, a few, or hundreds of images, it directly predicts **camera intrinsics and extrinsics, depth maps, pointmaps and dense 3D point tracks**, typically in under one second.

```
# VGGT (schematic)
Input: images {I_1..I_N}, patchified by DINO → tokens; + camera token & register tokens per frame
Backbone: L blocks of Alternating-Attention
           frame-wise self-attention   (tokens within I_i)
           global self-attention       (tokens across all frames)
Heads:   camera head  → (q_i, t_i, f_i)      — rotation quaternion, translation, FoV
         DPT heads    → depth D_i, pointmap X_i (in frame-1 coords), confidences
         tracking head→ features for CoTracker-style point tracks
Loss: L = L_camera + L_depth + L_pmap + λ L_track   (multi-task; first frame = reference)
Optional: feed predictions to BA for further refinement
```

The authors found that the redundant targets (depth + camera vs. pointmap) are complementary, and that combining predicted depth with predicted cameras gives more accurate 3D points than the pointmap head alone. VGGT's predictions can be exported to COLMAP format, so it serves as a drop-in replacement for SfM before 3DGS training. Training code was released in July 2025. A separately licensed **VGGT-1B-Commercial** checkpoint permits commercial (non-military) use.

### Successors

- **π³** [Wang2025b, ICLR 2026] (Shanghai AI Lab, ZJU, SII) argues that VGGT's dependence on a fixed reference frame is an inductive bias that hurts robustness. It uses a **fully permutation-equivariant** architecture with no designated reference view and predicts affine-invariant camera poses plus scale-invariant local pointmaps. The authors report state-of-the-art results on pose, video depth and pointmap benchmarks. Pi3X (Dec 2025) adds conditional inputs and approximate metric scale.
- **MapAnything** [Keetha2026, 3DV 2026] (Meta Reality Labs + CMU) is a "universal" metric reconstructor. It accepts images plus *any* subset of intrinsics, poses and depth, and outputs factorized depth, local ray maps, poses and a global **metric scale factor**. One model covers 12+ tasks (uncalibrated SfM, calibrated MVS, depth completion, localization). Code is Apache-2.0, with both CC-BY-NC and Apache-licensed weights.
- **Depth Anything 3** [Lin2026, ICLR 2026] (ByteDance Seed) shows that a *plain* DINO-style ViT with a single **depth-ray** target suffices for any-view geometry, trained with teacher–student distillation on public academic data only. The authors report average gains over VGGT of 35.7% in pose accuracy and 23.6% in geometric accuracy on their new visual-geometry benchmark. DA3 also supports feed-forward 3DGS and long-video DA3-Streaming in <12 GB of GPU memory.
- **WorldMirror** [WorldMirror2025, ICML 2026] (Tencent Hunyuan) adds *any-prior prompting*: optional poses, intrinsics and depth are encoded as tokens. It outputs pointmaps, depth, cameras, surface normals and 3D Gaussians in one pass.
- **VGGT-Ω** [Wang2026, CVPR 2026 Oral] (Oxford VGG + Meta) is the direct successor of VGGT, released in May 2026 with 1B-parameter checkpoints, including a text-aligned variant. The official repo reports peak memory of ~6 GB for 1 frame to ~43 GB for 500 frames (A100, 624×416).

### Streaming and Memory-Based Variants

- **CUT3R** [Wang2025c, CVPR 2025] (Q. Wang, Zhang, Holynski, Efros, Kanazawa) is a recurrent model with a **persistent state** that is updated per frame and read out as metric pointmaps in a shared frame. It handles video or unordered photos, static or dynamic scenes, and can query unobserved virtual views.
- **MUSt3R** [Cabon2025, CVPR 2025] (NAVER LABS) is a symmetric multi-view extension of DUSt3R with a multi-layer memory for online pose and structure prediction.
- **StreamVGGT** [StreamVGGT2025, ICLR 2026] is a causal-attention VGGT with cached memory tokens for incremental, frame-by-frame 4D reconstruction.

### Dynamic Scenes: D4RT (CVPR 2026 Best Paper)

**"Efficiently Reconstructing Dynamic Scenes One D4RT at a Time"** (Google DeepMind, University College London, University of Oxford, including Andrew Zisserman) won the **CVPR 2026 Best Paper Award**, selected from ~16k submissions. A single transformer jointly infers depth, spatio-temporal correspondence and camera parameters from a single video of a dynamic scene. This unifies tasks that dynamic SfM systems such as MegaSaM (a CVPR 2025 Best Paper honorable mention) handled with optimization pipelines. As of the award, secondary sources reported that code and weights had not been publicly released.

---

## Foundation-Model SLAM (2025–2026)

### MASt3R-SLAM [Murai2025, CVPR 2025]

Riku Murai, Eric Dexheimer and Andrew J. Davison (Imperial College London) built a **real-time monocular dense SLAM** system from the ground up on the two-view MASt3R prior. Its components are:

1. **Pointmap matching**: efficient iterative projective matching of MASt3R pointmaps using ray-based errors.
2. **Tracking and local fusion**: running weighted averaging of pointmaps into keyframes.
3. **Graph construction and loop closure**: retrieval via MASt3R encoder features (ASMK).
4. **Second-order global optimization** over keyframe poses with a **generic central camera model**. Only a unique camera centre is assumed, so no parametric calibration is needed.

It runs at **~15 FPS** with globally consistent poses and dense geometry. With known calibration it reaches state-of-the-art accuracy on several benchmarks (TUM RGB-D, EuRoC, ETH3D-SLAM, 7-Scenes), and it works on in-the-wild video with unknown or changing intrinsics.

### VGGT-SLAM [Maggio2025, NeurIPS 2025]

Dominic Maggio, Hyungtae Lim and Luca Carlone (MIT SPARK) build submaps with VGGT and align them incrementally and globally. Their key observation is that, for **uncalibrated** monocular cameras, reconstructions are defined only up to a 15-DoF **projective** ambiguity, not a 7-DoF similarity. Submaps and loop closures are therefore aligned by homographies optimized on the **SL(4)** manifold:

```
# VGGT-SLAM factor graph (schematic)
Nodes:    H_k ∈ SL(4)  (4×4 projective transform of submap k, det = 1)
Factors:  relative H_{k,k+1} from overlapping frames;  loop closures H_{i,j}
min_{H}  Σ_{(i,j)} || Log( H_{ij}^{-1} · H_i^{-1} H_j ) ||^2_Ω
```

This allows long videos that are infeasible for VGGT alone due to GPU memory.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Structure-from-Motion Revisited (COLMAP) | Schönberger, Frahm | 2016 | CVPR | Incremental SfM with SIFT; gold standard offline pipeline |
| SuperPoint: Self-Supervised Interest Point Detection and Description | DeTone, Malisiewicz, Rabinovich | 2018 | CVPR DeepVision | Self-supervised keypoint detector via homographic adaptation |
| SuperGlue: Learning Feature Matching with Graph Neural Networks | Sarlin, DeTone, Malisiewicz, Rabinovich | 2020 | CVPR (Oral) | Attentional GNN + Sinkhorn optimal transport for feature matching |
| ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D | Mur-Artal, Tardós | 2017 | IEEE TRO | Classical SLAM with ORB features; loop closure; relocalization |
| ORB-SLAM3: Visual, Visual-Inertial, and Multimap SLAM | Campos, Elvira, Gómez, Montiel, Tardós | 2021 | IEEE TRO | Visual-inertial tightly coupled; multi-map atlas; fisheye |
| DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D | Teed, Deng | 2021 | NeurIPS (Oral) | Differentiable dense bundle adjustment; end-to-end trainable SLAM |
| LightGlue: Local Feature Matching at Light Speed | Lindenberger, Sarlin, Pollefeys | 2023 | ICCV | Adaptive depth/width pruning; 4–10× faster than SuperGlue |
| DUSt3R: Geometric 3D Vision Made Easy | Wang, Leroy, Cabon, Chidlovskii, Revaud | 2024 | CVPR | ViT pointmap regression; no known poses/intrinsics required |
| Grounding Image Matching in 3D with MASt3R | Leroy, Cabon, Revaud | 2024 | ECCV | DUSt3R + dense matching head; metric SfM without calibration |
| SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM | Keetha et al. | 2024 | CVPR | 3DGS as SLAM map; joint Gaussian optimization and camera tracking |
| Global Structure-from-Motion Revisited (GLOMAP) | Pan, Baráth, Pollefeys, Schönberger | 2024 | ECCV | Global positioning of cameras + points; 1–2 orders faster than COLMAP; merged into COLMAP 4 |
| VGGT: Visual Geometry Grounded Transformer | Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny | 2025 | CVPR (Best Paper) | Feed-forward cameras, depth, pointmaps and tracks for 1 to 100s of views in <1 s |
| MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors | Murai, Dexheimer, Davison | 2025 | CVPR | Calibration-free real-time (~15 FPS) dense monocular SLAM on MASt3R |
| Continuous 3D Perception Model with Persistent State (CUT3R) | Wang, Zhang, Holynski, Efros, Kanazawa | 2025 | CVPR | Recurrent persistent state; online metric pointmaps for static/dynamic scenes |
| MUSt3R: Multi-view Network for Stereo 3D Reconstruction | Cabon et al. (NAVER LABS) | 2025 | CVPR | Symmetric multi-view DUSt3R with multi-layer memory |
| VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold | Maggio, Lim, Carlone | 2025 | NeurIPS | Projective (15-DoF) submap alignment for uncalibrated VGGT SLAM |
| π³: Permutation-Equivariant Visual Geometry Learning | Wang, Zhou et al. | 2026 | ICLR | Reference-free, permutation-equivariant multi-view geometry |
| Depth Anything 3: Recovering the Visual Space from Any Views | Lin et al. (ByteDance Seed) | 2026 | ICLR | Plain ViT + depth-ray target; any-view depth, pose and 3DGS |
| MapAnything: Universal Feed-Forward Metric 3D Reconstruction | Keetha et al. (Meta, CMU) | 2026 | 3DV | Optional multimodal inputs; factorized metric outputs; 12+ tasks |
| Efficiently Reconstructing Dynamic Scenes One D4RT at a Time | Google DeepMind, UCL, Oxford | 2026 | CVPR (Best Paper) | Single transformer for depth, correspondence and cameras in dynamic video |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| COLMAP (SIFT) | ETH3D | F1 (sparse SfM) | ~0.62 | Classical; slow but accurate |
| COLMAP + SuperGlue | ETH3D | F1 | ~0.76 | +14 pts over SIFT |
| ORB-SLAM2 (Mono) | TUM RGB-D | ATE RMSE (m) | 0.016 | Monocular; scale ambiguity |
| ORB-SLAM3 (Mono+IMU) | EuRoC | ATE RMSE (m) | 0.007 | Visual-inertial; best classical |
| DROID-SLAM (Mono) | TUM RGB-D | ATE RMSE (m) | 0.010 | Deep; 2–5× better than classical on hard scenes |
| DUSt3R | ScanNet (pose est.) | RRA@15° | 91.3% | No known camera params |
| MASt3R | Map-free Reloc. | Median Pos. Error | ~0.14m | SOTA; outperforms SuperGlue+COLMAP |
| SplaTAM | TUM RGB-D | ATE RMSE (m) | 0.009 | Dense Gaussian map + trajectory |
| GLOMAP | Various SfM benchmarks | Runtime vs COLMAP | 10–100× faster | On-par or better accuracy (authors' claim) |
| VGGT (1B) | Co3D | Camera pose AUC@30 | ≈90 | Feed-forward, no BA; per official repo |
| Depth Anything 3 | Authors' visual-geometry benchmark | Pose / geometry accuracy | +35.7% / +23.6% vs VGGT (avg.) | Authors' claim |
| MASt3R-SLAM | TUM RGB-D, EuRoC, ETH3D-SLAM, 7-Scenes | Throughput | ~15 FPS | Real-time dense; SOTA with known calibration (authors' claim) |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Classical SfM (COLMAP) | High accuracy; robust via RANSAC; widely validated; modular pipeline | Slow (hours for large datasets); brittle on low-texture; requires feature overlap |
| Deep Matching (SuperGlue/LightGlue) | Handles wide baselines, repetitive textures, low light; state-of-the-art accuracy | Requires GPU; not interpretable; training requires ground-truth matches |
| End-to-End (DUSt3R/MASt3R) | No camera calibration needed; handles unconstrained in-the-wild images; single forward pass | Memory-intensive (ViT); slower per-pair than classical; global alignment still needed for many images |
| Feed-forward multi-view (VGGT, π³, MapAnything, DA3) | All views jointly in seconds; no global alignment; cameras + dense depth + tracks; good SfM/3DGS initializers | GPU memory grows with view count (~43 GB for 500 frames for VGGT-Ω); accuracy still benefits from BA refinement; restrictive licences on some weights |
| Foundation-model SLAM (MASt3R-SLAM, VGGT-SLAM) | Calibration-free; dense maps; robust on in-the-wild video | Below classical VIO speed; GPU required; drift/projective ambiguity must be handled explicitly |

---

## Open Problems & Research Gaps

- **Scalability of neural SfM:** DUSt3R/MASt3R process pairs efficiently but global alignment of thousands of images is still a computational bottleneck. Hierarchical or distributed approaches analogous to graph-based SfM are needed. VGGT-class models remove pairwise alignment, but global attention costs grow quadratically in total tokens. Streaming (StreamVGGT, CUT3R, DA3-Streaming) and submap approaches (VGGT-SLAM) trade global consistency for memory, and city-scale feed-forward reconstruction is unsolved.
- **Feed-forward vs. optimization:** Feed-forward reconstructors are now strong initializers, but BA refinement still improves accuracy. Whether learned models can match COLMAP/GLOMAP precision on large, high-resolution collections without test-time optimization is open, and so is a principled way to fuse the two (learned priors inside BA).
- **Dynamic and 4D reconstruction:** D4RT (CVPR 2026 Best Paper) shows that a single feed-forward model can recover depth, correspondence and cameras in dynamic video. Long-horizon, multi-object dynamics and reproducible open baselines remain gaps.
- **Long-term SLAM:** Classical ORB-SLAM3 handles multi-session mapping but struggles with perceptual aliasing (visually similar but geometrically different places). Deep place recognition (NetVLAD, SuperPoint-based) partially addresses this.
- **Dynamic environments:** All SfM/SLAM methods assume a static world. Handling moving objects (pedestrians, vehicles) requires explicit dynamic segmentation (Mask-SLAM, DynaSLAM) or probabilistic filtering, adding significant complexity.
- **Real-time 3DGS SLAM:** SplaTAM and MonoGS optimize Gaussians online but are slower than ORB-SLAM3. Achieving real-time Gaussian SLAM at ORB-SLAM3 speed with dense map quality is an important open problem. MASt3R-SLAM reaches ~15 FPS for dense pointmap SLAM, but has no photorealistic map and no IMU fusion.
- **Metric scale from monocular:** Monocular SfM and SLAM recover scene structure only up to scale. Without IMU or known object sizes, metric scale is unavailable. Depth priors from monocular depth estimation models (DPT, Depth Anything) offer a partial solution.
- **Adversarial conditions:** Rain, fog, night, and motion blur severely degrade all current methods. Robust all-weather SLAM for autonomous vehicles requires cross-modal fusion (LiDAR, radar) and learned robustness.
- **Privacy-preserving SfM:** Feature descriptors leak visual information. Cryptographic approaches to privacy-preserving localization (using quantized or hashed descriptors) are emerging but not mature.

---

## Further Reading

- [COLMAP Documentation](https://colmap.github.io/) — comprehensive user guide and API reference
- [SuperGlue Project Page (Magic Leap)](https://psarlin.com/superglue/) — paper, slides, and pretrained models
- [LightGlue GitHub (ETH CVG)](https://github.com/cvg/LightGlue) — code and benchmarks
- [DROID-SLAM GitHub (Princeton VL)](https://github.com/princeton-vl/DROID-SLAM) — reference implementation
- [DUSt3R GitHub (NAVER)](https://github.com/naver/dust3r) — code with demo
- [MASt3R GitHub (NAVER)](https://github.com/naver/mast3r) — including MASt3R-SfM pipeline
- [VGGT GitHub (Meta / Oxford VGG)](https://github.com/facebookresearch/vggt) — CVPR 2025 Best Paper; training code and commercial checkpoint
- [VGGT-Ω GitHub](https://github.com/facebookresearch/vggt-omega) — CVPR 2026 Oral successor
- [π³ GitHub](https://github.com/yyfz/Pi3) — permutation-equivariant geometry (ICLR 2026); Pi3X
- [MapAnything GitHub (Meta)](https://github.com/facebookresearch/map-anything) — universal metric reconstruction; unified wrappers for many baselines
- [Depth Anything 3 GitHub (ByteDance Seed)](https://github.com/ByteDance-Seed/Depth-Anything-3) — any-view depth/pose/3DGS
- [WorldMirror GitHub (Tencent Hunyuan)](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) — any-prior prompted reconstruction (ICML 2026)
- [MASt3R-SLAM GitHub](https://github.com/rmurai0610/MASt3R-SLAM) — CVPR 2025 real-time dense SLAM
- [VGGT-SLAM GitHub (MIT SPARK)](https://github.com/MIT-SPARK/VGGT-SLAM) — SL(4) submap SLAM
- [COLMAP Releases](https://github.com/colmap/colmap/releases) — 4.x with integrated GLOMAP global mapper
