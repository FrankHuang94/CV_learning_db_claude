# 3D Representations: Point Clouds, Voxels, Meshes, Implicit Surfaces, and Neural Fields

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [NeRF and Variants](./01_nerf_and_variants.md) · [Gaussian Splatting](./02_gaussian_splatting.md) · [SfM and SLAM](./03_sfm_and_slam.md) · [3D Generation](./04_3d_generation.md)

---

## Overview

The choice of 3D representation is the foundational architectural decision in any 3D vision system. Each representation makes fundamentally different trade-offs among memory footprint, geometric fidelity, rendering speed, editability, and learnability. Over the past decade, the field has moved from classical discrete representations (point clouds, voxels, meshes) toward continuous implicit functions (SDFs, occupancy fields) and, more recently, toward neural fields (NeRF) and explicit probabilistic primitives (3D Gaussian Splatting). Understanding these trade-offs at a rigorous level is essential for selecting the right backbone for tasks ranging from robotic manipulation to photorealistic content creation.

The taxonomy of 3D representations divides broadly into **explicit** and **implicit** paradigms. Explicit representations (point clouds, voxels, meshes, Gaussians) directly store geometric primitives in memory, affording fast rendering and easy manipulation but limited resolution-efficiency trade-offs. Implicit representations encode geometry as the level-set of a learned or analytic scalar field — signed distance functions, occupancy probabilities, or radiance fields — enabling theoretically unlimited resolution at the cost of expensive query operations. The boundary between these categories has blurred considerably with the advent of hybrid approaches such as Instant-NGP [MüllerEtAl2022], which embeds implicit MLP queries into explicit hash-grid lookups.

A crucial but often underappreciated dimension is **supervision signal compatibility**: point clouds arise naturally from LiDAR and stereo, voxels from CT/MRI, meshes from artist pipelines and MVS marching cubes, while implicit neural fields are trained directly from posed RGB images. This diversity of acquisition pathways means that real-world 3D vision pipelines frequently must transcode between representations — e.g., fitting a NeRF from images, then extracting a mesh for physics simulation, then converting to a point cloud for downstream learning.

---

## Point Clouds

A point cloud is the most primitive 3D representation: an unordered set $\mathcal{P} = \{p_i \in \mathbb{R}^d\}_{i=1}^N$ where each point carries at minimum $(x, y, z)$ coordinates and optionally color, normal, and semantic attributes. Point clouds are the direct output of LiDAR sensors and stereo/depth reconstruction algorithms and require no connectivity information.

### PointNet and PointNet++ [Qi2017a, Qi2017b]

The seminal breakthrough for deep learning on point clouds was PointNet [Qi2017a], published at CVPR 2017 by Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas (Stanford). PointNet addresses the core challenge of **permutation invariance**: a point cloud $\{p_1, \ldots, p_N\}$ represents the same shape regardless of the ordering of points, which standard convolutions cannot handle. PointNet's solution is architecturally elegant: apply a shared MLP pointwise to lift each point to a high-dimensional feature, then aggregate with a symmetric function (global max pooling) to obtain an order-invariant global descriptor.

```
# PointNet global feature extraction (pseudocode)
for p_i in point_cloud:
    f_i = MLP(p_i)           # shared weights, point-independent
global_feat = maxpool({f_i}) # symmetric aggregation
```

A critical regularizer is the **T-Net** (mini-PointNet producing a 3×3 then 64×64 alignment matrix), which learns to canonicalize the input before feature extraction. PointNet achieves strong classification and part-segmentation on ModelNet40 and ShapeNet Part respectively, but lacks a mechanism to capture local geometric structure.

PointNet++ [Qi2017b], published at NeurIPS 2017 by the same group, addresses this by recursively applying PointNet on nested local neighborhoods. The architecture introduces **farthest point sampling** (FPS) to select representative centroids, **ball query** to group neighbors within radius $r$, and a **multi-scale grouping** (MSG) strategy for non-uniform density. Formally, a set abstraction level maps $\{(x_i, f_i)\}$ to $\{(y_j, g_j)\}$ where $y_j$ are subsampled centroids and $g_j$ are features aggregated from local neighborhoods.

**Limitations of point clouds:** No intrinsic topology, making surface rendering expensive (e.g., requires splatting or meshing); storage scales linearly with scene complexity; fixed density creates artifacts near boundaries; and deep networks operating on points face $O(N^2)$ attention or $O(N \log N)$ nearest-neighbor costs at scale.

---

## Voxel Grids and Sparse Voxels

Voxel representations discretize 3D space into a regular grid of cubic cells, enabling straightforward application of 3D convolutions. A dense voxel grid at resolution $V^3$ requires $O(V^3)$ memory — $64^3 = 262,144$ cells, $256^3 = 16.7M$ cells — which becomes prohibitive for large scenes or high-resolution geometry.

### MinkowskiNet [Choy2019]

Choy, Gwak, and Savarese at CVPR 2019 introduced **4D Spatio-Temporal ConvNets (MinkowskiNet)**, which operationalizes **sparse convolutions** via the Minkowski Engine. Rather than allocating memory for all $V^3$ voxels, only occupied voxels are stored as a sparse tensor indexed by integer coordinates. Generalized sparse convolutions apply standard kernels only at occupied sites, achieving dramatic memory and compute savings for real-world 3D data (which is typically less than 1% occupied). The library handles automatic differentiation for backpropagation through sparse operations and was essential for enabling large-scale outdoor 3D semantic segmentation (e.g., on SemanticKITTI).

**VoxNet, Occupancy Grid, and TSDF:** Earlier works (Maturana & Scherer 2015, ICRA) operated on dense occupancy voxel grids for object detection. The **Truncated Signed Distance Function (TSDF)** — a voxel grid where each cell stores the truncated distance to the nearest surface — is the standard dense 3D map format in RGB-D SLAM (e.g., KinectFusion, InfiniTAM).

---

## Polygon Meshes

A triangle mesh $\mathcal{M} = (\mathcal{V}, \mathcal{F})$ with vertex set $\mathcal{V} \subset \mathbb{R}^3$ and face set $\mathcal{F} \subset \mathbb{N}^3$ is the dominant representation in computer graphics, animation, and physical simulation. Meshes are compact, GPU-rasterizable, and support UV texturing. However, mesh generation from raw sensor data or neural optimization is notoriously difficult due to topological constraints — differentiable mesh extraction typically requires knowing the topology in advance or using marching cubes post-hoc.

**Differentiable Rendering with Meshes:** PyTorch3D (Ravi et al. 2020) and SoftRas (Liu et al. 2019) introduced differentiable rasterizers that allow gradients to flow back to vertex positions from rendered images, enabling mesh-based inverse rendering. **NMR (Neural Mesh Renderer)** [Kato 2018] was an early approximation. MeshGPT [Siddiqui2024, CVPR 2024] revisits meshes from a generative perspective, training a decoder-only GPT-style transformer to autoregressively generate triangle sequences, directly producing compact, artist-quality meshes rather than density fields.

---

## Implicit Representations: SDF and Occupancy Networks

### DeepSDF [Park2019]

Park, Florence, Straub, Newcombe, and Lovegrove (Facebook Reality Labs) introduced **DeepSDF** at CVPR 2019. A signed distance function $f: \mathbb{R}^3 \to \mathbb{R}$ maps each query point $x$ to its signed distance to the nearest surface: negative inside, positive outside, zero on the surface. The surface is the zero level-set $\mathcal{S} = \{x : f(x) = 0\}$.

DeepSDF parameterizes $f_\theta(x, z)$ as a feed-forward MLP conditioned on a latent code $z \in \mathbb{R}^{256}$ via a **probabilistic auto-decoder**: latent codes and decoder weights are jointly optimized at training time; at test time, $z$ is inferred by gradient descent to minimize reconstruction loss on the observed surface. This auto-decoder design (no encoder needed at training) achieves better generalization than encoder-decoder architectures.

```
# SDF surface condition
f(x) = 0   ← surface
f(x) < 0   ← inside
f(x) > 0   ← outside

# Eikonal constraint for valid SDF:
||∇_x f(x)|| = 1   (almost everywhere)
```

### Occupancy Networks [Mescheder2019]

Mescheder, Oechsle, Niemeyer, Nowozin, and Geiger (Max Planck / University of Tübingen) presented **Occupancy Networks** at CVPR 2019 concurrently with DeepSDF. Rather than signed distance, they predict a binary occupancy probability $o_\theta(x, z) \in [0, 1]$ — whether point $x$ is inside the shape given latent $z$. The surface is the $0.5$ iso-surface. A 5-block ResNet processes $(x, z)$ jointly. Surfaces are extracted at inference via multiresolution MISE (Marching Cubes with adaptive subdivision).

**Key difference from DeepSDF:** Occupancy is easier to supervise (binary labels) and naturally handles non-watertight surfaces, but DeepSDF's signed distance provides richer geometric information (gradients give surface normals directly).

---

## Neural Radiance Fields (NeRF)

NeRF [Mildenhall2020, ECCV 2020] redefines scene representation as a continuous volumetric function mapping 5D input $(x, y, z, \theta, \phi)$ — 3D position and 2D viewing direction — to volume density $\sigma$ and view-dependent color $c = (r, g, b)$. Novel views are synthesized via differentiable volume rendering (see [01_nerf_and_variants.md](./01_nerf_and_variants.md) for the full rendering equation). NeRF's key properties are: (1) implicit representation requiring no explicit geometry; (2) view-dependent appearance from the directional input; (3) training purely from posed RGB images via photometric loss; (4) continuity enabling arbitrary-resolution rendering.

The main weaknesses are slow training (hours to days per scene) and slow rendering (seconds per frame), motivating the acceleration line Instant-NGP → Zip-NeRF → and eventually 3DGS which replaces the implicit field entirely with explicit Gaussian primitives.

---

## 3D Gaussian Splatting (3DGS)

Kerbl, Kopanas, Leimkühler, and Drettakis (INRIA) introduced **3D Gaussian Splatting** at SIGGRAPH 2023, representing a scene as a set of anisotropic 3D Gaussians — each defined by a mean $\mu \in \mathbb{R}^3$, a covariance matrix $\Sigma \in \mathbb{R}^{3 \times 3}$ (parameterized via rotation $R$ and scale $S$ for positive semi-definiteness), opacity $\alpha$, and spherical harmonic coefficients for view-dependent color. Rendering proceeds via tile-based differentiable rasterization (splatting) rather than volume marching, achieving 100+ FPS at 1080p. See [02_gaussian_splatting.md](./02_gaussian_splatting.md) for full details.

---

## Representation Trade-off Summary

```
Representation | Memory     | Rendering Speed | Geometry Quality | Learnability | Editability
---------------------------------------------------------------------------
Point Cloud    | O(N)       | Medium (splat)  | Medium           | High         | High
Voxel (dense)  | O(V^3)     | Fast (raster)   | Medium           | High         | Medium
Voxel (sparse) | O(N_occ)   | Fast            | Medium-High      | High         | Medium
Mesh           | O(V+F)     | Very Fast (GPU) | High             | Medium       | Very High
DeepSDF/OccNet | O(θ)       | Slow (query)    | Very High        | High         | Low
NeRF           | O(θ)       | Very Slow       | High             | High         | Low
3DGS           | O(N_gauss) | Very Fast       | High             | Medium       | Medium
```

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | Qi, Su, Mo, Guibas | 2017 | CVPR | First permutation-invariant deep network on raw point clouds via shared MLP + global max pooling |
| PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space | Qi, Yi, Su, Guibas | 2017 | NeurIPS | Hierarchical feature learning with farthest-point sampling and ball-query grouping |
| 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks | Choy, Gwak, Savarese | 2019 | CVPR | Sparse 3D/4D convolutions via Minkowski Engine enabling large-scale voxel learning |
| DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation | Park, Florence, Straub, Newcombe, Lovegrove | 2019 | CVPR | Auto-decoder for class-conditional continuous SDF; latent shape interpolation |
| Occupancy Networks: Learning 3D Reconstruction in Function Space | Mescheder, Oechsle, Niemeyer, Nowozin, Geiger | 2019 | CVPR | Implicit occupancy field; MISE-based mesh extraction at arbitrary resolution |
| NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng | 2020 | ECCV | 5D radiance field + differentiable volume rendering for photorealistic novel view synthesis |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl, Kopanas, Leimkühler, Drettakis | 2023 | SIGGRAPH | Anisotropic 3D Gaussians + tile rasterization; real-time novel view synthesis at 100+ FPS |
| MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers | Siddiqui, Alliegro, Artemov et al. | 2024 | CVPR | Autoregressive triangle mesh generation via GPT-style transformer on learned mesh vocabulary |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| PointNet++ (MSG) | ModelNet40 | Classification Acc. | 91.9% | Outperforms PointNet (89.2%) |
| MinkowskiNet | SemanticKITTI | mIoU | 65.4% | Sparse 3D conv, 21 classes |
| DeepSDF | ShapeNet Cars | Chamfer-L1 | — | Better interpolation than AtlasNet |
| OccNet | ShapeNet | IoU (3D) | 0.761 | vs voxel baselines ~0.66 |
| NeRF | NeRF-Synthetic | PSNR | 31.01 dB | 8 scenes, hours training per scene |
| Instant-NGP | NeRF-Synthetic | PSNR | 33.18 dB | ~5 min training; SIGGRAPH Best Paper 2022 |
| 3DGS | Tanks & Temples | PSNR | 23.14 dB | Real-time rendering, 1080p |
| Mip-Splatting | Mip-NeRF 360 | PSNR | 27.92 dB | CVPR 2024 Best Student Paper |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Point Clouds | Direct LiDAR/stereo output; permutation-invariant learning (PointNet++); easy streaming | No connectivity; rendering requires splatting or reconstruction; density varies |
| Implicit Neural (SDF/OccNet) | Unlimited resolution; smooth geometry; compact (MLP weights); analytically differentiable | Slow per-point queries; no direct rendering; meshing step loses continuity |
| 3D Gaussian Splatting | Real-time rendering; fast optimization; high PSNR; easy densification/pruning | No surface normals by default; large memory per scene; floater artifacts; limited topology |

---

## Open Problems & Research Gaps

- **Representation unification:** No single representation dominates all metrics simultaneously. Active research investigates hybrid representations (e.g., Gaussian primitives anchored to mesh surfaces) that combine rendering speed with geometric precision.
- **Scaling to city-scale:** Current methods (NeRF, 3DGS) scale poorly beyond bounded scenes. City-scale neural representations require hierarchical or streaming designs — an unsolved engineering and algorithmic challenge.
- **Dynamic scene representation:** Extending static representations to 4D (space-time) without exponential memory growth remains open. Current 4D-GS [Wu2023] and SC-GS [Huang2024] work on short monocular clips but struggle with long-horizon, multi-object dynamics.
- **Physics-aware representations:** Representations that jointly encode geometry and physical properties (material, deformability) for simulation are nascent. Neural SDFs can be coupled with elasticity solvers but the training pipeline is complex.
- **Generalization across scenes:** Current per-scene fitting methods (NeRF, 3DGS) do not generalize — a new optimization is needed per scene. Feed-forward generalizable models (LRM [Hong2023], TripoSR [Tochilkin2024]) are emerging but sacrifice per-scene quality.
- **Semantic and editable representations:** Adding semantic fields (LangSplat [Qin2023]) or edit-friendly structure (Scaffold-GS [Lu2023]) to neural representations without sacrificing rendering quality is an active area.
- **Topology changes and non-rigid:** Implicit representations and Gaussians both struggle with large topological changes (e.g., a object being torn). Neural deformable representations that handle topology change are an open problem.

---

## Further Reading

- [NeRF Project Page (Mildenhall et al.)](https://www.matthewtancik.com/nerf) — original NeRF with videos and pretrained models
- [3D Gaussian Splatting Official Repo (Kerbl et al.)](https://github.com/graphdeco-inria/gaussian-splatting) — reference implementation
- [Neural Fields in Visual Computing (Xie et al. 2022)](https://arxiv.org/abs/2111.11426) — comprehensive survey of implicit neural representations
- [PointNet Project Page (Qi et al.)](https://stanford.edu/~rqi/pointnet/) — Stanford project page with code and data
- [Occupancy Networks Blog (Autonomous Vision Group)](https://autonomousvision.github.io/occupancy-networks/) — accessible overview
- [Open3D Documentation](http://www.open3d.org/docs/release/) — standard library for point cloud and mesh processing
