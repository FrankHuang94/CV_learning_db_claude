# Neural Radiance Fields: NeRF and Variants

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [3D Representations](./00_3d_representations.md) · [Gaussian Splatting](./02_gaussian_splatting.md) · [SfM and SLAM](./03_sfm_and_slam.md)

---

## Overview

Neural Radiance Fields (NeRF), introduced by Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, and Ng at ECCV 2020, constitute one of the most transformative ideas in 3D computer vision of the 2020s. The core insight — that a continuous 5D function from position and viewing direction to color and density, encoded in an MLP and optimized via differentiable volume rendering from posed 2D images, can reconstruct photorealistic 3D scenes — opened an entirely new research paradigm. Within three years the original paper had generated hundreds of follow-up works addressing its key limitations: slow training and rendering, aliasing artifacts at different scales, poor performance on unbounded outdoor scenes, and inability to generalize across scenes.

The NeRF lineage follows a clear progression. The original NeRF (2020) demonstrated proof-of-concept photorealistic synthesis but required 1–5 days of training and seconds per rendered frame on a high-end GPU. Mip-NeRF (Barron et al., ICCV 2021) addressed aliasing by casting anti-aliased conical frustums instead of infinitesimally thin rays, halving error rates. Mip-NeRF 360 (Barron et al., CVPR 2022) extended this to unbounded outdoor scenes via scene contraction and a proposal-based sampler. Instant-NGP (Müller et al., SIGGRAPH 2022) achieved the critical engineering breakthrough of reducing training to ~5 minutes by replacing the MLP bottleneck with a multiresolution hash grid. Zip-NeRF (Barron et al., ICCV 2023) then combined the anti-aliasing of Mip-NeRF 360 with the speed of Instant-NGP.

Despite this rapid progress, the NeRF paradigm has been substantially superseded for real-time novel-view synthesis by 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), which achieves comparable or better quality at 100+ FPS versus NeRF's ~1 FPS. However, NeRF remains important for: (1) scenes requiring analytically smooth geometry (SDF-based NeRFs); (2) generative 3D models (DreamFusion, LRM); and (3) as a theoretical foundation for understanding neural scene representation.

---

## Volume Rendering Equation

The core of NeRF is the volume rendering integral, which computes the expected color $C(\mathbf{r})$ of a camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$:

```
# Volume Rendering Integral (continuous form)
C(r) = ∫_{t_n}^{t_f}  T(t) · σ(r(t)) · c(r(t), d) dt

where:
  T(t) = exp( -∫_{t_n}^{t} σ(r(s)) ds )   ← transmittance (prob. of no hit up to t)
  σ(r(t))                                  ← volume density at r(t)
  c(r(t), d)                               ← emitted radiance at r(t) in direction d

# Discrete approximation (quadrature, N samples):
C_hat(r) = Σ_{i=1}^{N}  T_i · (1 - exp(-σ_i · δ_i)) · c_i

where:
  T_i = exp( -Σ_{j=1}^{i-1} σ_j · δ_j )   ← accumulated transmittance
  δ_i = t_{i+1} - t_i                       ← sample interval width
  α_i = 1 - exp(-σ_i · δ_i)                ← alpha composite weight
```

This integral is differentiable w.r.t. the MLP parameters $\theta$ producing $(\sigma, c)$, allowing end-to-end training via:

```
L = Σ_{r ∈ R} || C_hat(r) - C_gt(r) ||_2^2
```

where $R$ is the set of training rays sampled from posed images.

---

## NeRF Architecture and Training

The original NeRF uses a **hierarchical sampling** strategy with a coarse and a fine MLP. The coarse network uses stratified sampling to produce a rough density estimate; the fine network then samples more densely in regions of high expected density using inverse CDF sampling. Both networks are trained jointly.

```mermaid
flowchart TD
    A[Input: Posed Images + Camera Intrinsics] --> B[Sample Camera Rays]
    B --> C[Stratified Sampling along Ray\n t_i ~ Uniform on bins]
    C --> D[Positional Encoding\n γ(x) = sin/cos at log frequencies]
    D --> E[Coarse MLP\nf_c: (x,d) → (σ, c)]
    E --> F[Coarse Volume Render\nC_coarse]
    E --> G[PDF Sampling\nSample more near surfaces]
    G --> H[Fine MLP\nf_f: (x,d) → (σ, c)]
    H --> I[Fine Volume Render\nC_fine]
    F --> J[Loss = ||C_coarse - C_gt||^2\n+ ||C_fine - C_gt||^2]
    I --> J
    J --> K[Backprop through\nDifferentiable Rendering]
```

**Positional Encoding:** NeRF encodes $(x,y,z)$ and $(\theta, \phi)$ using Fourier features to enable the MLP to represent high-frequency spatial detail:

```
# Positional Encoding
γ(p) = [sin(2^0 π p), cos(2^0 π p),
         sin(2^1 π p), cos(2^1 π p),
         ...
         sin(2^{L-1} π p), cos(2^{L-1} π p)]

where L=10 for position (x,y,z), L=4 for direction (d)
```

This encoding, inspired by work on spectral bias in neural networks [Rahaman et al. 2019], is critical — without it, the MLP converges to a blurry low-frequency solution.

---

## Mip-NeRF: Anti-Aliased Conical Frustums [Barron2021]

Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan (Google Research) published Mip-NeRF at ICCV 2021. The insight is that a ray is a zero-width abstraction — the actual imaged region is a **conical frustum** (a truncated cone segment) whose cross-section varies with distance. When training images span multiple scales or the same scene is viewed at different distances, a ray-based NeRF aliases because the same query point is asked to represent vastly different spatial extents.

Mip-NeRF approximates the frustum with an **integrated positional encoding (IPE)**: rather than encoding a point $\mu$, it encodes the Gaussian distribution $\mathcal{N}(\mu, \Sigma)$ characterizing the frustum, computing the expected value of the Fourier features analytically:

```
# Integrated Positional Encoding
E[γ(x)] for x ~ N(μ, Σ)

= [sin(2^k π μ) exp(-2^{2k-1} π^2 Σ_{jj}),
   cos(2^k π μ) exp(-2^{2k-1} π^2 Σ_{jj})]

← High-frequency components suppressed when variance is large
```

This produces a single-MLP model (removing the coarse/fine hierarchy) that is 7% faster and 17% lower error than NeRF on the original benchmark, and 60% lower error on a new multiscale evaluation set.

---

## Mip-NeRF 360: Unbounded Scenes [Barron2022]

Barron, Mildenhall, Verbin, Srinivasan, and Hedman (Google Research) extended anti-aliased NeRF to **unbounded outdoor scenes** at CVPR 2022. The key challenges for 360° scenes are: (a) content at vastly different distances (nearby foreground to infinity); (b) the camera can look in any direction; (c) sky and background with no clear depth.

Their solution combines three ingredients:
1. **Non-linear scene contraction**: map unbounded world coordinates to a bounded ball $\{x : ||x|| \leq 2\}$ using $\text{contract}(x) = x / ||x||$ for $||x|| > 1$
2. **Proposal-based sampling**: a small MLP produces a probability distribution over depth used to guide sampling, trained with a histogram loss rather than through volume rendering
3. **Distortion-based regularizer**: penalizes weight distributions that are spread over large intervals, discouraging the "floater" artifacts common in unbounded scenes

Mip-NeRF 360 reduces mean squared error by 57% vs. Mip-NeRF on the Mip-NeRF 360 dataset (introduced in the same paper, comprising 9 outdoor and indoor unbounded scenes).

---

## Instant-NGP: Hash Encoding Breakthrough [Müller2022]

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller (NVIDIA) published Instant Neural Graphics Primitives at SIGGRAPH 2022, winning the **Best Paper Award**. The fundamental bottleneck in NeRF is that the MLP must be large to encode all scene content, making it slow. Instant-NGP decouples encoding capacity from computation by replacing MLP layers with a **multiresolution hash table** of trainable feature vectors.

```
# Multiresolution Hash Encoding
For query point x ∈ R^3:
  For each resolution level l = 1..L:
    1. Compute voxel corners at resolution r_l = floor(N_min * b^l)
    2. Hash each corner: h(v) = (v_x XOR v_y*p_1 XOR v_z*p_2) mod T
       where T = hash table size, p_1,p_2 are large primes
    3. Look up feature vectors at hashed corners
    4. Trilinearly interpolate → feature f_l ∈ R^F
  Concatenate features: [f_1, ..., f_L] → tiny MLP → (σ, c)

Key: hash collisions are resolved implicitly by gradient descent —
colliding entries average their features, which works because
nearby spatial positions collide with similar features.
```

The result: NeRF training converges in **~5 minutes** (vs. ~12 hours for original NeRF) on a single NVIDIA RTX 3090. PSNR on NeRF-Synthetic improves from 31.01 to 33.18 dB. The system also handles signed distance functions, neural images, and neural volumes under a unified "Neural Graphics Primitives" framework.

---

## Zip-NeRF: Combining Mip-NeRF with Hash Grids [Barron2023]

Barron, Mildenhall, Verbin, Srinivasan, and Hedman (Google Research) published Zip-NeRF at ICCV 2023, addressing an incompatibility: Instant-NGP uses point sampling (aliased), while Mip-NeRF 360 uses integrated positional encoding (anti-aliased but slow). Zip-NeRF proposes **multisampling** — approximate the IPE over a conical frustum using a small number of point samples positioned along a circle within the frustum cross-section, then average their hash-grid lookups. This is combined with the Mip-NeRF 360 proposal network and distortion regularizer.

Results: 8–77% lower error than either Instant-NGP or Mip-NeRF 360 alone, with training time 24× faster than Mip-NeRF 360. Zip-NeRF represents the peak of pure NeRF-based novel view synthesis quality before 3DGS.

---

## Why 3DGS Superseded NeRF for Real-Time NVS

Despite Zip-NeRF achieving state-of-the-art quality, NeRF fundamentally cannot render faster than ~1–5 FPS because every ray requires hundreds of MLP queries. 3DGS (Kerbl et al. 2023) achieves comparable quality (PSNR within 0.5–1 dB) at 100+ FPS because:

1. **Explicit primitives**: 3D Gaussians are directly splatted to screen, requiring no ray marching
2. **GPU rasterization**: tile-based sorting and alpha compositing map perfectly to GPU shader pipelines
3. **Densification**: adaptive Gaussian splitting/pruning allows the representation to grow where needed, equivalent to NeRF's fine sampling but without the per-frame cost

NeRF retains advantages in: (1) memory efficiency for large scenes (MLP weights << millions of Gaussians); (2) cleaner geometry for meshing/physics; (3) as an architectural backbone for feed-forward 3D models (LRM uses NeRF decoder); (4) text-to-3D (SDS loss requires differentiable rendering that NeRF provides naturally).

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng | 2020 | ECCV | 5D radiance field + differentiable volume rendering; photorealistic NVS from posed images |
| Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields | Barron, Mildenhall, Tancik, Hedman, Martin-Brualla, Srinivasan | 2021 | ICCV | Conical frustum sampling + integrated positional encoding; 17–60% lower error |
| Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields | Barron, Mildenhall, Verbin, Srinivasan, Hedman | 2022 | CVPR | Scene contraction + proposal network + distortion regularizer for unbounded scenes |
| Instant Neural Graphics Primitives with a Multiresolution Hash Encoding | Müller, Evans, Schied, Keller | 2022 | SIGGRAPH | Hash-grid encoding; training in ~5 min; SIGGRAPH Best Paper |
| Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields | Barron, Mildenhall, Verbin, Srinivasan, Hedman | 2023 | ICCV | Multisampling to combine Mip-NeRF 360 with hash grids; 24× faster than Mip-NeRF 360 |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl, Kopanas, Leimkühler, Drettakis | 2023 | SIGGRAPH | Replaces NeRF with explicit Gaussians; real-time 100+ FPS at 1080p |
| Block-NeRF: Scalable Large Scene Neural View Synthesis | Tancik et al. (Waymo) | 2022 | CVPR | City-scale NeRF via independent block sub-networks with appearance conditioning |
| Deformable Neural Radiance Fields (Nerfies) | Park et al. | 2021 | ICCV | Deformable template NeRF for non-rigidly moving subjects (selfie videos) |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| NeRF | NeRF-Synthetic (Blender) | PSNR | 31.01 dB | Original, ~100k iterations, ~1 day training |
| Instant-NGP | NeRF-Synthetic | PSNR | 33.18 dB | ~5 min training; hash grid encoding |
| Mip-NeRF | NeRF-Synthetic | PSNR | 33.09 dB | Single MLP; anti-aliased frustums |
| Mip-NeRF 360 | Mip-NeRF 360 Dataset | PSNR | 27.69 dB | Unbounded indoor+outdoor scenes |
| Zip-NeRF | Mip-NeRF 360 Dataset | PSNR | 28.54 dB | Best NeRF-family result on this benchmark |
| 3DGS | Mip-NeRF 360 Dataset | PSNR | 27.21 dB | Comparable quality, 100+ FPS vs ~1 FPS |
| Mip-Splatting | Mip-NeRF 360 Dataset | PSNR | 27.92 dB | Anti-aliased 3DGS variant |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Original NeRF | Photorealistic; no explicit geometry needed; purely image-supervised | 12+ hours training; seconds/frame rendering; per-scene optimization |
| Hash-Grid NeRFs (Instant-NGP) | Fast training (~5 min); high quality; open-source CUDA implementation | Still ~1–10 FPS rendering; requires GPU; per-scene |
| Mip-NeRF family | Correct multi-scale anti-aliasing; handles zoom changes cleanly | Complex implementation; proposal network overhead; no real-time rendering |

---

## Open Problems & Research Gaps

- **Real-time NeRF:** The fundamental conflict between implicit MLP queries and real-time rendering has pushed the field to 3DGS. Baking NeRF to meshes (BakingNeRF) or voxels (PlenOctrees) offers partial solutions but loses the representation compactness.
- **Generalizable NeRF:** Per-scene optimization is impractical for deployment. Feed-forward models (pixelNeRF, MVSNeRF, LRM) require training on large 3D datasets, which remain scarce and expensive to acquire.
- **Dynamic NeRF:** Extending to dynamic scenes requires either per-frame optimization (expensive) or learning temporal consistency (hard). Deformable NeRF methods (Nerfies, D-NeRF, HyperNeRF) work on short clips but struggle with large motions.
- **Scene editing:** NeRF's implicit nature makes semantic editing difficult — changing one object requires modifying MLP weights globally. Compositional NeRF methods (ObjectNeRF, NSG) address this partially.
- **Outdoor/large-scale:** Mip-NeRF 360 handles single outdoor scenes; truly large (block, city) scale requires hierarchical architectures (Block-NeRF, Mega-NeRF) with engineering challenges around memory and consistency.
- **Geometry extraction:** Converting NeRF density to surfaces is ill-posed — density ≠ SDF, and naive marching cubes produces noisy meshes. SDF-based NeRF variants (NeuS, VolSDF) improve this but add complexity.
- **Photometric calibration:** NeRF assumes known intrinsics and extrinsics; handling unknown/imperfect camera models (lens distortion, rolling shutter, auto-exposure) requires careful pose refinement during training.

---

## Further Reading

- [NeRF Project Page](https://www.matthewtancik.com/nerf) — original paper, videos, and code
- [Instant-NGP GitHub (NVlabs)](https://github.com/NVlabs/instant-ngp) — CUDA implementation with GUI
- [NeRF Explosion 2020 Blog (Dellaert)](https://dellaert.github.io/NeRF/) — survey of early NeRF variants
- [Jon Barron's Homepage](https://jonbarron.info/mipnerf360/) — Mip-NeRF 360 project page with results
- [Zip-NeRF ICCV 2023 Talk (YouTube)](https://www.youtube.com/watch?v=Sk3wU-VMoCI) — author presentation
- [Neural Fields Survey (Xie et al. 2022, arXiv)](https://arxiv.org/abs/2111.11426) — comprehensive 100+ page survey
