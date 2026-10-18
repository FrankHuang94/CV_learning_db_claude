# 3D Content Generation: Text-to-3D and Image-to-3D

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [3D Generation (Generative Vision)](../10_generative_vision/03_3d_generation.md) · [Gaussian Splatting](./02_gaussian_splatting.md) · [3D Representations](./00_3d_representations.md) · [Spatial Intelligence](../12_research_frontier_2024_2026/06_spatial_intelligence.md)

> **🆕 October 2026 update:**
> - New section on **native 3D latent generative models**, now the dominant paradigm. It covers TRELLIS (CVPR 2025 Spotlight, SLAT latents) and **TRELLIS.2** ("Native and Compact Structured Latents for 3D Generation", **CVPR 2026 Best Student Paper**; O-Voxel, 4B flow transformers, up to 1536³ with PBR materials).
> - **Hunyuan3D** lineage: 2.0 (Jan 2025), 2.1 (Jun 2025, first fully open-source PBR pipeline), 2.5, Omni (Sep 2025, controllable), and 3.0 (Sep 2025, 1536³, served via Tencent Cloud). Also covers TripoSG and Direct3D-S2 (NeurIPS 2025).
> - **SAM 3D** (Meta, Nov 2025): SAM 3D Objects (shape, texture and layout from a single image, robust to occlusion) and SAM 3D Body (promptable human mesh recovery with the Momentum Human Rig).
> - From objects to worlds: HunyuanWorld 1.0 and World Labs' Marble/Spark. Updated pipeline diagram, Key Papers, Benchmark, Pros & Cons and Open Problems.

---

## Overview

3D content generation — creating 3D assets from text descriptions or single images — is one of the most commercially significant open problems in computer vision and computer graphics. The field bifurcated into two main paradigms: **optimization-based approaches** that use 2D diffusion priors to supervise 3D optimization (DreamFusion, Magic3D), and **feed-forward approaches** that train large transformers or diffusion models directly on 3D data to predict 3D structure in a single forward pass (LRM, TripoSR, Shap-E). The latter now dominates practical applications due to its speed (seconds vs. hours), though it requires large-scale 3D training data.

DreamFusion [Poole2022] established the SDS (Score Distillation Sampling) loss as the mechanism for distilling 2D diffusion knowledge into 3D representations — a NeRF is optimized so that renders from all viewpoints look like plausible outputs of a pre-trained text-to-image diffusion model. This was a conceptual breakthrough enabling text-to-3D without any 3D supervision, but it suffers from the notorious **Janus problem** (multi-face artifacts), over-saturation, and slow optimization (1.5–3 hours per object). Magic3D [Lin2023] addressed quality through a two-stage coarse-to-fine approach; DreamGaussian [Tang2023] replaced NeRF with 3D Gaussians to achieve ~2 minute generation.

Feed-forward approaches emerged in 2023 with Zero-1-to-3 [Liu2023] (viewpoint-conditioned novel view synthesis from a single image via fine-tuned diffusion), LRM [Hong2023] (large transformer predicting NeRF directly from a single image in 5 seconds), and TripoSR [Tochilkin2024] (open-source LRM with improved data and training). Shap-E and Point-E (OpenAI) explored direct generation of implicit functions and point clouds respectively. MeshGPT [Siddiqui2024] closed the loop with autoregressive native mesh generation.

From 2025, a third paradigm became dominant: **native 3D latent generation**. A 3D VAE compresses assets into a structured (usually sparse-voxel or vector-set) latent, and a large diffusion/flow transformer is trained directly in that latent space, conditioned on an image or text. TRELLIS [Xiang2025], Hunyuan3D 2.x [Tencent2025], TripoSG [Li2025] and Direct3D-S2 [Wu2025b] established the recipe. TRELLIS.2 [Xiang2026] won the **CVPR 2026 Best Student Paper Award** by extending it to arbitrary topology and PBR materials at up to 1536³. Meta's SAM 3D [SAM3DTeam2025] moved the problem from clean object images to *in-the-wild* scenes with occlusion and layout.

---

## Score Distillation Sampling (SDS)

DreamFusion's core contribution is the **SDS loss** — a method to use a frozen 2D diffusion model $\epsilon_\phi$ as a 3D optimization oracle:

```
# Score Distillation Sampling (SDS) Loss
Given:
  θ         — 3D representation parameters (NeRF weights)
  g(θ, c)   — differentiable renderer (produces image x for camera c)
  ε_φ(x_t, t, y) — frozen pretrained diffusion model, text y
  t         — diffusion timestep ~ Uniform(t_min, t_max)
  ε         — Gaussian noise

Algorithm:
  1. Render image: x = g(θ, c)  for random camera pose c
  2. Add noise: x_t = √ᾱ_t · x + √(1-ᾱ_t) · ε
  3. Predict noise: ε̂_φ = ε_φ(x_t, t, y)   (no grad w.r.t. φ)
  4. SDS gradient w.r.t. θ:
     ∇_θ L_SDS = E_{t,ε,c} [ w(t) · (ε̂_φ(x_t, t, y) - ε) · ∂x/∂θ ]

Key: this is NOT backprop through the diffusion model — it uses the
     predicted noise as a pseudo-gradient direction, weighted by w(t).
     The U-Net Jacobian ∂ε̂_φ/∂x is dropped (expensive + empirically hurts).

Intuition: at each step, the diffusion model says "this render looks noisy;
           move the 3D scene in this direction to make it look like the text".
```

The SDS loss has well-known pathologies: **over-saturation** (the diffusion model maps to high-confidence modes), **over-smoothing** (multiple views are supervised independently, creating averaging artifacts), and the **Janus problem** (see below). ProlificDreamer [Wang2023] improved SDS with **Variational Score Distillation (VSD)**, treating the rendered image distribution as a distribution rather than a point, achieving higher-fidelity results.

### The Janus Problem

The Janus problem (named after the two-faced Roman god) is the canonical failure mode of SDS-based 3D generation: the 3D representation produces a plausible 2D image from any viewpoint, but the scene has **no consistent 3D structure** — instead, each viewpoint independently "looks like" the prompt, resulting in objects with multiple fronts (e.g., a human face visible from all angles around the head) or inconsistent geometry.

```
# Janus problem root cause:
The 2D diffusion prior p_φ(x | y) independently scores each rendered view.
It has no mechanism to enforce 3D consistency between views.
A 3D representation that places a face texture on all sides of the geometry
minimizes SDS loss from all azimuths, but is geometrically nonsensical.

Mitigations:
- Pose-conditioned diffusion (Zero123): conditions on relative camera pose
- 3D-aware diffusion models (trained on multi-view data)
- Geometry regularizers (depth smoothness, normal consistency)
- Multi-view SDS: evaluate multiple views simultaneously (MVDream)
```

---

## DreamFusion [Poole2022]

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall (Google Brain / Google Research) introduced DreamFusion as an arXiv preprint in September 2022 (published at ICLR 2023). It combines NeRF with the Imagen text-to-image diffusion model as a prior. The NeRF is initialized with density accumulated near a sphere, then optimized for ~10k–15k iterations. Additional regularizers include: normal consistency (neighboring points should have similar normals), orientation loss (densities should face the camera), and background regularization. Textureless regions are handled by sampling random albedo colors as a "textureless" baseline.

DreamFusion requires access to a large pre-trained text-to-image model (Imagen, which is not public) — this initially limited reproducibility. Subsequent works used publicly available models (Stable Diffusion, DeepFloyd IF).

---

## Magic3D [Lin2023, CVPR 2023]

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin (NVIDIA) published Magic3D at CVPR 2023. It addresses DreamFusion's slow optimization and low resolution by introducing a **two-stage coarse-to-fine pipeline**:

```mermaid
flowchart LR
    A[Text Prompt] --> B[Stage 1: Coarse NeRF\nInstant-NGP + low-res diffusion\n64×64 resolution\n~20 min]
    B --> C[DMTet Mesh Extraction\nDifferentiable mesh from density]
    C --> D[Stage 2: Fine Mesh Refinement\nHigh-res diffusion 512×512\nOptimize vertex positions + texture\n~20 min total]
    D --> E[High-quality 3D Mesh\n8× higher resolution than DreamFusion]
```

DMTet (Deep Marching Tetrahedra) enables differentiable mesh optimization from an initial coarse mesh. Magic3D achieves 8× higher resolution than DreamFusion (512 vs. 64 supervision) and 2× faster optimization (40 min vs. 1.5+ hours). User study shows 61.7% preference over DreamFusion.

---

## Zero-1-to-3 [Liu2023, ICCV 2023]

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick (Columbia University) introduced Zero-1-to-3 at ICCV 2023. Rather than generating 3D from scratch, Zero-1-to-3 uses a large pre-trained image diffusion model (Stable Diffusion) fine-tuned on Objaverse to generate **novel views** of an object from a single image at a specified relative camera pose $(\Delta R, \Delta T)$.

```
# Zero-1-to-3 formulation:
Input:  single image I_ref, relative camera pose (ΔR, Δt)
Output: novel view image I_novel

Architecture: 
  - Stable Diffusion U-Net fine-tuned on Objaverse renders
  - Conditioning: CLIP features of I_ref + (ΔR, Δt) injected via cross-attention
  - Trained on 800k objects × multiple pose pairs

Applications:
  - Direct use: single-image novel view synthesis
  - 3D reconstruction: Score Distillation using Zero-1-to-3 as diffusion prior
    (Zero123-XL, One-2-3-45)
```

Zero-1-to-3 demonstrates strong zero-shot generalization to in-the-wild images and paintings despite training on synthetic Objaverse data. It became the foundation for a family of single-image 3D reconstruction methods.

---

## LRM: Large Reconstruction Model [Hong2023]

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan (Adobe Research) introduced LRM (Large Reconstruction Model) as an arXiv preprint in November 2023. LRM is the first **feed-forward large-scale 3D reconstruction model**: a single forward pass predicts a full NeRF from one image in ~5 seconds.

```
# LRM Architecture:
Input: single image I ∈ R^{256×256×3}

1. Image Encoder: DINOv2 ViT-L (frozen) → image tokens [cls, patch_1,...,patch_N]
2. Cross-attention: image tokens → 3D triplane tokens
   - Triplane: three orthogonal 2D feature planes (XY, XZ, YZ)
   - Each plane: 192 × 192 × C resolution
3. NeRF Decoder: triplane features interpolated at query points → MLP → (density, color)
4. Volume rendering → predicted views for photometric loss

Training: ~1M 3D objects (Objaverse synthetic + MVImgNet real)
          Multi-view supervision with camera-conditioned decoder
Inference: 5 seconds on a single GPU
```

LRM's transformer-based architecture (500M parameters) enables learning geometry priors over large-scale 3D data, allowing generalization to diverse in-the-wild input images.

---

## TripoSR [Tochilkin2024]

Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao (Tripo AI + Stability AI) released TripoSR (arXiv March 2024) as an open-source, MIT-licensed large reconstruction model building on the LRM architecture. Key improvements over LRM:
- **Better data processing**: improved rendering pipeline, more object-centric crops
- **Training improvements**: improved loss weighting, better background augmentation
- **Speed**: under 0.5 seconds for 3D mesh from a single image
- **Open-source**: full code and weights released, unlike LRM

TripoSR achieves superior quantitative and qualitative performance vs. other open-source single-image 3D methods and is widely deployed in commercial 3D generation workflows.

---

## DreamGaussian [Tang2023, ICLR 2024 Oral]

Jiaxiang Tang et al. published DreamGaussian (ICLR 2024 Oral), replacing NeRF with 3D Gaussian Splatting as the optimization target in SDS-based generation. Key innovations:
1. **Progressive densification**: Gaussian splitting/cloning converges faster for generative tasks than NeRF's occupancy pruning
2. **Mesh extraction + UV refinement**: after Gaussian optimization, extracts a textured mesh via marching cubes on Gaussian opacity, then refines texture in UV space using the 2D diffusion prior

Results: ~2 minute generation vs. 1.5+ hours for DreamFusion/Magic3D. The speed improvement comes from 3DGS's faster rendering and the more direct optimization landscape provided by explicit primitives.

---

## Shap-E and Point-E (OpenAI)

Heewoo Jun and Alex Nichol (OpenAI) released two text-to-3D systems:

**Point-E** (2022): A two-stage pipeline — a text-to-image model generates a single rendered view, then a point cloud diffusion model conditioned on this image generates a 3D point cloud. Fast but limited by point cloud quality.

**Shap-E** (2023, arXiv:2305.02463): Trains a **latent diffusion model on implicit neural representations (INR)**. An encoder maps 3D assets (from ShapeNet/Cap3D) to latent vectors of a small MLP that represents the object. A conditional diffusion model then samples new latent vectors given text or image inputs. The same latent can be decoded as both a textured mesh and a NeRF. Shap-E produces 3D assets in seconds but at lower quality than optimization-based methods.

---

## MeshGPT [Siddiqui2024, CVPR 2024]

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov et al. published MeshGPT at CVPR 2024. Rather than generating density fields and extracting meshes, MeshGPT directly **autoregressively generates triangle meshes** — the native representation of 3D graphics.

```
# MeshGPT pipeline:
1. Graph Convolution Encoder: triangle mesh → per-triangle features
2. Residual Vector Quantization (RVQ): compress triangle features → discrete tokens
3. GPT-style Transformer Decoder: 
   autoregressively predict next triangle token
   (vertex triplets encoded as sorted (v1, v2, v3) tuples)
4. VQ-VAE Decoder: tokens → triangle geometries

Training: supervised on ShapeNet + 3D-FUTURE
Output: compact meshes with sharp edges, ~150–300 triangles
        (vs. marching cubes which generates 10,000+)

Key advantage: meshes resemble artist-created topology —
               efficient, clean, with proper edge loops
```

MeshGPT achieves 9% better shape coverage and 30-point FID improvement over prior mesh generation methods. It does not yet handle textures or complex scenes.

---

## Native 3D Latent Generative Models (2025–2026)

### TRELLIS: Structured 3D Latents [Xiang2025, CVPR 2025 Spotlight]

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong and Jiaolong Yang (Tsinghua, USTC, Microsoft Research) introduced **SLAT (Structured LATent)**. Each asset becomes a set of active voxels in a sparse $64^3$ grid, and each voxel carries a latent vector aggregated from dense multi-view DINOv2 features. Separate decoders map the *same* SLAT to 3D Gaussians, radiance fields or meshes.

```
# TRELLIS two-stage rectified-flow generation (schematic)
Stage 1 (structure):  sparse-structure flow transformer  → occupied voxel set {p_i} ⊂ {0..63}^3
Stage 2 (latents):    sparse flow transformer (SLAT)      → z_i ∈ R^C at each p_i
Decode:               D_GS(z) | D_RF(z) | D_Mesh(z)       ← Gaussians / radiance field / mesh
Rectified flow:       x_t = (1 - t) x_0 + t ε ;  minimize E || v_θ(x_t, t, cond) - (ε - x_0) ||^2
```

Models with up to 2B parameters (TRELLIS-text-xlarge) were trained on the 500K-object TRELLIS-500K dataset. TRELLIS-image-large has 1.2B parameters. Generation takes seconds, local 3D editing is supported, and the release is MIT-licensed.

### TRELLIS.2: Native and Compact Structured Latents [Xiang2026, CVPR 2026 Best Student Paper]

TRELLIS.2 (Tsinghua, Microsoft Research, USTC, Microsoft AI) addresses two limits of SLAT: dependence on 2D-feature aggregation, and the closed-surface assumptions of SDF/occupancy-based decoders. Its main contributions are:

- **O-Voxel**: a *field-free* sparse voxel structure that natively encodes both geometry and **PBR material** attributes. It handles open surfaces, non-manifold geometry and enclosed internal structures.
- **Sparse Compression VAE**: compresses O-Voxels into a compact latent space, so far fewer tokens are needed at high resolution.
- **~4B-parameter flow-matching transformers** trained on this latent, generating assets at $512^3$, $1024^3$ or $1536^3$. The official repo reports roughly 3 s, 17 s and 60 s respectively on an H100 for shape plus material. Code and weights are MIT-licensed.

### Hunyuan3D (Tencent)

Tencent's Hunyuan3D series is the most widely deployed *open* native-3D pipeline. It decouples geometry and texture:

- **Hunyuan3D 2.0** (Jan 2025, arXiv 2501.12202): **Hunyuan3D-DiT**, a flow-based diffusion transformer over a ShapeVAE vector-set latent, plus **Hunyuan3D-Paint**, a multi-view texture diffusion. Variants include 2mini (0.6B), 2mv (multi-view input) and Turbo/FlashVDM-accelerated versions (Mar 2025).
- **Hunyuan3D 2.1** (Jun 2025, arXiv 2506.15442) is described by its authors as the first fully open-source **PBR** asset pipeline. It includes the shape model (3.3B) and a PBR paint model (2B) that generates albedo, metallic and roughness maps, together with released data processing and training code.
- **Hunyuan3D 2.5** (technical report Jun 2025) and **Hunyuan3D-Omni** (Sep 2025, arXiv 2509.21245) are, respectively, a higher-fidelity scaled version and a controllable framework built on 2.1. Omni accepts point-cloud, voxel, skeleton and bounding-box controls.
- **Hunyuan3D 3.0** (announced 16 Sep 2025) reports 1536³ geometric resolution and roughly 3× precision over prior versions, according to Tencent's announcement. It is accessed via the Hunyuan3D engine and Tencent Cloud API, not as open weights.

### Other Open Native-3D Generators

- **TripoSG** [Li2025] (VAST): a 1.5B rectified-flow transformer over an SDF VAE latent (2,048 tokens). The VAE is trained with SDF, surface-normal and eikonal losses on ~2M image-SDF pairs; MIT licence.
- **Direct3D-S2** [Wu2025b, NeurIPS 2025] (DreamTech): introduces **Spatial Sparse Attention** for sparse-volume DiTs, reporting 3.9× forward and 9.6× backward speed-ups. This enables 1024³ SDF generation trained on only 8 GPUs; MIT licence.

### SAM 3D: 3Dfy Anything in Images [SAM3DTeam2025]

Meta's SAM 3D (released 19 Nov 2025, arXiv 2511.16624) brings the SAM philosophy of promptable models trained with a model-in-the-loop data engine to 3D. It has two models:

- **SAM 3D Objects** reconstructs **shape, texture and layout** (object pose and scale in the camera frame) of a masked object from a single natural image. It is robust to occlusion and clutter, and several objects can be composed into a scene. Meta's data engine produced roughly 3.14M meshes for almost 1M real images by having humans rank and select model-generated candidates. Training is progressive: synthetic pre-training followed by real-image post-training to close the sim-to-real gap. Meta introduced the **SA-3DAO** (3D Artist Objects) benchmark and reports strong human-preference wins over prior single-image methods (a ≥5:1 win rate has been reported). Code and weights are released under the SAM License.
- **SAM 3D Body** is promptable (masks, 2D keypoints) full-body human mesh recovery, including hands and feet. It uses the new **Momentum Human Rig (MHR)**, a parametric model that decouples skeleton from surface shape.

SAM 3D was reported among the CVPR 2026 Best Paper honorable mentions. Meta notes that current limitations include one-object-at-a-time prediction and no reasoning about physical contact or interpenetration.

### From Objects to Worlds

Native-3D generators increasingly feed **world generation**. Tencent's **HunyuanWorld 1.0** (Jul 2025) generates explorable, simulation-ready 3D worlds. World Labs' **Marble** product generates persistent 3D worlds. World Labs also maintains **Spark**, an open-source THREE.js Gaussian-splat renderer, reflecting how much world generation relies on splats as an output format. WorldMirror (ICML 2026) and other feed-forward reconstructors close the loop from generated video to 3D. See [Spatial Intelligence](../12_research_frontier_2024_2026/06_spatial_intelligence.md).

---

## Pipeline Comparison

```mermaid
flowchart TD
    T[Text Prompt] --> O[Optimization-Based]
    T --> F[Feed-Forward]
    I[Single Image] --> O
    I --> F

    O --> DS[DreamFusion\nNeRF + SDS\n1.5-3h, Janus issue]
    O --> M3[Magic3D\n2-stage coarse-to-fine\n40min, high-res mesh]
    O --> DG[DreamGaussian\n3DGS + SDS + UV refine\n2min]

    F --> Z123[Zero-1-to-3\nDiffusion NVS\nfine-tuned SD, ICCV 2023]
    F --> LRM[LRM\nViT → Triplane NeRF\n5 sec]
    F --> TSR[TripoSR\nOpen-source LRM\n<0.5 sec]
    F --> SE[Shap-E\nLatent diffusion on INR\nSeconds]
    F --> MG[MeshGPT\nAutoregressive triangles\nCompact mesh]

    T --> N[Native 3D Latent]
    I --> N
    N --> TR[TRELLIS / TRELLIS.2\nSparse-voxel latent + flow\nPBR, up to 1536³]
    N --> HY[Hunyuan3D 2.x / 3.0\nShape DiT + PBR Paint]
    N --> S3D[SAM 3D Objects\nIn-the-wild shape + layout]
```

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| DreamFusion: Text-to-3D using 2D Diffusion | Poole, Jain, Barron, Mildenhall | 2022 | ICLR 2023 | SDS loss; NeRF optimization via 2D diffusion prior; text-to-3D without 3D data |
| Magic3D: High-Resolution Text-to-3D Content Creation | Lin, Gao, Tang et al. (NVIDIA) | 2023 | CVPR | 2-stage coarse NeRF → fine DMTet mesh; 8× higher res, 2× faster than DreamFusion |
| Zero-1-to-3: Zero-shot One Image to 3D Object | Liu, Wu, Van Hoorick et al. | 2023 | ICCV | Fine-tuned diffusion for pose-conditioned NVS; enables image-to-3D without 3D supervision |
| LRM: Large Reconstruction Model for Single Image to 3D | Hong, Zhang et al. (Adobe) | 2023 | arXiv (ICLR 2024) | 500M param ViT transformer; triplane NeRF prediction in 5 sec; trained on 1M objects |
| TripoSR: Fast 3D Object Reconstruction from a Single Image | Tochilkin, Pankratz et al. (Tripo AI / Stability AI) | 2024 | arXiv | Open-source LRM variant; <0.5 sec; MIT license |
| DreamGaussian: Generative Gaussian Splatting for 3D Content Creation | Tang, Ren et al. | 2023 | ICLR 2024 (Oral) | 3DGS as SDS target; mesh extraction + UV refinement; 2-min generation |
| Shap-E: Generating Conditional 3D Implicit Functions | Jun, Nichol (OpenAI) | 2023 | arXiv | Latent diffusion on INR weights; text/image to 3D in seconds |
| MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers | Siddiqui, Alliegro, Artemov et al. | 2024 | CVPR | Autoregressive triangle sequence generation; compact artist-quality meshes |
| ProlificDreamer: High-Fidelity and Diverse Text-to-3D via VSD | Wang et al. | 2023 | NeurIPS | Variational Score Distillation; higher quality than SDS; reduced over-saturation |
| Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS) | Xiang, Lv, Xu, Deng, Wang, Zhang, Chen, Tong, Yang | 2025 | CVPR (Spotlight) | SLAT sparse-voxel latent; rectified-flow transformers up to 2B; multi-format decoding |
| Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation | Tencent Hunyuan3D Team | 2025 | arXiv | Shape DiT on vector-set latent + multi-view Paint model; open weights |
| Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material | Tencent Hunyuan3D Team | 2025 | arXiv | Fully open PBR pipeline (albedo/metallic/roughness); training code released |
| TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models | Li et al. (VAST) | 2025 | arXiv | 1.5B rectified flow over SDF VAE; ~2M training pairs |
| Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention | Wu et al. (DreamTech) | 2025 | NeurIPS | Spatial Sparse Attention; 1024³ SDF generation on 8 GPUs |
| SAM 3D: 3Dfy Anything in Images | SAM 3D Team (Meta) | 2025 | arXiv | In-the-wild single-image shape, texture and layout; data engine; SA-3DAO benchmark |
| Native and Compact Structured Latents for 3D Generation (TRELLIS.2) | Xiang et al. (Tsinghua, MSR, USTC, Microsoft AI) | 2026 | CVPR (Best Student Paper) | O-Voxel with PBR; arbitrary topology; 4B flow models up to 1536³ |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| LRM | GSO (Google Scanned Objects) | PSNR (novel views) | not publicly reported | Qualitative comparison in paper |
| TripoSR | GSO | PSNR | not publicly reported | Claims superior to open-source LRM variants |
| Zero-1-to-3 | GSO | PSNR (novel view) | 20.13 dB | Single image, held-out test views |
| DreamGaussian | — | CLIP Similarity | 0.287 | Text-to-3D; 2 min vs. DreamFusion 1.5h |
| MeshGPT | ShapeNet | FID (mesh quality) | 30-pt improvement | vs. prior mesh generation methods |
| DreamFusion | — | CLIP R-Precision | ~0.75 | Text alignment; Janus problem visible |
| Shap-E | ShapeNet Eval | FID | not publicly reported | Faster generation but lower quality |
| TRELLIS.2 (4B) | — | Generation time (H100) | ~3 s @512³ / ~17 s @1024³ / ~60 s @1536³ | Shape + PBR material; per official repo |
| Direct3D-S2 | — | Attention speed-up (SSA) | 3.9× fwd / 9.6× bwd | vs dense attention on sparse volumes; authors' claim |
| SAM 3D Objects | SA-3DAO + human preference | Win rate vs prior methods | ≥5:1 (reported) | Single in-the-wild image; Meta's claim |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Optimization-based (SDS) | No 3D training data needed; highly flexible text/image conditioning; latest 2D diffusion quality | Slow (minutes to hours); Janus problem; over-saturation; each object requires new optimization |
| Feed-forward (LRM/TripoSR) | Fast (<1 sec – 5 sec); deterministic; scalable deployment; generalizes from large data | Requires large 3D training datasets; limited detail for complex scenes; train-test distribution mismatch |
| Native mesh generation (MeshGPT) | Compact, clean artist-quality topology; directly usable in rendering/simulation | Limited to ShapeNet-like objects; no texture generation yet; limited generalization |
| Native 3D latent flow (TRELLIS.2, Hunyuan3D, TripoSG) | High-resolution geometry (up to 1536³); PBR materials; seconds to a minute; open weights for several models | Dense remeshing/retopology still needed for production; large 3D datasets required; licences vary (some API-only) |
| In-the-wild reconstruction-generation (SAM 3D) | Handles occlusion, clutter and scene layout; promptable | One object at a time; no physical contact reasoning; SAM License terms |

---

## Open Problems & Research Gaps

- **Janus problem remains unsolved at scale:** Multi-view consistent generation fundamentally requires either multi-view diffusion training (MVDream, Zero123++) or explicit 3D consistency constraints. Current methods still exhibit Janus-like artifacts on complex prompts.
- **Texture quality and detail:** Feed-forward methods (LRM, TripoSR) produce blurry textures compared to optimization-based approaches. High-frequency texture generation at scale is a key open problem. Native-3D models with PBR heads (Hunyuan3D 2.1, TRELLIS.2) have substantially narrowed this gap. Relightable, physically correct materials and UV-free texture representations (cf. HKTex, ECCV 2026 Best Paper, in [00_3d_representations.md](./00_3d_representations.md)) are the next frontier.
- **Scene generation (not just objects):** 3D generation has mostly focused on single objects. Text-to-scene (multiple objects with spatial relationships, backgrounds, lighting) remains hard; SceneWiz3D and Set-the-Scene are early attempts. In 2025–2026, SAM 3D added per-object layout from real images, and world-generation systems (HunyuanWorld, Marble) produce explorable scenes. Compositional, editable, physically consistent multi-object scenes are still open.
- **Topology and production readiness:** O-Voxel (TRELLIS.2) handles open and non-manifold surfaces. Artist-grade topology, UVs, rigging and part decomposition straight from the generator remain largely unsolved.
- **Physical plausibility:** Generated 3D assets often violate physics — floating parts, intersecting geometry, incorrect proportions. Incorporating physical priors or simulation-based feedback during generation is an emerging area.
- **Video-to-3D:** Given the abundance of video data, learning 3D generation from monocular video (rather than posed multi-view data) is highly valuable but requires solving appearance vs. geometry ambiguity.
- **Controllable generation:** Fine-grained control over generated 3D assets (shape, material, part articulation) is limited. Text is too coarse for precise geometric specification; CAD-like control interfaces are needed.
- **Evaluation metrics:** PSNR/FID/CLIP similarity inadequately capture 3D consistency, geometric accuracy, and perceptual quality simultaneously. A unified 3D generation benchmark is lacking. SAM 3D's SA-3DAO and its Hugging Face leaderboard (2026) are a step toward real-image evaluation, but most native-3D papers still rely on user studies and in-house comparisons.

---

## Further Reading

- [DreamFusion Project Page (Google)](https://dreamfusion3d.github.io/) — paper, videos, and SDS explanation
- [Zero-1-to-3 GitHub (Columbia CVLab)](https://github.com/cvlab-columbia/zero123) — code and pretrained models
- [TripoSR GitHub (VAST-AI)](https://github.com/VAST-AI-Research/TripoSR) — open-source code, MIT license
- [DreamGaussian GitHub](https://github.com/dreamgaussian/dreamgaussian) — ICLR 2024 Oral implementation
- [MeshGPT Project Page](https://nihalsid.github.io/mesh-gpt/) — paper and generation results
- [Shap-E GitHub (OpenAI)](https://github.com/openai/shap-e) — code and model weights
- [TRELLIS GitHub (Microsoft)](https://github.com/microsoft/TRELLIS) — SLAT, TRELLIS-500K, CVPR 2025 Spotlight
- [TRELLIS.2 GitHub (Microsoft)](https://github.com/microsoft/TRELLIS.2) — O-Voxel, CVPR 2026 Best Student Paper (MIT)
- [Hunyuan3D-2.1 GitHub (Tencent)](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — open PBR asset pipeline
- [Hunyuan3D-Omni GitHub (Tencent)](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni) — controllable generation
- [SAM 3D Objects GitHub (Meta)](https://github.com/facebookresearch/sam-3d-objects) — single-image in-the-wild 3D; SA-3DAO
- [SAM 3D Body GitHub (Meta)](https://github.com/facebookresearch/sam-3d-body) — promptable human mesh recovery with MHR
- [TripoSG GitHub (VAST)](https://github.com/VAST-AI-Research/TripoSG) — 1.5B rectified-flow shape model
- [Direct3D-S2 GitHub](https://github.com/DreamTechAI/Direct3D-S2) — NeurIPS 2025 sparse-attention gigascale generation
