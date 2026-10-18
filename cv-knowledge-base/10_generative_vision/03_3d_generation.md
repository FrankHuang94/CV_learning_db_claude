# 3D Generation: From Score Distillation to Feed-Forward Reconstruction

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](./01_diffusion_models.md) | [3D Vision and Scene Generation](../04_3d_vision_and_scene/04_3d_generation.md) | [Video Generation](./02_video_generation.md) | [GANs and VAEs](./00_gans_and_vaes.md)

> **🆕 October 2026 update:**
> - New section on the third paradigm that dominated 2025–2026: **native 3D latent diffusion** (Hunyuan3D 2.x, TRELLIS, TripoSG, Direct3D-S2, TRELLIS.2 — whose paper "Native and Compact Structured Latents for 3D Generation" was named CVPR 2026 Best Student Paper).
> - New section on **SAM 3D** (Meta, Nov 2025; CVPR 2026 honorable mention) for in-the-wild single-image object and human-body reconstruction.
> - New section on **object-to-world generation** (World Labs Marble, HunyuanWorld 1.0, Genie 3 as an implicit alternative).
> - Paradigm map, Key Papers, Pros & Cons, Open Problems and Further Reading updated.

---

## Overview

Generating 3D objects and scenes from text or images is significantly harder than 2D generation because ground-truth 3D supervision is scarce and expensive, the output space (meshes, point clouds, NeRFs, Gaussian splats) is high-dimensional and non-Euclidean, and evaluation of 3D quality requires multi-view consistency rather than a single-image FID score. The field bifurcated into two major paradigms: **optimization-based** methods that use 2D diffusion priors to guide iterative 3D optimization (DreamFusion, Magic3D), and **feed-forward** methods that train a large model to directly regress 3D structure from one or more images in a single forward pass (LRM, TripoSR).

The optimization-based paradigm was inaugurated by DreamFusion [Poole2022], which introduced Score Distillation Sampling (SDS) — using a frozen 2D diffusion model as a differentiable score function to update a NeRF representation. SDS is slow (hours per object) and suffers from over-saturation and the "Janus problem" (multi-face objects), but it requires no 3D training data. Subsequent work (Magic3D [Lin2023], ProlificDreamer [Wang2023], Fantasia3D) incrementally addressed these issues through coarse-to-fine strategies, variational distillation, and decoupled geometry-appearance optimization.

The feed-forward paradigm emerged from the Large Reconstruction Model (LRM) [Hong2023], which demonstrated that a transformer trained on Objaverse could reconstruct a NeRF or triplane representation from a single image in ~5 seconds. TripoSR [Tochilkin2024], an improved LRM trained with better data curation, reduced this to under 0.5 seconds. DreamGaussian [Tang2023] bridged the two paradigms by using 2D diffusion for image generation and a fast Gaussian splatting optimization for 3D lift, achieving minutes-per-object speed.

Since 2025 a third paradigm has become the production default: **native 3D latent diffusion**. A 3D VAE compresses large curated mesh collections (Objaverse/Objaverse-XL plus licensed or in-house assets) into a compact latent (vector sets, sparse voxels or structured latents), and a DiT trained with flow matching generates that latent directly from an image or text, followed by a separate texture/PBR stage. Hunyuan3D 2.x [Hunyuan3D2025], TRELLIS [Xiang2025] and TripoSG [Li2025TripoSG] made this approach open and practical, and commercial tools (Hunyuan3D, Tripo, Meshy, Rodin) now produce game-ready assets in tens of seconds. In parallel, the scope has expanded from isolated objects to **in-the-wild reconstruction** (SAM 3D) and **explorable worlds** (World Labs Marble, HunyuanWorld).

---

## Score Distillation Sampling (SDS)

DreamFusion [Poole2022] introduced SDS to distill knowledge from a 2D diffusion model $\epsilon_\phi$ into a 3D representation $\theta$ (NeRF parameters) without 3D supervision:

```
SDS Loss:
  grad_theta L_SDS = E_{t, epsilon, c} [
    w(t) * (epsilon_phi(z_t; y, t) - epsilon) * (partial z / partial theta)
  ]

where:
  z = g(theta)        -- render image from current NeRF parameters
  z_t = alpha_t*z + sigma_t*epsilon  -- noisy version
  epsilon_phi(z_t; y, t) -- diffusion model noise prediction given text y
  w(t)                -- weighting function (SNR-dependent)

Interpretation:
  Pushes rendered images toward high-probability regions of p(x|y)
  as defined by the diffusion model's score function.

Problems:
  1. Oversaturation: predicted noise != added noise -> oversmoothed colors
  2. Janus problem: each view independently maximizes p(x|y) -> multi-faced objects
  3. Speed: ~30-60 min per object on A100
```

Variational Score Distillation (VSD, ProlificDreamer [Wang2023]) addresses oversaturation by using a LoRA-fine-tuned diffusion model as a variational posterior rather than the fixed base model, treating the 3D representation as a particle in the variational distribution.

---

## The Janus Problem

The Janus problem is the defining failure mode of optimization-based 3D generation: because SDS renders the 3D object from random viewpoints and each view is independently evaluated by a 2D diffusion model, the model tends to place a "frontal" appearance (e.g., a face with two eyes, a frontal body) on every side of the object, producing multi-headed or multi-faced artifacts named after the two-faced Roman god Janus.

```
Root cause:
  2D diffusion model p(x|"a dog") -- agnostic to 3D camera pose context
  Rendered back: optimized for p(x|y) from all azimuths independently
  => Each 90-degree view converges to the "most likely" frontal appearance

Mitigations:
  1. View-dependent prompting: "front view of a dog" / "side view of a dog"
     injected as part of the conditioning text (DreamFusion)
  2. Zero-1-to-3 [Liu2023]: View-conditioned diffusion model trained on
     Objaverse; provides consistent multi-view prior
  3. MVDiffusion / SyncDreamer: Jointly generate N consistent views
     with explicit cross-view attention before 3D optimization
  4. Feed-forward models: Sidestep optimization entirely
```

---

## Magic3D and the Coarse-to-Fine Pipeline

Magic3D [Lin2023] (CVPR 2023) established a two-stage coarse-to-fine pipeline:
1. **Coarse stage**: Fast SDS optimization of a NeRF at low resolution (64³ grid) to obtain overall shape.
2. **Fine stage**: Extract an explicit mesh (marching cubes) and refine with a high-resolution diffusion model, using a differentiable mesh renderer (nvdiffrast) for gradient flow through triangle rasterization.

```
Stage 1: NeRF theta <- SDS with fast low-res diffusion model (Imagen)
  ~45 min on A100

Stage 2: theta -> mesh M via marching cubes
  M <- SDS with high-res diffusion (DeepFloyd IF)
  Differentiable render: L = LSDS(render(M)) + Lreg
  ~25 min on A100

Result: 8x higher resolution than DreamFusion; sharper texture details.
```

---

## Zero-1-to-3: View-Conditioned Diffusion

Zero-1-to-3 [Liu2023] (ICCV 2023) fine-tuned Stable Diffusion to synthesize a novel view of an object given a reference image and a camera transformation $(R, T)$:

```
Input:  (x, R, T)  -- reference image x, relative camera extrinsics (R, T)
Output: x'         -- novel view of the same object

Conditioning: CLIP image features of x + (R, T) encoded as tokens
Model: SD1.5 fine-tuned on Objaverse renderings + real objects (CO3D)

Inference for 3D generation:
  1. Synthesize N=8-64 views using Zero-1-to-3
  2. Optimize a NeRF using multi-view reconstruction loss on synthetic views
```

The key contribution is that a 2D diffusion model trained on synthetic multi-view data retains strong zero-shot generalization to real-world objects, enabling SDS-free 3D generation from a single image.

---

## LRM and TripoSR: Feed-Forward Reconstruction

The Large Reconstruction Model [Hong2023] trains a transformer to map a single image to a triplane NeRF representation:

```
Input:  single RGB image (x)
Encoder: ViT image encoder -> image tokens [N_img tokens]
Decoder: Transformer cross-attends image tokens -> triplane features
  Triplane: 3x (H x W) feature planes (XY, XZ, YZ) ~ 64x64 resolution
Output: NeRF queried via triplane interpolation -> volume render

Training: ~1M 3D objects (Objaverse); rendered 32 views/object
Loss: L2 render loss + LPIPS
Inference: ~5s on V100

TripoSR [Tochilkin2024]: LRM with:
  - Better data curation (higher quality Objaverse subset + filtering)
  - Improved renderer (volume render with masked background)
  - Faster inference: <0.5s on A100 (batch=1)
  - Open weights (MIT license)
```

---

## DreamGaussian

DreamGaussian [Tang2023] uses 3D Gaussian Splatting (3DGS) as the representation, enabling ~3-minute text/image-to-3D generation:

```
Stage 1: Generative Gaussian Splatting
  - Initialize 3DGS from point cloud (CLIP-guided or random)
  - SDS optimization on rendered views from 3DGS
  - 3DGS allows faster rendering than NeRF (rasterization vs. ray marching)

Stage 2: Mesh extraction and UV texture refinement
  - Convert 3DGS -> mesh via alpha-blended marching cubes
  - Refine texture in UV space using diffusion inpainting
  - Output: mesh + UV texture map (exportable .obj/.glb)

Speed: ~3 min (vs. 30-60 min for DreamFusion)
```

---

## Shap-E and Point-E (OpenAI)

**Point-E** [Nichol2022] generates a colored 3D point cloud from text or image input using a two-stage pipeline: (1) generate a single rendered view of the object, (2) use a diffusion model conditioned on the rendered view to generate a 4096-point RGB point cloud. Fast (~1-2 min) but produces coarse geometry.

**Shap-E** [Jun2023] improves on Point-E by directly generating the parameters of an implicit function (NeRF or SDF) rather than a point cloud. A transformer encoder-decoder maps 3D assets to implicit function parameters; a diffusion model is trained over these latents. Shap-E can produce meshes, point clouds, and NeRF representations from text or image conditioning. Both are open-source OpenAI models.

---

## MeshGPT

MeshGPT [Siddiqui2024] (CVPR 2024) directly generates triangle meshes using an autoregressive decoder-only Transformer, treating mesh generation as a sequence prediction task:

```
Tokenization:
  - Sort mesh faces by vertex position (z-first Morton ordering)
  - Each triangle -> 9 coordinate values (3 vertices x 3D)
  - Quantize to K-bit integers (K=7, 128 levels per axis)
  - Each face = 3 tokens (each encoding x,y,z of one vertex)
  - Codebook learned via VQ-VAE on face sequences

Autoregressive generation:
  p(mesh) = prod_{i=1}^{N} p(face_i | face_{1..i-1}, class_token)

Conditional on ShapeNet-Core class labels.
Produces valid, compact, non-watertight meshes; no NeRF rendering needed.
```

---

## Native 3D Latent Diffusion (2025–2026)

The recipe that replaced SDS and triplane LRMs for object generation has two stages — shape, then appearance:

```
Stage A: 3D shape VAE
  Input:  surface point cloud P (sampled from a watertight mesh, with normals)
  Encode: E(P) -> latent z   (e.g. vector set of ~1-4K tokens, or sparse voxel features)
  Decode: D(z, q) -> SDF / occupancy at query point q  -> marching cubes -> mesh

Stage B: shape DiT (flow matching in latent space)
  z_t = (1 - t) * z + t * eps
  L = E || v_theta(z_t, t, c_img) - (eps - z) ||^2
  c_img = DINOv2 features of the conditioning image (cross-attention)

Stage C: texture / PBR
  multi-view diffusion conditioned on rendered normals + positions
  -> back-projection / UV baking -> albedo, roughness, metallic maps
```

| System | Released | Latent representation | Notes |
|--------|----------|-----------------------|-------|
| Hunyuan3D 2.0 / 2.1 [Hunyuan3D2025] | Jan / Jun 2025 | Vector-set shape latent (ShapeVAE) + flow DiT | Open; Hunyuan3D-Paint for textures, 2.1 adds open PBR material generation |
| TRELLIS [Xiang2025] | Dec 2024 (CVPR 2025) | Structured LATent (SLAT): sparse voxels carrying DINOv2-aggregated features | Decodes to Gaussians, radiance fields or meshes from the same latent |
| TripoSG [Li2025TripoSG] | Feb 2025 | SDF-based VAE + rectified-flow transformer | High-fidelity sharp geometry; open weights |
| Direct3D-S2 | May 2025 | Sparse SDF volumes with spatial sparse attention | Gigascale (high-resolution) 3D generation at modest training cost |
| TRELLIS.2 | Dec 2025 | Native, compact structured latents | Larger successor to TRELLIS for higher-resolution geometry and PBR appearance; its paper "Native and Compact Structured Latents for 3D Generation" received the CVPR 2026 Best Student Paper award |

Two lessons carry over from 2D: **latent design dominates** (sparse, structure-aware latents scale better than dense triplanes), and **flow matching on DiTs scales cleanly** once data quality is controlled. The weak point is data: high-quality, licensed 3D assets number in the low millions, three orders of magnitude below image corpora.

---

## SAM 3D: Single-Image 3D in the Wild

Meta's **SAM 3D** (Nov 2025, released alongside SAM 3) comprises two models: **SAM 3D Objects**, which reconstructs full geometry, texture and layout of objects from a single natural image — including occluded and cluttered scenes — and **SAM 3D Body**, which recovers full-body human mesh and pose. Instead of relying solely on synthetic Objaverse renders, SAM 3D uses a **model-in-the-loop data engine**: annotators select and rank candidate 3D reconstructions for real images, progressively building a large real-image-to-3D training set, followed by synthetic pre-training and preference-style post-training. Meta reports large human-preference wins over prior single-image 3D methods on real-world inputs and released weights, code and an evaluation benchmark. SAM 3D received an honorable mention for the CVPR 2026 Best Paper award. It is significant because it moves image-to-3D from curated studio-style object photos to everyday photographs, the setting that matters for robotics, AR and e-commerce (see [3D Vision and Scene Generation](../04_3d_vision_and_scene/04_3d_generation.md)).

---

## From Objects to Worlds

- **World Labs Marble** (generally available Nov 2025): generates persistent, explorable 3D worlds from text, images, video or coarse 3D layouts, with editing tools and export to Gaussian splats and meshes — an explicit-3D answer to scene generation.
- **HunyuanWorld 1.0** (Tencent, Jul 2025): open model that generates panoramic proxies and lifts them into layered, explorable 3D meshes.
- **Genie 3** (Google DeepMind, Aug 2025) takes the *implicit* route: an interactive video world model with no explicit 3D representation, trading geometric exactness for open-ended dynamics (see [Video Generation](./02_video_generation.md) and [World Models](../06_robotics_and_embodied_ai/02_world_models.md)).

Explicit worlds (splats/meshes) are directly usable in game engines and simulators and are geometrically consistent by construction; implicit world models handle dynamics and interaction more naturally but drift over time. Hybrids that cache explicit 3D memory inside a video generator are an active direction.

---

## Mermaid: 3D Generation Paradigm Map

```mermaid
graph TD
    subgraph Optimization-Based
        T1[Text Prompt] --> SDS[Score Distillation Sampling]
        I1[Single Image] --> Zero123[Zero-1-to-3 Views]
        SDS --> NeRF[NeRF Optimization]
        Zero123 --> NeRF
        NeRF --> Mesh[Mesh Extraction]
    end

    subgraph Feed-Forward
        I2[Single Image] --> LRM[LRM / TripoSR Transformer]
        LRM --> Triplane[Triplane NeRF]
        Triplane --> Render[Volume Render]
    end

    subgraph Hybrid
        I3[Image/Text] --> DG[DreamGaussian SDS]
        DG --> 3DGS[3D Gaussian Splats]
        3DGS --> Mesh2[Mesh + UV Texture]
    end

    subgraph Discrete
        T2[Text/Class] --> MeshGPT[MeshGPT Autoregressive]
        MeshGPT --> TriMesh[Triangle Mesh Tokens]
    end

    subgraph Native3D
        I4[Image/Text] --> DiT3D[Flow DiT in 3D Latent]
        DiT3D --> VAE3D[3D VAE Decoder]
        VAE3D --> SDFMesh[SDF Mesh]
        SDFMesh --> PBR[Multi-view PBR Texturing]
    end
```

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| DreamFusion | Poole et al. | 2022 | ICLR 2023 | Score Distillation Sampling; text-to-NeRF without 3D data |
| Magic3D | Lin et al. | 2023 | CVPR | Coarse-to-fine NeRF→mesh; 8× higher resolution than DreamFusion |
| Zero-1-to-3 | Liu et al. | 2023 | ICCV | View-conditioned diffusion on Objaverse; zero-shot image-to-3D |
| Large Reconstruction Model (LRM) | Hong et al. | 2023 | ICLR 2024 | Feed-forward transformer; single image → triplane NeRF in 5s |
| TripoSR | Tochilkin et al. | 2024 | arXiv | LRM variant; <0.5s inference; open weights (MIT license) |
| DreamGaussian | Tang et al. | 2023 | ICLR 2024 | 3DGS as optimization target; mesh extraction; 3 min/object |
| Point-E | Nichol et al. | 2022 | arXiv | Two-stage point cloud generation; fast but coarse |
| Shap-E | Jun & Nichol | 2023 | arXiv | Implicit function parameter generation; text/image to mesh/NeRF |
| MeshGPT | Siddiqui et al. | 2024 | CVPR | Autoregressive triangle mesh generation; VQ tokenization |
| SyncDreamer | Liu et al. | 2023 | ICLR 2024 | Joint multi-view diffusion with cross-view attention; Janus mitigation |
| Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS) | Xiang et al. | 2024 | CVPR 2025 | Sparse structured latent (SLAT); decodes to Gaussians, NeRF or mesh |
| Hunyuan3D 2.0 | Tencent Hunyuan3D Team | 2025 | arXiv | Open ShapeVAE + flow DiT shape model with separate texture synthesis |
| TripoSG | Li et al. | 2025 | arXiv | SDF VAE + large rectified-flow transformer for high-fidelity shapes |
| SAM 3D Objects / SAM 3D Body | Meta AI | 2025 | Tech report (CVPR 2026 honorable mention) | In-the-wild single-image 3D via model-in-the-loop data engine |
| Native and Compact Structured Latents for 3D Generation (TRELLIS.2) | Microsoft Research et al. | 2025 | CVPR 2026 (Best Student Paper) | Native compact structured latents for high-resolution 3D generation |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| DreamFusion | COCO (text prompts) | CLIP R-Precision | 74.9% | Rendered views vs. text |
| Magic3D | COCO | CLIP R-Precision | 79.1% | Higher quality than DreamFusion |
| Zero-1-to-3 | GSO | PSNR | 21.3 | Novel view synthesis; Google Scanned Objects |
| LRM | GSO | PSNR | 22.6 | Single-image reconstruction |
| TripoSR | GSO | CD (Chamfer) | 0.041 | Lower is better; <0.5s inference |
| Shap-E | ShapeNet-Core | FID (renders) | not publicly reported | OpenAI internal evaluation |
| MeshGPT | ShapeNet-Core | Coverage (COV) | 56.8% | Autoregressive mesh generation |
| DreamGaussian | T3Bench | Quality Score | 28.1 | Human evaluation on T3Bench |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| **Optimization-based (SDS)** | No 3D training data needed; generalizes to arbitrary text prompts; produces smooth geometry | Very slow (30–60 min/object); Janus problem; over-saturation artifacts; NeRF to mesh conversion lossy |
| **Feed-forward (LRM/TripoSR)** | Near real-time (<0.5s); consistent 3D structure; no per-object optimization | Requires large-scale 3D training data (Objaverse); limited to reconstruction, poor extrapolation beyond training distribution |
| **Gaussian splatting (DreamGaussian)** | Faster than NeRF-based optimization; direct mesh export; good for asset pipelines | Gaussian-to-mesh conversion still imperfect; floaters and discontinuities in exported meshes |
| **Discrete mesh (MeshGPT)** | CAD-quality topology; directly useful for downstream 3D applications | Limited to single-category training domains; does not handle open-vocabulary text prompts |
| **Native 3D latent diffusion (Hunyuan3D 2.x, TRELLIS, TripoSG)** | Seconds per asset; sharp, watertight geometry; open weights; PBR texturing pipelines | Bounded by scarce, licence-sensitive 3D data; texture stage still relies on 2D multi-view diffusion; topology not artist-ready without retopology |
| **In-the-wild reconstruction (SAM 3D)** | Works on cluttered, occluded real photos; human body and objects | Hallucinated back sides; scale/layout ambiguity from a single view |

---

## Open Problems & Research Gaps

- **Janus problem at scale**: Despite mitigation strategies (view-conditioned diffusion, multi-view joint generation), the Janus problem resurfaces at fine-grained detail levels when the 2D prior is strongly opinionated about appearance. A principled solution based on 3D-aware diffusion is still lacking.
- **Open-vocabulary text-to-3D generalization**: Feed-forward models overfit to Objaverse categories (toys, household objects). Generalizing to rare objects, complex scene compositions, or abstract 3D concepts requires either larger datasets or better compositional representations.
- **Dynamic 3D generation**: Generating animatable 3D assets (rigged characters, physics-animated objects) remains largely unsolved. Current methods produce static geometry; 4D generation (3D + time) is emerging but far from practical.
- **Scene-level 3D generation**: Object generation is now largely solved at the asset level, and products like Marble generate explorable worlds, but text-to-scene with correct physical layout, articulation, functional semantics and simulation-ready collision geometry (needed for robotics) remains open.
- **3D data scale and licensing**: Native 3D latent models are capped by a few million usable 3D assets. Model-in-the-loop data engines (SAM 3D), distillation from video/world models and synthetic procedural data are competing remedies; which scales best is unknown.
- **Artist-ready topology**: Generated SDF meshes need retopology and UV unwrapping; autoregressive mesh generators (MeshGPT successors) produce clean topology but at limited face counts. Combining the two is open.
- **Efficient 3D representations**: NeRF representations are compact but slow to render; 3DGS is fast but mesh export is lossy; meshes are application-friendly but hard to generate directly. No single representation satisfies all downstream use-case requirements simultaneously.
- **Evaluation gap**: No widely-adopted benchmark equivalent to FID exists for 3D generation that jointly measures geometry accuracy, texture quality, 3D consistency, and diversity. T3Bench, ULIP-2 evaluations, and human preference studies are inconsistent across papers.
- **Material and lighting separation**: Most generated 3D assets bake illumination into texture, producing wrong appearance under novel lighting conditions. Disentangled material/lighting representation (PBR material generation) remains an active research frontier.

---

## Further Reading

- [Poole et al. (2022), "DreamFusion: Text-to-3D using 2D Diffusion," ICLR 2023](https://arxiv.org/abs/2209.14988)
- [Liu et al. (2023), "Zero-1-to-3: Zero-shot One Image to 3D Object," ICCV 2023](https://arxiv.org/abs/2303.11328)
- [Hong et al. (2023), "LRM: Large Reconstruction Model for Single Image to 3D," ICLR 2024](https://arxiv.org/abs/2311.04400)
- [Tochilkin et al. (2024), "TripoSR: Fast 3D Object Reconstruction from a Single Image"](https://arxiv.org/abs/2403.02151)
- [Tang et al. (2023), "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation," ICLR 2024](https://arxiv.org/abs/2309.16653)
- [Siddiqui et al. (2024), "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers," CVPR 2024](https://arxiv.org/abs/2311.15475)
- [Xiang et al. (2024), "Structured 3D Latents for Scalable and Versatile 3D Generation" (TRELLIS), CVPR 2025](https://arxiv.org/abs/2412.01506)
- [Tencent Hunyuan3D Team (2025), "Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation"](https://arxiv.org/abs/2501.12202)
- [Li et al. (2025), "TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models"](https://arxiv.org/abs/2502.06608)
- [TRELLIS.2 repository (Microsoft)](https://github.com/microsoft/TRELLIS.2)
- [SAM 3D Objects repository (Meta)](https://github.com/facebookresearch/sam-3d-objects)
- [HunyuanWorld 1.0 repository (Tencent)](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)
