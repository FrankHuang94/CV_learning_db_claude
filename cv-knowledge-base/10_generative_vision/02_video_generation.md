# Video Generation: From Sora to Open-Source World Simulators

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](./01_diffusion_models.md) | [GANs and VAEs](./00_gans_and_vaes.md) | [World Models](../06_robotics_and_embodied_ai/02_world_models.md) | [Evaluation and Safety](./04_evaluation_and_safety.md)

---

## Overview

Video generation occupies a qualitatively different difficulty tier from image synthesis: temporal coherence, physics plausibility, camera motion, long-range dependencies, and multi-second duration all impose demands that cannot be addressed by independently generating frames. The dominant paradigm prior to 2024 — recurrent networks, video GANs, or autoregressive models over frame sequences — struggled to produce more than a few seconds of visually consistent footage. The release of OpenAI's Sora in early 2024 demonstrated that a single architecture (Diffusion Transformer operating on spacetime patches) trained at scale could generate up to 60 seconds of photorealistic, physically plausible video from text or image prompts, fundamentally changing expectations for the field.

The core architectural insight behind the modern generation of models is the unification of spatial and temporal compression into a shared 3D latent space (via a 3D causal VAE), followed by a Transformer-based diffusion process operating on "spacetime patches" — contiguous 3D tokens spanning a few frames and a spatial tile. Because patches can be extracted from videos of any resolution, duration, and aspect ratio without re-architecture, the same model handles a wide variety of generation tasks. Peebles and Xie [Peebles2023] first demonstrated that replacing the U-Net backbone with a Transformer (DiT — Diffusion Transformer) scaled more cleanly than U-Net on image generation; Sora and its successors extended DiT to the spatiotemporal domain.

As of mid-2026, the video generation landscape splits into proprietary closed models (Sora 2, Veo 3.1, Runway Gen-4, Kling 3.0) and a maturing open-source ecosystem (HunyuanVideo, Wan 2.2, CogVideoX, LTX-Video). The open models lag behind proprietary systems on motion realism and physics, but close the gap rapidly. All major systems have converged on the spacetime-patch DiT architecture combined with flow matching rather than DDPM, reflecting the influence of SD3/FLUX lessons on video.

---

## Core Architecture: Spacetime Patch DiT

The Sora-style architecture decomposes a video $V \in \mathbb{R}^{T \times H \times W \times 3}$ into spatiotemporal patches via a 3D convolution:

```
Video V (T x H x W x 3)
  --> 3D Causal VAE Encoder
  --> Latent L (T/t_c x H/s x W/s x C)    [t_c=4, s=8 typical]
  --> Patchify (p_t x p_h x p_w tokens)
  --> Flatten to sequence of N tokens
  --> Transformer layers (self-attention over all N tokens)
  --> Unpatchify
  --> 3D Causal VAE Decoder
  --> Output video

N = (T/t_c / p_t) * (H/s / p_h) * (W/s / p_w)

For 480p, 5s @ 24fps:
  T=120, H=480, W=854
  After VAE: 30 x 60 x 107 x C
  With patch (1,2,2): N = 30*30*54 = 48,600 tokens
```

The Transformer applies full spatiotemporal self-attention (or efficient approximations — window attention in Wan 2.2, RoPE-3D positional encoding in HunyuanVideo). Text conditioning is injected via cross-attention to encoder embeddings (T5-XXL in most modern models). The 3D causal VAE is critical: the "causal" structure means future frames cannot attend to future latents during encoding, allowing streaming generation and temporal consistency.

---

## Sora (OpenAI, 2024)

OpenAI's Sora [OpenAI2024] uses a DiT operating on spacetime patches of video latent codes. Key properties:

- **Architecture**: Diffusion Transformer (DiT) with variable-resolution and variable-duration spacetime patches; text conditioning via recaptioned DALL-E-style descriptions.
- **Duration**: Up to 60 seconds in initial release; Sora 2 (2026) extends to longer clips with native audio.
- **Spacetime patches**: Patches are 3D tokens spanning multiple frames and spatial regions simultaneously, making the model agnostic to aspect ratio, resolution, and frame rate — no separate encoders for different formats.
- **Physics and 3D consistency**: Sora exhibits emergent 3D consistency, object permanence, and some physical plausibility (fluid dynamics, collisions) without explicit physics simulation, attributed to scale of training data.
- **Training data**: Not publicly disclosed; estimated to involve tens of millions of video clips with synthetic re-captioning for text alignment.

OpenAI published a technical report rather than a full research paper; many architectural details remain undisclosed.

---

## Stable Video Diffusion (SVD)

Stable Video Diffusion [Blattmann2023] adapts Stable Diffusion XL to video by:
1. Training a video-specific 3D U-Net (temporal attention layers inserted into spatial U-Net).
2. Image-to-video conditioning: a given reference frame conditions generation of 14–25 frames at 512×512 to 1024×576.
3. Three-stage curriculum: image pretraining → video pretraining → high-quality video fine-tuning.

SVD does not support text prompts in the base version; motion amplitude is controlled via `motion_bucket_id` and `augmentation_level`. SVD-XT extends output to 25 frames.

---

## Runway Gen-3 Alpha and Kling

**Runway Gen-3 Alpha** (2024) is a proprietary model supporting up to 10-second, 1280×768 clips from text or image prompts, emphasizing cinematic camera motion and director-style control through natural language.

**Kling** (Kuaishou, 2024) generates up to 5-minute videos at 1080p using a diffusion transformer with a proprietary spatial-temporal joint attention mechanism. Kling 1.5/2.0 added "high-consistency" mode for maintaining subject identity across shots. By Kling 3.0 (2026), native 4K output with 60fps upsampling is supported via the Omni One architecture.

---

## Veo 2 and Veo 3

**Veo 2** (Google DeepMind, 2024): Successor to VideoPoet, uses a latent diffusion model with video-specific transformer backbone. Supports 1080p output up to 60 seconds, with camera trajectory control (pan, zoom, tilt) specified in natural language. Demonstrated state-of-the-art FVD on UCF-101 and Kinetics at time of release.

**Veo 3** (2025): Added native audio generation synchronized to video, model architecture details not publicly disclosed. Supports up to 8-second clips at 1080p with audio; extend feature allows multi-minute assembly.

---

## CogVideoX (Tsinghua / Zhipu AI)

CogVideoX [Yang2024] is a fully open-source (Apache 2.0) video generation model built on a 3D causal VAE + Expert Transformer architecture:
- **CogVideoX-5B**: 5B parameters; generates 720×480, 6-second clips at 8fps; English prompts up to 226 tokens; 12GB VRAM minimum.
- **Architecture**: Expert Transformer with 3D RoPE; full-frame (temporal + spatial) attention.
- The model achieved competitive FVD scores on UCF-101 and demonstrated strong text-to-video alignment on VBench.

---

## Movie Gen (Meta, 2024)

Meta's Movie Gen [Polyak2024] is a 30B-parameter foundation model for video and audio generation:
- Text-to-video at 1080p, 16 seconds @ 16fps; audio co-generation model (13B params).
- Trained on ~1 billion video-text pairs; personalized generation via image conditioning.
- Architecture: Transformer-based temporal autoencoder + flow-matching diffusion; outperformed Kling and Runway Gen-3 on human preference evaluations at time of release.
- Not open-source; detailed technical report published.

---

## Wan 2.x (Alibaba)

Wan [WanTeam2025] is a family of open-source video generation models from Alibaba:
- **Wan 2.1** (February 2025): First open-source model to match proprietary quality on several benchmarks; text-to-video and image-to-video at 720p.
- **Wan 2.2** (July 2025): Mixture-of-Experts (MoE) architecture; same VRAM footprint as 2.1; trained on significantly expanded image+video data. Leads open-source models in photorealism for human subjects (facial detail, skin texture, hair).

---

## HunyuanVideo (Tencent)

HunyuanVideo [KongTeam2024] is the largest publicly-released open-source video model at initial release:
- **13B parameters** base; 3D causal VAE compresses video 16× spatially and 4× temporally.
- Architecture: Full Transformer with 3D RoPE; dual-stream text conditioning (CLIP + LLM).
- **HunyuanVideo 1.5** (2025): Trimmed to 8.3B parameters; runs on 14GB VRAM with offloading; excels at physics (fluid dynamics, cloth simulation).
- Human preference evaluations show it competes with closed models on motion quality.

---

## Model Comparison Table

| Model | Organization | Open | Max Length | Max Res | Architecture | Audio | Key Strength |
|-------|-------------|------|-----------|---------|-------------|-------|-------------|
| Sora 2 | OpenAI | No | ~12s | 1080p | DiT spacetime patches | Yes | Photorealism, physics |
| Veo 3.1 | Google DeepMind | No | 8s (+extend) | 1080p | LDM Transformer | Yes | Camera control, audio sync |
| Runway Gen-4 | Runway | No | 10s | 1280×768 | Proprietary DiT | No | Cinematic motion |
| Kling 3.0 | Kuaishou | No | 10s | 4K | Omni One DiT | No | 4K, multi-shot |
| Movie Gen | Meta | No | 16s | 1080p | 30B Flow Transformer | Yes | Longest hi-res, audio |
| HunyuanVideo 1.5 | Tencent | Yes | 13s | 720p | 8.3B DiT + 3D VAE | No | Physics, open weights |
| Wan 2.2 | Alibaba | Yes | 10s | 720p | MoE DiT | No | Photorealistic humans |
| CogVideoX-5B | Zhipu AI | Yes | 6s | 720×480 | 5B Expert Transformer | No | Lightweight, accessible |

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Scalable Diffusion Models with Transformers (DiT) | Peebles & Xie | 2023 | ICCV | Replaced U-Net with Transformer; showed ViT scaling laws apply to diffusion |
| Video generation models as world simulators (Sora) | OpenAI | 2024 | Technical Report | Spacetime patches; variable-res/-dur generation; 60s video |
| Stable Video Diffusion (SVD) | Blattmann et al. | 2023 | arXiv | Image-to-video with temporal U-Net; 3-stage training curriculum |
| CogVideo | Hong et al. | 2022 | ICLR | First large-scale text-to-video pretrained transformer |
| CogVideoX | Yang et al. | 2024 | arXiv | Open expert transformer; 3D causal VAE; competitive FVD |
| Movie Gen | Polyak et al. (Meta) | 2024 | arXiv | 30B model; joint video+audio; 1080p 16s |
| HunyuanVideo | Kong et al. (Tencent) | 2024 | arXiv | 13B open model; dual-stream conditioning; 3D RoPE |
| Wan | Wan Team (Alibaba) | 2025 | arXiv | MoE open model; image+video pretraining; human preference SOTA |
| Emu Video | Girdhar et al. (Meta) | 2023 | CVPR 2024 | Factorized text+image-to-video; temporal consistency |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| Sora | UCF-101 | FVD | not publicly reported | Technical report; no FVD disclosed |
| SVD | UCF-101 | FVD | 242 | 14 frames, 256px |
| CogVideoX-5B | VBench Total | Score | 81.97 | Comprehensive 16-dim benchmark |
| HunyuanVideo | VBench Total | Score | 85.09 | Best open-source at time of release |
| Wan 2.1 | VBench Total | Score | 83.2 | Open model; human subject quality |
| Movie Gen | Human Preference vs Kling | Win Rate | 73% | Meta internal evaluation |
| Veo 2 | EvalCrafter | Overall | not publicly reported | Google evaluation on proprietary benchmark |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| **Spacetime DiT** | Unified architecture handles any resolution/duration; strong temporal consistency; scales well with compute | Quadratic attention over long sequences expensive; long videos require windowed attention approximations |
| **Open-source models** | HunyuanVideo/Wan competitive with closed models; large community; fine-tunable | Lag 6–12 months behind proprietary frontier; 40GB+ VRAM needed at full quality |
| **Flow matching for video** | Faster sampling, better consistency than DDPM; SD3/FLUX lessons transfer | Training instability at very long sequences; motion ODE integration error accumulates over time |
| **Text conditioning** | Natural language control; zero-shot generalization to new concepts | Precise motion, trajectory, or timing control via text is unreliable; "camera zooms slowly" semantics unclear |
| **Proprietary frontier** | Sora/Veo/Movie Gen achieve near-photographic realism with audio | Closed weights; no community fine-tuning; costly API access; black-box failure modes |

---

## Open Problems & Research Gaps

- **Long video coherence**: Maintaining consistent identity, lighting, and scene layout beyond 30 seconds remains unsolved. Memory-augmented architectures and hierarchical latent structures are promising but not yet deployed at scale.
- **Physics accuracy**: Current models exhibit emergent physics plausibility but fail predictably on rigid-body collisions, fluid splashing, and deformable-body dynamics. Integrating differentiable physics simulators or physics-informed losses is an active direction.
- **Camera trajectory control**: Natural language camera instructions ("slowly push in", "180° arc") are poorly grounded. Explicit camera conditioning (extrinsic matrix sequences) is being explored in video models, analogous to ControlNet for images.
- **Efficient long-video generation**: Generating 5+ minute videos at 1080p requires either hierarchical latent representations or streaming architectures that are not yet available in any open model. Memory requirements grow linearly with duration at current token densities.
- **Audio-visual synchronization**: Movie Gen and Veo 3 show that joint audio-video diffusion is feasible, but precise phoneme-viseme alignment and diegetic sound-source binding remain imprecise.
- **Evaluation**: VBench and EvalCrafter capture some dimensions of quality, but no benchmark adequately measures physical realism, 3D consistency, or fine-grained temporal faithfulness. See [World Models](../06_robotics_and_embodied_ai/02_world_models.md) for discussion of video generation as world modeling.
- **Personalization and consistency**: Maintaining subject identity (a specific person, object) across shots without fine-tuning requires robust reference-image conditioning that generalizes better than IP-Adapter-style approaches.

---

## Further Reading

- [OpenAI Sora Technical Report (2024)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Peebles & Xie (2023), "Scalable Diffusion Models with Transformers," ICCV 2023](https://arxiv.org/abs/2212.09748)
- [Kong et al. (2024), "HunyuanVideo: A Systematic Framework For Large Video Generation Model"](https://arxiv.org/abs/2412.03603)
- [Polyak et al. (2024), "Movie Gen: A Cast of Media Foundation Models," Meta arXiv](https://arxiv.org/abs/2410.13720)
- [Yang et al. (2024), "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"](https://arxiv.org/abs/2408.06072)
- [Blattmann et al. (2023), "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets"](https://arxiv.org/abs/2311.15127)
