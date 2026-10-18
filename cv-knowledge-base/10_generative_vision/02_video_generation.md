# Video Generation: From Sora to Open-Source World Simulators

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](./01_diffusion_models.md) | [GANs and VAEs](./00_gans_and_vaes.md) | [World Models](../06_robotics_and_embodied_ai/02_world_models.md) | [Evaluation and Safety](./04_evaluation_and_safety.md)

> **🆕 October 2026 update:**
> - Corrected the Sora 2 timeline (released 30 Sep 2025, not 2026) and added Veo 3.1 (Oct 2025) and Google's Gemini Omni (I/O, May 2026; Omni Flash developer preview Jun 2026).
> - New section on **native audio-video generation** (Veo 3/3.1, Sora 2, Kling 2.6, Wan 2.5, LTX-2) and a section on **Seedance** (ByteDance).
> - New section on **autoregressive, few-step and real-time video** (CausVid, Self Forcing, MAGI-1, Seaweed-APT) and the link to interactive world models (Genie 3).
> - New section on **video models as zero-shot visual reasoners** (Veo 3); comparison table, Key Papers, Open Problems and Further Reading updated.

---

## Overview

Video generation occupies a qualitatively different difficulty tier from image synthesis: temporal coherence, physics plausibility, camera motion, long-range dependencies, and multi-second duration all impose demands that cannot be addressed by independently generating frames. The dominant paradigm prior to 2024 — recurrent networks, video GANs, or autoregressive models over frame sequences — struggled to produce more than a few seconds of visually consistent footage. The release of OpenAI's Sora in early 2024 demonstrated that a single architecture (Diffusion Transformer operating on spacetime patches) trained at scale could generate up to 60 seconds of photorealistic, physically plausible video from text or image prompts, fundamentally changing expectations for the field.

The core architectural insight behind the modern generation of models is the unification of spatial and temporal compression into a shared 3D latent space (via a 3D causal VAE), followed by a Transformer-based diffusion process operating on "spacetime patches" — contiguous 3D tokens spanning a few frames and a spatial tile. Because patches can be extracted from videos of any resolution, duration, and aspect ratio without re-architecture, the same model handles a wide variety of generation tasks. Peebles and Xie [Peebles2023] first demonstrated that replacing the U-Net backbone with a Transformer (DiT — Diffusion Transformer) scaled more cleanly than U-Net on image generation; Sora and its successors extended DiT to the spatiotemporal domain.

As of October 2026, the video generation landscape splits into proprietary closed models (Sora 2, Veo 3.1 and Google's Gemini Omni, Runway Gen-4/4.5, Kling 2.6/3.0, Seedance 1.x/2.0) and a maturing open-weight ecosystem (Wan 2.1/2.2, HunyuanVideo 1.5, LTX-Video/LTX-2, CogVideoX). The defining shift of 2025–2026 is **native, synchronized audio** — dialogue, sound effects and music generated jointly with the frames — pioneered at scale by Veo 3 (May 2025) and Sora 2 (Sep 2025). A second shift is **real-time, autoregressive generation** (Self Forcing, Genie 3), which blurs the line between video generators and interactive world models. Open models still lag the proprietary frontier on motion realism, physics and audio, but the gap is months rather than years. Almost all major systems use a spacetime-patch DiT trained with flow matching rather than DDPM, reflecting the influence of SD3/FLUX lessons on video.

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

The Transformer applies full spatiotemporal self-attention with 3D RoPE positional encoding (HunyuanVideo, Wan), or efficient approximations (sparse / sliding-window attention in long-video and distilled real-time variants). Text conditioning is injected via cross-attention to encoder embeddings (T5-XXL in most modern models). The 3D causal VAE is critical: the "causal" structure means future frames cannot attend to future latents during encoding, allowing streaming generation and temporal consistency.

---

## Sora (OpenAI, 2024)

OpenAI's Sora [OpenAI2024] uses a DiT operating on spacetime patches of video latent codes. Key properties:

- **Architecture**: Diffusion Transformer (DiT) with variable-resolution and variable-duration spacetime patches; text conditioning via recaptioned DALL-E-style descriptions.
- **Duration**: Up to 60 seconds in the 2024 research preview. **Sora 2** (released 30 Sep 2025 together with a social iOS "Sora" app) generates synchronized dialogue and sound effects, follows multi-shot instructions, and introduced consent-based "cameos" (users can insert a verified likeness of themselves); clips were initially 10 s, later extended to 15 s (25 s for Pro storyboards). Sora 2 outputs carry a visible moving watermark plus C2PA metadata (see [Evaluation and Safety](./04_evaluation_and_safety.md)).
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

**Veo 3** (Google I/O, May 2025): Added native audio generation synchronized to video (dialogue with lip-sync, ambient sound, effects); architecture details not publicly disclosed. Supports up to 8-second clips at 1080p with audio; extend feature allows multi-minute assembly.

**Veo 3.1** (Oct 2025): Richer audio, stronger prompt adherence, reference-image ("ingredients") conditioning, first/last-frame interpolation and scene extension, exposed through Google's Flow filmmaking tool and the Gemini API (including a cheaper Veo 3.1 Fast tier).

**Gemini Omni** (Google I/O, May 2026): Google's any-to-video model that turns combinations of text, images and audio into video and supports conversational, multi-turn video editing; a **Gemini Omni Flash** developer preview opened on 30 Jun 2026 (initially 10-second generations in the API). It marks the absorption of video generation into the general multimodal Gemini line, analogous to how Nano Banana absorbed image generation (see [Diffusion Models](./01_diffusion_models.md)).

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
- **Wan 2.2** (July 2025): Mixture-of-Experts (MoE) architecture — the A14B models route *high-noise* denoising steps (global layout) and *low-noise* steps (detail) to two separate ~14B experts, so per-step compute and VRAM stay at the 2.1 level; trained on significantly expanded image+video data with cinematic aesthetic labels. A dense **TI2V-5B** model uses a high-compression 16×16×4 VAE to produce 720p@24fps on a single consumer GPU. Leads open-source models in photorealism for human subjects (facial detail, skin texture, hair).
- **Wan 2.5** (Sept 2025, preview): Adds native audio-visual generation (voice, sound effects, lip-sync); initially offered through Alibaba Cloud rather than as open weights.

---

## HunyuanVideo (Tencent)

HunyuanVideo [KongTeam2024] is the largest publicly-released open-source video model at initial release:
- **13B parameters** base; 3D causal VAE compresses video 16× spatially and 4× temporally.
- Architecture: Full Transformer with 3D RoPE; dual-stream text conditioning (CLIP + LLM).
- **HunyuanVideo 1.5** (2025): Trimmed to 8.3B parameters; runs on 14GB VRAM with offloading; excels at physics (fluid dynamics, cloth simulation).
- Human preference evaluations show it competes with closed models on motion quality.

---

## Native Audio-Video Generation (2025–2026)

The 2025–2026 frontier treats a clip as a joint audio-visual signal. Most systems are closed, but published open designs share a pattern: a video DiT stream and an audio DiT stream (operating on spectrogram or audio-codec latents) exchange information through cross-attention or shared joint-attention blocks, with temporally aligned positional encodings so that lip motion and impact sounds line up.

```
Joint audio-video flow matching (schematic):
  z_v ~ video latent (T_v x H x W x C_v),  z_a ~ audio latent (T_a x C_a)
  x_t = (z_v,t , z_a,t) = (1 - t) * (z_v, z_a) + t * (eps_v, eps_a)
  L = E || v_theta^V(x_t, t, c) - (eps_v - z_v) ||^2
    + lambda_a * E || v_theta^A(x_t, t, c) - (eps_a - z_a) ||^2
  Cross-modal sync: shared time-aligned RoPE so frame k <-> audio window [k*T_a/T_v, (k+1)*T_a/T_v]
```

| System | Released | Audio capability | Notes |
|--------|----------|------------------|-------|
| Veo 3 / 3.1 | May / Oct 2025 | Dialogue, SFX, ambience | First widely deployed joint audio-video model |
| Sora 2 | Sep 2025 | Dialogue, SFX | Cameos (consented likeness); social app distribution |
| Wan 2.5 | Sep 2025 (preview) | Voice, SFX, lip-sync | Alibaba Cloud service |
| Kling 2.6 | Dec 2025 (reported) | Native audio | Kuaishou's first native-audio Kling |
| LTX-2 | Late 2025 (open weights) | Synchronized audio + video | Lightricks; open DiT-based audio-video model targeting consumer GPUs |
| Seedance 1.5 / 2.0 | Late 2025 / Feb 2026 (reported) | Joint audio-video | ByteDance; multimodal references |
| Gemini Omni | May 2026 | Audio as input and output | Any-to-video with conversational editing |

---

## Seedance (ByteDance)

**Seedance 1.0** [Seedance2025] (June 2025) is ByteDance Seed's video foundation model. Its technical report emphasizes (i) multi-source data curation with precise dense video captions, (ii) a DiT with decoupled spatial and temporal layers and interleaved multimodal positional encoding that natively supports **multi-shot** generation (consistent subjects across cuts) for both text-to-video and image-to-video, (iii) video-specific RLHF with multiple reward models, and (iv) multi-stage distillation plus system-level optimizations, reporting roughly 10× inference acceleration (a 5-second 1080p clip in ~41 s on an NVIDIA L20). At release Seedance 1.0 led third-party text-to-video and image-to-video arenas. **Seedance 2.0** (Feb 2026, reported) adds joint audio and multi-reference conditioning on images, video and audio; viral clips featuring celebrities and copyrighted characters triggered public objections from Hollywood studios and industry bodies, illustrating the likeness and copyright issues discussed in [Evaluation and Safety](./04_evaluation_and_safety.md).

---

## Autoregressive, Few-Step and Real-Time Video

Bidirectional DiTs denoise the whole clip jointly, so latency scales with clip length and nothing can be shown until the end. A 2025 line of work converts them into **causal, streaming generators**:

- **CausVid** [Yin2025CausVid] (CVPR 2025) distills a bidirectional teacher into a block-causal autoregressive student with DMD, using KV-caching to stream frames.
- **Self Forcing** [Huang2025SelfForcing] (NeurIPS 2025) trains the autoregressive student on its *own* rolled-out context (rather than ground-truth frames) with a holistic video-level distribution-matching loss, closing the train-test exposure-bias gap and reaching real-time streaming on a single GPU.
- **MAGI-1** (Sand AI, 2025) is an open, chunk-wise autoregressive video diffusion model designed for streaming and unbounded extension.
- **Seaweed-APT** [Lin2025APT] shows that adversarial post-training can produce *one-step* 720p video generation (see [GANs and VAEs](./00_gans_and_vaes.md)).

```
Self Forcing (schematic):
  for chunk k = 1..K:
      x^k ~ G_theta( . | x^{<k}_generated , c )   # few-step denoise, KV cache of own past outputs
  L = D( p_theta(x^{1:K}) || p_data(x^{1:K}) )     # video-level DMD / GAN / SiD objective
  => no teacher forcing; errors seen during training match those at inference
```

The same ingredients underpin **interactive world models**: DeepMind's **Genie 3** (Aug 2025) generates navigable 720p environments at 24 fps in real time with minute-scale consistency and promptable world events, and a broader "Project Genie" access followed in early 2026. These systems are covered in [World Models](../06_robotics_and_embodied_ai/02_world_models.md).

---

## Video Models as Zero-Shot Visual Reasoners

Wiedemer et al. [Wiedemer2025] ("Video models are zero-shot learners and reasoners") prompt Veo 3 with an input image and a text instruction and read the answer from the generated video. Without task-specific training, Veo 3 performs segmentation, edge detection, physical-property reasoning, image editing, and simple visual reasoning such as maze solving and symmetry completion — with large gains over Veo 2. The authors argue that video models may become general-purpose vision foundation models in the same way LLMs did for language ("chain-of-frames" as the analogue of chain-of-thought). Physics-focused benchmarks such as Physics-IQ, however, show that visual realism and physical understanding remain poorly correlated (see [Evaluation and Safety](./04_evaluation_and_safety.md)).

---

## Model Comparison Table

| Model | Organization | Open | Max Length | Max Res | Architecture | Audio | Key Strength |
|-------|-------------|------|-----------|---------|-------------|-------|-------------|
| Sora 2 | OpenAI | No | 10–25s | 1080p | DiT spacetime patches | Yes | Photorealism, physics, cameos |
| Veo 3.1 | Google DeepMind | No | 8s (+extend) | 1080p | LDM Transformer | Yes | Camera control, audio sync, reference images |
| Gemini Omni | Google DeepMind | No | ~10s (Flash API) | n/d | Multimodal Gemini (undisclosed) | Yes | Any-to-video, conversational editing |
| Runway Gen-4 | Runway | No | 10s | 1280×768 | Proprietary DiT | No | Cinematic motion |
| Kling 3.0 | Kuaishou | No | 10s | 4K (vendor-reported) | Omni One DiT (vendor-reported) | Yes (since 2.6) | 4K, multi-shot |
| Seedance 1.0 Pro | ByteDance | No | 10s | 1080p | Multi-shot DiT + RLHF | No (audio in 1.5/2.0) | Multi-shot narrative, speed |
| LTX-2 | Lightricks | Yes | ~10s (reported) | up to 4K (vendor-reported) | Audio-video DiT | Yes | Open audio-video, efficiency |
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
| From Slow Bidirectional to Fast Autoregressive Video Diffusion (CausVid) | Yin et al. | 2025 | CVPR | DMD distillation into block-causal streaming generator |
| Wan: Open and Advanced Large-Scale Video Generative Models | Wan Team (Alibaba) | 2025 | arXiv | Technical report for Wan 2.1 (14B/1.3B); Wan-VAE; open weights |
| Seedance 1.0 | ByteDance Seed | 2025 | arXiv | Native multi-shot T2V/I2V; video RLHF; ~10× distilled inference |
| Self Forcing | Huang et al. | 2025 | NeurIPS | Train AR video diffusion on own rollouts; real-time streaming |
| Diffusion Adversarial Post-Training (Seaweed-APT) | Lin et al. (ByteDance) | 2025 | arXiv | One-step video generation via adversarial post-training |
| Video models are zero-shot learners and reasoners | Wiedemer et al. (Google DeepMind) | 2025 | arXiv | Veo 3 solves perception/manipulation/reasoning tasks zero-shot; "chain-of-frames" |

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
| Seedance 1.0 | Artificial Analysis Video Arena (T2V, I2V) | Rank | #1 at release (Jun 2025) | Third-party human-preference arena; rankings change frequently |
| Self Forcing (Wan-1.3B base) | Streaming latency | Throughput | real-time (~17 fps on one H100, reported) | Sub-second first-frame latency |

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

- **Long video coherence**: Maintaining consistent identity, lighting, and scene layout beyond 30 seconds remains unsolved for cinematic generators. Interactive world models (Genie 3) reach minute-scale consistency at 720p, but via autoregressive memory that still drifts; explicit scene memory (3D caches, retrieval of past frames) versus implicit context remains an open design question.
- **Exposure bias in autoregressive video**: Self Forcing-style training reduces, but does not eliminate, error accumulation over long rollouts; combining real-time causal generation with the quality of bidirectional teachers is open.
- **Physics accuracy**: Current models exhibit emergent physics plausibility but fail predictably on rigid-body collisions, fluid splashing, and deformable-body dynamics. Integrating differentiable physics simulators or physics-informed losses is an active direction.
- **Camera trajectory control**: Natural language camera instructions ("slowly push in", "180° arc") are poorly grounded. Explicit camera conditioning (extrinsic matrix sequences) is being explored in video models, analogous to ControlNet for images.
- **Efficient long-video generation**: Generating 5+ minute videos at 1080p requires either hierarchical latent representations or streaming architectures that are not yet available in any open model. Memory requirements grow linearly with duration at current token densities.
- **Audio-visual synchronization**: Movie Gen, Veo 3/3.1, Sora 2 and LTX-2 show that joint audio-video generation is feasible and commercially viable, but precise phoneme-viseme alignment, multi-speaker dialogue attribution and diegetic sound-source binding remain imprecise, and there is no standard benchmark for audio-visual sync quality.
- **Likeness, consent and copyright**: Consent-gated cameos (Sora 2) and studio objections to Seedance 2.0 outputs show that controlling generation of real people and protected characters is now a first-order product constraint; technical means (identity filters, likeness registries, provenance) are immature.
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
- [OpenAI (2025), "Sora 2 is here"](https://openai.com/index/sora-2/)
- [Google DeepMind, Veo model page](https://deepmind.google/models/veo/)
- [TechCrunch (2026), "Google's Gemini Omni turns images, audio, and text into video"](https://techcrunch.com/2026/05/19/googles-gemini-omni-turns-images-audio-and-text-into-video-and-thats-just-the-start/) and [Google (2026), "Start building with Nano Banana 2 Lite and Gemini Omni Flash"](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-omni-flash-nano-banana-2-lite/)
- [Wan Team (2025), "Wan: Open and Advanced Large-Scale Video Generative Models"](https://arxiv.org/abs/2503.20314) and [Wan 2.2 repository](https://github.com/Wan-Video/Wan2.2)
- [ByteDance Seed (2025), "Seedance 1.0: Exploring the Boundaries of Video Generation Models"](https://arxiv.org/abs/2506.09113)
- [Yin et al. (2025), "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models" (CausVid), CVPR 2025](https://arxiv.org/abs/2412.07772)
- [Huang et al. (2025), "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"](https://arxiv.org/abs/2506.08009)
- [Wiedemer et al. (2025), "Video models are zero-shot learners and reasoners"](https://arxiv.org/abs/2509.20328)
- [Google DeepMind (2025), "Genie 3: A new frontier for world models"](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
