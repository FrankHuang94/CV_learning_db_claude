# 2024–2026 Computer Vision Research: State of the Field

> **Last Updated:** June 2026
> **Level:** Research Frontier
> **Related Sections:**
> - [Robotics & Embodied AI Overview](../06_robotics_and_embodied_ai/00_overview_roadmap.md)
> - [Future Trends](./07_future_trends.md)
> - [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md)
> - [World Models for Robotics](../06_robotics_and_embodied_ai/02_world_models.md)

---

## Overview

The period from 2024 to 2026 represents the most consequential inflection point in computer vision since the ImageNet moment of 2012. The defining shift is not merely one of scale but of *paradigm*: the field has moved from training task-specific models—each excelling at detection, segmentation, depth, or tracking in isolation—toward training universal visual representations that can be prompted, composed, and deployed across an open-ended space of perception and control tasks. Three distinct but converging battlegrounds characterize this moment: (1) **Vision-Language Models (VLMs)** for open-world perception, exemplified by the explosion of CLIP-family models and multi-modal LLMs; (2) **Vision-Language-Action (VLA) models** for physical control, where language-conditioned policies are displacing classical robot pipelines; and (3) **World Models** for simulation and planning, where generative video models are beginning to serve as differentiable physics engines for training embodied agents.

The catalyst for all three battlegrounds is a shared infrastructure advance: the emergence of very large, openly available foundation models that can be fine-tuned or composed for downstream tasks at a fraction of the original training cost. DINOv2 [Oquab2023] established that purely self-supervised ViT encoders trained on curated large-scale image data could match or exceed supervised representations across nearly every vision benchmark. SAM [Kirillov2023] demonstrated that a single promptable segmentation model trained on 1 billion masks could generalize zero-shot to domains its creators never imagined. SAM2 [Ravi2024] extended this to video, enabling real-time prompted tracking through occlusion. Depth Anything V2 [Yang2024] showed that monocular depth—long considered intractable without metric-scale supervision—could be solved to near-lidar quality by combining labeled data with massive pseudo-labeled unlabeled imagery. Collectively, these models define the *any-model paradigm*: a new class of foundation models that operate across arbitrary inputs via lightweight prompting rather than task-specific retraining.

The economic context cannot be ignored. The year 2024–2025 saw more than $10 billion invested in robotics and embodied AI startups globally, with robotics-focused funding in 2025 alone exceeding the full-year 2024 total. Physical Intelligence (pi.ai) raised a $600M Series B, valuing the company at $5.6B, on the strength of its π₀ [BlackEtAl2024] generalist robot policy. Figure AI completed a $1B Series C at a $39B valuation. NVIDIA placed a comprehensive platform bet through Isaac GR00T N1 [NVIDIA2025], the first openly released humanoid robot foundation model, and the Cosmos world foundation model family for physics-aware synthetic data generation. These investments reflect a consensus that general-purpose robotic intelligence—long regarded as decades away—is now a near-term engineering problem, not a fundamental science problem, contingent on getting visual representations and world models right.

---

## The Three Battlegrounds

### 1. VLMs for Open-World Perception

The CLIP-family has undergone dramatic scaling. EVA-CLIP-18B [Sun2024], with 18 billion parameters, is the largest openly released contrastive vision-language model and achieves 80.7% zero-shot top-1 accuracy averaged across 27 classification benchmarks. SigLIP 2 [Zhai2025], Google's successor to the sigmoid-loss contrastive encoder, adds multilingual support (109 languages), dense feature improvements, and strong open-vocabulary localization. DFN-5B—trained on a 5B-image filtered subset of 43B noisy web pairs via Data Filtering Networks [Fang2023]—showed that filtering quality dominates raw quantity. MetaCLIP [Xu2024] demonstrated that curating training data against CLIP's own concept distribution—rather than opaque proprietary filtering—allows 72.4% ImageNet zero-shot at 1B scale while beating OpenAI CLIP.

The emergence of multi-modal large language models (MLLMs) that fuse strong language model backbones with visual encoders has created a new SOTA tier: GPT-4V, Gemini 1.5 Pro, Claude 3.5, InternVL, LLaVA-NeXT, and Qwen-VL all post above 80% on MMMU (Massive Multi-discipline Multimodal Understanding), a benchmark that was near-random for CLIP-family models. The key architectural pattern is a frozen or lightly tuned vision encoder (typically DINOv2 + SigLIP) feeding into a large language model through a learned projector (MLP or Q-Former).

### 2. VLAs for Physical Control

Vision-Language-Action models represent the most direct application of the VLM revolution to robotics. The canonical architecture takes language instruction tokens and image tokens as input and outputs continuous or discretized robot actions. RT-2 [BrohmanEtAl2023] was the inflection point, showing that co-fine-tuning a 55B PaLI-X VLM on robot trajectory data produced emergent reasoning—the robot could navigate by counting objects described in a scene. OpenVLA [KimEtAl2024], a 7B-parameter open-source alternative built on Llama 2 with DINOv2+SigLIP visual encoder, demonstrated that open-source VLAs trained on Open X-Embodiment (970K trajectories) could outperform RT-2 on standard manipulation benchmarks.

π₀ [BlackEtAl2024] by Physical Intelligence pushed further: a flow-matching diffusion VLA trained on diverse dexterous robot hardware (single-arm, dual-arm, mobile manipulators) using PaliGemma 2B as the language backbone. It produces fluid, compliant motions that previous autoregressive discrete-action VLAs could not. Octo [GhoshEtAl2024], published at RSS 2024, is an open-source transformer policy trained on 800K trajectories that can be fine-tuned to new robot platforms in hours on a single GPU. GR00T N1 [NVIDIA2025], released March 2025, introduces a dual-system architecture: a slow VLM "System 2" for reasoning over language and scene, feeding a fast diffusion-transformer "System 1" for real-time action generation; the 2.2B-parameter model can sample 16 actions in 63.9 ms on an L40 GPU.

### 3. World Models for Simulation and Planning

World models—generative models of environment dynamics that can be used for planning and data augmentation—crossed a threshold of visual realism and physical plausibility in 2024–2025. Genie 2 [ParmarEtAl2024], announced by Google DeepMind in December 2024, uses an autoregressive latent diffusion model to generate interactive 3D environments from a single image, maintaining physical consistency for up to one minute, enabling rapid prototyping of RL environments. NVIDIA Cosmos [NVIDIA2025], launched at CES 2025 and updated through CoRL 2025, provides a family of open world foundation models for physics-aware video prediction conditioned on text, image, video, and robot action data; Cosmos Predict 2.5 enables multi-view outputs and up to 30-second video generation. V-JEPA 2 [Meta2025] takes a non-generative route: a 1.2B-parameter latent-space predictive model pre-trained on 1 million hours of video that achieves 77.3% top-1 on Something-Something v2 and enables zero-shot robot planning via model-predictive control in latent space.

---

## The Open-Source Moment

A defining characteristic of 2024–2026 is that the most impactful models are openly released:

| Model | Type | Release | Parameters |
|---|---|---|---|
| DINOv2 [Oquab2023] | Self-supervised ViT | Apr 2023 | 1.1B (ViT-g) |
| SAM [Kirillov2023] | Promptable Segmentation | Apr 2023 | ~630M |
| SAM2 [Ravi2024] | Video Segmentation | Aug 2024 | ~230M–450M |
| Depth Anything V2 [Yang2024] | Monocular Depth | Jun 2024 | 25M–1.3B |
| OpenVLA [KimEtAl2024] | Vision-Language-Action | Jun 2024 | 7B |
| Octo [GhoshEtAl2024] | Robot Policy | May 2024 | 93M |
| GR00T N1 [NVIDIA2025] | Humanoid VLA | Mar 2025 | 2.2B |
| V-JEPA 2 [Meta2025] | World Model | Jun 2025 | 1.2B |

This open-source moment is not altruistic—it reflects a strategic bet that the value in this stack lies in the platform and deployment layer, not the model weights themselves. Meta, NVIDIA, and Google DeepMind release weights to cultivate ecosystems; Physical Intelligence and Figure AI keep their best models proprietary to defend competitive moats in hardware-integrated deployment.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|---|---|---|---|---|
| Segment Anything (SAM) | Kirillov et al. | 2023 | ICCV | Promptable segmentation at scale; SA-1B dataset (1B masks, 11M images) |
| SAM 2: Segment Anything in Images and Videos | Ravi et al. | 2024 | ICLR 2025 | Extends SAM to video with streaming memory; 6× faster than SAM on images |
| Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data | Yang et al. | 2024 | CVPR | Semi-supervised depth foundation model; 62M+ unlabeled images |
| Depth Anything V2 | Yang et al. | 2024 | NeurIPS | Synthetic data replaces real labels; 97.1% on DA-2K; 10× faster than diffusion methods |
| DINOv2: Learning Robust Visual Features without Supervision | Oquab et al. | 2023 | TMLR | Self-supervised ViT features that match supervised on all benchmarks |
| OpenVLA: An Open-Source Vision-Language-Action Model | Kim et al. | 2024 | CoRL | 7B open-source VLA on 970K robot demos; outperforms RT-2 |
| Octo: An Open-Source Generalist Robot Policy | Ghosh et al. | 2024 | RSS | 93M transformer policy on 800K trajectories; fine-tunable in hours |
| π₀: A Vision-Language-Action Flow Model | Black et al. | 2024 | arXiv | Flow-matching diffusion VLA; multi-robot dexterous manipulation |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | NVIDIA | 2025 | arXiv | Dual-system VLA for humanoids; 40% gain with synthetic+real data |
| V-JEPA 2 | Meta | 2025 | arXiv | 1.2B latent world model; 77.3% SSv2; zero-shot robot planning |
| EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters | Sun et al. | 2024 | arXiv | Largest open CLIP; 80.7% avg zero-shot on 27 benchmarks |
| SigLIP 2 | Zhai et al. | 2025 | arXiv | Multilingual CLIP (109 langs); dense feature improvements |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|---|---|---|---|---|
| SAM2-L | SA-V (video) | J&F | ~82 | Zero-shot; 3× fewer clicks than prior VOS methods |
| SAM2-L | MOSE | J&F | 76.4 | Zero-shot on challenging occluded objects |
| Depth Anything V2-L | DA-2K | Accuracy | 97.1% | vs. Marigold ≤88.1%; 10× faster |
| EVA-CLIP-18B | ImageNet-1K | Zero-shot Top-1 | 80.7% (avg 27 datasets) | 18B params |
| DINOv2 (ViT-g) | ImageNet-1K | Linear probe Top-1 | 86.5% | 1.1B params, no labels |
| OpenVLA-7B | BridgeV2 / RT-2 benchmark | Task success | Outperforms RT-2 | vs. 55B RT-2 |
| Octo | WidowX manipulation | Task success | 25% above RT-1-X | Language-conditioned |
| V-JEPA 2 | Something-Something v2 | Top-1 accuracy | 77.3% | Attentive probe, no fine-tune |
| GR00T N1 | Sim manipulation (multi-embodiment) | Task success | SOTA vs. IL baselines | +40% with synthetic data |
| SigLIP 2 B/16 (256px) | ImageNet-1K | Zero-shot Top-1 | 79.1% | vs. SigLIP 76.7% |

---

## Pros & Cons

| Aspect | Pros | Cons |
|---|---|---|
| Universal Foundation Models | Single pretrained backbone covers detection, segmentation, depth, tracking—dramatically reducing per-task R&D | Catastrophic forgetting and task interference when fine-tuning on narrow domains; very large compute footprint |
| VLA Paradigm | Enables natural-language task specification; generalizes across robot morphologies; benefits from internet-scale pretraining | Action space tokenization remains lossy; low-frequency control limits fine dexterity; safety is poorly characterized |
| Open-Source Ecosystem | Accelerates research; reproducibility; community contributions extend models to new domains | Quality gap between open and proprietary closed models in production settings; no standardized safety audit process |
| World Models for RL/Planning | Unlimited synthetic data; safe exploration; latent MPC enables planning without explicit rollouts | Physical plausibility breaks down over long horizons; sim-to-real gap persists; photorealistic models are compute-prohibitive |
| Data Engines (SAM, Depth Anything) | Bootstrap labeled datasets at internet scale; reduce annotation cost by orders of magnitude | Pseudo-label noise compounds; domain gaps between synthesized and real-world distribution remain hard to quantify |

---

## Open Problems & Research Gaps

1. **Long-horizon task generalization in VLAs.** Current VLA models excel at 5–30 second manipulation primitives but fail on tasks requiring minutes of coherent multi-step execution. Neither π₀ nor OpenVLA have demonstrated reliable tabletop-to-full-kitchen task completion without human intervention.

2. **Physics fidelity of world models.** Genie 2, Cosmos, and V-JEPA 2 generate visually plausible futures but do not yet accurately model contact forces, deformable objects, or fluid dynamics. Using these models as drop-in physics engines for dexterous manipulation training remains an open problem.

3. **Compositional generalization.** VLMs can describe scenes containing novel object combinations but fail at compositional spatial reasoning at the 3D level required for manipulation planning (e.g., "put the blue cup inside the red bowl that is to the left of the stack"). Neuro-symbolic hybrids remain under-explored.

4. **Efficient adaptation with minimal demonstrations.** Few-shot and one-shot adaptation to new tasks and robot morphologies is not solved. Octo and OpenVLA both require hundreds of fine-tuning trajectories to reliably specialize; this is impractical for rare-task deployment.

5. **Safety and uncertainty quantification.** No current VLA or world model framework provides calibrated confidence scores or provable safety certificates. Deploying these systems in close proximity to humans requires addressing both epistemic uncertainty (novel scenes) and aleatoric uncertainty (contact ambiguity).

6. **Standardized evaluation.** The field lacks a universally accepted benchmark analogous to ImageNet for embodied intelligence. LIBERO, BridgeV2, and Open X-Embodiment partially fill this gap but do not cover mobile manipulation, human-robot collaboration, or long-horizon tasks.

7. **Annotation-free continual learning.** Foundation models are static after training; adapting to distribution shift from a live deployment without labeled feedback or human annotation is essentially unsolved despite being the norm in real-world deployment.

---

## Further Reading

- [SAM2 paper (arXiv:2408.00714)](https://arxiv.org/abs/2408.00714) — Ravi et al. 2024
- [V-JEPA 2 paper (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985) — Meta 2025
- [GR00T N1 paper (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734) — NVIDIA 2025
- [Physical Intelligence π₀ blog post](https://www.pi.website/blog/pi0) — Black et al. 2024
- [NVIDIA Cosmos overview](https://www.nvidia.com/en-us/ai/cosmos/) — NVIDIA 2025
- [OpenVLA project page](https://openvla.github.io/) — Kim et al. 2024
