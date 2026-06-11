# Vision-Language-Action (VLA) Models

> **Last Updated:** June 2026
> **Level:** Research Frontier
> **Related Sections:**
> - [Embodied AI Overview and Roadmap](./00_overview_roadmap.md)
> - [Multimodal Vision-Language Models](../05_multimodal_vision_language/00_overview.md)
> - [Research Frontier 2024–2026](../12_research_frontier_2024_2026/04_vla_embodied_2025_2026.md)
> - [Datasets and Benchmarks](../11_datasets_and_benchmarks/00_overview.md)

---

## Overview

Vision-Language-Action (VLA) models are neural architectures that condition robot control policies jointly on visual observations and natural-language instructions, generating actions directly as model outputs. The defining characteristic of a VLA is the co-location of perception, language understanding, and action generation within a single differentiable computation graph—as opposed to modular pipelines that treat these as separate subsystems. This design choice is motivated by the hypothesis that large-scale pretraining on Internet vision-language data produces representations that transfer beneficially to robot control [Brohan2023], and by the empirical observation that emergent capabilities (multi-step reasoning, novel-object generalisation) appear in VLAs trained on web data but not in purely robot-trained policies.

Architecturally, every VLA shares three components. First, a **vision encoder** maps one or more RGB images (and sometimes depth, wrist cameras, or proprioceptive readings) into a sequence of visual tokens. Second, a **language model backbone** processes the interleaved visual and text tokens, producing a rich contextual representation. Third, an **action head** maps the final hidden states of the backbone to robot actions. The design choices that differentiate VLAs are: (a) the scale and pretraining regime of the backbone, (b) the architecture and output format of the action head, and (c) the data pipeline that bridges robot trajectories with Internet-scale pretraining data.

The field has moved rapidly. RT-1 [Brohan2022] (late 2022) trained a compact transformer from scratch on 130k in-house demonstrations, achieving impressive results within a fixed kitchen environment but no web pretraining. RT-2 [Brohan2023] (mid-2023) demonstrated that co-fine-tuning a 55B vision-language model on robot data transferred Internet knowledge—including object concepts and chain-of-thought reasoning—to robot policies, more than doubling success on unseen objects relative to RT-1. The 2024–2025 generation introduced flow-matching action heads (π₀ [Black2024]), open-weight models (OpenVLA [Kim2024]), and humanoid-specific dual-system architectures (GR00T N1 [NVIDIA2025]). By mid-2026, VLAs are being deployed in commercial settings—BMW factories, domestic pilots—and the research frontier has shifted to long-horizon generalisation, on-device inference efficiency, and multi-robot coordination.

---

## Architecture Deep-Dive

### Vision Encoder

Early VLAs (RT-1) used FiLM-conditioned EfficientNet-B3 to extract 81 image tokens per frame, conditioning the visual features on language embeddings at each residual layer. This produced task-relevant representations without requiring a pretrained VLM backbone. Modern VLAs instead reuse the vision encoder from a pretrained VLM: SigLIP-400M in PaliGemma (used by π₀ [Black2024]), the fused DINOv2 + SigLIP encoder in OpenVLA [Kim2024], and SigLIP-2 in GR00T N1's Eagle-2 backbone [NVIDIA2025]. The advantage is that these encoders already represent semantic object categories, affordances, and spatial relationships learned from billions of image-caption pairs, reducing the amount of robot-specific visual learning required.

Multi-camera setups are the norm for dexterous manipulation: a head/wrist camera provides global context while a wrist camera provides close-up task-relevant detail. GR00T N1 and π₀ both consume multi-view RGB streams. Camera tokens are typically linearised and prepended to the language tokens before being fed into the transformer backbone.

### Language Model Backbone

The backbone provides the "reasoning engine" of the VLA. Its scale has grown substantially:

| Era | VLA | Backbone | Params |
|-----|-----|----------|--------|
| 2022 | RT-1 | Custom Transformer | 35M |
| 2023 | RT-2 | PaLI-X / PaLM-E | 12B–55B |
| 2024 | OpenVLA | Llama 2 (Prismatic-7B) | 7B |
| 2024 | π₀ | PaliGemma | 3B |
| 2025 | GR00T N1 | Eagle-2 (full) | 34B |
| 2025 | GR00T N1 (released) | Eagle-2 (distilled) | 2.2B |

The backbone processes a flat token sequence: `[image tokens] [task description tokens] [history tokens]`, and its output at the final layer is used by the action head. For discrete-action VLAs (RT-2, OpenVLA), the backbone's language modelling head is repurposed directly to predict action tokens. For continuous-action VLAs (π₀, GR00T N1), the backbone produces a latent vector that conditions a separate generative action head.

### Action Head Design

This is the most architecturally diverse component. Three paradigms dominate:

#### Discrete Tokenization (RT-2, OpenVLA)

Each action dimension is discretised into 256 bins spanning the action range, and each bin maps to a token in the existing LLM vocabulary (tokens are "hijacked" from the least-used vocabulary entries). The backbone then performs autoregressive prediction over action tokens as if they were text tokens, keeping the training procedure identical to standard next-token prediction.

```
# RT-2 / OpenVLA action tokenization
action_range = [-1, 1]   # normalised per dimension
n_bins = 256
bin_id = floor((action + 1) / 2 * n_bins)  # integer in [0, 255]
token_id = vocab_offset + bin_id           # maps into LLM vocabulary
```

**Advantage:** No architectural change; leverages LLM world knowledge for in-context generalisation.  
**Disadvantage:** Lossy discretisation; independent per-dimension sampling ignores action correlations; repurposed tokens may interfere with language representations.

#### Continuous Regression (Octo, ACT)

A lightweight MLP or transformer head maps the backbone's output latent directly to a continuous action vector (or a short action chunk). Typically trained with mean-squared error or behavioural cloning.

```
# Action chunk prediction (ACT-style)
action_chunk = MLP(z_backbone)  # shape: (H, action_dim)
loss = MSE(action_chunk, demonstration_chunk)
```

**Advantage:** Exact continuous outputs; fast inference.  
**Disadvantage:** Unimodal—cannot represent bimodal or multimodal action distributions, causing the "averaging" failure mode on demonstrations that contain multiple valid strategies.

#### Flow Matching (π₀, GR00T N1)

Flow matching [Lipman2022] defines a vector field `v_θ(x_t, t)` that transports samples from a source distribution (Gaussian noise) to the target action distribution along straight-line probability flows. The model is trained by regressing the predicted flow to the conditional flow:

```
# Flow matching training objective
# x_0 ~ N(0, I): noise sample
# x_1: target action from demonstration
# x_t = (1 - t) * x_0 + t * x_1   (linear interpolation, t ~ U[0,1])
# Condition c: visual + language latent from backbone

loss = ||v_θ(x_t, t, c) - (x_1 - x_0)||^2

# Inference: integrate ODE from t=0 to t=1
# dx/dt = v_θ(x_t, t, c)
# In practice, 10 Euler steps suffice for π₀
```

Because flow matching samples the entire action chunk `(H, action_dim)` simultaneously, it captures temporal correlations across the chunk and handles multimodal distributions without mode averaging. π₀ generates H=50 future actions at once, at 50 Hz, giving 1 second of lookahead. The inference time of ~73 ms on a consumer GPU makes this viable for real-time control.

**Advantage:** Continuous, multimodal, temporally coherent action chunks; no discretisation artefacts; fast sampling (single ODE solve, ~10 steps).  
**Disadvantage:** Slightly higher inference latency than one-step regression; requires an additional action expert module (~300M parameters for π₀).

---

## VLA Inference Stack

```mermaid
graph TD
    subgraph Sensors
        CAM1[Head Camera\nRGB 224×224]
        CAM2[Wrist Camera\nRGB 224×224]
        PROP[Proprioception\nJoint angles · velocities]
        LANG[Natural Language\nTask Instruction]
    end

    subgraph VisionEncoder ["Vision Encoder (SigLIP / DINOv2+SigLIP)"]
        VE[Patch Tokenizer\n→ Visual Tokens N×D]
    end

    subgraph LMBackbone ["Language Model Backbone (Llama-2 7B / PaliGemma 3B / Eagle-2 34B)"]
        TOK[Tokenize Instruction]
        CONCAT[Concatenate Tokens\n[visual | text | proprioception]]
        ATTN[Transformer Layers\nCausal + Cross-Attention]
        LATENT[Backbone Latent z\nshape: D_model]
    end

    subgraph ActionHead ["Action Head"]
        DISC[Discrete Path\nNext-token prediction\nRT-2 · OpenVLA]
        FLOW[Flow Matching Path\nODE integration over H steps\nπ₀ · GR00T N1 System 1]
    end

    subgraph Output
        ACT[Action Chunk\nH × action_dim\ne.g., 50 × 7]
        ROBOT[Robot Controller\nJoint-space or EE-space]
    end

    CAM1 --> VE
    CAM2 --> VE
    LANG --> TOK
    VE --> CONCAT
    TOK --> CONCAT
    PROP --> CONCAT
    CONCAT --> ATTN
    ATTN --> LATENT
    LATENT --> DISC
    LATENT --> FLOW
    DISC --> ACT
    FLOW --> ACT
    ACT --> ROBOT

    style FLOW fill:#2d6a4f,color:#fff
    style DISC fill:#1d3557,color:#fff
```

---

## Model-by-Model Technical Summaries

### RT-1 (Brohan et al., 2022, CoRL / RSS 2023)

RT-1 is the first demonstration that a compact (35M parameter) transformer, trained at scale on real robot demonstrations, can achieve broad task coverage within a fixed environment. The architecture applies FiLM conditioning to inject language embeddings into each residual block of an EfficientNet-B3 visual encoder, producing 81 language-conditioned visual tokens per frame. A TokenLearner module then compresses these to 8 tokens via learned weighted pooling, dramatically reducing the sequence length fed to the 8-layer transformer backbone. Six image frames are concatenated (6 × 8 = 48 tokens total), and the transformer outputs an 11-dimensional discrete action token (3D translation, 3D rotation, gripper, base x/y/yaw, episode termination), each discretised to 256 bins.

**Training data:** 130,000 episodes collected over 17 months with 13 EDR robots across a kitchen environment. Natural language instructions were annotated per episode.

**Results:** 97% success on 700+ seen instructions; 76% on unseen instructions (+24% over best baseline); 83% with distractor objects; 59% with background changes.

**Key limitation:** No Internet pretraining; brittle to novel objects, new scenes, or embodiment changes; language conditioning is FiLM-based (early fusion) rather than attention-based, limiting compositional generalisation.

---

### RT-2 (Brohan et al. / Zitkovich et al., 2023, CoRL)

RT-2 is the pivotal paper establishing VLAs as a paradigm. It co-fine-tunes two pretrained vision-language models—PaLI-X (5B and 55B versions) and PaLM-E-12B—on a mixture of robot trajectory data and the original Internet VLM training tasks (visual question answering, image captioning). Actions are represented as text tokens: each action dimension is discretised to 256 bins, and the bins are appended to the VLM's vocabulary as special tokens.

The co-fine-tuning objective is standard next-token prediction over both text and action tokens, making robot trajectories indistinguishable from text sequences to the optimiser. This allows the model to maintain its language and visual understanding while acquiring action generation capability.

**Emergent behaviours:** RT-2 exhibits multi-step chain-of-thought reasoning (e.g., can be prompted to reason about which object is recyclable before picking it), and transfers physical/semantic knowledge from web pretraining (e.g., correctly interprets "pick up the lion toy" when no "lion" appeared in robot training data).

**Results:** Both RT-2-PaLI-X-55B and RT-2-PaLM-E-12B achieve ~62% average success on unseen-object tasks, compared to ~32% for RT-1 baselines. On the Language Table simulation benchmark, RT-2 achieves 90% success vs. 77% for prior state-of-the-art.

**Key limitation:** Discrete action tokenisation introduces lossy quantisation and treats each dimension independently; model weights not released; inference requires serving a 55B model.

---

### Open X-Embodiment / RT-X (Open X-Embodiment Collaboration, 2023, ICRA 2024 — Best Paper)

RT-X is not a single model but a collaborative dataset and model family assembled from 22 different robot embodiments across 21 institutions, covering 527 distinct skills and 160,266 task episodes. The central hypothesis is that cross-embodiment data provides a form of regularisation and coverage that single-embodiment datasets cannot, and that policies trained on this mixture exhibit "positive transfer" — better performance on any single embodiment than a model trained only on that embodiment's data.

Two model variants are released (RT-1-X and RT-2-X), both trained on the full Open X-Embodiment mixture. OpenVLA later demonstrated that a 7B open-weight model trained on the same mixture matches the performance of RT-2-X on several benchmarks [Kim2024].

**Key insight:** Cross-embodiment training can substitute for scale in data-scarce settings; a robot that has seen laundry folding on a Franka arm can transfer some of that structure to a different arm.

---

### π₀ (Black et al., 2024, Physical Intelligence)

π₀ is the first VLA to demonstrate competitive performance on genuinely dexterous manipulation tasks—laundry folding, bag unzipping, tabletop tidying—that require smooth, high-frequency, multi-joint coordination rather than discrete pick-and-place actions. The key architectural innovation is the **action expert**: a separate transformer module of ~300M parameters that processes the PaliGemma backbone's output latent alongside a flow-matching noise vector and outputs a continuous action chunk.

The backbone is PaliGemma (3B parameters, Google's open VLM built on SigLIP + Gemma), which provides the visual and language understanding. The action expert is initialised randomly and trained jointly with the backbone (with the backbone fine-tuned at a lower learning rate). During inference, flow matching is solved via 10 Euler integration steps over the ODE `dx/dt = v_θ(x_t, t, c)`, producing H=50 future actions simultaneously. This yields ~73 ms inference latency on consumer hardware, sufficient for real-time control.

**Training data:** Over 10,000 hours of robot trajectory data collected from multiple robot platforms (single-arm, dual-arm, mobile manipulators), with natural language instruction annotations.

**Results (independent benchmark, [benchmarkpaper2024]):**
- In-distribution (4-task macro avg): ~72% success
- Spatial OOD: ~67% (−5 pp)
- Unseen objects (instance + spatial OOD): ~59% (−13 pp)
- Comparison: ACT achieves 48% in-distribution, drops to ~6% on unseen objects

**Key limitation:** Training data is proprietary and not released; the model itself is not fully open-weight; inference latency (73 ms) precludes operation above ~14 Hz without speculative decoding or distillation.

---

### OpenVLA (Kim et al., 2024, CoRL)

OpenVLA is the first fully open-weight, open-data VLA matching the performance of RT-2 on standard benchmarks. The backbone is Prismatic-7B, which pairs Llama 2 (7B) with a fused vision encoder combining DINOv2-ViT-L (1.1B) and SigLIP-ViT-So400M (400M). The fused encoder produces richer visual representations than either encoder alone, combining DINOv2's spatial/geometric features with SigLIP's semantic language-aligned features. Actions are discretised to 256 bins per dimension (7 dimensions for a 7-DoF manipulator) and predicted autoregressively, matching RT-2's output format.

The model is trained on the full Open X-Embodiment dataset (970k trajectories), making it the most broadly trained open-weight robot policy as of its release. Both model weights and training code are released under permissive licences.

**Results:**
- Google Robot tasks: 85.0 ± 4.6% mean success (vs. RT-2-X: 78.3 ± 5.4%)
- BridgeV2 (zero-shot): 56–65% success on unseen tasks
- Both OpenVLA and RT-2-X significantly outperform RT-1-X and Octo on these benchmarks

**Key limitation:** Discrete action tokenisation limits dexterity; 7B parameter backbone requires a GPU with ≥20 GB VRAM for inference; operates at single-step predictions rather than action chunks.

---

### GR00T N1 (NVIDIA, March 2025, arXiv)

GR00T N1 introduces a dual-system architecture explicitly inspired by the System 1 / System 2 distinction in cognitive science. System 2 (slow, deliberate) is Eagle-2, a 34B-parameter VLM (SigLIP-2 vision encoder + SmolLM2 language backbone), running at ~10 Hz to interpret multi-view RGB observations and natural language instructions and produce a high-level context vector. System 1 (fast, automatic) is a Diffusion Transformer (DiT) motor controller running at 50+ Hz that generates fluid, continuous joint-position trajectories conditioned on the System 2 context vector and the current proprioceptive state.

The DiT action decoder uses adaptive layer normalisation to condition on both the diffusion timestep and the System 2 context, allowing it to generate embodiment-specific actions via a learned linear projection head (the "Action Decoder") that maps latent trajectories to joint-position commands for variable-DoF humanoids (10-DoF bimanual arms to 50+ DoF full-body systems).

**Training data:** 50,000+ robot trajectories + 3 million egocentric human video clips + synthetic Isaac Sim trajectories. NVIDIA's synthetic data pipeline (GR00T Blueprint) can generate 780k trajectories in 11 hours.

**Results:** +40% task success improvement over baseline when combining synthetic and real training data (on NVIDIA's internal benchmark). The released 2.2B-parameter variant (1.34B VLM + ~860M DiT) is available on Hugging Face.

**Key strength:** Explicitly designed for humanoid whole-body control; publicly released model weights; strong synthetic data infrastructure; operates at realistic whole-body frequencies.

---

### π₀.5 (Hejna et al., Physical Intelligence, April 2025, arXiv)

π₀.5 extends π₀ to achieve meaningful generalisation to entirely novel home environments unseen during training. The core training change is **heterogeneous co-training**: the model is trained simultaneously on robot manipulation trajectories, high-level semantic prediction tasks (predicting what action to take next given an image), and web-scale vision-language data. Ablations show that web data is the single most important factor for out-of-distribution object generalisation, while cross-robot data is critical for general task competence.

Architectural changes from π₀ include replacing MLP-based diffusion timestep fusion with AdaRMSNorm (adaptive RMS normalisation), and discretising the robot state into 256 bins over `[-1, 1]` and incorporating it as discrete tokens in the language prefix (rather than as raw floats), which reportedly stabilises training.

**Key result:** π₀.5 successfully performs household manipulation tasks in entirely new homes (novel furniture, novel objects, novel lighting) with no environment-specific fine-tuning, a capability that previous VLAs could not demonstrate reliably.

---

### Helix (Figure AI, 2025)

Helix is Figure AI's proprietary VLA for whole-body humanoid dexterity on the Figure 02 platform. Its distinguishing features are (a) running entirely on onboard embedded low-power GPUs (no cloud inference), (b) outputting simultaneous continuous control of the full humanoid upper body—wrists, torso, head, and individual fingers at high rate, and (c) enabling two-robot coordination, where two Helix-equipped robots share a long-horizon manipulation task involving objects neither robot has encountered before.

Helix was developed entirely in-house after Figure AI ended its partnership with OpenAI in 2025 to build a fully proprietary AI stack. Commercial deployment at BMW's Spartanburg, South Carolina plant has logged 1,250+ runtime hours and 90,000+ parts loaded across 30,000 vehicles.

Benchmark numbers for Helix have not been publicly reported against standard academic benchmarks.

---

## Comparative Model Table

| Model | Backbone | Action Head | Data Scale | Open Source | Key Strength | Key Weakness |
|-------|----------|-------------|------------|-------------|--------------|--------------|
| RT-1 | FiLM-EfficientNet + custom Transformer (35M) | Discrete 256-bin tokens, 11-dim | 130k episodes, 13 robots, 17 months | No | High success on known tasks (97%); efficient inference at 3 Hz | No web pretraining; brittle to novel objects/scenes |
| RT-2 | PaLI-X 55B or PaLM-E 12B | Discrete action tokens (VLM vocab) | Robot data + Internet VLM data | No | Emergent chain-of-thought; 2× improvement on unseen objects vs. RT-1 | 55B inference cost; discrete actions; proprietary weights |
| OpenVLA | Prismatic-7B (Llama-2 + DINOv2 + SigLIP) | Discrete 256-bin per-dimension tokens | 970k episodes (Open X-Embodiment) | Yes (weights + code) | First open-weight VLA matching RT-2; strong benchmark performance | Discrete action head; no action chunking; 7B inference memory |
| π₀ | PaliGemma 3B | Flow matching action expert (~300M params), H=50 chunk | 10,000+ hrs proprietary multi-robot data | Partial (architecture described; weights not fully released) | Dexterous high-frequency manipulation; multimodal action distribution; 50 Hz | Proprietary data; 73 ms inference latency; limited open-world generalisation |
| GR00T N1 | Eagle-2 (34B full / 2.2B released) | Diffusion Transformer (DiT) + embodiment-specific Action Decoder | 50k trajectories + 3M egocentric videos + Isaac Sim synthetic | Yes (2.2B variant on HuggingFace) | Humanoid whole-body control; strong sim-to-real pipeline; variable-DoF support | Full 34B model not released; benchmark numbers limited to internal evaluation |

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| RT-1: Robotics Transformer for Real-World Control at Scale | Brohan, Brown et al. | 2022 | CoRL / RSS 2023 | First large-scale Transformer robot policy; 130k demo training; 97% on 700+ tasks |
| RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | Brohan/Zitkovich et al. | 2023 | CoRL | VLM co-fine-tuning for robot control; action-as-text-token paradigm; emergent chain-of-thought |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | Open X-Embodiment Collaboration | 2023 | ICRA 2024 (Best Paper) | 22-embodiment, 527-skill cross-embodiment dataset; positive transfer across robots |
| OpenVLA: An Open-Source Vision-Language-Action Model | Kim et al. | 2024 | CoRL | First open-weight, open-data 7B VLA matching RT-2 on standard benchmarks |
| π₀: A Vision-Language-Action Flow Model for General Robot Control | Black et al. | 2024 | arXiv (Physical Intelligence) | Flow-matching action head; PaliGemma backbone; dexterous 50 Hz manipulation |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | NVIDIA (Thor et al.) | 2025 | arXiv | Dual-system (VLM + DiT) humanoid foundation model; open weights for 2.2B variant |
| π₀.5: a Vision-Language-Action Model with Open-World Generalization | Hejna et al. | 2025 | arXiv (Physical Intelligence) | Co-training on heterogeneous + web data for novel-home generalisation |
| Gemini Robotics: Bringing AI into the Physical World | Google DeepMind | 2025 | arXiv | Gemini 2.0-based VLA for dexterous manipulation; Gemini-ER for spatial reasoning |
| FAST: Efficient Action Tokenization for Vision-Language-Action Models | Physical Intelligence | 2025 | arXiv | DCT-based frequency-space action tokenizer; 5× training speedup vs. π₀ flow matching |

---

## Benchmark Performance

| Model | Dataset / Benchmark | Metric | Score | Notes |
|-------|---------------------|--------|-------|-------|
| RT-1 | 700+ seen tasks (real robot) | Success rate | 97% | 13 EDR robots, kitchen domain |
| RT-1 | Unseen instructions | Success rate | 76% | +24% over best baseline |
| RT-1 | Distractor objects | Success rate | 83% | Same setup, added visual distractors |
| RT-2-PaLI-X-55B | Unseen objects (hard, real robot) | Success rate | 62% | vs. ~32% for RT-1 baseline |
| RT-2-PaLM-E-12B | Unseen objects (hard, real robot) | Success rate | 62% | Average tied with PaLI-X-55B |
| RT-2-PaLI-X-55B | Language Table (simulation) | Success rate | 90% | vs. 77% prior SOTA |
| RT-2-PaLI-X-55B | Symbol understanding (emergent) | Success rate | 82% | Not explicitly trained; transfers from web pretraining |
| OpenVLA | Google Robot tasks | Mean success rate | 85.0 ± 4.6% | vs. RT-2-X: 78.3 ± 5.4% |
| OpenVLA | BridgeV2 (zero-shot) | Success rate | 56–65% | Widow-X robot, unseen task splits |
| π₀ | In-distribution 4-task macro avg | Success rate | ~72% | Clean dish, sponge, bag unzip, shorts fold |
| π₀ | Spatial OOD | Success rate | ~67% | −5 pp from in-distribution |
| π₀ | Unseen objects (instance + spatial OOD) | Success rate | ~59% | −13 pp from in-distribution; vs. ACT −42 pp |
| GR00T N1 | Sim + real combined (internal) | Task success | +40% over baseline | Synthetic + real data combination |
| Helix (Figure AI) | Commercial benchmark (BMW) | Runtime hours | 1,250+ hrs | 90,000+ parts; benchmark numbers not publicly reported vs. academic tasks |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Discrete action tokenisation (RT-2, OpenVLA) | Architecturally simple; leverages full LLM generation stack; easy to implement; benefits from LLM world knowledge at prediction time | Lossy quantisation; independent per-dimension sampling ignores action correlations; repurposed vocabulary tokens may interfere with semantic representations |
| Flow matching / diffusion action head (π₀, GR00T N1) | Continuous outputs; handles multimodal distributions; temporally coherent action chunks; state-of-the-art dexterous manipulation performance | Additional architectural component (~300M params); 10-step ODE integration adds ~60–70 ms latency; training more complex than cross-entropy |
| Large VLM backbone (RT-2 55B, GR00T N1 34B) | Stronger emergent generalisation; richer language understanding; better novel-object recognition | 55B inference cost prohibitive for onboard deployment; requires model parallelism; fine-tuning expensive |
| Open-weight VLAs (OpenVLA, GR00T N1 2.2B) | Reproducible; community fine-tuning; enables academic benchmarking; reduces deployment barrier | Smaller scale limits generalisation; released variants may lag proprietary versions by months–years |
| Cross-embodiment training (Open X-Embodiment) | Positive transfer across robots; reduces per-robot data requirements; more robust representations | Action space heterogeneity (joint angles vs. EE pose vs. delta actions) complicates joint training; dataset quality and annotation consistency vary |

---

## Open Problems & Research Gaps

- **Bridging discrete and continuous action representations at scale:** Discrete tokenisation preserves VLM structure but loses action fidelity; flow matching gains fidelity but loses the simplicity of next-token prediction. A principled approach that achieves both simultaneously—possibly via vector-quantised latent action spaces or adaptive tokenisation—is an open problem.
- **On-device inference for large backbones:** Deploying a 7B+ parameter VLA at 50 Hz on embedded hardware (e.g., NVIDIA Jetson Orin, 40 TOPS NPUs) requires aggressive quantisation, speculative decoding, or dedicated inference engines. The 2025 Gemini Robotics On-Device release is a significant step, but generalising the approach across architectures remains unsolved.
- **Contact-rich and deformable manipulation:** VLAs trained primarily on pick-and-place tasks generalise poorly to insertion, peg-in-hole, cloth manipulation, or liquid pouring. These tasks require force feedback integration into the VLA loop, which no current architecture handles robustly.
- **Action distribution shift between human demonstrations and robot execution:** Demonstrations collected via teleoperation exhibit human motion artefacts (hesitation, overshoot, retries) that are suboptimal for the robot to imitate. Learning to extract intent rather than trajectory-level behaviour from imperfect demonstrations is an active research direction (see DAgger variants, implicit behavioural cloning).
- **Principled uncertainty quantification:** VLAs do not have well-calibrated uncertainty estimates. A policy should recognise when it is out-of-distribution and either request human intervention or attempt a conservative fallback—capabilities that current autoregressive and flow-matching heads do not provide.
- **Multi-task interference in shared backbone weights:** Co-fine-tuning a VLM on both text tasks and robot control risks catastrophic forgetting of language capabilities and task interference among robot tasks. The optimal data mixture ratios and training curricula are not well understood theoretically.
- **Evaluation standardisation:** The field lacks a universally adopted benchmark comparable to ImageNet or BOP for manipulation. Each paper uses different robot platforms, tasks, and success criteria, making cross-paper comparisons unreliable. Community-driven efforts (LIBERO, SimplerEnv, RoboSuite) are emerging but have not yet achieved universal adoption.

---

## Further Reading

- [RT-1 paper (arXiv 2212.06817)](https://arxiv.org/abs/2212.06817) — Original Robotics Transformer paper with full architecture and training details
- [RT-2 paper (arXiv 2307.15818)](https://arxiv.org/abs/2307.15818) — Co-fine-tuning VLMs for robot control; emergent chain-of-thought in policies
- [π₀ paper (arXiv 2410.24164)](https://arxiv.org/abs/2410.24164) — Flow matching action head; PaliGemma backbone; dexterous manipulation results
- [OpenVLA paper (arXiv 2406.09246)](https://arxiv.org/abs/2406.09246) — Open-weight 7B VLA; full training pipeline and benchmark results
- [GR00T N1 paper (arXiv 2503.14734)](https://arxiv.org/abs/2503.14734) — Dual-system humanoid foundation model; Isaac Sim synthetic data pipeline
- [π₀.5 paper (arXiv 2504.16054)](https://arxiv.org/abs/2504.16054) — Open-world generalisation via heterogeneous co-training
