# Diffusion Policy: Visuomotor Control via Action Diffusion

> **Last Updated:** October 2026
> **Level:** Advanced | Research Frontier
> **Related Sections:**
> - [Cross-Embodiment Learning](./04_cross_embodiment.md)
> - [Simulation and Data for Embodied AI](./05_simulation_and_data.md)
> - [Vision-Language-Action Models](./01_vla_models.md)
> - [Generative Models and Diffusion](../10_generative_vision/01_diffusion_models.md)

> **🆕 October 2026 update:**
> - Added **TRI Large Behavior Models** (diffusion-transformer policies at ~1,700 h scale, rigorous statistical evaluation) and diffusion/flow action heads inside VLAs (GR00T N1.5–N1.7 DiT, SmolVLA, π0.5).
> - Added one-step / few-step policies (MP1 MeanFlow, ManiFlow consistency-flow) and **Real-Time Chunking** for flow policies.
> - New section on **RL fine-tuning of diffusion and flow policies** (DPPO, DSRL, πRL, SAC Flow) and on **video-diffusion policies** (UVA, Cosmos Policy, DreamZero).
> - Corrected inaccurate statements about π0 (10 Euler steps, ~73 ms) and π0-FAST (autoregressive FAST tokens, not rectified flow); fixed broken links; added rows to tables and Further Reading.

---

## Overview

Diffusion Policy [Chi2023] reframes robot imitation learning as a conditional denoising problem: rather than regressing a single "best" action or computing a categorical distribution over discretized actions, a policy network iteratively denoises a sequence of Gaussian noise into a coherent action trajectory conditioned on the current visual observation. The underlying generative mechanism is a Denoising Diffusion Probabilistic Model (DDPM) [Ho2020], where a U-Net or transformer backbone learns to reverse a fixed forward noising process. At inference time the model starts from pure noise in action space and takes $K$ denoising steps guided by the observation embedding, producing a receding-horizon action chunk that is executed open-loop for a short horizon before the next planning cycle begins.

The key behavioral advantage is the handling of **multimodal action distributions**. In demonstrations of everyday manipulation, a human may reach for an object from the left or from the right depending on subtle context; a standard behavior-cloning MSE loss averages both modes and produces a hesitant middle trajectory. Diffusion Policy's energy-based implicit score-matching objective naturally represents such distributions without mode collapse, yielding significantly higher task success on contact-rich and precision tasks. [Chi2023] demonstrated an average 46.9% improvement over prior state-of-the-art behavioral cloning baselines (IBC, BET) across 15 simulated and real tasks.

The work also introduced two architecture variants: a **CNN-based** policy encoding spatial observations via a ResNet-18 feature pyramid, and a **Transformer-based** policy (DiT-style) that cross-attends over observation tokens to condition each denoising step. The transformer variant shows superior generalization on long-horizon tasks but incurs higher computational cost. Subsequent work in 2024-2026 has extended the paradigm to 3D point cloud conditioning (DP3), flow-matching accelerators (pi-zero), and large-scale pretraining, making diffusion-based action generation the de facto standard for visuomotor imitation learning.

By October 2026 the diffusion-policy idea has been absorbed into almost every robot foundation model: GR00T N1.x uses a flow-matching Diffusion Transformer (DiT) action head, π0/π0.5/π0.6 and SmolVLA use flow-matching action experts, and TRI's Large Behavior Models are multitask diffusion transformers trained on ~1,700 hours of robot data [TRI2025]. Three research threads dominate the recent literature: (i) **inference speed**—one-step policies (MeanFlow, consistency flow) and Real-Time Chunking; (ii) **RL fine-tuning** of diffusion/flow policies, which requires either tractable likelihoods (DPPO, πRL) or steering in noise space (DSRL); and (iii) **video-diffusion policies**, where the denoiser is a pretrained video generator that also predicts future frames (UVA, Cosmos Policy, DreamZero).

---

## Technical Background: DDPM for Action Generation

### Forward and Reverse Processes

Given a demonstration action trajectory $\mathbf{a}^0 \in \mathbb{R}^{T_a \times D_a}$ (a chunk of $T_a$ future joint-position or end-effector waypoints), the forward process adds Gaussian noise over $K$ steps:

```latex
q(\mathbf{a}^k | \mathbf{a}^{k-1}) = \mathcal{N}\!\left(\mathbf{a}^k;\,\sqrt{1-\beta_k}\,\mathbf{a}^{k-1},\,\beta_k \mathbf{I}\right)
```

where $\beta_1, \ldots, \beta_K$ is a fixed variance schedule (cosine or linear). After $K$ steps $\mathbf{a}^K \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$. The policy network $\epsilon_\theta$ is trained to predict the noise at each step conditioned on the visual observation embedding $\mathbf{o}$:

```latex
\mathcal{L}_\text{DP} = \mathbb{E}_{k,\,\mathbf{a}^0,\,\boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \epsilon_\theta\!\left(\mathbf{a}^k,\,k,\,\mathbf{o}\right)\right\|^2\right]
```

At inference, starting from $\mathbf{a}^K \sim \mathcal{N}(\mathbf{0},\mathbf{I})$, the reverse process iterates:

```latex
\mathbf{a}^{k-1} = \frac{1}{\sqrt{\alpha_k}}\!\left(\mathbf{a}^k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}}\,\epsilon_\theta(\mathbf{a}^k, k, \mathbf{o})\right) + \sigma_k\,\mathbf{z}
```

where $\alpha_k = 1-\beta_k$, $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$, and $\mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$.

### Receding-Horizon Control

Diffusion Policy generates an action chunk of horizon $T_a$ (e.g., 16 steps at 10 Hz = 1.6 s) but only executes the first $T_e < T_a$ actions before re-planning. This **action chunking** improves temporal consistency and reduces compounding errors compared to per-step prediction.

```mermaid
flowchart LR
    OBS["Visual Obs\n(RGB + depth)"] --> ENC["CNN / ViT\nEncoder"]
    ENC --> COND["Observation\nEmbedding o"]
    NOISE["a^K ~ N(0,I)"] --> D1["Denoise\nstep K"]
    COND --> D1
    D1 --> D2["Denoise\nstep K-1"]
    COND --> D2
    D2 --> DOTS["..."]
    DOTS --> D0["Denoise\nstep 0"]
    COND --> D0
    D0 --> CHUNK["Action Chunk\na^0: T_a steps"]
    CHUNK --> EXEC["Execute T_e\nsteps on robot"]
    EXEC --> OBS
```

### Inference Acceleration: DDIM and Beyond

Standard DDPM requires $K=100$ denoising steps, yielding ~1-2 Hz on a GPU — insufficient for reactive control. **DDIM** (Denoising Diffusion Implicit Models) [Song2021] reformulates inference as a deterministic ODE, enabling 10-16 steps while preserving sample quality. With 10 DDIM steps on an Nvidia RTX 3080, [Chi2023] reports ~0.1 s inference latency, achieving ~10 Hz control. Further acceleration:

- **Consistency Distillation** [Song2023]: distills the multi-step process into a 1-4 step model.
- **Flow Matching** [Lipman2022]: replaces the score-matching loss with straight conditional flows, enabling few-step inference (π0 [Black2024] integrates 10 Euler steps for a 50-step action chunk; ~73 ms end-to-end on an RTX 4090). GR00T N1.x pairs flow matching with a DiT denoiser.
- **Autoregressive alternative — FAST tokens**: π0-FAST replaces the flow head with DCT-compressed discrete action tokens decoded autoregressively; it trains faster but decodes more slowly than the flow expert.
- **One-step flows (2025)**: MP1 applies MeanFlow (learning the *average* velocity over an interval) to 3D point-cloud policies for single-step generation (arXiv 2507.10543); ManiFlow (CoRL 2025, arXiv 2509.01819) combines flow matching with consistency training for 1–2-step dexterous control.
- **Real-Time Chunking (RTC, NeurIPS 2025)** [Black2025RTC]: rather than reducing steps, hides latency by generating the next chunk during execution, freezing the actions that will execute during the delay and inpainting the remainder with guided flow sampling; a training-time variant (arXiv 2512.05964) removes the guidance overhead.

---

## Architecture Variants

### CNN Diffusion Policy (DP-C)

- Observation encoder: ResNet-18 with group normalization, outputs spatial feature map flattened to a 1D token.
- Denoiser: 1-D temporal U-Net over the action sequence, conditioned via FiLM layers from observation embedding.
- Training: 100 DDPM steps, cosine noise schedule, AdamW optimizer.
- Inference: 10-16 DDIM steps; adequate for 10 Hz tabletop control.

### Transformer Diffusion Policy (DP-T)

- Observation encoder: ViT or ResNet, outputs a sequence of tokens.
- Denoiser: transformer decoder where noisy action tokens cross-attend over observation tokens at each denoising step.
- Stronger on tasks requiring global spatial reasoning; ~2x compute vs. DP-C.

### 3D Diffusion Policy (DP3) [Ze2024]

Replaces 2D image encoder with a sparse **point cloud** encoder (PointNet++ MLP backbone) capturing metric 3D structure. Action conditioning uses the compact 3D representation. Published at RSS 2024; achieves 24.2% relative improvement over 2D DP across 72 simulation tasks, and 85% real-robot success on 4 dexterous tasks (Roll-Up, Dumpling, Drill, Pour) with only 40 demonstrations each. **iDP3** (arXiv 2410.10803) extends DP3 to humanoids with egocentric (camera-frame) 3D representations, removing the need for camera calibration and point-cloud segmentation.

### Multitask Diffusion Transformers at Scale: Large Behavior Models (TRI, 2025)

TRI's LBM study (arXiv 2507.05331) trains language-conditioned, multi-camera **diffusion-transformer** policies on ~1,700 hours of bimanual robot data (internal teleoperation, curated Open X-Embodiment subsets and simulation) and evaluates them with blind, randomised trials and statistical confidence intervals in both simulation and the real world. Reported conclusions: multitask pretraining improves success and robustness, and fine-tuned LBMs reach a given performance level with substantially less task-specific data than single-task Diffusion Policies; but improvements are often smaller than single-demo videos imply, and detecting them requires many more evaluation trials than is customary. LeRobot now also ships a "Multitask DiT Policy" implementation in this spirit.

### Diffusion / Flow Heads Inside VLAs

| Model | Denoiser | Conditioning | Steps | Notes |
|---|---|---|---|---|
| π0 / π0.5 | ~300M-class action expert sharing attention with the VLM | Prefix VLM tokens (+ discretised state tokens in π0.5) | 10 (flow) | π0.5 adds AdaRMSNorm timestep injection and Knowledge Insulation |
| GR00T N1.5 / N1.6 / N1.7 | DiT (16 / 32 / 16 layers) | Cross-attention to VLM features; embodiment-specific encoders/decoders | ~4 (flow) | N1.6/N1.7 predict state-relative action chunks |
| SmolVLA | Flow-matching expert interleaving self- and cross-attention | Truncated SmolVLM2 features | Few | 450M total; asynchronous inference |
| Discrete Diffusion VLA (arXiv 2508.20072) | The VLM transformer itself | Masked discrete diffusion over action tokens | Iterative | Unifies diffusion-style refinement with a cross-entropy-trained backbone |

---

## Comparison: Diffusion Policy vs. Related Methods

### Behavioral Cloning Baselines

| Method | Distribution Model | Mode Averaging? | Inference Latency |
|--------|-------------------|-----------------|-------------------|
| MSE Regression BC | Unimodal Gaussian | Yes (collapses modes) | <1 ms |
| IBC [Florence2021] | Energy-based (MCMC) | No | ~100 ms (MCMC) |
| BET [Shafiullah2022] | Discrete VQ + Offset | Partial | ~5 ms |
| Diffusion Policy [Chi2023] | DDPM score field | No | 10-100 ms (DDIM) |
| ACT [Zhao2023] | CVAE + Transformer | Partial (CVAE) | ~5 ms |
| pi-zero [Black2024] | Flow Matching + VLM | No | ~73 ms (full model, RTX 4090) |
| GR00T N1.x [NVIDIA2025] | Flow-matching DiT + VLM | No | ~64 ms for a 16-step chunk (N1, L40) |
| MP1 / ManiFlow (2025) | One-/few-step flow | No | ~1 denoising step |
| Cosmos Policy / DreamZero (2026) | Video diffusion (2–14B) | No | Hundreds of ms to seconds |

### ACT (Action Chunking with Transformers) [Zhao2023]

ACT [Zhao2023] introduced action chunking (predicting $k=100$ timesteps at 50 Hz) with a CVAE-Transformer architecture on the ALOHA bimanual platform. It achieves 80-95% success on fine-grained bimanual tasks vs. 20-50% for standard BC, and requires only ~50 demonstrations. ACT uses temporal ensemble (average overlapping action predictions) rather than open-loop execution of chunks. Diffusion Policy outperforms ACT on highly multimodal tasks; ACT is faster at inference and simpler to implement.

### pi-zero (pi0) [Black2024]

Physical Intelligence's pi-zero [Black2024] builds a **Vision-Language-Action Flow Model** on top of the PaliGemma 3B VLM backbone, replacing DDPM with flow matching for action generation. Trained on 10,000+ hours of demonstrations across 7 robot platforms and 68 tasks, pi-zero achieves state-of-the-art zero-shot and fine-tuned performance on dexterous tasks (laundry folding, box assembly, table bussing). The flow matching head outputs 50-step chunks for 50 Hz control. This represents the convergence of diffusion/flow-based policies with large-scale language-grounded pretraining.

---

## RL Fine-Tuning of Diffusion and Flow Policies

Diffusion and flow policies are trained by imitation, but their iterative sampling complicates RL: the action log-likelihood needed for policy gradients is intractable for a deterministic ODE sampler. Four strategies emerged:

```latex
\text{DPPO: treat the K-step denoising chain as an MDP; each step } p_\theta(\mathbf{a}^{k-1}\mid\mathbf{a}^k,\mathbf{o}) = \mathcal{N}(\mu_\theta,\sigma_k^2\mathbf{I}) \text{ has a closed-form log-likelihood}
```

```latex
\text{DSRL: keep } \pi_\text{base}(\mathbf{a}\mid\mathbf{o},\mathbf{w}) \text{ frozen; learn a latent-noise policy } \mathbf{w}\sim\pi^{\mathcal{W}}_\phi(\cdot\mid\mathbf{o}) \text{ with SAC to maximise } Q(\mathbf{o},\pi_\text{base}(\mathbf{o},\mathbf{w}))
```

| Method | Year / venue | Idea | Scale demonstrated |
|---|---|---|---|
| DPPO — Diffusion Policy Policy Optimization (arXiv 2409.00588) | ICLR 2025 | Two-layer MDP over environment steps × denoising steps; PPO on Gaussian denoising transitions | Sim-to-real manipulation; strong fine-tuning stability vs. other diffusion-RL methods |
| DSRL — Steering Your Diffusion Policy with Latent Space RL (arXiv 2506.15799) | 2025 | RL over the *input noise* of a frozen policy; black-box, sample-efficient | Applied to generalist policies including π0 |
| SAC Flow (arXiv 2509.25756) | 2025 | Reparameterises flow velocity as a sequential model for off-policy SAC | Sample-efficient flow-policy RL |
| πRL (arXiv 2510.25889) | 2025 | Flow-Noise (learned noise injection) or Flow-SDE (ODE→SDE) gives tractable likelihoods for PPO/GRPO | π0/π0.5 VLAs on LIBERO, ManiSkill3, MetaWorld, CALVIN |
| Recap (π\*0.6, arXiv 2511.14759) | 2025 | Avoids likelihoods entirely: advantage-conditioned supervised training of the flow VLA | Real-world, multi-hour tasks |

---

## Video-Diffusion Policies

A 2025–2026 extension uses a **pretrained video diffusion model as the denoiser** for actions, jointly denoising future frames and action chunks so that the policy inherits physical priors from Internet video:

- **UVA** (RSS 2025, arXiv 2503.00200): a shared latent with decoupled video and action diffusion heads; actions can be decoded without generating video.
- **Cosmos Policy** (NVIDIA, arXiv 2601.16163): fine-tunes Cosmos-Predict2-2B so that latent frames also encode actions, proprioception and values; 5 denoising steps for actions in the released LIBERO configuration; optional best-of-N planning using its own predicted futures and values.
- **DreamZero** (NVIDIA, arXiv 2602.15922): a 14B Wan2.1-based World Action Model trained only on DROID that reached #1 on RoboArena in February 2026.

These blur the boundary between Diffusion Policy and world models; see [World Models](./02_world_models.md).

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | Chi et al. | 2023 | RSS 2023 / IJRR 2024 | DDPM over action trajectories; receding-horizon control; 46.9% avg improvement over SOTA BC |
| Denoising Diffusion Probabilistic Models | Ho et al. | 2020 | NeurIPS 2020 | Original DDPM framework used as the generative backbone |
| Denoising Diffusion Implicit Models | Song et al. | 2021 | ICLR 2021 | DDIM accelerated inference (10-16 steps vs. 100); enables robot-frequency deployment |
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT) | Zhao et al. | 2023 | RSS 2023 | Action chunking + CVAE-Transformer on ALOHA; key baseline/alternative to DP |
| 3D Diffusion Policy (DP3) | Ze et al. | 2024 | RSS 2024 | Point-cloud conditioning; 24.2% improvement on 72 sim tasks; 85% real-world success |
| pi-zero: A Vision-Language-Action Flow Model for General Robot Control | Black et al. (Physical Intelligence) | 2024 | arXiv 2410.24164 | Flow matching + PaliGemma 3B VLM; 7 platforms, 68 tasks, 10K+ demo hours |
| Diffusion Transformer Policy | Ye et al. | 2024 | ICLR 2025 | DiT-based denoiser with cross-attention to observation tokens; strong on long-horizon tasks |
| Diffusion Policy Policy Optimization (DPPO) | Ren et al. | 2024 | ICLR 2025 | PPO over the denoising MDP; stable RL fine-tuning of diffusion policies |
| Unified Video Action Model (UVA) | Li et al. | 2025 | RSS 2025 | Joint video-action latent diffusion with decoupled decoding |
| Steering Your Diffusion Policy with Latent Space Reinforcement Learning (DSRL) | Wagenmaker et al. | 2025 | arXiv 2506.15799 | RL in the noise space of a frozen diffusion/flow policy |
| Real-Time Execution of Action Chunking Flow Policies (RTC) | Black, Galliker, Levine | 2025 | NeurIPS 2025 | Inpainting-based asynchronous chunk generation |
| A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation | Toyota Research Institute | 2025 | arXiv 2507.05331 | ~1,700 h multitask diffusion-transformer pretraining; rigorous statistics |
| Mean Flow Tames Policy Learning in 1-step for Robotic Manipulation (MP1) | — | 2025 | arXiv 2507.10543 | MeanFlow one-step point-cloud policy |
| ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training | — | 2025 | CoRL 2025 | Consistency-flow training for 1–2-step dexterous control |
| πRL: Online RL Fine-tuning for Flow-based VLA Models | RLinf team | 2025 | arXiv 2510.25889 | Tractable likelihoods for flow VLAs via Flow-Noise / Flow-SDE |
| Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning | NVIDIA | 2026 | arXiv 2601.16163 | Video diffusion model as policy, world model and value function |

---

## Benchmark Performance

| Model | Dataset / Task | Metric | Score | Notes |
|-------|---------------|--------|-------|-------|
| Diffusion Policy (DP-C) | Robomimic Lift | Success Rate | 96.9% | CNN variant, 100 demos |
| Diffusion Policy (DP-C) | Robomimic Can | Success Rate | 91.3% | CNN variant |
| Diffusion Policy (DP-T) | Robomimic Transport | Success Rate | 78.8% | Transformer variant; bimanual |
| Diffusion Policy (DP-C) | Push-T | Coverage Score | 0.804 | Multimodal pushing task |
| Diffusion Policy avg | 12 Robomimic tasks | Success Rate | SOTA 11/12 | First publication result |
| DP3 | 72 simulation tasks | Avg Success Rate | +24.2% rel. over DP-C | Point-cloud 3D observations |
| DP3 | 4 real dexterous tasks | Success Rate | 85% | 40 demos each; Allegro Hand + gripper |
| ACT | ALOHA sim insertion | Success Rate | 95% | 50 demos; CVAE-Transformer |
| ACT | ALOHA real battery slot | Success Rate | 80% | Contact-rich fine-grained |
| pi-zero | Laundry folding (real) | Task Completion | not publicly reported | Zero-shot after pretraining |
| Diffusion Policy (DP-C) vs IBC | 11 tasks average | Success Rate | +46.9% relative | Original paper comparison |
| GR00T N1.6 (flow DiT head) | LIBERO 4-suite avg | Success Rate | ≈97.0% | Repo-reported |
| π0.5 (flow expert) | LIBERO 4-suite avg | Success Rate | 96.85% | openpi-reported |
| π0 + πRL (Flow-Noise) | MetaWorld MT50 | Success Rate | 50.8% → 85.8% | RL after few-shot SFT |
| π0.5 + πRL (Flow-SDE) | CALVIN ABC→D | Success Rate | 61.3% → 87.0% | RL after few-shot SFT |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Distribution modeling | Naturally represents multimodal and discontinuous action distributions; no mode collapse | Training requires large datasets to avoid overfitting the score field |
| Inference quality | Action chunks produce temporally smooth, physically consistent trajectories | Standard DDPM (100 steps) too slow (1-2 Hz) for reactive tasks; requires DDIM or flow matching |
| Architecture flexibility | Modular: any visual encoder (CNN, ViT, point cloud) can condition the denoiser | Hyperparameter sensitivity: noise schedule, chunk length, step count, architecture size all matter |
| Generalization | Handles unseen object positions well when combined with strong visual encoders | Struggles with extreme distribution shift; still requires task-specific demonstrations |
| Training stability | Score-matching loss is well-behaved; compatible with standard deep learning toolchains | More complex to implement than MSE-BC; 2-5x longer training time |
| Scalability | Scales to larger models and datasets (DiT, VLM backbones) | Compute cost at inference scales with denoising steps; edge deployment is challenging |
| RL compatibility | DPPO / πRL / DSRL / Recap make RL fine-tuning feasible; noise-space steering keeps base weights frozen | Likelihoods are approximate or require architectural changes; RL on real hardware remains expensive |
| Video-diffusion denoisers | Inherit physical priors from video pretraining; jointly predict futures for planning | 2–14B parameters; latency far above classical DP; heavy training compute (e.g., 64 H100s × 48 h for Cosmos Policy on LIBERO) |

---

## Open Problems & Research Gaps

1. **Real-time inference at high control frequency.** Even with DDIM, achieving reliable 50-100 Hz control on embedded hardware (Jetson Orin, etc.) without specialized distillation remains unsolved. Consistency models and flow matching reduce steps but introduce training complexity. One-step flows (MeanFlow, consistency flow), Real-Time Chunking and TensorRT export (GR00T N1.7 on Jetson Thor) narrowed the gap in 2025–2026, but video-diffusion policies reopened it.

2. **Long-horizon task composition.** Diffusion Policy excels at short manipulation primitives but chains of 5+ subtasks (pick, stack, hand-off, insert, close) still suffer from error accumulation. Hierarchical diffusion or language-conditioned sub-goal selection is an open direction.

3. **Sim-to-real transfer for diffusion policies.** The multimodal score field may overfit to simulation visual statistics. Domain randomization interacts poorly with denoising steps tuned for specific visual noise levels; bridging remains ad hoc.

4. **Data efficiency.** Current SOTA results use 50-200 demonstrations per task. Few-shot and one-shot diffusion policies that generalize from a single demo via foundation model priors are not yet reliable in contact-rich settings.

5. **Action space heterogeneity.** Diffusion Policy is typically applied to a fixed action space (e.g., end-effector delta or joint positions). Extending to whole-body humanoid control (50+ DoF) or deformable tool manipulation with force/torque feedback is largely unexplored.

6. **Theoretical guarantees on coverage and safety.** The stochastic denoising process can occasionally produce dynamically infeasible actions. Constrained diffusion (projecting samples onto feasible manifolds) or diffusion with safety certificates is an open theoretical challenge.

7. **Integration with model-based planning.** Combining the expressive generative model of Diffusion Policy with physics-based model predictive control (MPC) — using diffusion to propose candidate trajectories and MPC to filter — has shown promise but lacks a principled training framework. Cosmos Policy's best-of-N planning with self-predicted values is one learned alternative.

8. **Is multimodality the real reason diffusion works?** Large-scale results (LBM) and strong L1-regression baselines (OpenVLA-OFT, 97.1% LIBERO with continuous regression) suggest that much of the benefit attributed to multimodal distribution modelling may come from action chunking, architecture and data scale. Careful ablations that separate these factors are still scarce.

9. **Stable RL for iterative samplers.** Which likelihood approximation (denoising-MDP, SDE conversion, noise-space policies, advantage conditioning) scales best to multi-billion-parameter flow VLAs and to real-world training is unsettled.

---

## Further Reading

- [Diffusion Policy project page and code (Columbia)](https://diffusion-policy.cs.columbia.edu/)
- [3D Diffusion Policy (DP3) — arXiv:2403.03954](https://arxiv.org/abs/2403.03954)
- [pi-zero paper — arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- [ACT (ALOHA) — arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- [DDIM — arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- [Diffusion Transformer Policy — OpenReview ICLR 2025](https://openreview.net/forum?id=PvvXDazPMs)
- [DPPO — arXiv:2409.00588](https://arxiv.org/abs/2409.00588)
- [DSRL — arXiv:2506.15799](https://arxiv.org/abs/2506.15799)
- [Real-Time Chunking — arXiv:2506.07339](https://arxiv.org/abs/2506.07339) and [training-time RTC — arXiv:2512.05964](https://arxiv.org/abs/2512.05964)
- [TRI Large Behavior Models — arXiv:2507.05331](https://arxiv.org/abs/2507.05331)
- [MP1 (MeanFlow policy) — arXiv:2507.10543](https://arxiv.org/abs/2507.10543)
- [πRL — arXiv:2510.25889](https://arxiv.org/abs/2510.25889)
- [Cosmos Policy — arXiv:2601.16163](https://arxiv.org/abs/2601.16163)
- [LeRobot Diffusion / Multitask DiT policies](https://github.com/huggingface/lerobot)
