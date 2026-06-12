# End-to-End Autonomous Driving: From UniAD to Neural World Models

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [AV Stack Overview](./00_overview.md) · [Perception Stack](./01_perception_stack.md) · [Occupancy Prediction](./02_occupancy_prediction.md) · [World Models](../06_robotics_and_embodied_ai/02_world_models.md)

---

## Overview

End-to-end (E2E) autonomous driving refers to architectures that learn a direct mapping from raw sensor observations to vehicle control outputs—steering, throttle, and braking—without requiring hand-engineered intermediate representations such as 3D bounding boxes, HD map polygons, or explicit motion forecasting modules. The conceptual lineage traces to ALVINN [Pomerleau1989], which trained a shallow network on camera-to-steering mappings on Carnegie Mellon's campus, and DAVE-2 [Bojarski2016], which applied the same idea with deep CNNs at NVIDIA. Modern E2E driving differs from these precursors in scale (billions of parameters, millions of training hours), representational richness (multi-camera transformer encoders, BEV lifting, language conditioning), and task scope (full urban driving including complex interactions, not just lane-following on highways).

The architectural evolution of E2E driving over 2022–2025 can be characterized along two axes: **structural granularity** and **supervisory richness**. Early E2E systems (CILRS [Codevilla2019], Transfuser [Chitta2022]) used a single-stream backbone that collapsed perception and planning into one forward pass, with minimal intermediate supervision. This made them brittle outside training distribution and difficult to debug. The 2023 generation—epitomized by **UniAD** [Hu2023] (CVPR 2023 Best Paper Award), **VAD** [Jiang2023], and **SparseDrive**—retained task-specific query streams (detection queries, map queries, motion queries, planning queries) connected by cross-attention, enabling E2E gradient flow while preserving interpretable intermediate outputs. The 2024–2025 generation incorporates **language-conditioned reasoning** (DriveVLM [Tian2024], DriveLM [Sima2024]) and **generative world models** (GAIA-1 [Hu2023GAIA], DriveDreamer, MAGICDRIVE) that can simulate future sensor frames conditioned on ego-actions, enabling closed-loop self-play training without real-world data.

The shift from modular to E2E systems in production is most visibly represented by **Tesla FSD v12** (2024), which replaced approximately 300,000 lines of C++ rule-based code with a single large neural network trained by imitation learning on ~10 million labeled driving video clips [Tesla2024]. The engineering significance is that FSD v12 required no module-level specification, no hand-written corner-case handlers, and improved rapidly with additional data—demonstrating the data-scalability advantage of E2E architectures that the academic community had theorized but production teams had not previously validated at this scale.

---

## The Modular-to-E2E Transition: Structural Analysis

The modular stack factors the driving policy as a product of independently optimized components. Let $x_t$ be the sensor observation at time $t$, $s_t$ the latent scene state, and $a_t$ the control action. The modular decomposition is:

$$p(a_t \mid x_t) = \int p(a_t \mid s_t) \cdot p(s_t \mid x_t)\; ds_t$$

where $p(s_t \mid x_t)$ is the perception module and $p(a_t \mid s_t)$ is the planning module, each optimized with a separate loss $\mathcal{L}_\text{perc}$ and $\mathcal{L}_\text{plan}$.

The pathology of this decomposition is **information bottleneck**: $s_t$ must be expressive enough to fully determine $a_t$, but any information in $x_t$ not captured by the chosen representation $s_t$ is permanently discarded. Soft failures—ambiguous observations, unusual geometries, partially occluded agents—that fall outside the specification of $s_t$ produce errors that compound silently downstream.

E2E systems optimize $p(a_t \mid x_t)$ jointly:

$$\theta^* = \arg\min_\theta \mathbb{E}_{(x, a^*) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), a^*) \right]$$

The gradient $\partial \mathcal{L} / \partial x$ flows through all layers simultaneously, allowing perceptual representations to specialize for planning-relevant features. The practical consequence is that E2E systems implicitly learn scene features the designer did not specify—e.g., road wetness affecting braking distance, or the behavioral pattern of a specific pedestrian type.

---

## UniAD: Planning-Oriented End-to-End Driving (CVPR 2023 Best Paper)

**UniAD** (*Unified Autonomous Driving*) [Hu2023UniAD, arXiv:2212.10156] from OpenDriveLab (SenseTime + Shanghai AI Lab) won the CVPR 2023 Best Paper Award. It represents the clearest architectural statement of the **structured E2E** paradigm: a sequence of transformer decoder modules, each solving a standard driving sub-task, connected by shared query interfaces.

### Architecture

```
Camera images (surround-view)
        ↓
BEV Encoder (BEVFormer backbone)
        ↓
TrackFormer  →  MapFormer  →  MotionFormer  →  OccFormer  →  PlanFormer
(detection+   (online HD    (multi-agent      (occupancy     (ego trajectory
 tracking)     mapping)      forecasting)      flow)          planning)
```

Each module outputs a set of typed query embeddings that flow forward. Crucially, all five modules share a common BEV feature representation and are **jointly trained end-to-end** with task-specific losses at each intermediate output:

$$\mathcal{L}_\text{total} = \lambda_1 \mathcal{L}_\text{track} + \lambda_2 \mathcal{L}_\text{map} + \lambda_3 \mathcal{L}_\text{motion} + \lambda_4 \mathcal{L}_\text{occ} + \lambda_5 \mathcal{L}_\text{plan}$$

### Performance on nuScenes

Compared to prior SOTA (BEV-Planner and individual task-specific models):
- **+20% MOTA** in multi-object tracking
- **+30% accuracy** in online lane mapping (IoU-based)
- **−38% minFDE** in motion forecasting
- **−28% planning L2 error** at 3 s horizon
- **Planning L2 at 3s**: 0.71 m (vs. ~1.0 m for non-E2E planning baselines)

UniAD demonstrates that **joint optimization improves every sub-task simultaneously**, not just planning—validating the theoretical argument that shared representations benefit all modules.

---

## VAD: Vectorized Autonomous Driving (ICCV 2023)

**VAD** (*Vectorized Autonomous Driving*) [Jiang2023VAD, arXiv:2303.12077] proposes a fully vectorized scene representation for E2E driving, replacing rasterized BEV feature maps with sparse instance-level vector representations:

- **Agent instances**: Represented as polyline tokens (sampled trajectory points).
- **Map elements**: Represented as polyline tokens (lane centerlines, road boundaries, crosswalks).
- **Ego query**: A single learnable token interacting with all agent and map vectors via attention.

The vectorized approach eliminates the high computational cost of dense BEV grids while preserving geometric relationships via relative position encodings. VAD achieves:
- **L2 error** at 1s/2s/3s: 0.41 / 0.70 / 1.05 m
- **Average collision rate**: 0.22%
- **4× faster inference** than UniAD at comparable accuracy

VAD's key insight is that planning does not require pixel-level BEV features; the structural topology of agent interactions is the relevant information, and vectors encode topology more efficiently than grids.

---

## DriveVLM: Vision-Language Models for Autonomous Driving

**DriveVLM** [Tian2024DriveVLM, arXiv:2402.12289] from Tsinghua MARS Lab integrates large vision-language models (VLMs) into the driving pipeline to handle **complex and rare scenarios** that standard E2E networks fail to reason about explicitly. The key architectural insight is a **dual-system design**:

- **DriveVLM-Dual**: A slow VLM reasoning system handles scene understanding and meta-action generation (e.g., "there is a child chasing a ball near the crosswalk; apply anticipatory braking"); a fast spatial E2E module handles standard trajectory planning.
- **SUP task**: DriveVLM formally defines *Scene Understanding and Planning* (SUP) as a new benchmark task, with evaluation metrics for both descriptive correctness and planning trajectory quality.

On nuScenes validation, DriveVLM-Dual achieves:
- **L2 at 1s/2s/3s**: 0.18 / 0.34 / 0.68 m
- **Avg collision rate**: 0.27%

The VLM component provides **chain-of-thought reasoning** about unusual scenarios, improving few-shot generalization to novel scenes not in the training distribution—addressing the long-tail problem that pure imitation learning cannot solve.

---

## GAIA-1: Generative World Model for Autonomous Driving (Wayve)

**GAIA-1** (*Generative AI for Autonomy*) [Hu2023GAIA, arXiv:2309.17080] from Wayve represents a fundamental architectural shift: rather than training an E2E policy directly, GAIA-1 trains a **generative world model** that can predict future video frames conditioned on ego-vehicle actions and text/image conditioning.

### Architecture

GAIA-1 casts world modeling as unsupervised sequence prediction by tokenizing all inputs:
- **Video tokens**: VQ-VAE tokenization of each camera frame → discrete tokens.
- **Action tokens**: Ego vehicle state (speed, steering, acceleration) → discretized tokens.
- **Text tokens**: Scene description (weather, location type, hazards) → BPE tokens.

A large autoregressive transformer (GPT-style) predicts the next token jointly over video, actions, and text:

$$p(v_{t+1}, a_{t+1} \mid v_{1:t}, a_{1:t}, \text{text})$$

**Key properties**:
- **Emergent scene dynamics**: Despite no explicit physics supervision, GAIA-1 learns coherent vehicle motion, occlusion handling, and scene consistency across frames.
- **Controllable generation**: By conditioning on different action sequences, users can synthesize rare scenarios (near-misses, adversarial pedestrians, edge-case lighting).
- **Training data**: ~4,700 hours of proprietary driving footage collected in London (2019–2023) at 25 Hz; ~420 million unique frames.
- **Scale**: Base model; later scaled to **9 billion parameters** [Wayve2024Scaling].

GAIA-1 is not itself a driving policy; it is a **simulator** that can be used to augment training data, evaluate policies in closed-loop without real-world deployment, and potentially as a reward model for RL-based policy training. This positions it as a neural alternative to physics-based simulators like CARLA.

---

## Tesla FSD v12: Production End-to-End Neural Driving

Tesla FSD v12 (released early 2024) is the highest-profile production E2E AV system as of June 2026. Key technical details:

- **Input**: Eight cameras (360° surround, 120–250 m range), no LiDAR at inference.
- **Output**: Direct vehicle control (steering angle, throttle, braking torque) at ~10–20 Hz.
- **Architecture**: A large transformer network; internal architectural details not published. The system involves 48 distinct neural network components [FredPope2024], though whether these are truly independent or constitute heads within a shared architecture is not publicly clarified.
- **Training**: Imitation learning from ~10 million human driving clips; human interventions serve as negative samples; fleet data provides pseudo-labels via offline labeling. Trained on Dojo supercomputer.
- **Capability uplift**: FSD v12 demonstrated qualitative improvements in "natural driving behavior"—smoother lane changes, better yielding, reduced abrupt braking—compared to the rule-based FSD v11. These are difficult to quantify from public data; Tesla does not publish per-maneuver evaluation metrics.

**Critical limitation**: Tesla does not publish formal safety statistics (miles per disengagement, accident rates per distance, comparison to human baseline) that would allow academic or regulatory evaluation. Safety claims are based on quarterly accident reports filed with NHTSA and voluntary corporate reporting.

---

## nuScenes Planning Metrics

The nuScenes planning benchmark evaluates E2E driving planners on a held-out set of 150 real driving scenarios using two primary metrics computed over multi-horizon predictions:

1. **L2 displacement error**: Average Euclidean distance between predicted ego-trajectory and expert trajectory at $t \in \{1, 2, 3\}$ seconds.

$$\text{L2}(t) = \frac{1}{N}\sum_{i=1}^N \|\hat{p}_i^t - p_i^{t*}\|_2$$

2. **Collision rate**: Proportion of predicted trajectories that intersect with any annotated agent bounding box at any predicted timestep.

| Model | L2 @1s | L2 @2s | L2 @3s | Collision Rate (avg) |
|-------|--------|--------|--------|----------------------|
| IL baseline | ~0.80 | ~1.20 | ~1.60 | ~0.60% |
| ST-P3 [Hu2022] | 1.33 | 2.11 | 3.24 | 0.71% |
| UniAD [Hu2023] | 0.36 | 0.58 | 0.71 | 0.12% |
| VAD [Jiang2023] | 0.41 | 0.70 | 1.05 | 0.22% |
| DriveVLM [Tian2024] | 0.18 | 0.34 | 0.68 | 0.27% |

*Note: nuScenes open-loop planning metrics are criticized for not reflecting closed-loop driving quality; low L2 can be achieved by following the expert trajectory without safety reasoning. Treat these numbers as relative comparisons, not absolute safety indicators.*

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| UniAD | Hu et al. | 2023 | CVPR (Best Paper) | Planning-oriented E2E; joint optimization of full AV task stack |
| VAD | Jiang et al. | 2023 | ICCV | Vectorized scene representation; 4× faster than UniAD |
| DriveVLM | Tian et al. | 2024 | CoRL | VLM + E2E dual system; superior few-shot generalization |
| GAIA-1 | Hu et al. | 2023 | arXiv | Generative world model; 9B parameters; 4,700h training data |
| Transfuser | Chitta et al. | 2022 | TPAMI | Sensor fusion transformer; strong CARLA benchmark baseline |
| CILRS | Codevilla et al. | 2019 | ICCV | Conditional imitation learning; established E2E benchmark |
| ST-P3 | Hu et al. | 2022 | ECCV | First camera-only E2E planning on nuScenes; BEV + safety cost |
| DriveLM | Sima et al. | 2024 | ECCV | Graph VQA for driving; language-grounded E2E planning |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| UniAD | nuScenes val | Planning L2 @3s | 0.71 m | CVPR 2023 Best Paper |
| VAD | nuScenes val | Planning L2 @3s | 1.05 m | Vectorized; faster inference |
| DriveVLM | nuScenes val | Planning L2 @3s | 0.68 m | VLM reasoning; best on standard metrics |
| Transfuser | CARLA Town05 | Route completion | 54.5% | Camera + LiDAR fusion |
| UniAD | nuScenes val | Avg collision rate | 0.12% | Lowest collision rate published |

---

## Pros & Cons

| Aspect | End-to-End Neural | Modular Stack |
|--------|-----------------|---------------|
| Joint optimization | Full gradient through all tasks | Independent per-module optimization |
| Data scalability | Scales with raw driving footage | Requires per-module annotations |
| Interpretability | Opaque intermediate representations | Explicit outputs at each stage |
| Failure mode diagnosis | Difficult — which stage failed? | Module-level logging and replay |
| Regulatory validation | Immature safety case methodology | Established independent verification |
| Generalization to novel scenes | Better with VLM integration | Limited by hand-written rules |

---

## Open Problems & Research Gaps

- **Open-loop vs. closed-loop gap**: nuScenes L2/collision metrics use open-loop evaluation; a model that mimics the expert trajectory but cannot avoid a newly inserted obstacle will score well but fail in deployment. Closed-loop evaluation (nuPlan, CARLA, Waymax) is necessary but expensive.
- **Long-tail generalization**: E2E systems trained on imitation learning are bounded by the training distribution; adversarial or novel scenarios (black-swan events) require either massive dataset coverage or principled out-of-distribution detection.
- **Causal confounding in imitation learning**: The agent learns spurious correlations from demonstration data (e.g., always slowing near schools because the expert does, not because the agent understands why). Causal interventional training is an open research direction.
- **VLM latency for real-time control**: Large VLMs (7B+ parameters) cannot run at real-time inference rates on automotive hardware; the DriveVLM dual-system approach is a workaround, not a solution. Hardware-efficient VLM architectures for AV are needed.
- **Safety certification of neural policies**: No accepted methodology exists for formally bounding the probability of catastrophic failure of a large neural network policy. This is the central obstacle to L4 deployment without human oversight.
- **Multi-agent interaction modeling**: Current E2E systems model other agents as obstacles to avoid rather than rational agents with goals; explicit theory-of-mind and multi-agent game-theoretic planning is nascent.
- **Reward signal for RL-based E2E**: Imitation learning converges to average behavior; RL can optimize for safety-critical scenarios but requires a reliable reward signal. World models like GAIA-1 as RL environments are a promising but immature direction.

---

## Further Reading

- [arXiv:2212.10156 UniAD](https://arxiv.org/abs/2212.10156) — CVPR 2023 Best Paper
- [arXiv:2309.17080 GAIA-1](https://arxiv.org/abs/2309.17080) — Wayve world model
- [arXiv:2402.12289 DriveVLM](https://arxiv.org/abs/2402.12289) — VLM integration
- [OpenDriveLab/UniAD GitHub](https://github.com/OpenDriveLab/UniAD) — official code
- [arXiv:2401.08658 E2E Planning Survey 2022-2023](https://arxiv.org/abs/2401.08658) — comprehensive survey
- [Wayve scaling GAIA-1 blog post](https://wayve.ai/thinking/scaling-gaia-1/) — 9B parameter results
