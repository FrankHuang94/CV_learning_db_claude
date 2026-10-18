# End-to-End Autonomous Driving: From UniAD to Neural World Models

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [AV Stack Overview](./00_overview.md) · [Perception Stack](./01_perception_stack.md) · [Occupancy Prediction](./02_occupancy_prediction.md) · [World Models](../06_robotics_and_embodied_ai/02_world_models.md) · [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md) · [Simulation & Data](./04_simulation_and_data.md)

> **🆕 October 2026 update:**
> - New section **"2025–2026: Driving Foundation Models and Reasoning VLAs"**: Waymo EMMA, NVIDIA Alpamayo-R1 / open Alpamayo 1 (10B), SimLingo, the latent-reasoning trend, and production VLA / world-action stacks (XPeng VLA 2.0, Li Auto, Huawei ADS 4 WEWA).
> - New section **"Diffusion and Multi-Modal Planners"**: Hydra-MDP and DiffusionDrive (88.1 PDMS on NAVSIM).
> - New section **"Beyond nuScenes: NAVSIM (v1/v2) and Bench2Drive"**, with the PDM Score, the v2 pseudo-simulation / EPDMS, and Bench2Drive closed-loop results; new benchmark rows.
> - GAIA section extended with **GAIA-2** (Mar 2025) and **GAIA-3** (Dec 2025). Tesla section updated with **FSD v14** and the unsupervised robotaxi (Austin, Jan 2026).
> - New Open Problems (reasoning faithfulness, reasoning latency, RL in world models) and Further Reading.

---

## Overview

End-to-end (E2E) autonomous driving refers to architectures that learn a direct mapping from raw sensor observations to vehicle control outputs—steering, throttle, and braking—without requiring hand-engineered intermediate representations such as 3D bounding boxes, HD map polygons, or explicit motion forecasting modules. The conceptual lineage traces to ALVINN [Pomerleau1989], which trained a shallow network on camera-to-steering mappings on Carnegie Mellon's campus, and DAVE-2 [Bojarski2016], which applied the same idea with deep CNNs at NVIDIA. Modern E2E driving differs from these precursors in scale (billions of parameters, millions of training hours), representational richness (multi-camera transformer encoders, BEV lifting, language conditioning), and task scope (full urban driving including complex interactions, not just lane-following on highways).

The architectural evolution of E2E driving over 2022–2025 can be characterized along two axes: **structural granularity** and **supervisory richness**. Early E2E systems (CILRS [Codevilla2019], Transfuser [Chitta2022]) used a single-stream backbone that collapsed perception and planning into one forward pass, with minimal intermediate supervision. This made them brittle outside training distribution and difficult to debug. The 2023 generation—epitomized by **UniAD** [Hu2023] (CVPR 2023 Best Paper Award), **VAD** [Jiang2023], and **SparseDrive**—retained task-specific query streams (detection queries, map queries, motion queries, planning queries) connected by cross-attention, enabling E2E gradient flow while preserving interpretable intermediate outputs. The 2024–2025 generation incorporates **language-conditioned reasoning** (DriveVLM [Tian2024], DriveLM [Sima2024]) and **generative world models** (GAIA-1 [Hu2023GAIA], DriveDreamer, MAGICDRIVE) that can simulate future sensor frames conditioned on ego-actions, enabling closed-loop self-play training without real-world data. The **2025–2026 generation** goes further in two directions. First, it builds on *pretrained multimodal foundation models*: MLLM-native planners such as EMMA [Hwang2024EMMA] and reasoning VLAs such as Alpamayo-R1 [Wang2025Alpamayo], trained with supervised fine-tuning on reasoning traces followed by RL post-training. Second, it replaces regression heads with *generative trajectory decoders* (diffusion / flow matching) that represent multi-modal futures. Evaluation has meanwhile shifted from nuScenes open-loop L2 to NAVSIM and Bench2Drive.

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

### GAIA-2 and GAIA-3 (2025)

**GAIA-2** [Russell2025GAIA2] (arXiv 2503.20523, March 2025) replaces GAIA-1's discrete-token autoregressive world model with a **latent diffusion** world model. It generates up to five spatially and temporally consistent camera streams (448×960). Generation is conditioned on structured inputs: ego dynamics, the position, orientation and size of other agents, road semantics, geography (UK/US/Germany), weather and time of day. It can also condition on external latents such as CLIP embeddings or a proprietary driving embedding. **GAIA-3** (December 2025) was presented by Wayve as a ~15B-parameter model trained on roughly 10× more data than GAIA-2, with a 2× larger video tokenizer. It is explicitly positioned as **evaluation infrastructure**: Wayve reports that policy rankings computed inside GAIA-3 correlate closely with rankings from physical road testing (a company claim, not independently verified). The trajectory GAIA-1 → GAIA-3 mirrors the field's move from "world model as data generator" to "world model as validator".

---

## 2025–2026: Driving Foundation Models and Reasoning VLAs

### EMMA (Waymo)

**EMMA** [Hwang2024EMMA] (arXiv 2410.23262) builds an E2E driving model on a Gemini-based multimodal LLM. Camera images go in; all other inputs (navigation command, ego history) and outputs (planner trajectory waypoints, 3D boxes, road-graph polylines) are serialised as **plain text**. Task selection is done by prompting. EMMA reported state-of-the-art motion planning on nuScenes and competitive results on the Waymo Open Motion Dataset and camera-only 3D detection on WOD. Joint training on planning, detection and road-graph tasks improved all three, echoing UniAD's multi-task finding inside an MLLM. Its stated limitations (few frames, no LiDAR/radar, high inference cost) define the agenda for production-grade VLAs.

### Alpamayo-R1 / Alpamayo 1 (NVIDIA)

**Alpamayo-R1** [Wang2025Alpamayo] (arXiv 2511.00088) targets the long tail with explicit causal reasoning:

- **Chain of Causation (CoC) dataset**: decision-grounded, causally linked reasoning traces produced by a hybrid auto-labeling + human-in-the-loop pipeline.
- **Architecture**: a **Cosmos-Reason** VLM backbone (NVIDIA's physical-AI reasoning VLM) produces reasoning, and a **diffusion-based trajectory decoder** generates dynamically feasible plans in real time.
- **Training**: SFT for reasoning, then **RL post-training** that rewards reasoning quality and reasoning-action consistency.
- **Reported results**: up to 12% better planning accuracy on hard cases vs. a trajectory-only baseline; 35% fewer close encounters in closed-loop simulation; RL improves reasoning quality by 45% and reasoning-action consistency by 37%; on-vehicle road tests at 99 ms latency.

At CES (5 January 2026) NVIDIA released the family openly as **Alpamayo 1**, a 10B-parameter VLA with weights on Hugging Face. It is intended as a *teacher* to be fine-tuned and distilled into on-vehicle stacks, and ships together with the **AlpaSim** closed-loop simulator and >1,700 h of open driving data.

### Language-Grounded Closed-Loop Driving: SimLingo

**SimLingo** [Renz2025SimLingo] (CVPR 2025) is a camera-only VLA for CARLA. It jointly performs driving, VQA and instruction following, and introduces "Action Dreaming" to enforce *alignment between language instructions and actions*. Its predecessor CarLLaVA won the 2024 CARLA Autonomous Driving Challenge, the first time a VLM-based driver topped the official closed-loop leaderboard.

### Latent Reasoning and Production VLAs

Explicit text reasoning is expensive at 10–30 Hz. A 2026 trend is **latent or one-step reasoning**: the model reasons in a compact latent space and decodes language only on demand for explanation. Examples include Xiaomi's OneVL (arXiv 2604.18486) and TacitVLA (arXiv 2610.17540), alongside work on accelerating Alpamayo 1's trajectory generation (arXiv 2605.08975). In production, XPeng's **VLA 2.0** integrates perception, reasoning and action in a single model and removes the intermediate language layer to cut latency. It runs vision-only on in-house Turing chips (robotaxi road tests from March 2026; global deliveries planned for 2027). **Li Auto** ships a VLA driver model (MindVLA lineage). **Huawei** ADS 4 instead adopts a **World Engine + World Action (WEWA)** design that maps sensors directly to actions, with no language bottleneck. A cloud diffusion "world engine" generates hard scenarios for training; Huawei claims about 50% lower end-to-end latency, 20% higher traffic efficiency and 30% fewer hard-braking events (company figures).

---

## Diffusion and Multi-Modal Planners

Imitation with an L2/L1 regression head averages over modes (e.g., "pass left" and "pass right" average into a collision). Two families address this:

- **Scoring a trajectory vocabulary.** **Hydra-MDP** [Li2024HydraMDP] (NVIDIA) scores a large fixed vocabulary of candidate trajectories with multiple heads. Each head is distilled from a rule-based teacher (the PDM sub-scores: collision, drivable area, TTC, progress, comfort) as well as from human demonstrations. It won the CVPR 2024 NAVSIM challenge.
- **Truncated diffusion.** **DiffusionDrive** [Liao2025DiffusionDrive] (CVPR 2025) denoises from an *anchored* Gaussian mixture centred on clustered trajectory anchors rather than pure noise. Only **2 denoising steps** are needed. It reaches **88.1 PDMS** on NAVSIM navtest at ~45 FPS (RTX 4090), above the TransFuser baseline (84.0).

The same generative decoder now appears inside VLAs (Alpamayo-R1's diffusion decoder; flow-matching action heads borrowed from robotics, see [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md)).

```
DiffusionDrive (truncated diffusion):
  anchors {a_k} ──► add small noise ──► x_T  (T ≪ 1000)
  x_T ──► denoiser(BEV/scene feats, ego status) ──► x_{T-1} ──► … ──► x_0
  x_0 + per-anchor confidence ──► select trajectory
```

---

## Tesla FSD v12: Production End-to-End Neural Driving

Tesla FSD v12 (released early 2024) is the highest-profile production E2E AV system as of June 2026. Key technical details:

- **Input**: Eight cameras (360° surround, 120–250 m range), no LiDAR at inference.
- **Output**: Direct vehicle control (steering angle, throttle, braking torque) at ~10–20 Hz.
- **Architecture**: A large transformer network; internal architectural details not published. The system involves 48 distinct neural network components [FredPope2024], though whether these are truly independent or constitute heads within a shared architecture is not publicly clarified.
- **Training**: Imitation learning from ~10 million human driving clips; human interventions serve as negative samples; fleet data provides pseudo-labels via offline labeling. Trained on Dojo supercomputer.
- **Capability uplift**: FSD v12 demonstrated qualitative improvements in "natural driving behavior"—smoother lane changes, better yielding, reduced abrupt braking—compared to the rule-based FSD v11. These are difficult to quantify from public data; Tesla does not publish per-maneuver evaluation metrics.

**Update: FSD v13–v14 and the robotaxi (2025–2026).** FSD v14 began rolling out in October 2025, adding features such as emergency-vehicle yielding and rerouting around blocked roads. Musk had originally promised a ~10× larger network for v14. Later reporting indicates the shipped v14.x line still runs a refined "small" model (reportedly ~1B parameters), with the large model deferred to v15; these figures come from Musk statements relayed by enthusiast press, not technical documentation. Tesla's robotaxi service launched in Austin on 22 June 2025 with in-car safety monitors. **Unsupervised public rides** (no one in the car, remote monitoring) began on 22 January 2026 using a dedicated "unsupervised" FSD v14 branch, and Tesla reported unsupervised rides in Miami, Orlando and Tampa from July 2026. These are the first driverless deployments of a camera-only E2E policy, at fleet sizes far smaller than Waymo's.

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

*Note: nuScenes open-loop planning metrics are criticized for not reflecting closed-loop driving quality; low L2 can be achieved by following the expert trajectory without safety reasoning. AD-MLP [Zhai2023] and Li et al. [Li2024EgoStatus] (CVPR 2024) showed that a model using only **ego status** (velocity, acceleration, history), with no camera input, matches or beats UniAD/VAD on these metrics. Treat these numbers as relative comparisons, not absolute safety indicators.*

---

## Beyond nuScenes: NAVSIM (v1/v2) and Bench2Drive

### NAVSIM

**NAVSIM** [Dauner2024NAVSIM] (NeurIPS 2024 Datasets & Benchmarks) sits between open-loop and full closed-loop evaluation. Built on OpenScene (a redistribution of nuPlan), it runs a **non-reactive simulation** of the planned 4 s trajectory using an LQR controller and a kinematic bicycle model. It scores the result with the **PDM Score (PDMS)**, which aggregates rule-based sub-metrics:

$$\text{PDMS} = \underbrace{\text{NC} \times \text{DAC}}_{\text{multiplicative penalties}} \times \frac{5\,\text{TTC} + 2\,\text{C} + 5\,\text{EP}}{12}$$

Here NC is no at-fault collision, DAC drivable-area compliance, TTC time-to-collision, C comfort and EP ego progress. The `navtest` split filters out trivial scenes (e.g., constant-velocity-solvable), which removes much of the ego-status shortcut. **NAVSIM v2** ("pseudo-simulation", CoRL 2025) adds synthetic follow-up observations, rendered with 3D Gaussian splatting from perturbed start states, so a planner is scored on how it *recovers* from its own deviations. It also introduces the **Extended PDM Score (EPDMS)**, which adds terms for lane keeping, driving-direction compliance, traffic-light compliance and extended comfort. Since 2024 NAVSIM has served as the benchmark for the end-to-end driving track of the CVPR Autonomous Grand Challenge.

### Bench2Drive

**Bench2Drive** [Jia2024Bench2Drive] (NeurIPS 2024 Datasets & Benchmarks) is a closed-loop benchmark in CARLA v2. It has an official training set of ~2M annotated frames, and evaluation covers **220 short routes** across **44 interactive scenarios** (cut-ins, overtaking, detours, emergency braking, give-way). Metrics are Driving Score, Success Rate, Efficiency, Comfortness and per-ability "Multi-Ability" scores. Its baseline results exposed how weak 2023 nuScenes-trained E2E stacks are in closed loop: UniAD-Base reached a Driving Score of 45.81 and VAD 42.35, below expert-distillation methods such as ThinkTwice and DriveAdapter. Since then VLM-based and diffusion-based planners (SimLingo, and many 2025 methods) have pushed Bench2Drive scores substantially higher. The benchmark is now the default closed-loop check for academic E2E papers.

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
| EMMA | Hwang et al. (Waymo) | 2024 | arXiv 2410.23262 | Gemini-based MLLM driving; trajectories, boxes and road graph as text |
| Hydra-MDP | Li et al. (NVIDIA) | 2024 | arXiv / CVPR 2024 challenge | Multi-target distillation from rule-based scorers; NAVSIM challenge winner |
| NAVSIM | Dauner et al. | 2024 | NeurIPS (D&B) | Non-reactive simulation benchmark; PDM Score |
| Bench2Drive | Jia et al. | 2024 | NeurIPS (D&B) | Multi-ability closed-loop E2E benchmark in CARLA v2 |
| GAIA-2 | Russell et al. (Wayve) | 2025 | arXiv 2503.20523 | Multi-camera controllable latent-diffusion world model |
| DiffusionDrive | Liao et al. | 2025 | CVPR | Truncated diffusion planner; 88.1 PDMS on NAVSIM at real-time speed |
| SimLingo | Renz et al. | 2025 | CVPR | Camera-only VLA with language-action alignment; CARLA challenge lineage |
| Alpamayo-R1 | Wang et al. (NVIDIA) | 2025 | arXiv 2511.00088 | Reasoning VLA (Chain of Causation + RL); released openly as Alpamayo 1 (10B) |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| UniAD | nuScenes val | Planning L2 @3s | 0.71 m | CVPR 2023 Best Paper |
| VAD | nuScenes val | Planning L2 @3s | 1.05 m | Vectorized; faster inference |
| DriveVLM | nuScenes val | Planning L2 @3s | 0.68 m | VLM reasoning; best on standard metrics |
| Transfuser | CARLA Town05 | Route completion | 54.5% | Camera + LiDAR fusion |
| UniAD | nuScenes val | Avg collision rate | 0.12% | Lowest among 2023 E2E methods; ego-status caveat applies |
| TransFuser | NAVSIM navtest | PDMS | 84.0 | NAVSIM reference baseline |
| DiffusionDrive | NAVSIM navtest | PDMS | 88.1 | 2-step truncated diffusion; ~45 FPS |
| UniAD-Base | Bench2Drive | Driving Score | 45.81 | Closed-loop; Bench2Drive paper baseline |
| VAD | Bench2Drive | Driving Score | 42.35 | Closed-loop; Bench2Drive paper baseline |

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
- **Reward signal for RL-based E2E**: Imitation learning converges to average behavior; RL can optimize for safety-critical scenarios but requires a reliable reward signal. World models like GAIA-1 as RL environments are a promising but immature direction. By 2026, RL post-training of VLAs (Alpamayo-R1) and world-model-based evaluation (GAIA-3, AlpaSim) have made this mainstream, but reward hacking against imperfect simulators is a live concern.
- **Faithfulness of reasoning**: Reasoning VLAs output human-readable causal explanations, but 2026 studies probe whether these explanations actually determine the trajectory. Examples are *"Is VLA Reasoning Faithful? Probing Safety of Chain-of-Causation"* (arXiv 2605.17268), and *"Lost in Fog"* (arXiv 2605.21446), which finds that sensor perturbations expose reasoning fragility. Unfaithful explanations are worse than none for safety cases.
- **Reasoning latency vs. benefit**: Explicit chain-of-thought adds hundreds of tokens per decision. Whether latent or one-step reasoning (OneVL, TacitVLA) keeps the long-tail gains without the latency is unresolved.
- **Benchmark saturation and validity**: NAVSIM navtest and Bench2Drive scores rose rapidly in 2025. Pseudo-simulation (NAVSIM v2) and generative-world-model evaluation aim to close the remaining gap to reactive closed-loop testing, but none is yet validated against real-world safety outcomes.

---

## Further Reading

- [arXiv:2212.10156 UniAD](https://arxiv.org/abs/2212.10156) — CVPR 2023 Best Paper
- [arXiv:2309.17080 GAIA-1](https://arxiv.org/abs/2309.17080) — Wayve world model
- [arXiv:2402.12289 DriveVLM](https://arxiv.org/abs/2402.12289) — VLM integration
- [OpenDriveLab/UniAD GitHub](https://github.com/OpenDriveLab/UniAD) — official code
- [arXiv:2401.08658 E2E Planning Survey 2022-2023](https://arxiv.org/abs/2401.08658) — comprehensive survey
- [Wayve scaling GAIA-1 blog post](https://wayve.ai/thinking/scaling-gaia-1/) — 9B parameter results
- [arXiv:2410.23262 EMMA](https://arxiv.org/abs/2410.23262) — Waymo MLLM-based E2E driving
- [arXiv:2511.00088 Alpamayo-R1](https://arxiv.org/abs/2511.00088) — reasoning VLA with Chain of Causation
- [arXiv:2503.20523 GAIA-2](https://arxiv.org/abs/2503.20523) — multi-camera controllable world model
- [NAVSIM GitHub](https://github.com/autonomousvision/navsim) — benchmark code, v1/v2 and leaderboards
- [Bench2Drive GitHub](https://github.com/Thinklab-SJTU/Bench2Drive) — closed-loop multi-ability benchmark
- [DiffusionDrive GitHub](https://github.com/hustvl/DiffusionDrive) — truncated diffusion planner
