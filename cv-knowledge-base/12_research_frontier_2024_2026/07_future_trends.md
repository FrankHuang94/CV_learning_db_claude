# Future Trends in Computer Vision and Embodied AI (2024–2026)

> **Last Updated:** June 2026
> **Level:** Research Frontier
> **Related Sections:**
> - [../06_robotics_and_embodied_ai/11_generalist_vs_specialist.md](../06_robotics_and_embodied_ai/11_generalist_vs_specialist.md) — Architectural debate underlying embodied foundation models
> - [../04_3d_vision_and_scene/](../04_3d_vision_and_scene/) — 3D and 4D scene understanding foundations
> - [../05_multimodal_vision_language/](../05_multimodal_vision_language/) — VLM architectures that are being scaled
> - [../09_efficiency_and_deployment/](../09_efficiency_and_deployment/) — Tiny model deployment for edge robotics

---

## Overview

This section distills the research agenda that is actively driving funding, publications, and industrial investment in computer vision and embodied AI as of mid-2026. It is grounded in peer-reviewed papers, researcher statements, and observable engineering trends — not speculation. Each trend is presented as a problem statement, a summary of current evidence, and a frank assessment of what remains unsolved.

The organizing theme is that the field is transitioning from *perceiving* the world (classical CV) to *acting* within it (embodied AI), and the missing infrastructure for that transition defines most of the open problems below.

---

## Trend 1: Embodied Foundation Models at Scale

### The Bet

The central hypothesis driving the largest robotics investments of 2025–2026 is that a single foundation model trained on sufficiently large and diverse robot data will generalize across robot embodiments, task categories, and physical environments — analogous to how GPT-4 generalizes across domains of text. This is not yet proven, but there is growing empirical evidence.

### Current Evidence

Open X-Embodiment [Collaboration2023] demonstrated that cross-embodiment training — pooling data from 22 different robots across 33 institutions — yields meaningful transfer. RT-1-X improved ~50% in success rates compared to training on a single robot's data. RT-2-X achieved 3× emergent-skill success over RT-1-X by adding internet-scale pre-training.

Physical Intelligence's π₀ [Black2024] was trained on 10,000+ hours of robot demonstrations across 68 tasks and 7 robot configurations, and demonstrated state-of-the-art performance on deformable manipulation and long-horizon household tasks, categories that had previously resisted end-to-end learning.

Generalist AI's GEN-0 and GEN-1 [GeneralistAI2025] represent the first published scaling curves for robot manipulation, reporting performance on a benchmark suite trained on 270,000 hours of real-world manipulation data from 1,000+ homes, warehouses, and workplaces across multiple countries. GEN-1 achieves 99% success on held-in tasks (versus 62% for GEN-0 and 42% for a no-pretraining baseline), though scaling to 30–40 specialist tasks simultaneously shows a performance cliff (~55% and ~30% respectively), indicating that naive scaling alone is insufficient for combinatorial generalization.

### Data-Scale Estimates

Current best-guess estimates for reaching strong generalization across common household tasks:

```
Current frontier (2025):  ~10,000–50,000 hours of demonstrations
Estimated requirement:     ~500,000–2,000,000 hours
Current synthetic throughput (Genesis @ 43M FPS): ~9 months of
  real-time simulation in 11 hours of compute
```

Simulation fidelity remains the bottleneck: Genesis [Genesis2024] achieves 43 million frames per second on a single RTX 4090 and can train locomotion policies in 26 seconds, but the sim-to-real gap for contact-rich manipulation tasks (e.g., folding laundry, inserting flexible connectors) remains an active research problem.

### Timeline Uncertainty

No credible researcher has stated a specific timeline for a "general household robot." The current trajectory suggests that models with strong in-distribution generalization (familiar environments, familiar object categories) are achievable by 2027–2028, while out-of-distribution robustness in arbitrary homes remains a 5–10 year research problem.

---

## Trend 2: Spatial Intelligence as the Missing Capability

### The Problem

Vision-language models that score near-human on text benchmarks fail systematically on spatial reasoning tasks: identifying whether object A is to the left or right of object B, estimating metric distances, reasoning about occlusion, or understanding above/below/inside relationships. This is not a minor limitation — it is a prerequisite for any embodied AI that must navigate or manipulate in physical space.

A concrete example: GPT-4V and LLaVA-1.5, state-of-the-art VLMs, perform near-chance on metric distance estimation questions ("Which object is closer to the camera, the mug or the book?") [Chen2024-SpatialVLM].

### Key Work

**SpatialVLM** [Chen2024-SpatialVLM] (CVPR 2024, authored by Chen, Xu, Kirmani, Ichter, Sadigh, Guibas, Xia) introduced the first internet-scale 3D spatial reasoning dataset in metric space, generated by annotating spatial relationships from 3D reconstructions of indoor scenes. Co-training a VLM on this dataset produced the first model capable of answering quantitative spatial questions ("approximately how many meters is the cup from the edge of the table?") with reasonable accuracy. The approach enables chain-of-thought spatial reasoning and was demonstrated to improve downstream robot manipulation planning.

**SpatialBot** [SpatialBot2024] (ICRA 2025) integrates depth estimation directly into the VLM pipeline, using RGB-D inputs to provide grounded spatial grounding for robot manipulation. SpatialBot introduces a SpatialQA dataset combining depth perception and object grounding, supporting multi-frame inputs for dynamic scenes.

**SpatialRGPT** [SpatialRGPT2024] extends region-level spatial reasoning, allowing models to answer questions about specific image regions rather than whole-scene spatial relationships, which is critical for precise manipulation.

**3D-LLM** connects language models to 3D point cloud representations, enabling natural language queries over reconstructed scenes with spatial answers.

### Why This Is a Prerequisite for Household Robots

```
Task: "Put the blue mug on the shelf above the microwave."

Required spatial reasoning:
  1. Identify the microwave → object detection ✓ (solved)
  2. Identify "above" in 3D space → metric spatial reasoning ✗ (hard)
  3. Identify the specific shelf (one of N possible) → ordinal spatial ✗ (hard)
  4. Generate a reaching trajectory that clears the shelf edge → contact spatial ✗ (open problem)
```

Current VLMs solve step 1 reliably; steps 2–4 remain active research.

---

## Trend 3: 4D Scene Understanding — From Static 3D to Temporal Dynamic Worlds

### The Trajectory

The progression in scene representation has been:
- 2020–2022: NeRF (static 3D implicit representation from RGB images)
- 2023: 3D Gaussian Splatting (3DGS) — real-time rendering of static scenes
- 2024: 4D-GS — extending Gaussians to model dynamics over time
- 2025–2026: World models — predicting *future* scene states under robot actions

### Current Work

**4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** [Wu2024] (CVPR 2024) uses HexPlane to build temporal Gaussian features from monocular video, achieving 82 FPS at 800×800 on an RTX 3090 while modeling dynamic scene elements (moving objects, articulated motion).

**4D-Rotor Gaussian Splatting** [SIGGRAPH2024] represents dynamic scenes with anisotropic 4D XYZT Gaussians, which are temporally sliced to produce dynamic 3D Gaussians. This provides a unified representation for both geometric and temporal queries.

**ST-4DGS** [SIGGRAPH2024] focuses on spatial-temporal consistency — ensuring that a person walking across a scene maintains consistent geometry and texture across frames, which is required for stable robot scene representations.

### Connection to World Models for Robotics

The critical insight linking 4D scene understanding to robotics is that a robot needs not just a *representation* of the current scene, but a *predictive model* of how the scene will change under different actions:

```
World model goal:
  Given: scene state S_t, robot action a_t
  Predict: scene state S_{t+1}

4D representation enables:
  - Planning: "If I push this object left, what state results?"
  - Counterfactual reasoning: "Which action reaches the goal state fastest?"
  - Safe exploration: "Does any reachable state lead to damage?"
```

V-JEPA 2 [Meta2025] from Meta, released June 2025, addresses exactly this problem. A 1.2B-parameter model trained on 1 million hours of video and 1 million images learns abstract predictive representations of physical dynamics. In a second stage, action-conditioned learning on ~62 hours of robot data enables planning in image space. V-JEPA 2-AC was deployed zero-shot on Franka arms in two different labs and achieved pick-and-place via image-goal planning without collecting any environment-specific data — a significant demonstration of world model utility for robotics.

---

## Trend 4: Video as a Robot Data Engine

### The Core Idea

Internet video contains vast amounts of human manipulation behavior — cooking, assembling, cleaning, repairing — that encodes implicit physics priors, affordance knowledge, and task structure. Rather than collecting all robot data via expensive teleoperation, can we extract these priors from unlabeled video?

### Datasets and Methods

**Ego4D** [Grauman2022] (Facebook AI Research): 3,670 hours of egocentric video from 923 participants in 74 locations across 9 countries, covering daily activities including extensive hand-object interaction. Encoders pre-trained on Ego4D transfer more effectively to robot manipulation than ImageNet-pretrained encoders [LBW2024].

**HowTo100M** [Miech2019]: 136 million instructional video clips scraped from YouTube with ASR narrations. However, Ego4D narrations are significantly better aligned temporally with video content than HowTo100M ASR captions [EgoVLP2022], limiting HowTo100M's utility for precise action labeling.

**UniPi** [Du2023] (ICLR 2024): Casts sequential decision-making as text-conditioned video generation. A video prediction model generates future frames; an inverse dynamics model between frames recovers actions. Demonstrates planning in video space for maze navigation and block manipulation.

**SuSIE** [Black2023]: Uses a fine-tuned InstructPix2Pix model to generate goal images from language commands, then runs a downstream diffusion policy conditioned on the generated goal. Demonstrates language-conditioned long-horizon manipulation.

**V-JEPA** [Assran2024] (Meta, 2024): Video-JEPA predicts abstract feature representations of future video frames rather than raw pixels, avoiding the mode collapse and blurriness of pixel-prediction models. Shows strong transfer to action recognition benchmarks with limited fine-tuning.

**Latent Action Pre-Training from Videos (LAPT)** [LAPT2024]: Trains a VQ-VAE to discretize video-frame transitions into "latent actions," allowing pre-training of a language-conditioned policy from unlabeled video. Reduces robot demo requirements for downstream fine-tuning.

### The Action Label Problem

```
Gap: Internet video has visual content but no robot-executable action labels.

Approaches:
  (A) Inverse dynamics models: predict actions between consecutive frames
      Problem: ambiguous (many actions produce same visual transition)
  
  (B) Latent action spaces: learn compressed action representations
      from videos without grounding to real robot DOFs
      Problem: latent actions may not transfer to physical robot

  (C) Hand/body keypoint tracking: infer grasp pose and arm trajectory
      from video, then retarget to robot kinematics
      Problem: human arm ≠ robot arm; retargeting is approximate

  (D) Video generation as policy: predict video, then act to match
      Problem: requires accurate low-level control to execute
      generated videos in real time
```

None of these approaches has fully solved the action label problem. It remains one of the most actively pursued open problems in embodied AI.

---

## Trend 5: Physical Simulation Renaissance

### Context

Physical simulation for robot training is not new — MuJoCo has been used for RL since 2012. What is new in 2024–2026 is the combination of GPU-parallel simulation, differentiable physics, and generative scene creation that makes simulation practically viable as a data source at the scales required for foundation models.

### Current Platforms

**Genesis** [Genesis2024] (December 2024, open source): A universal physics engine built entirely in Python. Benchmarks:
- 43 million FPS simulating a Franka arm on a single RTX 4090
- 430,000× faster than real-time
- 10–80× faster than Isaac Gym/Sim/Lab and MuJoCo MJX
- Trains locomotion policies in 26 seconds
- Supports MPM (soft bodies), SPH (fluids), FEM (deformable), PBD, Stable Fluids, and rigid bodies in a unified framework
- Cross-platform: CPU, NVIDIA GPU, AMD GPU, Apple Metal

**Isaac Lab** [NVIDIA2024]: GPU-accelerated RL framework built on Isaac Sim. Supports massively parallel environment instances, enabling thousands of simultaneous robot policy rollouts. Used by NVIDIA for GR00T N1 synthetic data generation.

**MuJoCo MJX / MuJoCo Playground** [DeepMind2025]: MJX is MuJoCo compiled to JAX/XLA, enabling GPU/TPU-parallel simulation with autodiff through the physics. MuJoCo Playground (January 2025) won the Outstanding Demo Paper Award at RSS 2025. Supports quadrupeds, humanoids, dexterous hands, and arms with zero-shot sim-to-real transfer from state and pixel inputs.

**Newton** [NVIDIA-Google-Disney2025]: A unified physics engine announced by NVIDIA, Google DeepMind, and Disney Research, building on MuJoCo-Warp. Claims up to 152× speedup for locomotion and 313× for manipulation on RTX 4090 compared to CPU MuJoCo.

### When Is Sim Data Sufficient?

The sim-to-real transfer question is task-dependent:

| Task class | Sim-to-real gap | Current status |
|---|---|---|
| Locomotion (rigid body) | Low | Solved: Isaac Lab policies deploy zero-shot [Kumar2021] |
| Pick-and-place (rigid objects) | Medium | Largely solved with domain randomization |
| Contact-rich assembly | High | Active research; Genesis FEM promising |
| Deformable manipulation (cloth, rope) | Very High | Partially solved; Genesis MPM shows progress |
| Liquid handling | Extremely High | Open problem |
| In-hand dexterous manipulation | High | Active research; requires tactile feedback |

The consensus is that sim data is *necessary but not sufficient* for contact-rich manipulation, and that real-world data remains required for tasks involving deformable objects, liquids, or highly variable object properties (friction, compliance).

---

## Trend 6: Cross-Modal Tactile-Visual Fusion

### Why Dexterity Needs Touch

Vision alone cannot reliably determine:
- Grip force (is the object slipping?)
- Object compliance (is it rigid or deformable?)
- Contact geometry (how is the object touching the finger surface?)
- Texture at contact point

Humans have ~17,000 mechanoreceptors per hand. Dexterous robot hands without equivalent sensing cannot achieve human-level manipulation reliability regardless of visual system quality.

### Sensor Technology

**GelSight** (MIT/GelSight Inc.): A camera-based tactile sensor that images the deformation of a gel elastomer under contact, recovering 3D contact geometry at sub-millimeter resolution. The GelSight Mini is commercially available; a Phase II SBIR contract was awarded to develop ruggedized versions for industrial deployment.

**DIGIT** [Lambeta2020] (Meta AI / GelSight): A compact, reproducible high-resolution tactile sensor designed for multi-fingered hands. Published specifications: 18×24 mm sensing area, 60 Hz frame rate, USB connectivity. Used extensively in Meta's dexterous manipulation research. The Digit 360 (2024) extends this to a full fingertip shape with 18+ sensing modalities.

**DVTac, FingerVision**: Alternative vision-based tactile sensors with varying resolution/form factor tradeoffs.

### Visuotactile Models

The key research challenge is fusing visual and tactile information in a shared representation that enables policies to reason about contact:

```python
# Conceptual visuotactile fusion
visual_features = vision_encoder(rgb_image)      # global scene context
tactile_features = tactile_encoder(gel_image)    # local contact detail
proprioceptive = mlp(joint_angles, velocities)

combined = cross_attention(
    query=proprioceptive,
    key_value=concat(visual_features, tactile_features)
)
action = policy_head(combined)
```

Recent work [ViTaSCOPE2025] uses implicit neural representations to estimate in-hand pose and extrinsic contacts from visuotactile observations. FlowTouch [FlowTouch2025] addresses view-invariant tactile prediction, enabling policies trained with one camera viewpoint to generalize to others.

The fundamental open problem is the lack of large-scale visuotactile datasets. Most published work trains on fewer than 1,000 demonstrations per task; a foundation model for manipulation requiring real tactile feedback would need orders of magnitude more data.

---

## Trend 7: Tiny Foundation Models for Edge Robotics

### The Deployment Constraint

Production robots — particularly mobile platforms and low-cost cobots — have strict compute budgets:

```
NVIDIA Jetson Orin NX:  16 GB LPDDR5, 20 TOPS, ~25W TDP
NVIDIA Jetson Orin AGX: 32–64 GB, 275 TOPS, ~60W TDP
Typical large VLA:      7B–55B params, requires A100/H100 (400W+)
```

7B-parameter VLAs exceed the memory and power budgets of Jetson-class hardware. The field has responded with distillation, quantization, and architecture redesign.

### Current Work

**EdgeVLA** [EdgeVLA2025]: Eliminates autoregressive end-effector position prediction (the main inference bottleneck in VLAs) in favor of a separate lightweight head. Achieves 7× speedup in inference. Runs in 5–6 ms on Jetson AGX Orin with model sizes down to 56–86 MB. Uses Qwen2-0.5B as the language backbone, enabling sub-2 GB memory footprint.

**MobileVLM** (Meituan/BAAI, 2023–2024): Sub-3B vision-language model optimized for mobile deployment, using an efficient visual projector and grouped-query attention. MobileVLM V2 adds task-specific restore tokens.

**SwiftVLA**: Achieves 1–7× parameter reduction and 10–18× memory reduction compared to large VLA baselines while maintaining comparable task success rates through structured pruning and knowledge distillation.

**NanoVLA** [NanoVLA2025]: Routes decoupled vision-language understanding to a nano-sized policy, separating perception (handled by a quantized VLM) from action generation (handled by a tiny reactive network).

### The Performance-Efficiency Tradeoff

```
Model size:   55B      7B       1B      500M      56M
Success rate: RT-2-X   OpenVLA  ~SOTA   EdgeVLA   ~75% of OpenVLA
Inference:    ~500ms   ~150ms   ~50ms   ~10ms     ~5ms
Platform:     H100     A100     L40     Orin AGX  Orin NX
```

(Numbers are approximate; exact benchmarks vary by task and environment.)

The current frontier is maintaining >80% of large-model performance at sub-100M parameters on Jetson-class hardware. This is achievable for narrow task domains but remains an open problem for general household manipulation.

---

## Trend 8: Neuromorphic and Event-Based Vision

### Motivation

Standard CMOS cameras have two fundamental limitations for high-speed robot control:
1. **Latency**: Frame readout requires ~33 ms at 30 FPS, creating unavoidable control lag
2. **Motion blur**: Fast-moving objects or robot end-effectors blur at standard shutter speeds

Event cameras (dynamic vision sensors) address both:

```
Standard frame camera:
  Output: Full frame I(x,y,t) at fixed intervals (30–120 Hz)
  Latency: Frame period (8–33 ms)
  Power: ~1–5W for sensor

Event camera:
  Output: Stream of events (x, y, t, polarity) triggered by
          log-intensity changes: ΔL(x,y,t) > θ
  Latency: <1 μs per event
  Dynamic range: >120 dB (vs. ~60 dB for CMOS)
  Power: ~10–100 mW
```

### Hardware

**Prophesee EVK4** (Prophesee): 1.28 Mpixel event camera, 120 dB dynamic range, available commercially.
**DVXplorer** (iniVation): VGA resolution, 110 dB dynamic range, 165 million events per second throughput, USB 3.0 connectivity.
**DVXplorer Lite**: QVGA resolution, lower cost entry point for research.

### Applications in Robotics

- **Surgical robotics**: Tracking instrument motion and patient breathing with sub-millimeter precision using event streams
- **High-speed grasping**: Tracking fast-moving objects on conveyor belts where standard cameras blur
- **Slip detection**: Detecting early tactile slip events before grasp failure propagates
- **Drone navigation**: Obstacle avoidance at high speed in low-light conditions

### The Missing Foundation Model

Despite demonstrated advantages, there are no large-scale foundation models trained on event camera data as of mid-2026. Key obstacles:
1. **Data scarcity**: Event datasets contain thousands of sequences; ImageNet-scale event datasets do not exist
2. **Representation heterogeneity**: Event streams are sparse, asynchronous, and variable-density — poor fit for standard CNN/ViT architectures without conversion to frame-based representations (voxel grids, time surfaces)
3. **Simulation gap**: Simulating realistic event data from synthetic scenes remains approximate

This is one of the clearest gaps between sensor capability and machine learning readiness in the field.

---

## Trend 9: Safety and Alignment in Embodied AI

### Physical vs. Digital Harm

The safety stakes for embodied AI differ categorically from those for language models:

| | Language Model | Embodied AI |
|---|---|---|
| Failure mode | Incorrect/harmful text | Physical injury, property damage |
| Reversibility | Retractions possible | Dropped objects, collision damage often irreversible |
| Harm radius | Reader | Co-located humans, infrastructure |
| Regulatory regime | Content moderation | Product liability, CE marking, ISO 10218 (industrial robots) |

### Current Approaches

**Constraint-Based Safety Layers**: Physical safety constraints (workspace limits, velocity limits, force limits) encoded as hard constraints on the output of the neural policy. The policy operates freely within the safe region; a constraint-enforcing layer clips or rejects actions that violate limits. Drake [Tedrake2019] supports formal constraint verification at design time.

**RLHF for Robot Policies**: Adapting reinforcement learning from human feedback to physical action spaces, where human evaluators rate trajectory quality rather than text responses. Challenges: human evaluation of 3D motion trajectories is cognitively demanding; reward models trained on trajectory ratings may generalize poorly to novel physical configurations.

**Formal Verification of Contact Dynamics**: Recent work [FormalMethods2026] surveys co-learning of planning and control policies constrained by differentiable logic specifications. LTL (Linear Temporal Logic) specifications derived from natural language safety requirements are translated to Büchi automata via LLM, then constrained decoding ensures policy outputs remain on accepting paths. This approach is demonstrated on simple manipulation but has not yet scaled to household environments.

**Provable Probabilistic Safety** [ProvoSafe2025]: Derived from conformal prediction, provides probabilistic guarantees ("the robot will not exceed safe force limits with probability ≥ 0.99") without requiring full formal verification of the neural network.

### The Specification Problem

The hardest open problem in embodied AI safety is not technical but conceptual:

> **How do you specify what a household robot should and should not do, comprehensively enough that the specification is safe, but narrowly enough that it is not cripplingly restrictive?**

"Do not harm humans" is underspecified (what counts as harm? at what probability threshold?). A comprehensive formal specification of household safety requirements is not known to exist. This is the physical analogue of the AI alignment problem, with the additional constraint that violations are immediate and potentially irreversible.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|---|---|---|---|---|
| SpatialVLM: Endowing VLMs with Spatial Reasoning Capabilities | Chen, Xu, Kirmani, Ichter, Sadigh, Guibas, Xia | 2024 | CVPR 2024 | First internet-scale metric spatial reasoning dataset; VLM co-training for distance estimation |
| 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering | Wu et al. | 2024 | CVPR 2024 | HexPlane temporal Gaussians; 82 FPS at 800×800 on RTX 3090 |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | Meta AI | 2025 | arXiv 2025 | 1.2B world model; 1M hours video + 62h robot data; zero-shot pick-and-place |
| UniPi: Learning Universal Policies via Text-Guided Video Generation | Du et al. | 2023 | ICLR 2024 | Video generation as policy; inverse dynamics recovers actions from predicted frames |
| EdgeVLA: Efficient Vision-Language-Action Models | — | 2025 | arXiv | 7× inference speedup; 5–6 ms on Jetson AGX Orin; 56–86 MB model |
| Genesis: Universal Physics Engine | Genesis Team | 2024 | Open source | 43M FPS on RTX 4090; unified MPM/SPH/FEM/rigid; locomotion in 26 seconds |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | NVIDIA Research | 2025 | arXiv 2025 | Dual-system humanoid VLA; 780k synthetic trajectories in 11 hours |
| Formal Methods in Robot Policy Learning and Verification | — | 2026 | Survey | LTL-based safety specifications; differentiable logic constraints; Büchi decoding |

---

## Benchmark Performance

| Model | Task | Metric | Score | Notes |
|---|---|---|---|---|
| SpatialVLM | Metric distance estimation (indoor scenes) | Accuracy vs. GPT-4V | Significant improvement over GPT-4V baseline | First model with quantitative spatial estimation |
| 4D-GS (CVPR 2024) | Dynamic scene novel view synthesis | FPS @ 800×800 | 82 FPS | RTX 3090; competitive PSNR with dynamic NeRF methods |
| V-JEPA 2-AC | Zero-shot pick-and-place (Franka) | Success rate | Demonstrated in two different labs | No task-specific data collected in deployment labs |
| Genesis | Franka arm simulation | Frames per second | 43,000,000 FPS | Single RTX 4090; 430,000× real-time |
| EdgeVLA | Tabletop manipulation | Inference latency | 5–6 ms | Jetson AGX Orin; 7× faster than baseline VLA |
| OpenVLA (7B) | 29-task manipulation suite | Absolute success | +16.5% over RT-2-X (55B) | 7× fewer parameters |

---

## Pros & Cons

| Aspect | Current Strengths | Current Weaknesses |
|---|---|---|
| **Embodied foundation models** | Strong in-distribution generalization; OXE-scale training validates cross-embodiment transfer; synthetic data (Genesis) now scalable | Out-of-distribution generalization remains poor; no solved contact-rich household tasks; data ceiling ~10× below NLP equivalent |
| **Spatial intelligence** | SpatialVLM demonstrates metric estimation is learnable; RGB-D integration (SpatialBot) improves grounding | Most VLMs still fail ordinal spatial questions near-randomly; 3D spatial reasoning not yet integrated into mainstream VLA pipelines |
| **4D scene / world models** | Real-time 4D-GS at 82 FPS; V-JEPA 2 zero-shot robot deployment; Genesis enables fast counterfactual simulation | Sim-to-real gap for deformable/liquid; world model action conditioning is approximate; predicting contact events reliably is open |
| **Safety/alignment** | Constraint layers deployable today; provable probabilistic bounds available; formal LTL methods demonstrated on simple tasks | Specification problem unsolved; RLHF for physical actions poorly validated; no ISO-equivalent safety standard for general-purpose AI-controlled robots yet ratified |

---

## Open Problems & Research Gaps

1. **The action label problem at scale.** Extracting robot-executable actions from unlabeled internet video — the key to 100× more training data — remains unsolved. Latent action pre-training is promising but has not been validated to match supervised demonstration quality on complex manipulation tasks.

2. **Deformable and liquid sim-to-real transfer.** Genesis and MuJoCo MJX handle rigid bodies and soft bodies, but reliably training policies that transfer to real fabric, rope, granular media, and liquids is an open problem. The fundamental issue is that small errors in material parameters compound across contact sequences.

3. **Event camera foundation models.** No large-scale pre-trained model for event camera data exists despite demonstrated advantages for high-speed robotics. The absence of an "ImageNet for event cameras" blocks the application of standard transfer learning methodology to neuromorphic sensing.

4. **Spatial reasoning in multi-step manipulation.** Current spatial VLMs answer static image questions. Spatial reasoning across a multi-step manipulation sequence — tracking how a scene's spatial relationships change as the robot acts — is barely studied.

5. **Tactile data at foundation model scale.** The largest published visuotactile datasets contain tens of thousands of contact examples. A foundation model for dexterous manipulation would require millions to billions of tactile observations. Synthetic tactile generation (simulating GelSight images from contact geometry) is a promising but underexplored direction.

6. **Formal safety for open-world robot deployment.** Translating natural language safety requirements into enforceable formal specifications, and verifying that a neural policy respects those specifications in open-world environments, does not yet have a satisfactory solution. This is arguably the gating problem for deploying general-purpose robots in homes and hospitals.

7. **Cross-embodiment policy transfer theory.** Empirical evidence shows that cross-embodiment training helps, but there is no theoretical account of when and why transfer should succeed or fail. A theory that predicts transfer quality from morphological similarity, task similarity, and training data statistics would enable principled architecture and data-mixing decisions.

---

## Further Reading

- [V-JEPA 2 — Meta AI](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) — World model for robot planning from video
- [Genesis Physics Engine](https://genesis-world.readthedocs.io/) — 43M FPS open-source universal simulator
- [SpatialVLM (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.html) — Spatial reasoning in VLMs
- [GR00T N1 arXiv](https://arxiv.org/abs/2503.14734) — Open humanoid foundation model with dual-system architecture
- [Formal Methods in Robot Policy Learning (Survey 2026)](https://arxiv.org/pdf/2602.06971) — LTL, Büchi automata, and differentiable safety constraints
- [Recent Event Camera Innovations Survey (ECCV 2024 Workshop)](https://arxiv.org/html/2408.13627v2) — Comprehensive review of neuromorphic vision for robotics
