# Future Trends in Computer Vision and Embodied AI (2024–2026)

> **Last Updated:** October 2026
> **Level:** Research Frontier
> **Related Sections:**
> - [../06_robotics_and_embodied_ai/11_generalist_vs_specialist.md](../06_robotics_and_embodied_ai/11_generalist_vs_specialist.md) — Architectural debate underlying embodied foundation models
> - [../04_3d_vision_and_scene/](../04_3d_vision_and_scene/) — 3D and 4D scene understanding foundations
> - [../05_multimodal_vision_language/](../05_multimodal_vision_language/) — VLM architectures that are being scaled
> - [../09_efficiency_and_deployment/](../09_efficiency_and_deployment/) — Tiny model deployment for edge robotics

> **🆕 October 2026 update:**
> - Added a **"Reality Check (October 2026)"** section that scores each trend's predictions against what actually happened from mid-2025 to October 2026.
> - Updated evidence per trend: π\*0.6 / π0.7, Gemini Robotics 2, GR00T N1.7, Generalist GEN-0/GEN-1 (corrected data figures), SenseNova-SI and Cambrian-S for spatial intelligence, D4RT (CVPR 2026 Best Paper) and VGGT-Ω for 4D, World Action Models (DreamZero, Cosmos Policy) for video-as-data.
> - Simulation and edge: Genesis World 1.0, Newton under the Linux Foundation, Cosmos 3 (incl. 4B Cosmos3-Edge), Jetson AGX Thor.
> - World models went mainstream (Genie 3, World Labs Marble, Cosmos 3, V-JEPA 2.1); event-camera and tactile foundation models remain largely missing.

---

## Overview

This section distills the research agenda that is actively driving funding, publications, and industrial investment in computer vision and embodied AI as of October 2026. It is grounded in peer-reviewed papers, researcher statements, and observable engineering trends — not speculation. Each trend is presented as a problem statement, a summary of current evidence, and a frank assessment of what remains unsolved.

The organizing theme is that the field is transitioning from *perceiving* the world (classical CV) to *acting* within it (embodied AI), and the missing infrastructure for that transition defines most of the open problems below. Because several of these trends were written as forecasts, the [Reality Check (October 2026)](#reality-check-october-2026-predictions-vs-outcomes) section near the end compares them with what was actually published and deployed between mid-2025 and October 2026.

---

## Trend 1: Embodied Foundation Models at Scale

### The Bet

The central hypothesis driving the largest robotics investments of 2025–2026 is that a single foundation model trained on sufficiently large and diverse robot data will generalize across robot embodiments, task categories, and physical environments — analogous to how GPT-4 generalizes across domains of text. This is not yet proven, but there is growing empirical evidence.

### Current Evidence

Open X-Embodiment [Collaboration2023] demonstrated that cross-embodiment training — pooling data from 22 different robots across 33 institutions — yields meaningful transfer. RT-1-X improved ~50% in success rates compared to training on a single robot's data. RT-2-X achieved 3× emergent-skill success over RT-1-X by adding internet-scale pre-training.

Physical Intelligence's π₀ [Black2024] was trained on 10,000+ hours of robot demonstrations across 68 tasks and 7 robot configurations, and demonstrated state-of-the-art performance on deformable manipulation and long-horizon household tasks, categories that had previously resisted end-to-end learning.

Generalist AI's GEN-0 and GEN-1 [GeneralistAI2025] represent the first published scaling curves for robot manipulation. **GEN-0** (November 2025) was pre-trained on ~270,000 hours of real-world manipulation data, collected largely with human-worn "data hands" grippers across homes, warehouses and workplaces, and was presented as evidence that scaling laws apply to robot learning. **GEN-1** (April 2026) was trained from scratch on roughly **half a million hours** of such data; Generalist reports average success rising to **99%** on tasks where previous models reached **64%**, ~3× faster execution, and ~1 hour of robot data per task. The company notes GEN-1 targets short (typically sub-minute) tasks with limited variability, so these results do not yet establish combinatorial or long-horizon generalization.

**2026 evidence.** Three further results strengthen the bet without proving it. Physical Intelligence's **π0.7** (April 2026) reports *early signs of compositional generalization*—recombining skills to operate unseen appliances and to fold laundry on a platform with no folding data—while **π\*0.6** (November 2025) shows VLAs can improve from their own deployment experience via advantage-conditioned RL (Recap), roughly doubling throughput on hard tasks. Google DeepMind's **Gemini Robotics 2** (July 2026) extends a frontier-model VLA to whole-body humanoid control. NVIDIA's **GR00T N1.7** (Apache 2.0) adds 20K hours of egocentric human video to robot pretraining. All are company-reported (see [VLA & Embodied AI 2025–2026](./04_vla_embodied_2025_2026.md)).

### Data-Scale Estimates

Current best-guess estimates for reaching strong generalization across common household tasks:

```
Current frontier (2025):  ~10,000–50,000 hours of demonstrations
Frontier (2026):          ~20,000 h robot teleop (e.g., LingBot-VLA) up to
                          ~500,000 h human-worn gripper data (GEN-1),
                          plus ~20,000 h human video (GR00T N1.7)
Estimated requirement:     ~500,000–2,000,000 hours
Current synthetic throughput (Genesis @ 43M FPS): ~9 months of
  real-time simulation in 11 hours of compute
```

Simulation fidelity remains the bottleneck: Genesis [Genesis2024] achieves 43 million frames per second on a single RTX 4090 and can train locomotion policies in 26 seconds, but the sim-to-real gap for contact-rich manipulation tasks (e.g., folding laundry, inserting flexible connectors) remains an active research problem.

### Timeline Uncertainty

No credible researcher has stated a specific timeline for a "general household robot." The current trajectory suggests that models with strong in-distribution generalization (familiar environments, familiar object categories) are achievable by 2027–2028, while out-of-distribution robustness in arbitrary homes remains a 5–10 year research problem. *October 2026 note:* raw data scale reached the lower bound of the estimate above sooner than expected (via human-worn data collection rather than teleoperation), but published success rates remain concentrated on short, in-distribution tasks; the 2027–2028 in-distribution forecast still looks plausible, and nothing published changes the out-of-distribution estimate.

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

### 2025–2026 Progress

The diagnosis moved from single images to video: **VSI-Bench** (*Thinking in Space*, CVPR 2025) showed that MLLMs watching a room-scale video build only local "cognitive maps" and that linguistic chain-of-thought does not help. The remedy that worked best was **scale and curation of spatial data**: **Cambrian-S** (VSI-590K, Nov 2025) and **SenseNova-SI** (8M samples, CVPR 2026) produced 7–8B open models that score ~67–69 on VSI-Bench versus 55.0 for GPT-5 under a common evaluation. At the same time, geometry encoders from feed-forward reconstruction (VGGT, CUT3R) were fused into MLLMs, and robotics labs shipped dedicated **embodied-reasoning** models (Gemini Robotics-ER 1.5 → ER 2, Molmo2-ER, Cosmos Reason) as the spatial layer above VLAs. Multi-image and perspective-taking benchmarks (MMSI-Bench, ViewSpatial) remain far from solved (best ~40–60). See [Spatial Intelligence](./06_spatial_intelligence.md).

---

## Trend 3: 4D Scene Understanding — From Static 3D to Temporal Dynamic Worlds

### The Trajectory

The progression in scene representation has been:
- 2020–2022: NeRF (static 3D implicit representation from RGB images)
- 2023: 3D Gaussian Splatting (3DGS) — real-time rendering of static scenes
- 2024: 4D-GS — extending Gaussians to model dynamics over time
- 2025–2026: World models — predicting *future* scene states under robot actions
- 2025–2026 (in parallel): **feed-forward 4D reconstruction** — VGGT (CVPR 2025 Best Paper) → D4RT (CVPR 2026 Best Paper), which recovers depth, 3D tracks and cameras from a single video in one model

### Current Work

**4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** [Wu2024] (CVPR 2024) uses HexPlane to build temporal Gaussian features from monocular video, achieving 82 FPS at 800×800 on an RTX 3090 while modeling dynamic scene elements (moving objects, articulated motion).

**4D-Rotor Gaussian Splatting** [SIGGRAPH2024] represents dynamic scenes with anisotropic 4D XYZT Gaussians, which are temporally sliced to produce dynamic 3D Gaussians. This provides a unified representation for both geometric and temporal queries.

**ST-4DGS** [SIGGRAPH2024] focuses on spatial-temporal consistency — ensuring that a person walking across a scene maintains consistent geometry and texture across frames, which is required for stable robot scene representations.

**D4RT** [Zhang2026] (Google DeepMind / UCL / Oxford, CVPR 2026 Best Paper) encodes a video once into a global scene representation and answers independent queries for the 3D position of any pixel at any time in any camera frame, unifying depth, point-cloud, camera-pose and 3D-tracking estimation; DeepMind reports up to 300× efficiency gains over prior methods. **VGGT-Ω** (CVPR 2026 Best Paper Finalist) shows that feed-forward reconstruction of static and dynamic scenes scales predictably with model and data size. See [4D and Dynamic Scenes](./05_4d_and_dynamic_scenes.md).

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

V-JEPA 2 [Meta2025] from Meta, released June 2025, addresses exactly this problem. A 1.2B-parameter model trained on 1 million hours of video and 1 million images learns abstract predictive representations of physical dynamics. In a second stage, action-conditioned learning on ~62 hours of robot data enables planning in image space. V-JEPA 2-AC was deployed zero-shot on Franka arms in two different labs and achieved pick-and-place via image-goal planning without collecting any environment-specific data — a significant demonstration of world model utility for robotics. **V-JEPA 2.1** (March 2026) updated the recipe to learn high-quality, temporally consistent *dense* features (dense predictive loss, deep self-supervision, multi-modal tokenizers).

**Generative world models (2025–2026).** In parallel, pixel-space and 3D-space world models matured rapidly: Google DeepMind's **Genie 3** (August 2025) generates interactive worlds in real time (720p, 24 fps) that stay consistent for minutes; World Labs launched **Marble** (November 2025), which produces persistent, exportable 3D worlds; NVIDIA's **Cosmos 3** (May 2026) merges reasoning, world prediction, domain transfer and action generation into one model family. Reported 2026 developments—e.g., broader public access to Genie-based world exploration, Waymo building a driving world model on Genie 3, and Yann LeCun's world-model start-up AMI Labs raising a large initial round—indicate that world models have become a primary investment target, though these business details come from press coverage rather than technical reports.

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

### 2025–2026 Progress

Two pragmatic answers gained traction. (1) **Shared action spaces for humans and robots**: NVIDIA's GR00T N1.7 represents actions as *relative end-effector deltas* consistent across human and robot data, which lets it pretrain on 20K hours of EgoScale egocentric human video; Generalist sidesteps retargeting by having humans wear robot-like "data hands" grippers, reaching ~500K hours. (2) **Video generation as policy (approach D) became competitive**: *World Action Models* that jointly predict video and actions—NVIDIA's **Cosmos Policy** and **DreamZero**—gained traction, with DreamZero reaching the top of the real-world RoboArena and MolmoSpaces leaderboards in early 2026 and adapting to new robots from ~30 minutes of play data. Inference cost (seconds per step on an H100 without caching) is the main remaining obstacle for approach D.

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

**Newton** [NVIDIA-Google-Disney2025]: A unified physics engine announced by NVIDIA, Google DeepMind, and Disney Research, building on MuJoCo-Warp. Claims up to 152× speedup for locomotion and 313× for manipulation on RTX 4090 compared to CPU MuJoCo. By 2026 Newton is an open-source **Linux Foundation** project (Apache-2.0) built on NVIDIA Warp with **MuJoCo Warp** as its primary backend, emphasizing GPU computation, OpenUSD support and differentiability.

**2026 updates.** The Genesis team released **Genesis World 1.0** (May 2026) together with a position piece on the role of simulation in scalable robotics. On the generative side, **Cosmos 3** (May 2026; 64B Super, 16B Nano, 4B Edge) and Cosmos-Predict 2.5 action-conditioned models are used to synthesize robot training data and as action-conditioned learned simulators, blurring the line between physics simulation and world models.

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

In industry, Figure's **Helix 02** (early 2026) highlights tactile sensing and palm cameras for fine manipulation such as extracting pills and dispensing small syringe volumes (company-reported), but no visuotactile *foundation model* comparable to vision-only VLAs had been published as of October 2026.

The fundamental open problem is the lack of large-scale visuotactile datasets. Most published work trains on fewer than 1,000 demonstrations per task; a foundation model for manipulation requiring real tactile feedback would need orders of magnitude more data.

---

## Trend 7: Tiny Foundation Models for Edge Robotics

### The Deployment Constraint

Production robots — particularly mobile platforms and low-cost cobots — have strict compute budgets:

```
NVIDIA Jetson Orin NX:  16 GB LPDDR5, 20 TOPS, ~25W TDP
NVIDIA Jetson Orin AGX: 32–64 GB, 275 TOPS, ~60W TDP
NVIDIA Jetson AGX Thor: 128 GB, up to ~2,070 FP4 TFLOPS, 40–130W (Aug 2025)
Typical large VLA:      7B–55B params, requires A100/H100 (400W+)
```

7B-parameter VLAs exceed the memory and power budgets of Jetson-class hardware. The field has responded with distillation, quantization, and architecture redesign.

### Current Work

**EdgeVLA** [EdgeVLA2025]: Eliminates autoregressive end-effector position prediction (the main inference bottleneck in VLAs) in favor of a separate lightweight head. Achieves 7× speedup in inference. Runs in 5–6 ms on Jetson AGX Orin with model sizes down to 56–86 MB. Uses Qwen2-0.5B as the language backbone, enabling sub-2 GB memory footprint.

**MobileVLM** (Meituan/BAAI, 2023–2024): Sub-3B vision-language model optimized for mobile deployment, using an efficient visual projector and grouped-query attention. MobileVLM V2 adds task-specific restore tokens.

**SwiftVLA**: Achieves 1–7× parameter reduction and 10–18× memory reduction compared to large VLA baselines while maintaining comparable task success rates through structured pruning and knowledge distillation.

**NanoVLA** [NanoVLA2025]: Routes decoupled vision-language understanding to a nano-sized policy, separating perception (handled by a quantized VLM) from action generation (handled by a tiny reactive network).

**2025–2026 developments.** The hardware ceiling rose with **Jetson AGX Thor** (August 2025), and vendors began shipping edge-targeted variants of frontier stacks: **GR00T N1.7** runs inference on a single 16 GB-class GPU including Jetson AGX Thor/Orin with ONNX/TensorRT export; **Gemini Robotics On-Device** (June 2025) and **On-Device 2** (July 2026) run locally on robots; **Cosmos3-Edge** (4B, July 2026) targets real-time robot policy and visual reasoning on Jetson; and **SmolVLA** (~0.45B, Hugging Face) targets consumer GPUs. Research on extreme compression continues, e.g., **ActQuant** (2026) explores sub-4-bit, action-guided quantization of VLAs.

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

Despite demonstrated advantages, there are no large-scale foundation models trained on event camera data as of October 2026 (to our knowledge, no event-camera model at the scale of VGGT, DINOv3 or V-JEPA 2 has been published). Key obstacles:
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

**Safety as a reasoning-layer capability (2026).** Frontier embodied-reasoning models now report safety-specific evaluations: Google's **Gemini Robotics ER 2** reports 97.9% on safety-instruction following (vs. 47.2% for ER 1.6), and ER 1.6 reported improved hazard identification. These are vendor benchmarks of *semantic* safety understanding, not guarantees on physical behavior. Two 2026 surveys—on VLA safety (threats, evaluations, mechanisms; arXiv:2604.23775) and on risks, attacks and defenses in embodied AI (arXiv:2605.02900)—map adversarial and jailbreak risks that are specific to language-conditioned robots.

### The Specification Problem

The hardest open problem in embodied AI safety is not technical but conceptual:

> **How do you specify what a household robot should and should not do, comprehensively enough that the specification is safe, but narrowly enough that it is not cripplingly restrictive?**

"Do not harm humans" is underspecified (what counts as harm? at what probability threshold?). A comprehensive formal specification of household safety requirements is not known to exist. This is the physical analogue of the AI alignment problem, with the additional constraint that violations are immediate and potentially irreversible.

---

## Reality Check (October 2026): Predictions vs. Outcomes

The trends above were first written as forecasts. The table scores them against verifiable releases between mid-2025 and October 2026.

| Trend | What this file predicted / assumed | What actually happened (mid-2025 → Oct 2026) | Verdict |
|---|---|---|---|
| 1. Embodied foundation models | Growing evidence for cross-embodiment generalization; data scale the bottleneck | π\*0.6 (RL from experience), π0.7 (early compositional generalization), Gemini Robotics 2 (whole-body), GR00T N1.7 (open, human-video pretraining); GEN-1 at ~500K hours | **On track / faster on data**; OOD robustness still unproven, results mostly self-reported |
| 2. Spatial intelligence | VLMs fail spatial reasoning; synthetic 3D data is the main lever | Video spatial benchmarks (VSI-Bench, VSI-SUPER, MMSI); 8M-sample spatial data lets 8B open models beat GPT-5 on VSI-Bench; embodied-reasoning models productized | **Partly realized** — large gains on benchmarks, unclear transfer to manipulation |
| 3. 4D scenes → world models | 4D-GS extends to predictive world models | Feed-forward 4D reconstruction won CVPR 2026 Best Paper (D4RT); Genie 3, Marble, Cosmos 3 made world models mainstream | **Exceeded** on reconstruction speed and generative worlds; action-conditioned physical accuracy still weak |
| 4. Video as robot data engine | Action-label problem unsolved | Shared human–robot action spaces (GR00T N1.7), wearable data capture (Generalist), World Action Models top RoboArena | **Meaningful progress** — not solved, but approach (D) became competitive |
| 5. Simulation renaissance | GPU/differentiable sim + generative scenes become a data source | Newton (Linux Foundation, MuJoCo Warp backend), Genesis World 1.0, Cosmos 3 for synthetic data | **On track**; deformables/liquids sim-to-real still open |
| 6. Tactile–visual fusion | Needed for dexterity; data-limited | Tactile sensing in commercial humanoids (Helix 02); no visuotactile foundation model published | **Slower than hoped** |
| 7. Tiny/edge foundation models | Compression to Jetson budgets | Jetson AGX Thor; GR00T N1.7 on Thor/Orin; Gemini Robotics On-Device 2; Cosmos3-Edge (4B); SmolVLA | **On track**, driven by larger edge hardware as much as by compression |
| 8. Event-based vision | No event-camera foundation model | Still none at foundation scale | **Unchanged** |
| 9. Safety and alignment | Specification problem unsolved | Vendor safety-reasoning benchmarks (ER 2); VLA safety surveys; no general-purpose robot safety standard broadly ratified | **Unchanged in substance** |

The main surprise relative to the original forecasts is how quickly **world models and feed-forward 3D/4D reconstruction** moved from research to the center of both academic awards and industrial roadmaps, and how much of the 2026 data scaling came from **human-centric data** (egocentric video, wearable grippers) rather than robot teleoperation.

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
| π\*0.6: a VLA That Learns From Experience | Physical Intelligence | 2025 | arXiv:2511.14759 | Recap: advantage-conditioned RL from rollouts + interventions |
| π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | Physical Intelligence | 2026 | arXiv:2604.15483 | Early compositional task generalization |
| Cambrian-S: Towards Spatial Supersensing in Video | Yang, Yang, Huang, Brown, …, LeCun, Fei-Fei, Xie | 2025 | arXiv:2511.04670 | VSI-SUPER; VSI-590K; predictive sensing |
| Scaling Spatial Intelligence with Multimodal Foundation Models (SenseNova-SI) | Cai et al. | 2026 | CVPR 2026 | 8M-sample spatial data scaling; open SOTA on spatial benchmarks |
| Efficiently Reconstructing Dynamic Scenes One D4RT at a Time | Zhang, Le Moing, …, Zisserman, Zhang, Sajjadi | 2026 | CVPR 2026 (Best Paper) | Query-based feed-forward 4D reconstruction and tracking |
| DreamZero: World Action Models Are Zero-Shot Policies | NVIDIA GEAR | 2026 | arXiv:2602.15922 | Joint video + action model; #1 RoboArena (Feb 2026) |

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
| SenseNova-SI-1.3 (8B) | VSI-Bench | Accuracy | 68.6 (vs 55.0 GPT-5) | Common EASI evaluation; author-reported |
| GEN-1 | Generalist task suite | Avg success | 99% (vs 64% prior) | Company-reported; short tasks |
| π\*0.6 (Recap) | Hard real-world tasks | Throughput / failure rate | ~2× / ~0.5× | Company-reported |
| Gemini Robotics ER 2 | Safety-instruction following | Accuracy | 97.9% (vs 47.2% ER 1.6) | Google-reported |
| D4RT | 4D reconstruction suites | Efficiency | Up to 300× vs prior methods | DeepMind-reported; CVPR 2026 Best Paper |

---

## Pros & Cons

| Aspect | Current Strengths | Current Weaknesses |
|---|---|---|
| **Embodied foundation models** | Strong in-distribution generalization; OXE-scale training validates cross-embodiment transfer; synthetic data (Genesis) now scalable | Out-of-distribution generalization remains poor; no solved contact-rich household tasks; data ceiling ~10× below NLP equivalent |
| **Spatial intelligence** | SpatialVLM demonstrates metric estimation is learnable; RGB-D integration (SpatialBot) improves grounding; curated spatial data lets 8B open models exceed GPT-5 on VSI-Bench (2026) | Many general VLMs still fail ordinal and multi-view spatial questions (MMSI-Bench best ~40–43); 3D spatial reasoning not yet integrated into mainstream VLA pipelines |
| **4D scene / world models** | Real-time 4D-GS at 82 FPS; feed-forward 4D reconstruction in seconds (D4RT); V-JEPA 2 zero-shot robot deployment; Genesis enables fast counterfactual simulation | Sim-to-real gap for deformable/liquid; world model action conditioning is approximate; predicting contact events reliably is open |
| **Safety/alignment** | Constraint layers deployable today; provable probabilistic bounds available; formal LTL methods demonstrated on simple tasks | Specification problem unsolved; RLHF for physical actions poorly validated; no ISO-equivalent safety standard for general-purpose AI-controlled robots yet ratified |
| **World Action Models / generative world models** | Exploit internet-scale video pretraining; top real-world leaderboards (DreamZero); interactive worlds (Genie 3, Marble) | Slow inference; physical accuracy and contact dynamics unverified; evaluation inside learned simulators may be optimistic |
| **Human-centric data scaling** | Egocentric video and wearable grippers reach 10⁵-hour scale cheaply (GR00T N1.7, GEN-1) | Embodiment gap and action retargeting remain approximate; data largely proprietary |

---

## Open Problems & Research Gaps

1. **The action label problem at scale.** Extracting robot-executable actions from unlabeled internet video — the key to 100× more training data — remains unsolved. Latent action pre-training is promising but has not been validated to match supervised demonstration quality on complex manipulation tasks. Shared relative end-effector action spaces (GR00T N1.7) and World Action Models (DreamZero, Cosmos Policy) narrowed the gap in 2026, but neither has yet shown that *unlabeled internet* video (as opposed to curated egocentric capture) yields equivalent gains.

2. **Deformable and liquid sim-to-real transfer.** Genesis and MuJoCo MJX handle rigid bodies and soft bodies, but reliably training policies that transfer to real fabric, rope, granular media, and liquids is an open problem. The fundamental issue is that small errors in material parameters compound across contact sequences.

3. **Event camera foundation models.** No large-scale pre-trained model for event camera data exists despite demonstrated advantages for high-speed robotics. The absence of an "ImageNet for event cameras" blocks the application of standard transfer learning methodology to neuromorphic sensing.

4. **Spatial reasoning in multi-step manipulation.** Current spatial VLMs answer static image questions. Spatial reasoning across a multi-step manipulation sequence — tracking how a scene's spatial relationships change as the robot acts — is barely studied.

5. **Tactile data at foundation model scale.** The largest published visuotactile datasets contain tens of thousands of contact examples. A foundation model for dexterous manipulation would require millions to billions of tactile observations. Synthetic tactile generation (simulating GelSight images from contact geometry) is a promising but underexplored direction.

6. **Formal safety for open-world robot deployment.** Translating natural language safety requirements into enforceable formal specifications, and verifying that a neural policy respects those specifications in open-world environments, does not yet have a satisfactory solution. This is arguably the gating problem for deploying general-purpose robots in homes and hospitals.

7. **Cross-embodiment policy transfer theory.** Empirical evidence shows that cross-embodiment training helps, but there is no theoretical account of when and why transfer should succeed or fail. A theory that predicts transfer quality from morphological similarity, task similarity, and training data statistics would enable principled architecture and data-mixing decisions.

8. **Independent, real-world evaluation.** Most headline 2026 robotics numbers (Gemini Robotics 2, π0.7, GEN-1, Helix 02) are self-reported on proprietary hardware and task suites. Shared real-world evaluation (e.g., RoboArena) exists but covers a narrow embodiment range; credible third-party benchmarks are a prerequisite for comparing trends 1, 4 and 7.

9. **Validating learned world models as simulators.** As generative world models (Genie 3, Cosmos 3, WAMs) are used for training and evaluating policies, methods to certify that their predicted dynamics—especially contact and failure cases—match reality are largely missing.

---

## Further Reading

- [V-JEPA 2 — Meta AI](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) — World model for robot planning from video
- [Genesis Physics Engine](https://genesis-world.readthedocs.io/) — 43M FPS open-source universal simulator
- [SpatialVLM (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.html) — Spatial reasoning in VLMs
- [GR00T N1 arXiv](https://arxiv.org/abs/2503.14734) — Open humanoid foundation model with dual-system architecture
- [Formal Methods in Robot Policy Learning (Survey 2026)](https://arxiv.org/pdf/2602.06971) — LTL, Büchi automata, and differentiable safety constraints
- [Recent Event Camera Innovations Survey (ECCV 2024 Workshop)](https://arxiv.org/html/2408.13627v2) — Comprehensive review of neuromorphic vision for robotics
- [π\*0.6 (arXiv:2511.14759)](https://arxiv.org/abs/2511.14759) and [π0.7 (arXiv:2604.15483)](https://arxiv.org/abs/2604.15483) — learning from experience; compositional generalization
- [Gemini Robotics 2 (DeepMind blog)](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) — whole-body VLA + ER 2
- [Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) — open humanoid VLA with human-video pretraining
- [D4RT (CVPR 2026 Best Paper)](https://d4rt-paper.github.io/) — feed-forward 4D reconstruction and tracking
- [Cambrian-S](https://github.com/cambrian-mllm/cambrian-s) and [SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI) — scaling spatial intelligence
- [DreamZero](https://github.com/dreamzero0/dreamzero) — World Action Models as zero-shot policies
- [NVIDIA Cosmos 3](https://github.com/NVIDIA/Cosmos) — unified world foundation models for physical AI
- [Newton physics engine](https://github.com/newton-physics/newton) — Linux Foundation GPU simulator (MuJoCo Warp backend)
- [V-JEPA 2 / 2.1 code](https://github.com/facebookresearch/vjepa2) — self-supervised video world models
- [Genie 3 (Google DeepMind)](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) — real-time interactive world model
- [VLA Safety survey (arXiv:2604.23775)](https://arxiv.org/abs/2604.23775) — threats, evaluations and mechanisms
