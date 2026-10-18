# Generalist vs. Specialist Robot Policies

> **Last Updated:** October 2026
> **Level:** Research Frontier
> **Related Sections:**
> - [../05_multimodal_vision_language/](../05_multimodal_vision_language/) — VLM foundations underpinning VLA models
> - [../12_research_frontier_2024_2026/07_future_trends.md](../12_research_frontier_2024_2026/07_future_trends.md) — Forward-looking embodied AI research agenda
> - [../04_3d_vision_and_scene/](../04_3d_vision_and_scene/) — 3D scene understanding for robot perception
> - [../09_efficiency_and_deployment/](../09_efficiency_and_deployment/) — Edge inference relevant to on-robot deployment

> **🆕 October 2026 update:**
> - New generalist evidence: π0.5 (unseen homes), **π\*0.6 with Recap RL** (generalist + on-task RL from experience and corrections), Gemini Robotics 1.5 Motion Transfer, Generalist AI's **GEN-0** scaling results, and TRI's careful **Large Behavior Model** study.
> - New specialist evidence: HIL-SERL (near-perfect success after 1–2.5 h of real-world RL) and the persistence of a dedicated, specially trained **whole-body/balance tier** in every humanoid stack, as a separate RL controller or, in Helix 02, as a learned ~1 kHz System 0.
> - Updated the dual-system lineage: GR00T N1.5 → N1.6 → **N1.7** (3B, Cosmos-Reason2 backbone, EgoScale human-video pre-training, GEAR-SONIC whole-body control), Helix → **Helix 02** on Figure 03 (three-tier System 0/1/2), Gemini Robotics 2 + ER 2, Galaxea G0, AgiBot GO-1.
> - New framing: the 2025–26 consensus is **"generalist pre-training, specialist post-training"**—a single foundation model specialized per deployment by fine-tuning and RL—rather than either pure camp.
> - New Key Papers/benchmark rows, open problems and Further Reading links.
> - Added the most direct 2026 evidence in this debate: **π0.7** (Apr 2026), a single generalist that reportedly matches the RL-fine-tuned π\*0.6 specialists on laundry, espresso and box building, plus **Gemini Robotics 2** (Jul 2026) whole-body control. Figure's 8-hour-shift claim is now qualified as company-reported, without throughput or error metrics.

---

## Overview

One of the deepest open debates in embodied AI is architectural: should a robot be controlled by a single large "robot GPT" trained on all available data, or by a modular pipeline of specialist components each optimized for its particular subtask? The question mirrors debates from NLP (BERT/GPT versus ensemble methods), but with a critical physical twist — robots must act in real time, respect hard physical constraints, and fail safely in ways that text generators do not. Neither camp has won decisively as of October 2026, and the emergent consensus is a *dual-system* or *hierarchical* hybrid that tries to capture the best of both paradigms.

The 2025–26 evidence sharpened this consensus into a recipe that mirrors LLM practice: **pre-train one generalist, then specialize it**. Physical Intelligence's π\*0.6 is a generalist VLA that is further improved on each target task with reinforcement learning from its own deployment experience; Gemini Robotics and GR00T ship as base models to be fine-tuned per embodiment; and even the most generalist humanoid stacks still rely on a dedicated balance tier, either a separately RL-trained whole-body controller or, in Figure's Helix 02, a learned ~1 kHz System 0 inside the stack. In April 2026 π0.7 pushed the other way: a single generalist reportedly matched the RL-fine-tuned π\*0.6 specialists on their own tasks [PI2026], suggesting that specialist gains can be folded back into the generalist. The live question is therefore less "one model or many?" than *where* the boundary between shared pre-training and per-task specialization should sit.

This section structures the debate as an intellectual confrontation, surveys the strongest evidence on each side, and describes the hybrid architectures that are currently attracting the most investment from both academia and industry.

---

## The Case for Generalist Policies (Vision-Language-Action Models)

### Core Argument

The NLP analogy is the generalist camp's most compelling argument: GPT-style pre-training on internet-scale data, followed by fine-tuning, produced capabilities no task-specific system could match. VLA (Vision-Language-Action) models attempt the same program for robotics — pre-train a large transformer jointly on web images/video/text and robot demonstration data, then fine-tune for specific deployments.

**Key proponents:** Sergey Levine (BAIR/Physical Intelligence), Chelsea Finn (Stanford/Physical Intelligence), Pieter Abbeel (UC Berkeley/Covariant).

### Emergent Cross-Task Generalization

RT-2 [Zitkovich2023] demonstrated that a model pre-trained on internet data could execute novel instructions never seen in robot training — e.g., picking up an object described as "something that could be used as a hammer" — by drawing on semantic knowledge from web pre-training. In evaluation across 6,000 trials, RT-2-X (55B parameters, trained on Open X-Embodiment) achieved a 3× improvement in emergent-skill success over RT-1-X. The Open X-Embodiment dataset [Collaboration2023] assembled demonstrations from 22 different robots across 33 academic labs, spanning 527 skills, and showed that cross-embodiment training yields meaningful transfer: RT-1-X trained on all robots improved ~50% over a model trained only on a single robot's data.

OpenVLA [Kim2024], a 7B-parameter open-source VLA trained on 970,000 trajectories from Open X-Embodiment, outperformed RT-2-X (55B parameters) by 16.5% absolute success rate across 29 tasks, demonstrating that well-designed smaller generalists can beat much larger closed systems — echoing the Chinchilla finding in NLP that data-efficiency matters as much as scale.

### Web Pre-Training as a Force Multiplier

The key claim is that internet-scale pre-training gives robots commonsense physics, object semantics, and linguistic grounding that would be impossible to encode from robot data alone. π₀ [Black2024] from Physical Intelligence used a PaliGemma-2B backbone pre-trained on web data, combined with flow matching action generation, and trained on over 10,000 hours of in-house demonstration data across 68 tasks and 7 robot configurations. The resulting model demonstrated significantly better generalization to novel household tasks compared to models trained only on robot data.

### New Generalist Evidence (2025–2026)

- **Open-world generalization.** π0.5 [PI2025] co-trains on web data, high-level sub-task labels and heterogeneous multi-robot data and performs long-horizon cleaning in **homes never seen in training**—the clearest real-world demonstration that breadth of pre-training buys out-of-distribution competence.
- **Generalist + RL beats generalist alone.** π\*0.6 [PI2025b] (Nov 2025) adds **Recap** (RL with Experience & Corrections via Advantage-conditioned Policies): the deployed generalist learns from its own rollouts, value estimates and human corrections. Physical Intelligence reported that on hard tasks (making espresso drinks, folding diverse laundry, assembling boxes) Recap **more than doubled throughput and roughly halved failure rates**. This blurs the debate—task-specific improvement is obtained *without* a task-specific architecture.
- **Cross-embodiment skill sharing.** Gemini Robotics 1.5 [GeminiRobotics2025] uses *Motion Transfer* so that skills learned on one robot (ALOHA) transfer to others (bi-arm Franka, Apptronik Apollo) without retraining per embodiment.
- **Industrial-scale scaling curves.** Generalist AI's GEN-0 [GeneralistAI2025] reports power-law improvement of downstream error with ~270,000 hours of pre-training data, and an apparent **~7B-parameter threshold** below which models stop benefiting from more data—direct, if company-reported, evidence that capacity and data must co-scale.
- **Careful multitask evidence.** TRI's Large Behavior Model study [TRI2025] used blind, statistically powered evaluation and found that fine-tuned multitask LBMs outperform single-task policies and reach a given performance with substantially less task-specific data, improving smoothly with pre-training scale. With Boston Dynamics, TRI also ran a single language-conditioned LBM controlling the whole body of the electric Atlas on long-horizon tasks (Aug 2025).
- **A generalist matches its own specialists.** π0.7 [PI2026] (Apr 2026; Gemma 3 4B backbone + ~860M action expert, trained with richer context such as subtask instructions and subgoal images) is reported to match the task-specific, RL-fine-tuned π\*0.6 specialists on laundry, espresso and box building with a single model. It also shows early **compositional generalization**, recombining skills to fold t-shirts on a UR5e with no laundry data for that robot and to use an air fryer after minimal coaching. This is the most direct evidence so far for the generalist camp. The caveats are that it needs step-by-step guidance for long tasks and has no standardized external benchmark.
- **Whole-body generalists.** Gemini Robotics 2 [GeminiRobotics2026] (Jul 2026) extends a single VLA from mostly upper-body control to whole-body humanoid control (walking, crouching and manipulating while reasoning). Its companion On-Device 2 model adapts to new embodiments within hours of data, although the VLA and On-Device models are limited to early-access partners.

### Simpler End-to-End Design

A single differentiable graph from perception to action allows gradients to flow through the entire system, which is architecturally elegant and removes the need to hand-engineer intermediate representations. The full system can be fine-tuned jointly on new task data, whereas modular systems face the "interface engineering" problem — one module's output format must match the next module's expected input.

### The NLP Scaling Precedent

```
Observation: In NLP, task-specific models (BERT fine-tuned per-task)
were eventually dominated by generalist GPT-style models.

Hypothesis: With sufficient data and scale, robot policies will follow
the same trajectory — a single large policy θ* will generalize better
than N specialized policies {θ_1, ..., θ_N}.
```

The relevant scaling question (see Section "The Chinchilla Question for Robotics" below) is whether this hypothesis holds given the physical constraints on robot data collection.

---

## The Case for Modular Specialist Systems

### Core Argument

Physical systems have hard constraints that language tokens do not. A sentence that is grammatically awkward is merely inelegant; a robot arm that exceeds its torque limits destroys itself. Modularity allows each component to be verified, bounded, and replaced independently.

**Key proponents:** Russ Tedrake (MIT/TRI), Dieter Fox (NVIDIA/UW), Marc Toussaint (TU Berlin).

### Physical Constraints Are Not in LLM Weights

Robot control requires:

```
Joint position:    q ∈ ℝ^n
Joint velocity:    q̇ ∈ ℝ^n,  ‖q̇‖ ≤ v_max
Joint torques:     τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
Contact forces:    λ ∈ K   (friction cone constraint)
```

These constraints govern every motion a robot makes, yet they are nowhere in the training distribution of a language model pre-trained on internet text and images. A specialist low-level controller designed using model-based robotics (e.g., Drake [Tedrake2019]) can provide hard guarantees on these quantities; a VLA inference call cannot.

Marc Toussaint's work on factored task-and-motion planning [Toussaint2024] shows that when tasks can be decomposed into subproblems with known structure (geometric constraints, kinematics, logic preconditions), optimization-based planners achieve dramatically better sample efficiency and formal correctness guarantees than end-to-end learned policies.

### Control Loop Latency Is Incompatible with LLM Inference

High-performance robot control loops operate at:

| Control type | Typical frequency | Required latency |
|---|---|---|
| Force/impedance control | 1–4 kHz | 0.25–1 ms |
| Joint-level servo | 250–500 Hz | 2–4 ms |
| Cartesian control | 100–250 Hz | 4–10 ms |
| Task-level planning | 1–10 Hz | 100–1000 ms |

A 7B-parameter LLM inference call takes 50–200 ms on embedded GPU hardware. Running it in the inner control loop is physically impossible for force control, servo control, or Cartesian control. This is not an engineering limitation to be engineered away — it is a consequence of the physics of feedback control. Fast disturbance rejection (e.g., not dropping a glass when someone bumps the robot) requires kilohertz-rate feedback that no LLM can provide.

### Safety and Interpretability

In manufacturing and healthcare applications, a regulator or customer may require:
- Formal proofs that the robot cannot reach a forbidden region of configuration space
- Explanation of why a specific action was chosen
- Independent auditing of the planning logic

Modular pipelines support all three. A single neural network policy is opaque by construction. Russ Tedrake's Drake toolbox provides Lyapunov-based stability certificates and reachability analysis that are only meaningful if the control system is separated from the perception system. When the two are fused into a single network, the mathematical machinery of control theory becomes inapplicable.

### Sample Efficiency

Training a VLA from scratch on a new task typically requires thousands to tens of thousands of demonstrations. Specialist approaches (imitation learning + model-based refinement) routinely achieve expert performance with 50–200 demonstrations. For deployment in facilities where robot data collection is expensive or dangerous (e.g., surgical suites, nuclear environments), this is not a minor academic distinction.

### New Specialist Evidence (2025–2026)

- **Real-world RL specialists.** HIL-SERL [Luo2025] (Science Robotics 2025) trains task-specific vision-based policies with human-in-the-loop RL directly on real robots and reaches **near-perfect success with fast cycle times after only 1–2.5 hours of training** on precise, dynamic and dual-arm tasks (e.g. assembly, insertion, flipping)—outperforming imitation baselines trained on far more data. For a known, fixed task in a factory cell this remains the most sample-efficient route to reliability.
- **Whole-body control stays specialized.** Through 2025, every deployed humanoid VLA sat on top of a separately trained RL locomotion/whole-body controller (e.g. NVIDIA's GEAR-SONIC controller shipped with GR00T N1.7 [NVIDIA2026]). In 2026 Figure's Helix 02 [Figure2026] folded balance into a single learned system, but it did so through a dedicated ~1 kHz **System 0** trained separately on human motion data and sim-to-real RL. Gemini Robotics 2 likewise controls the whole body. Specialization has therefore moved *inside* the stack rather than disappearing: the kHz balance loop is still a distinct, specially trained tier.
- **Reliability gap in deployment.** Generalist demos emphasize breadth, while commercial deployments (logistics tote handling, factory part loading) are narrow, repetitive and judged on uptime and throughput—metrics on which tuned specialists and heavily post-trained generalists, not zero-shot generalists, have delivered.

---

## Hybrid Middle Ground: Dual-System and Hierarchical Architectures

The architectures attracting the most engineering investment in 2024–2026 are hybrids that assign fast, safety-critical, physically-grounded computation to a specialist module, and slow, language-grounded, semantic reasoning to a large model.

### The Dual-System Architecture

Inspired by Kahneman's System 1/System 2 framework from cognitive science:

```mermaid
graph TD
    subgraph Slow ["System 2 — Slow (VLM / LLM)"]
        S2[Vision-Language Model\n7B–70B params\n100ms inference\nScene understanding\nTask planning\nLanguage grounding]
    end

    subgraph Fast ["System 1 — Fast (Reactive Policy)"]
        S1[Visuo-Motor Controller\n10M–200M params\n5ms inference\nJoint-level control\nForce/impedance\nReactive correction]
    end

    subgraph Env ["Environment"]
        CAM[RGB Camera(s)]
        PROP[Proprioception\n(joint angles, torques)]
        ACT[Actuators]
    end

    CAM -->|High-res image| S2
    CAM -->|Current frame| S1
    PROP --> S1
    S2 -->|Semantic goal / subgoal tokens| S1
    S1 -->|Motor commands @ 200Hz| ACT
    ACT -->|State feedback| PROP

    style Slow fill:#dbeafe,stroke:#3b82f6
    style Fast fill:#dcfce7,stroke:#16a34a
    style Env fill:#fef9c3,stroke:#ca8a04
```

### Deployed Dual-System Implementations

**GR00T N1** [NVIDIA2025]: Released March 2025. System 2 is a 1.34B-parameter VLM; System 1 is a fast reactive policy. Total 2.2B parameters. Samples 16 actions in 63.9 ms on L40 GPU. Trained on 780,000 synthetic + real trajectories across Fourier GR-1 and 1X humanoid embodiments. Outperforms SOTA imitation learning baselines on simulation benchmarks across multiple embodiments.

**Helix** [Figure2025]: Developed by Figure AI, released February 2025. System 2 is a 7B-parameter VLM operating at 7–9 Hz; System 1 is an 80M-parameter visuo-motor controller at 200 Hz. Both run on onboard embedded GPUs with no cloud dependency. Helix controls the upper body. For its successor, Helix 02, Figure claims 8-hour shifts without intervention but has released no throughput or error metrics.

**π₀** [Black2024]: Physical Intelligence's generalist policy uses flow matching (continuous normalizing flow) as the action head, with PaliGemma as the language backbone. π₀-FAST [PhysicalIntelligence2024] further accelerates action generation using DCT-space (FAST) tokenization, achieving 7× compression of action sequences and 5× faster training convergence.

**Successor systems (2025–2026).** The dual-system pattern has become the default industrial architecture:

- **GR00T N1.5 → N1.6 → N1.7** [NVIDIA2026]: N1.5 (mid-2025) froze the VLM to preserve language grounding; N1.6 used an Eagle-based VLM with a 32-layer DiT action head; **N1.7** (2026, GA release, 3B parameters) swaps in the reasoning VLM **Cosmos-Reason2-2B** (Qwen3-VL architecture) as System 2, halves the DiT head to 16 layers, uses a **relative end-effector action space shared by robot and human data**, pre-trains on ~20K hours of EgoScale human video, extends the action horizon from 16 to 40 steps and the state/action dimension from 29 to 132, and exports to TensorRT for Jetson-class deployment.
- **Helix on Figure 03** [Figure2025]: Figure's third-generation humanoid (Oct 2025) was designed around Helix, with a redesigned camera system and tactile fingertips feeding System 1.
- **Helix 02** [Figure2026] (late Jan 2026): a single learned system on Figure 03 that unifies perception, balance, locomotion and manipulation in three tiers. System 2 handles semantic reasoning, System 1 turns perception into whole-body motion at high frequency, and a new System 0 runs at ~1 kHz for balance and coordination. Its flagship demo is ~4 minutes of autonomous dishwasher unloading and reloading.
- **π0.5 / π0.6** [PI2025]: rather than two separate networks, a single VLA first decodes a high-level sub-task (slow, discrete) and then an action chunk through a flow-matching action expert (fast, continuous), trained with *knowledge insulation* so that action-expert gradients do not degrade the VLM.
- **Gemini Robotics 1.5 + ER 1.5** [GeminiRobotics2025]: an explicit two-model system in which an embodied-reasoning VLM orchestrates a "thinking" VLA (see [Language-Conditioned Control](./10_language_conditioned_control.md)).
- **Gemini Robotics 2 + ER 2** [GeminiRobotics2026] (Jul 2026): the same two-model split, now with a whole-body VLA executor and an ER 2 planner (available in Google AI Studio) that Google rates as its safest robotics model on ASIMOV-2 and evaluates on a new ASIMOV-Agentic benchmark.
- **π0.7** [PI2026] (Apr 2026): continues the single-network design of π0.5, but the prompt carries subtask instructions and subgoal images, so the slow "planning" signal can come from a human coach or a higher-level model.
- **Galaxea G0** and **AgiBot GO-1** (2025): open Chinese dual-system/latent-planner VLAs trained on single-vendor fleet data (see [Robot Data & Teleoperation](./09_robot_data_and_teleoperation.md)).

### Hierarchical Control

An alternative framing treats the large model as a task planner that decomposes goals into subgoals, each executed by a specialist:

```mermaid
graph LR
    NL["Natural Language Goal\n'Clean the kitchen'"]
    VLM["VLM Planner\n(GPT-4V / LLaMA-3)\n~1 Hz"]
    SUBGOAL["Subgoal Sequence\n['Pick sponge', 'Wipe counter',\n'Return sponge']"]
    SPEC1["Specialist: Pick\n(Diffusion Policy)"]
    SPEC2["Specialist: Wipe\n(Impedance Control)"]
    SPEC3["Specialist: Place\n(Model-Predictive)"]
    WORLD["Physical World"]

    NL --> VLM
    VLM --> SUBGOAL
    SUBGOAL --> SPEC1
    SUBGOAL --> SPEC2
    SUBGOAL --> SPEC3
    SPEC1 & SPEC2 & SPEC3 <--> WORLD

    style VLM fill:#dbeafe,stroke:#3b82f6
    style SPEC1 fill:#dcfce7,stroke:#16a34a
    style SPEC2 fill:#dcfce7,stroke:#16a34a
    style SPEC3 fill:#dcfce7,stroke:#16a34a
```

In this framing the LLM is explicitly *not* a motor controller — it is an interface between language and a library of verified specialist skills. Toussaint's factored TAMP [Toussaint2024] formalizes this as a factor graph over task-mode variables, with specialist controllers handling each mode.

---

## The Chinchilla Question for Robotics

The NLP community's "Chinchilla" finding [Hoffmann2022] showed that most large language models were significantly *undertrained* relative to their parameter count — optimal compute allocation requires far more data than was used. This raises the key scaling question for robotics:

> **Do robot policies follow LLM-style scaling laws, and if so, what is the optimal data-to-parameter ratio?**

**Evidence for scaling laws in robot data:**
- RT-1-X → RT-2-X: adding cross-embodiment pre-training yielded ~50% improvement in success rates on held-out tasks [Collaboration2023]
- OpenVLA trained on 10× more data than RT-2 fine-tuning data, achieving better generalization with 7× fewer parameters [Kim2024]
- Generalist AI's GEN-0 reports first empirical scaling curves for robot manipulation, trained on 270,000 hours of data from 1,000s of real environments [GeneralistAI2025]; it also reports "ossification" of models below ~7B parameters under heavy pre-training, i.e. a minimum capacity for scaling to hold
- NVIDIA's EgoScale finds a log-linear law between egocentric human-video hours (up to ~20.8k h) and validation loss, which predicts real-robot success [EgoScale2026]
- Physical Intelligence finds that the *ability to learn from human video* emerges only once robot pre-training is large and diverse enough—a qualitative, scale-dependent capability rather than a smooth gain [PI2025H2R]

**Counter-evidence (the "robot data is different" argument):**

```
LLM data characteristics:
  Source: Internet text (scrapeable, free, near-infinite)
  Collection cost: ~$0 per token
  Diversity: All of human written knowledge
  Scale: ~10^13 tokens feasibly available

Robot data characteristics:
  Source: Physical robot demonstrations (requires hardware + time)
  Collection cost: ~$1–100 per trajectory
  Diversity: Limited by lab environments and robot morphologies
  Scale: ~10^7 trajectories feasibly available (Open X-Embodiment ~10^6)
```

The conclusion is that robot policies *do* show scaling improvements with data, but the data ceiling is orders of magnitude lower than for LLMs. This makes the generalist vs. specialist question fundamentally about how to allocate scarce data — a question that pure scaling cannot resolve.

**2026 update to the counter-argument.** The data ceiling has risen faster than expected: proprietary corpora reached ~10⁵ hours (GEN-0), open fleet corpora passed 10⁶ trajectories (AgiBot World), and action-labeled human video added ~10⁴ hours (EgoScale) at much lower cost per hour. The gap to LLM-scale data remains several orders of magnitude, but the robot-data side is now growing by >10⁴ hours per week at a single company—which shifts the burden of proof toward the generalist camp for *pre-training*, while leaving per-task reliability to specialization.

**Simulation as a partial solution:** Genesis [Genesis2024] (43M FPS on a single RTX 4090, 430,000× faster than real-time) and Isaac Lab [NVIDIA2024] attempt to bridge the data gap by generating synthetic trajectories. NVIDIA generated 780,000 synthetic trajectories (equivalent of 9 months of human demonstrations) in 11 hours for GR00T N1 training, improving performance by 40% over real-data-only training [NVIDIA2025].

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|---|---|---|---|---|
| RT-2: Vision-Language-Action Models Transfer Web Knowledge | Zitkovich et al. (Google DeepMind) | 2023 | CoRL | First large-scale VLA; web pre-training → robot emergent skills; 6,000-trial eval |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | Open X-Embodiment Collaboration (33 institutions) | 2023 | CoRL Workshop / ICRA 2024 | 22 robots, 527 skills, cross-embodiment scaling laws; RT-1-X +50% |
| OpenVLA: An Open-Source Vision-Language-Action Model | Kim, Pertsch, Karamcheti et al. | 2024 | arXiv / CoRL 2024 | 7B VLA on OXE; outperforms RT-2-X (55B) by 16.5% at 7× fewer params |
| π₀: A Vision-Language-Action Flow Model for General Robot Control | Black, Brown, Driess, Finn, Levine et al. | 2024 | arXiv | Flow matching action head; 68 tasks, 7 robot configs; 10k+ demo hours |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | NVIDIA Research (Fox et al.) | 2025 | arXiv | Dual-system 2.2B VLA; 780k synthetic trajectories; 63.9 ms inference on L40 |
| Octo: An Open-Source Generalist Robot Policy | Ghosh et al. (Berkeley, Stanford, CMU, Google DeepMind) | 2024 | RSS 2024 | 93M transformer-diffusion policy; 800k episodes; modular observation heads |
| Factored Task and Motion Planning with Combined Optimization, Sampling and Learning | Toussaint et al. | 2024 | ICRA 2024 | Factor-graph TAMP; specialist controllers per logic mode; formal guarantees |
| Helix: A Vision-Language-Action Model for Generalist Humanoid Control | Figure AI | 2025 | Technical Report | 7B S2 + 80M S1; 200 Hz reactive; no cloud; upper-body control |
| π0.5: a Vision-Language-Action Model with Open-World Generalization | Physical Intelligence | 2025 | CoRL 2025 | Hierarchical in-model reasoning + co-training; long-horizon tasks in unseen homes |
| π\*0.6: a VLA That Learns From Experience (Recap) | Physical Intelligence | 2025 | Technical report | Advantage-conditioned RL on deployment experience + corrections; >2× throughput on hard tasks |
| Gemini Robotics 1.5 | Gemini Robotics Team (Google DeepMind) | 2025 | Technical report | ER orchestrator + thinking VLA; Motion Transfer across embodiments |
| A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation | TRI LBM Team | 2025 | arXiv:2507.05331 | Blind, statistically powered evaluation; pre-training reduces task data needs |
| HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL | Luo, Xu, Wu, Levine | 2024 | Science Robotics 2025 | Real-world RL specialists: near-perfect success in 1–2.5 h |
| GEN-0: Embodied Foundation Models That Scale with Physical Interaction | Generalist AI | 2025 | Technical blog | 270k h pre-training; data power law; ~7B capacity threshold |
| GR00T N1.7 | NVIDIA | 2026 | Open model release | 3B; Cosmos-Reason2 System 2; human-video pre-training; GEAR-SONIC whole-body control |
| Helix 02 | Figure AI | 2026 | Company blog | Three-tier System 0/1/2; learned ~1 kHz balance tier; whole-body control on Figure 03 |
| π0.7: a Steerable Generalist Robotic Foundation Model | Physical Intelligence | 2026 | arXiv:2604.15483 | Single generalist ≈ RL-fine-tuned π\*0.6 specialists; early compositional generalization |
| Gemini Robotics 2 / ER 2 / On-Device 2 | Google DeepMind | 2026 | DeepMind blog + model card | Whole-body humanoid VLA; ER 2 planner; On-Device 2 adapts to new embodiments within hours of data |

---

## Benchmark Performance

| Model | Dataset / Eval | Metric | Score | Notes |
|---|---|---|---|---|
| RT-2-X (55B) | Open X-Embodiment emergent tasks | Success rate | ~62% emergent | 6,000 trials; emergent = tasks in LLM pretraining not robot data |
| RT-1-X | Open X-Embodiment | Success rate | ~50% relative improvement over single-robot RT-1 | Cross-embodiment training |
| OpenVLA (7B) | 29 tasks, multiple embodiments | Absolute success rate | 16.5% higher than RT-2-X | 7B vs. 55B; BridgeData V2 and Google Robot |
| Octo-Base (93M) | Language-conditioned manipulation | vs. RT-2-X | Comparable on language tasks | 800k training episodes |
| GR00T N1 (2.2B) | Fourier GR-1 + 1X humanoid tasks | vs. SOTA IL | Outperforms SOTA simulation benchmarks | 40% improvement from +synthetic data |
| π₀ | 9 benchmark categories, 5 embodiments | Task completion | State-of-the-art on deformable + long-horizon manipulation | Flow matching; 10k+ demo hours |
| Helix / Helix 02 (Figure AI) | Factory and household manipulation | Deployment | 8-hour shifts claimed | Company claim; no throughput or error metrics released; Helix 02 demo: ~4-min autonomous dishwasher task |
| π\*0.6 + Recap | Espresso, laundry, box assembly | Throughput / failure rate | >2× / ~½ | Company-reported; generalist + task RL [PI2025b] |
| HIL-SERL | Real-world precise/dynamic tasks | Success after 1–2.5 h training | Near-perfect | Task-specific RL specialist [Luo2025] |
| π0.7 (single generalist) | Laundry, espresso, box building | Task performance vs. specialists | ≈ RL-fine-tuned π\*0.6 specialists | Company-reported; no standardized external benchmark [PI2026] |
| GEN-0 | Real-world downstream tasks | Post-training error vs data | Power law; ≥7B needed | Company-reported [GeneralistAI2025] |
| AgiBot GO-1 | AgiBot World tasks | Avg. perf. vs OXE pre-training | ~+30% | Fleet data vs aggregated data [AgiBot2025] |

---

## Pros & Cons

| Aspect | Generalist VLA Pros | Specialist/Modular Pros |
|---|---|---|
| **Generalization** | Zero-shot to novel objects and language instructions via web pre-training; emergent cross-task transfer [Zitkovich2023]; early compositional generalization (π0.7) | Provable performance guarantees within task domain; no distribution shift from out-of-domain instructions |
| **Data efficiency** | Amortizes data cost across all tasks; pre-training reduces per-task fine-tuning requirement | Requires only ~50–200 demonstrations per task; factor of 100× improvement in sample efficiency for known tasks |
| **Control latency** | Dual-system architectures achieve 5–10 ms for fast path; full-speed LLM call is 50–200 ms which is incompatible with kHz force control | Model-based specialists run at 1–4 kHz with hard timing guarantees; composable with real-time OS schedulers |
| **Safety / verifiability** | Black-box neural network; no formal stability certificates; interpretability research ongoing | Lyapunov functions, reachability analysis, formal verification (Drake) applicable; regulatorily auditable |
| **Maintenance / extensibility** | Single model fine-tuned for new tasks; no interface engineering | New tasks require new specialists and interface specifications; combinatorial explosion with task count |
| **Failure mode** | Hallucination of physically impossible actions; graceful degradation unclear | Brittleness at task boundaries; performance cliff outside trained regime |

---

## Open Problems & Research Gaps

1. **Closing the control frequency gap without losing generality.** The 5 ms System 1 / 100 ms System 2 split in dual-system models remains a pragmatic engineering hack rather than a principled solution. Can System 2 be made to predict physical state well enough that System 1 can extrapolate across a full 100 ms gap without re-querying? Quantifying the compounding error over multi-step extrapolation is an open research problem. Helix 02's learned ~1 kHz System 0 is the first industrial attempt to close this gap with a learned tier rather than a classical controller, but no metrics have been published.

2. **The robot Chinchilla problem.** No work has yet derived principled optimal compute allocation curves for robot policies of the form "given N real demonstrations and M simulation hours, what is the optimal policy size?" The Chinchilla methodology applied to robot data would likely yield qualitatively different results due to data ceiling effects.

3. **Contact physics in learned policies.** Current VLAs learn implicit contact models from demonstrations, which fails catastrophically in novel contact configurations. Embedding differentiable contact simulation (e.g., MJX, Drake) into the VLA's inner loop, or providing contact-aware auxiliary losses, is largely unexplored.

4. **Formal safety specifications from natural language.** Given a natural language safety constraint ("do not move faster than 0.5 m/s near a human"), translating this reliably into a formal constraint enforced at the System 1 level remains an open problem. LTL-based translation approaches [FormalMethods2026] show promise but have not been demonstrated at scale.

5. **Cross-morphology transfer.** Open X-Embodiment shows that training on diverse embodiments helps, but the mechanism of transfer remains poorly understood. A theoretical account of *why* a policy trained on a 6-DOF arm transfers to a 16-DOF humanoid would enable principled data mixing strategies.

6. **Sample efficiency with internet video.** Human internet video (Ego4D, HowTo100M) contains enormous amounts of task-relevant manipulation behavior but lacks action labels. The action label problem — recovering robot-executable actions from unlabeled video — is one of the central bottlenecks to scaling robot data. Latent Action Pre-Training (LAPT) [LAPT2024] is a promising direction but has not yet been demonstrated to match supervised demonstration quality.

7. **Evaluating generalization vs. memorization.** Current benchmarks conflate zero-shot generalization with performance on tasks that appear in pre-training web data. A clean benchmark that tests on tasks verifiably absent from web training distributions is needed to make the VLA generalization argument rigorous.

8. **Where to put the specialization boundary.** "Generalist pre-training, specialist post-training" (π\*0.6 Recap, per-embodiment GR00T fine-tuning) is now standard, but there is no theory of how much task-specific RL or fine-tuning erodes generality (catastrophic forgetting) or how to share post-training gains back into the base model. π0.7's reported parity with the π\*0.6 specialists is a first data point that such gains can be absorbed into a generalist, but it is company-reported.

9. **Reproducible real-world comparison.** Many 2025–26 generalist claims (GEN-0 scaling, Recap throughput, π0.7's parity with specialists, Helix / Helix 02 shift lengths) are company-reported. Blind, statistically powered protocols (TRI LBM, RoboArena) exist but are not yet adopted for headline results.

10. **Generalist–specialist distillation.** Can a generalist be distilled into compact, verifiable specialist controllers for safety-critical cells, or conversely can specialist RL policies (HIL-SERL) be absorbed into a generalist without losing their reliability?

---

## Further Reading

- [RT-2 Paper (CoRL 2023)](https://proceedings.mlr.press/v229/zitkovich23a/zitkovich23a.pdf) — Original VLA paper with 6,000-trial evaluation
- [Open X-Embodiment arXiv](https://arxiv.org/abs/2310.08864) — Cross-embodiment dataset and RT-X scaling results
- [OpenVLA arXiv](https://arxiv.org/abs/2406.09246) — Open-source 7B VLA beating RT-2-X
- [GR00T N1 arXiv](https://arxiv.org/abs/2503.14734) — NVIDIA's dual-system humanoid foundation model
- [π₀ arXiv](https://arxiv.org/html/2410.24164v1) — Physical Intelligence's flow-matching VLA
- [Russ Tedrake's Underactuated Robotics Course](https://underactuated.csail.mit.edu/) — Canonical reference for model-based robot control theory
- [π0.5 arXiv](https://arxiv.org/abs/2504.16054) — Open-world generalization via co-training and hierarchical inference
- [π\*0.6 / Recap (Physical Intelligence blog)](https://www.pi.website/blog/pistar06) — Generalist VLA improved by RL from deployment experience
- [Gemini Robotics 1.5 arXiv](https://arxiv.org/abs/2510.03342) — Embodied reasoning orchestrator + thinking VLA
- [TRI Large Behavior Models arXiv](https://arxiv.org/abs/2507.05331) — Careful multitask evaluation of LBMs
- [HIL-SERL arXiv](https://arxiv.org/abs/2410.21845) — Real-world human-in-the-loop RL specialists
- [GEN-0 (Generalist AI)](https://generalistai.com/blog/nov-04-2025-GEN-0) — Robot scaling laws from 270k hours of real-world data
- [Isaac GR00T (GitHub)](https://github.com/NVIDIA/Isaac-GR00T) — GR00T N1.x open humanoid foundation models
- [π0.7 arXiv](https://arxiv.org/abs/2604.15483) — Steerable generalist that reportedly matches the π\*0.6 specialists
- [Gemini Robotics 2 (DeepMind blog)](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) — Whole-body VLA, ER 2 and On-Device 2
- [Figure Helix 02](https://figure.ai/news/helix-02) — Three-tier whole-body humanoid control
