# Cross-Embodiment Learning for Robotics

> **Last Updated:** October 2026
> **Level:** Advanced | Research Frontier
> **Related Sections:**
> - [Diffusion Policy](./03_diffusion_policy.md)
> - [Simulation and Data for Embodied AI](./05_simulation_and_data.md)
> - [Vision-Language Models](../05_multimodal_vision_language/00_overview.md)
> - [Datasets and Benchmarks](../11_datasets_and_benchmarks/02_3d_and_robotics_datasets.md)

> **🆕 October 2026 update:**
> - New section on **2025–2026 cross-embodiment generalists**: π0.5, GR00T N1.5–N1.7 (embodiment tags → shared relative-EEF action space), Gemini Robotics 1.5 **Motion Transfer**, **X-VLA soft prompts**, HPT stems/heads.
> - New section on **humans as an embodiment**: latent-action pretraining (LAPA, UniVLA, villa-X), EgoScale (20K h human video in GR00T N1.7), Being-H0.5, EgoVerse.
> - Data infrastructure: LeRobotDataset (Parquet + MP4, v3) as a PyTorch-native alternative to RLDS; DROID and very-large proprietary corpora (Generalist GEN-0/GEN-1).
> - Added rows to the mermaid diagram, Key Papers and Benchmark tables (X-VLA, GR00T N1.6), new open problems, Further Reading; fixed broken links.

---

## Overview

Cross-embodiment learning asks whether a single policy — or a shared representation — can transfer knowledge across robots with fundamentally different kinematics, morphologies, sensor suites, and action spaces. The central hypothesis, confirmed empirically by the Open X-Embodiment project [OXE2023], is that **co-training on diverse robot data improves per-embodiment performance** even when the robot structures differ drastically. This mirrors findings in NLP and computer vision: more diverse pretraining data improves generalization on downstream tasks, and the same logic extends to visuomotor control.

The Open X-Embodiment (OXE) dataset [OXE2023], released in late 2023 and presented at ICRA 2024, aggregated over 1 million robot trajectories from 22 distinct embodiments (single-arm manipulators, bimanual systems, mobile manipulators, legged robots) contributed by 34 research institutions across 60 individual datasets, covering 527 distinct manipulation skills. The dataset is stored in the **RLDS** (Reinforcement Learning Datasets) format — a TensorFlow Datasets-backed episode structure — which provides a unified serialization layer that abstracts over heterogeneous sensor configurations. Training RT-1-X (a Robotics Transformer) on this mixture yielded a 50% average success-rate improvement across five diverse robot platforms compared to policies trained exclusively on per-embodiment data, establishing co-training as a powerful inductive bias.

The fundamental obstacle is the **embodiment gap**: each robot exposes a distinct observation space (monocular RGB, stereo, RGBD, proprioception dimensionality, force-torque) and a distinct action space (joint positions, end-effector Cartesian delta, absolute poses, gripper width). Bridging these gaps without manual alignment is the defining unsolved problem in cross-embodiment research, and recent architectures (Octo, CrossFormer, pi-zero) each propose different solutions ranging from modality-specific tokenizers to latent action spaces.

The 2025–2026 generation added four new answers. (1) **Learned embodiment prompts**: X-VLA attaches a small set of learnable "soft prompt" embeddings per data source so that a single 0.9B transformer absorbs heterogeneity and adapts to a new robot by learning only new prompts [Zheng2025]. (2) **Shared relative action spaces**: GR00T N1.6/N1.7 predict state-relative action chunks, and N1.7 uses a relative end-effector space common to robots *and* human hands [NVIDIA2026]. (3) **Motion Transfer**: Gemini Robotics 1.5 trains one checkpoint across ALOHA, bi-arm Franka and the Apollo humanoid so that skills learned on one embodiment execute zero-shot on another [GoogleDeepMind2025b]. (4) **Humans as just another embodiment**: tens of thousands of hours of egocentric human video (EgoScale) or human-centric unified action spaces (Being-H0.5) are mixed directly into VLA pretraining.

---

## The Embodiment Gap: Observation and Action Heterogeneity

### Observation Space Misalignment

Different robots attach cameras at different positions, use different resolutions, and may include modalities entirely absent on other platforms (wrist cameras, tactile sensors, depth rings). Approaches to alignment:

- **Canonical view normalization**: warp all images to a canonical egocentric perspective using estimated camera intrinsics/extrinsics.
- **Modality masking**: treat missing sensors as masked tokens; the model learns to predict actions from available subsets.
- **Modality-specific tokenizers**: each sensor type (RGB image, depth, joint state, language) passes through a dedicated encoder, and all output tokens are concatenated into a unified sequence before the shared transformer backbone.

### Action Space Misalignment

Action dimensionalities range from 4-DoF (simple gripper) to 30+ DoF (humanoid manipulation). Key strategies:

```mermaid
flowchart TD
    A[Heterogeneous\nAction Spaces] --> B[Dimension Normalization\nMinMax per-axis]
    A --> C[Latent Action Space\nShared encoder + embodiment-specific decoders]
    A --> D[Tokenized Actions\nVQ-VAE discrete tokens]
    A --> E[Language-Specified Actions\nRelative delta in named joint frames]
    A --> J[Embodiment Soft Prompts\nLearnable per-source embeddings]
    A --> K[Shared Relative EEF Space\nRobots + human hands]
    B --> F[RT-1-X / Octo approach]
    C --> G[Latent Action Diffusion\narXiv:2506.14608]
    D --> H[RT-2 / SayCan]
    E --> I[CrossFormer approach]
    J --> L[X-VLA 2025]
    K --> M[GR00T N1.7 / Being-H0.5 2026]
```

**MinMax normalization** (used in Octo [OctoTeam2024]) applies per-dimension affine rescaling to $[-1, 1]$ across all demonstrations in a training mixture, providing consistent numerical ranges for the denoiser/regressor while preserving relative magnitudes.

**Embodiment-specific action heads** (used in CrossFormer [Doshi2024]) attach separate small MLP heads per embodiment type to a shared transformer trunk, allowing the trunk to learn transferable representations while action decoders handle morphology-specific output structure.

---

## Open X-Embodiment Dataset and RT-X Models

### Dataset Statistics

- **22 robot embodiments**: includes Franka Panda, Google RT-1 robot, WidowX, Hello Robot Stretch, Boston Dynamics Spot, xArm, UR5, ALOHA, and others.
- **1,000,000+ trajectories** across **527 distinct manipulation skills** from **34 research labs** at **21 institutions**.
- **Data format**: RLDS (see below); all converted to a common episode schema with flexible observation/action fields.
- Released under CC-BY license; hosted on Google Cloud Storage.

### RLDS Data Format

RLDS (Reinforcement Learning Datasets) is the standardization layer that made OXE possible. Each dataset episode is stored as:

```
Episode:
  steps: [
    {
      observation: {image_0: uint8[H,W,3], image_1: ..., state: float32[D_obs]},
      action: float32[D_act],
      reward: float32,
      discount: float32,
      is_first: bool,
      is_last: bool,
      is_terminal: bool,
      language_instruction: string
    }, ...
  ]
```

The schema is backed by TensorFlow Datasets serialized as TFRecord shards. RLDS has become the de facto standard for robotic learning datasets, enabling plug-and-play mixing across labs without bespoke data loaders.

**PyTorch-native alternative (2025–2026).** Hugging Face's **LeRobotDataset** format (Parquet for states/actions + MP4 video, hosted and streamable on the Hub; v3.0 in 2025) has become the de facto format for new open datasets and policies. GR00T N1.x consumes a LeRobot-v2 "flavour" with an extra `meta/modality.json` that maps concatenated state/action vectors to named fields—effectively a lightweight embodiment schema—and openpi, SmolVLA and RLinf all train from LeRobot datasets. Conversions between RLDS (OXE, DROID) and LeRobot formats are now routine.

### RT-1-X and RT-2-X

The OXE paper evaluated two RT-X models:

- **RT-1-X**: the RT-1 Robotics Transformer architecture [Brohan2022] retrained on the full OXE mixture. Achieved **+50% average success rate** across 5 evaluation platforms vs. platform-specific RT-1 checkpoints.
- **RT-2-X**: the RT-2 vision-language-action model (PaLI-X backbone) co-trained on OXE. Achieved **3x performance** on real-world robotic skill execution. RT-2-X demonstrated emergent cross-embodiment generalization — skills seen only on one robot were executable on novel embodiments.

---

## Octo: Open-Source Generalist Robot Policy

**Octo** [OctoTeam2024] (RSS 2024) is the first fully open-source generalist robot policy pretrained on OXE. Key design choices:

- **Architecture**: transformer with readout tokens for task specification (language string or goal image), processing heterogeneous observations via modality-specific tokenizers.
- **Action head**: diffusion head (conditional DDPM) over 4-step action chunks, modeling expressive multimodal distributions.
- **Training data**: 800,000 robot demonstrations from the OXE mixture (a curated subset for compute feasibility).
- **Fine-tuning**: Octo can be fine-tuned to new observation spaces (e.g., force-torque inputs), new action spaces (e.g., joint-position control), and entirely new embodiments within minutes to hours, making it a practical starting point for new labs.
- **Results**: Octo matches or outperforms prior task-specific SOTA on 9 of 11 evaluation tasks in the OXE evaluation suite after fine-tuning.

```mermaid
flowchart LR
    subgraph Inputs
        L["Language\nInstruction"] --> LT["Language\nTokenizer"]
        I["Camera\nImages"] --> IT["Image\nTokenizer (ViT)"]
        S["Proprioception"] --> ST["State\nTokenizer"]
        G["Goal Image\n(optional)"] --> GT["Goal\nTokenizer"]
    end
    LT & IT & ST & GT --> TRF["Shared\nTransformer\nBackbone"]
    TRF --> RO["Readout\nTokens"]
    RO --> DH["Diffusion Head\n(action denoiser)"]
    DH --> AC["Action Chunk\na_0 ... a_{T_a}"]
```

---

## CrossFormer: Scaling Across Morphologies

**CrossFormer** [Doshi2024] (CoRL 2024, Oral — top 4%) extended cross-embodiment learning beyond manipulation to navigation, locomotion, and aerial vehicles:

- **Training data**: 900,000 trajectories across 20 embodiments, including single/dual-arm manipulators, wheeled robots, quadcopters, and quadrupeds.
- **Architecture**: modality-specific tokenizers → shared decoder-only transformer → embodiment-specific action heads (one per morphology type). Missing modalities are zero-masked.
- **Key finding**: a single network trained jointly outperforms separate per-embodiment policies on 12 of 20 evaluation tasks, with particularly large gains on data-scarce embodiments that benefit from knowledge transfer.
- **Action space handling**: each embodiment registers its action dimensionality and normalization statistics at training time; the corresponding head uses a small MLP to project from the shared latent.

---

## Cross-Embodiment Co-Training: Empirical Findings

A consistent empirical finding across OXE, Octo, and CrossFormer is that **adding diverse cross-embodiment data improves performance on the source embodiment**, even when the added data comes from morphologically very different robots. Mechanisms proposed:

1. **Shared visual representations**: diverse scenes and objects across datasets improve the visual encoder's generalization.
2. **Task-level transfer**: manipulation primitives (grasping, pushing, placing) appear across embodiments; sharing this signal improves primitive-level policies.
3. **Implicit data augmentation**: varied camera viewpoints, lighting, and backgrounds in cross-embodiment data act as augmentation for the target embodiment.

However, **negative transfer** occurs when datasets are too different in action statistics or task semantics; dataset mixture weighting (e.g., up-weighting in-distribution data) is critical for avoiding performance degradation.

---

## 2025–2026 Cross-Embodiment Generalists

### π0 / π0.5 (Physical Intelligence)

π0 was pretrained on 10,000+ hours from 7 robot configurations plus OXE; π0.5 adds multi-environment static robots, mobile manipulators (~400 h), and web/verbal data, and Physical Intelligence reports that **cross-embodiment robot data is critical for task competence** while web data matters most for novel-object generalisation. Actions from different robots are zero-padded to the largest action dimension, with the state/action layout implicitly identified by the model from the observations. The openpi release (Sept 2025) includes base checkpoints intended for fine-tuning to new embodiments; the README cautions that adaptation to very different platforms may not always succeed.

### GR00T N1.x: From Embodiment Tags to Shared Relative Actions

GR00T N1 used **embodiment-specific MLP encoders/decoders** around a shared DiT, selected by an *embodiment tag* (e.g., GR-1 humanoid, Franka, Unitree G1). N1.5 added heads for single-arm EEF control (OXE-DROID) and the AgiBot Genie-1 humanoid; N1.6 switched most embodiments to **state-relative action chunks** and added thousands of hours of bimanual YAM and AgiBot data; N1.7 expanded state/action dimensions to 132 and adopted a **relative end-effector action space shared across robots and humans**, which NVIDIA identifies as a key factor in its cross-embodiment performance [NVIDIA2026]. Every inference or fine-tuning call still requires an embodiment tag that selects the modality configuration (state/action keys and normalisation).

### Gemini Robotics 1.5: Motion Transfer

Gemini Robotics 1.5 is trained on heterogeneous data from ALOHA, bi-arm Franka and Apptronik Apollo with a **Motion Transfer (MT)** recipe that aligns motions across embodiments. DeepMind reports that tasks demonstrated only on one robot can be performed by the others without robot-specific data, and that MT improves performance even on embodiments with abundant data [GoogleDeepMind2025b]. Details of the mechanism are only partially disclosed.

### X-VLA: Soft Prompts for Heterogeneity

X-VLA (arXiv 2510.10274) treats each hardware setup/data source as a "task" encoded by learnable **soft prompts** injected at the earliest stage of multimodal fusion of a pure-Transformer, flow-matching VLA. X-VLA-0.9B is pretrained on 290K episodes from seven platforms (Phase I) and adapts to a new robot (Phase II) by learning a new prompt set with the backbone frozen; with only ~9M tuned parameters (1%) it approaches π0 on LIBERO and Simpler-WidowX. Fully fine-tuned checkpoints report 98.1% on LIBERO, 95.8% on Simpler-WidowX and 70% on RoboTwin 2.0.

### HPT: Embodiment-Specific Stems and Heads

Heterogeneous Pre-trained Transformers (HPT, NeurIPS 2024, arXiv 2409.20537) tokenise each embodiment's proprioception and vision through embodiment-specific **stems** into a fixed number of tokens processed by a shared trunk, with task-specific heads—an early formulation of the modular stem/trunk/head design adopted by many 2025 models, and an early study of scaling across 50+ datasets.

---

## Humans as an Embodiment

Human video dwarfs robot data, but lacks action labels and has a large morphology gap. Three strategies became mainstream:

1. **Latent actions.** LAPA (ICLR 2025, arXiv 2410.11758) learns a VQ latent-action model between frames of unlabelled video, pretrains a VLA to predict these latent actions, then fine-tunes to real actions with a small labelled set. UniVLA (arXiv 2505.06111) learns *task-centric* latent actions that suppress task-irrelevant motion; villa-X (arXiv 2507.23682) improves latent-action modelling and integrates it into VLA pretraining. NVIDIA's FLARE objective (in GR00T N1.5) aligns future latent representations to learn from human ego-video.
2. **Shared explicit action spaces.** EgoScale (arXiv 2602.16710; UC Berkeley, UT Austin, NVIDIA) scales dexterous manipulation pretraining with diverse egocentric human data; 20K hours of it are used in GR00T N1.7, made possible by the relative-EEF action space shared with robots. Being-H0.5 (arXiv 2601.12993; CoRL 2026) proposes a human-centric unified action space for cross-embodiment generalisation and releases the UniHand pretraining subset.
3. **Large egocentric datasets.** EgoVerse (arXiv 2604.07607) and similar efforts collect egocentric human manipulation data around the world specifically for robot learning.

In parallel, companies report very large proprietary *robot* corpora: Generalist AI's GEN-0 (Nov 2025) was trained on a very large corpus of real-world manipulation data across multiple embodiments and reported predictable scaling behaviour, with GEN-1 following in April 2026; DROID (76k trajectories, 564 scenes) remains the most widely used open single-embodiment "in-the-wild" dataset and the basis of the RoboArena evaluation.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | Padalkar, Walke, et al. (large consortium) | 2023 | arXiv 2310.08864; ICRA 2024 | 1M+ trajectories, 22 embodiments, 527 skills; RT-1-X +50% success; RT-2-X 3x improvement |
| Octo: An Open-Source Generalist Robot Policy | Octo Model Team | 2024 | RSS 2024 | First open-source generalist policy on OXE; diffusion head; fine-tunable to new embodiments |
| Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation (CrossFormer) | Doshi, Walke, Mees, Dasari, Levine | 2024 | CoRL 2024 (Oral, top 4%) | 900K traj, 20 embodiments incl. quadcopters; embodiment-specific heads on shared trunk |
| pi-zero: A Vision-Language-Action Flow Model for General Robot Control | Black et al. (Physical Intelligence) | 2024 | arXiv 2410.24164 | Flow matching + PaliGemma 3B; 7 platforms, 68 tasks, 10K+ demo hours; SOTA dexterous manipulation |
| RLDS: An Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning | Ramos et al. | 2021 | arXiv 2111.02767 | RLDS format — the standardization layer enabling OXE interoperability |
| RT-1: Robotics Transformer for Real-World Control at Scale | Brohan et al. | 2022 | RSS 2023 | Transformer action policy pretrained at scale; baseline architecture for RT-X |
| Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation | Hejna et al. | 2024 | arXiv 2402.19432 | Systematic study of negative transfer and mixture weighting strategies |
| Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (HPT) | Wang, Chen, Zhao, He | 2024 | NeurIPS 2024 | Embodiment-specific stems + shared trunk; scaling across heterogeneous datasets |
| Latent Action Pretraining from Videos (LAPA) | Ye et al. | 2024 | ICLR 2025 | VQ latent actions from unlabelled video for VLA pretraining |
| π₀.₅: a Vision-Language-Action Model with Open-World Generalization | Physical Intelligence | 2025 | arXiv 2504.16054 | Cross-embodiment + web co-training; cross-robot data critical for task competence |
| UniVLA: Learning to Act Anywhere with Task-centric Latent Actions | Bu et al. | 2025 | arXiv 2505.06111 | Task-centric latent actions for cross-embodiment pretraining |
| Gemini Robotics 1.5 | Google DeepMind | 2025 | arXiv 2510.03342 | Motion Transfer: one checkpoint across ALOHA, Franka, Apollo |
| X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA Model | Zheng et al. | 2025 | arXiv 2510.10274 | Per-source soft prompts; 0.9B; prompt-only adaptation to new robots |
| Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization | BeingBeyond | 2026 | arXiv 2601.12993; CoRL 2026 | Human-centric unified action space |
| EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data | Berkeley, UT Austin, NVIDIA et al. | 2026 | arXiv 2602.16710 | Egocentric human data at scale; 20K h used in GR00T N1.7 |

---

## Benchmark Performance

| Model | Evaluation | Metric | Score | Notes |
|-------|-----------|--------|-------|-------|
| RT-1-X | 5 diverse platforms (OXE eval) | Avg Success Rate | +50% vs. per-robot RT-1 | Co-training benefit on source embodiments |
| RT-2-X | Real-world manipulation skills | Success Rate | 3x vs. RT-2 single-embodiment | Emergent cross-embodiment generalization |
| Octo (fine-tuned) | 9 of 11 OXE eval tasks | Success Rate | Matches/exceeds SOTA | After minimal fine-tuning on target robot |
| CrossFormer | 20-embodiment eval suite | Avg Success Rate | Outperforms per-embodiment on 12/20 | Largest gains on data-scarce embodiments |
| pi-zero | 68 tasks, 7 platforms | Task Completion | SOTA on tested dexterous tasks | not publicly reported per-task numbers |
| Octo (zero-shot) | Berkeley Bi-Manual (novel platform) | Success Rate | not publicly reported | Demonstrates zero-shot new-embodiment capability |
| X-VLA-0.9B (full FT) | LIBERO / Simpler-WidowX / RoboTwin 2.0 | Success Rate | 98.1% / 95.8% / 70% | Repo-reported; Simpler Google Robot 83.5% (VM) / 76.4% (VA) |
| X-VLA-0.9B (prompt-only PEFT, ~9M params) | LIBERO Spatial / Object / Goal | Success Rate | 96.2% / 96.0% / 94.4% | Backbone frozen; new soft prompts only |
| GR00T N1.6 | LIBERO 4-suite avg | Success Rate | ≈97.0% | Single 3B cross-embodiment base, fine-tuned |
| GR00T N1.5 | GR-1 language following | Success Rate | 93.3% vs. 46.6% (N1) | Frozen VLM + FLARE + DreamGen data |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| Data efficiency | Cross-embodiment co-training acts as free augmentation; scarce-data embodiments benefit most from shared pretraining | Negative transfer can occur when action statistics or task semantics diverge significantly; requires careful mixture weighting |
| Generalization | Models learn platform-agnostic visual and task representations; better zero-shot generalization to unseen scenes and objects | Generalization within a novel embodiment is still limited; fine-tuning or adaptation data from the target robot is usually needed |
| Infrastructure | RLDS standardization enables plug-and-play dataset mixing; growing ecosystem of open datasets | RLDS/TFRecord infrastructure is non-trivial to adopt; PyTorch-native alternatives are still maturing |
| Architecture flexibility | Modality-specific tokenizers + masking elegantly handle missing sensors; embodiment-specific heads avoid conflation of different action spaces | Model capacity must scale with number of embodiments; storing per-embodiment heads adds parameters; memory cost grows with embodiment diversity |
| Soft prompts / embodiment tags | Cheap adaptation (prompt-only or tag + head); backbone shared | Still requires per-robot data and normalisation statistics; no zero-shot transfer to truly unseen morphologies |
| Human video as an embodiment | Orders of magnitude more data; rich dexterous behaviour | Morphology gap (hands vs. grippers); noisy hand-pose estimates; benefits depend on a shared action space |

---

## Open Problems & Research Gaps

1. **Principled mixture weighting.** How should training batches weight trajectories from 22+ embodiments? Current practice mixes by dataset size or manual heuristics; a theoretically principled approach (e.g., gradient-based data selection, influence functions) remains absent.

2. **Action space unification.** MinMax normalization and embodiment-specific heads are workarounds. A truly universal action representation — perhaps in a latent space learned by a cross-embodiment VAE or via relative joint-frame primitives — that allows genuine zero-shot transfer to unseen morphologies is an open problem.

3. **Proprioceptive and tactile integration.** Most cross-embodiment work focuses on RGB visual observations and ignores proprioception or force-torque signals, which are critical for contact-rich assembly. Tokenizing and aligning these signals across heterogeneous sensor configurations is largely unsolved.

4. **Human demonstration transfer.** Human video contains orders of magnitude more diverse manipulation data. Bridging the human-to-robot embodiment gap (morphology, kinematics, tool use) via retargeting, reward learning, or latent alignment is a major open direction (related: DexMimic, AnyDex, 2025-2026 work).

5. **Evaluation protocol standardization.** Every cross-embodiment paper defines its own evaluation suite, making comparisons across papers impossible. A standardized multi-embodiment benchmark analogous to ImageNet for manipulation generalization does not yet exist.

6. **Continual/lifelong cross-embodiment learning.** As new robots are added to a fleet, catastrophic forgetting of old embodiment skills is a real concern. Continual learning methods (EWC, LoRA-based adapters) for incremental embodiment addition are not yet mature.

7. **Scaling laws for cross-embodiment co-training.** It is unknown whether OXE-style co-training follows power-law scaling (more embodiments = monotonically better) or exhibits diminishing returns. Mapping the scaling curve requires systematic ablations not yet published. Company reports (Generalist GEN-0) claim predictable scaling with real-world data, but without public data or ablations.

8. **Which action space should be universal?** 2025–2026 models disagree: zero-padded joint vectors (π0), embodiment-tagged heads (GR00T N1–N1.5), relative EEF deltas shared with humans (GR00T N1.7, Being-H0.5), latent actions (LAPA, UniVLA), or learned prompts (X-VLA). Relative EEF spaces ease human transfer but discard joint-level information crucial for dexterous hands and whole-body humanoids.

9. **Transparent cross-embodiment claims.** Motion Transfer and similar closed recipes report striking zero-shot transfer, but without released data or code it is hard to know which components (shared kinematic retargeting, data mixture, model scale) are responsible.

---

## Further Reading

- [Open X-Embodiment project page](https://robotics-transformer-x.github.io/)
- [Octo model and code (octo-models.github.io)](https://octo-models.github.io/)
- [CrossFormer paper — arXiv:2408.11812](https://arxiv.org/abs/2408.11812)
- [RLDS paper — arXiv:2111.02767](https://arxiv.org/abs/2111.02767)
- [Open X-Embodiment GitHub (google-deepmind)](https://github.com/google-deepmind/open_x_embodiment)
- [pi-zero paper — arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- [Isaac GR00T (embodiment tags, relative EEF, EgoScale)](https://github.com/NVIDIA/Isaac-GR00T)
- [X-VLA — arXiv:2510.10274](https://arxiv.org/abs/2510.10274) and [code](https://github.com/2toINF/X-VLA)
- [Gemini Robotics 1.5 — arXiv:2510.03342](https://arxiv.org/abs/2510.03342)
- [HPT — arXiv:2409.20537](https://arxiv.org/abs/2409.20537)
- [LAPA — arXiv:2410.11758](https://arxiv.org/abs/2410.11758)
- [EgoScale — arXiv:2602.16710](https://arxiv.org/abs/2602.16710)
- [Being-H (H0 / H0.5 / H0.7)](https://github.com/BeingBeyond/Being-H)
- [LeRobotDataset v3 documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
