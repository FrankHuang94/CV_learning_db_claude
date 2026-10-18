# Key Labs & Research Groups

> **Last Updated:** October 2026
> **Level:** Foundational
> **Related Sections:**
> - [Embodied AI Overview & Roadmap](../06_robotics_and_embodied_ai/00_overview_roadmap.md)
> - [PhD Study Guide](./04_phd_study_guide.md)
> - [Key Papers](./00_key_papers.md)

> **🆕 October 2026 update:**
> - **Meta** consolidated its AI research under Meta Superintelligence Labs (2025). Its 2025–2026 vision releases include SAM 3 / SAM 3.1, SAM 3D, DINOv3 and V-JEPA 2 / 2.1. **Yann LeCun left Meta** at the end of 2025 and founded **AMI Labs**, a JEPA-style world-model startup with a large reported 2026 raise.
> - **Google DeepMind:** Genie 3, Gemini Robotics 1.5 / ER 1.5, Gemini 3, Veo 3.x, and the CVPR 2026 Best Paper D4RT (with Oxford VGG). **NVIDIA:** GR00T N1.6 → N1.7, Cosmos Reason 2, and the Newton physics engine (with Google DeepMind and Disney Research, now a Linux Foundation project).
> - Academic highlights: **Oxford VGG** won back-to-back CVPR Best Papers (VGGT 2025, D4RT 2026); **CMU** won the ICCV 2025 Marr Prize (BrickGPT); **Imperial** won the ECCV 2026 Best Paper (HKTex); **Tsinghua/Microsoft** won the CVPR 2026 Best Student Paper (TRELLIS.2).
> - Startup changes: π₀.₅ / π\*₀.₆ (Physical Intelligence), Figure 03 and the end of the Figure–OpenAI collaboration, 1X NEO consumer preorders, World Labs Marble, and Covariant's founders moving to Amazon. Chinese industry labs (Qwen, ByteDance Seed) added.

---

## Overview

Computer vision research is produced by a relatively concentrated set of industry labs and academic groups whose output disproportionately shapes the field's direction. Knowing who works where—and what each group's research program and "house style" is—is practically valuable for a PhD student: it informs who to read, whose code to trust, where to apply for internships, and how to interpret the provenance and likely follow-up of a given paper. The landscape bifurcates into **industry research labs** (FAIR/Meta, Google DeepMind, OpenAI, NVIDIA, Microsoft Research), which increasingly drive frontier-scale work requiring large compute, and **academic groups** (Berkeley BAIR, Stanford, CMU, MIT, Oxford VGG, and others), which contribute foundational ideas, train the field's researchers, and often partner with industry.

A defining feature of the 2024–2026 period is the rise of **embodied-AI startups**—Physical Intelligence, Figure AI, 1X, Skild AI, World Labs—often spun out of the academic groups below and absorbing their faculty and students (see [Embodied AI Overview](../06_robotics_and_embodied_ai/00_overview_roadmap.md)). This file maps the major groups, their focus areas, and representative contributions. (Personnel move frequently; affiliations below reflect the field as of October 2026 and should be verified for current status.) The 2025–2026 period also brought **frontier-lab reorganizations** (Meta Superintelligence Labs), senior departures to found startups (LeCun → AMI Labs), and faculty moves from industry to academia (Kaiming He to MIT in 2024).

---

## Industry Research Labs

- **FAIR / Meta AI** — open-research powerhouse; SAM/SAM 2/SAM 3 (concept segmentation, Nov 2025; SAM 3.1 Object Multiplex, Mar 2026), SAM 3D (CVPR 2026 Best Paper honorable mention), DINO/DINOv2/DINOv3 (Aug 2025), MAE, ImageBind, V-JEPA/V-JEPA 2 (Jun 2025)/V-JEPA 2.1 (Mar 2026), VGGT (with Oxford), MapAnything, Llama. Strong commitment to open weights and self-supervised learning (see [Self-Supervised Learning](../03_architectures/03_self_supervised_learning.md)). In 2025 Meta reorganized its AI efforts under **Meta Superintelligence Labs (MSL)**, with FAIR inside it; recent vision papers (e.g., SAM 3D) are credited to MSL. Yann LeCun, the architect of the JEPA world-model agenda, **left Meta at the end of 2025** to found AMI Labs (below).
- **Google DeepMind** — merged Google Brain + DeepMind; RT-1/2/X, Gemini (Gemini 3, Nov 2025), Gemini Robotics / Gemini Robotics 1.5 and ER 1.5 (Sept 2025), Genie/Genie 2/Genie 3 (Aug 2025; reportedly the basis of the Waymo World Model, 2026), Veo 3.x, Imagen, ViT, Flamingo, and D4RT (CVPR 2026 Best Paper, with Oxford VGG/UCL). Co-initiator of the Newton physics engine and maintainer of MuJoCo / MuJoCo Warp. Spans foundation VLMs, robotics, and world models.
- **OpenAI** — GPT-4V/4o/GPT-5, DALL·E, CLIP (origin), Sora / Sora 2 (Sept 2025). Its humanoid collaboration with Figure ended in Feb 2025, when Figure moved to its in-house Helix VLA.
- **NVIDIA Research** — Cosmos (Predict 2.5, Transfer, Reason / Reason 2), GR00T N1 → N1.5 → N1.6 → N1.7 (Apache-2.0, Cosmos-Reason2 / Qwen3-VL backbone, pretrained with 20k hours of EgoScale human video), Isaac Sim (6.x in 2026)/Isaac Lab, Newton (co-initiated with Google DeepMind and Disney Research; now a Linux Foundation project), NitroGen (CVPR 2026 honorable mention), FoundationPose, StyleGAN lineage, Instant-NGP; the dominant force in robot simulation and physical-AI infrastructure (see [Simulation and Data](../06_robotics_and_embodied_ai/05_simulation_and_data.md)).
- **Microsoft Research** — Swin Transformer, BEiT, LayoutLM, Florence, CvT, TRELLIS / TRELLIS.2 (CVPR 2026 Best Student Paper, with Tsinghua and USTC); strong document AI, backbone and 3D-generation work.
- **Apple, Allen Institute for AI (Ai2)** — MobileViT/efficient models (Apple); OLMo, ProcTHOR, Molmo, MolmoAct / MolmoAct2 open action-reasoning models (now integrated in LeRobot), embodied benchmarks (Ai2).
- **Chinese industry labs** — Alibaba **Qwen** (Qwen2.5-VL → Qwen3-VL, a widely used open VLM backbone that NVIDIA's Cosmos-Reason2 also builds on); **ByteDance Seed** (video generation and robot VLAs); Shanghai AI Lab (InternVL). These labs supply many of the open-weight VLM backbones that academic VLA work now builds on.

## Academic Groups (Robotics-CV focus)

- **UC Berkeley (BAIR)** — Sergey Levine, Pieter Abbeel, Trevor Darrell, Alyosha Efros, Jitendra Malik. Robot learning (BridgeData, OpenVLA co-development), generative models, representation learning—arguably the most influential academic CV/robotics group.
- **Stanford** — Fei-Fei Li (ImageNet, spatial intelligence, World Labs), Chelsea Finn (ALOHA, π₀ co-founder, meta-learning), Jiajun Wu, Karen Liu. Vision + robot learning + 3D.
- **CMU** — Robotics Institute; Deepak Pathak, Abhinav Gupta, Shubham Tulsiani, Deva Ramanan, Jun-Yan Zhu, Changliu Liu. Self-supervised robot learning, 3D, Genesis simulator (1.0 released May 2026); ICCV 2025 Marr Prize (BrickGPT).
- **MIT** — CSAIL; Russ Tedrake (model-based control, Diffusion Policy), Antonio Torralba, Phillip Isola, Vincent Sitzmann (neural fields), Bill Freeman, Kaiming He (joined MIT EECS in 2024 from FAIR; ResNet, Mask R-CNN, MAE).
- **Oxford VGG** — Andrew Zisserman, Andrea Vedaldi, and others. Won consecutive CVPR Best Papers: VGGT (2025, with Meta) and D4RT (2026, with Google DeepMind/UCL). It is the leading academic group in feed-forward 3D/4D reconstruction.
- **Other notable** — Imperial College London (HKTex, ECCV 2026 Best Paper; Zafeiriou, Birdal), Tsinghua (TRELLIS.2), Technion (FlowEdit, ICCV 2025 Best Student Paper), University of Toronto/Vector, Princeton, NYU, Tübingen, ETH Zürich, Shanghai AI Lab (InternVL/InternImage), HUST (Vision Mamba).

## Embodied-AI Startups (2024–2026)

- **Physical Intelligence (π)** — Karol Hausman, Sergey Levine, Chelsea Finn, Brian Ichter; π₀/π₀-FAST/π₀.₅ flow-matching VLAs (π₀.₅ weights open in openpi, with PyTorch support since Sept 2025), and π\*₀.₆ (Nov 2025), which adds RL from experience on top of the VLA; data collection at scale.
- **Figure AI** — Figure 02 and Figure 03 (Oct 2025) humanoids; Helix dual-system VLA (in-house since the end of the OpenAI collaboration in Feb 2025); BMW deployment.
- **1X Technologies** — NEO humanoid; OpenAI-backed; consumer home robots, with NEO preorders opened in Oct 2025 for 2026 delivery, supported by teleoperation for tasks the autonomy cannot yet handle.
- **World Labs** — Fei-Fei Li, Justin Johnson and co-founders; spatial intelligence and 3D world models. Launched **Marble** (Nov 2025), a commercial product that generates persistent, exportable 3D worlds (e.g., Gaussian splats) from text, images or video.
- **AMI Labs (Advanced Machine Intelligence)** — founded by Yann LeCun after he left Meta (announced Nov 2025; funding round reported in 2026). It pursues JEPA-style predictive world models rather than generative LLM scaling. Expect it to compete for the same self-supervised video and world-model talent as Meta, Google DeepMind and NVIDIA (see [World Models](../06_robotics_and_embodied_ai/02_world_models.md)).
- **Skild AI, Covariant (Pieter Abbeel), Wayve** — manipulation foundation models (Skild Brain, 2025), AV end-to-end driving. Covariant's founders (Abbeel, Peter Chen, Rocky Duan) joined Amazon in 2024 under a licensing deal, an example of the "acqui-hire" route by which robot-learning startups end up inside big tech.

---

## Group Comparison

| Group | Type | Focus | Representative work |
|-------|------|-------|---------------------|
| FAIR / Meta (MSL) | Industry | SSL, open foundation, world models | SAM 3, DINOv3, V-JEPA 2 |
| Google DeepMind | Industry | VLMs, robotics, world models | Gemini 3, Gemini Robotics 1.5, Genie 3 |
| NVIDIA | Industry | Simulation, physical AI | Cosmos, GR00T N1.7, Isaac, Newton |
| Oxford VGG | Academic | 3D/4D reconstruction, video | VGGT, D4RT |
| Berkeley BAIR | Academic | Robot learning, gen models | OpenVLA, BridgeData |
| Stanford | Academic | Vision, robot learning, 3D | ImageNet, ALOHA |
| Physical Intelligence | Startup | Manipulation VLAs | π₀, π₀.₅, π\*₀.₆ |
| World Labs | Startup | Spatial intelligence, 3D worlds | Marble |
| AMI Labs | Startup | JEPA world models | (founded 2026; early stage) |

---

## Pros & Cons (industry vs. academia for a PhD)

| Aspect | Industry lab | Academic group |
|--------|--------------|-----------------|
| Compute/data | Abundant; frontier scale | Limited; forces cleverness |
| Freedom | Product/strategy constraints | Open problem choice |
| Mentorship | Senior researchers, fast feedback | Long-term advising, teaching |
| Openness | Variable (some closed) | Generally open publication |

---

## Open Problems & Research Gaps (meta)

- **Compute divide.** Frontier results increasingly require industry-scale resources, narrowing what academia can pursue.
- **Brain drain.** Faculty and students flowing to startups/industry reshapes academic capacity.
- **Reproducibility of closed work.** Key results (Gemini Robotics, GPT-4o) cannot be independently replicated.
- **Open vs. closed tension.** The field's open-source tradition is in tension with commercial consolidation.
- **Talent concentration.** A few groups dominate, raising questions about diversity of ideas.
- **Data moats.** Robot-data collection advantages may entrench incumbents (see [Pros, Cons & Roadmaps](../06_robotics_and_embodied_ai/07_pros_cons_roadmaps.md)).
- **Lab instability.** Frequent reorganizations (e.g., Meta's MSL restructuring and 2025 layoffs) and founder departures make lab "house style" and code maintenance less predictable. Check whether a repository is still maintained before building on it.
- **Startup capture.** Acqui-hires and licensing deals (Covariant → Amazon) and nine-to-ten-figure startup raises (Physical Intelligence, Skild AI, AMI Labs) pull senior researchers out of open publication.

---

## Further Reading

- [BAIR Blog](https://bair.berkeley.edu/blog/) — Berkeley AI Research
- [Meta AI Research](https://ai.meta.com/research/) — FAIR publications
- [Google DeepMind](https://deepmind.google/research/) — research index
- [Physical Intelligence](https://www.physicalintelligence.company/) — π₀ and robot foundation models
- [Stanford Vision & Learning Lab (SVL)](https://svl.stanford.edu/) — vision + robotics
- [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/) — VGGT, D4RT
- [World Labs](https://www.worldlabs.ai/) — Marble and spatial-intelligence research
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — open humanoid VLA releases (N1.5 → N1.7)
- [Newton physics engine](https://github.com/newton-physics/newton) — Linux Foundation project initiated by Disney Research, Google DeepMind and NVIDIA
