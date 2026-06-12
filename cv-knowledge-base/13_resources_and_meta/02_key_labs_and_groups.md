# Key Labs & Research Groups

> **Last Updated:** June 2026
> **Level:** Foundational
> **Related Sections:**
> - [Embodied AI Overview & Roadmap](../06_robotics_and_embodied_ai/00_overview_roadmap.md)
> - [PhD Study Guide](./04_phd_study_guide.md)
> - [Key Papers](./00_key_papers.md)

---

## Overview

Computer vision research is produced by a relatively concentrated set of industry labs and academic groups whose output disproportionately shapes the field's direction. Knowing who works where—and what each group's research program and "house style" is—is practically valuable for a PhD student: it informs who to read, whose code to trust, where to apply for internships, and how to interpret the provenance and likely follow-up of a given paper. The landscape bifurcates into **industry research labs** (FAIR/Meta, Google DeepMind, OpenAI, NVIDIA, Microsoft Research), which increasingly drive frontier-scale work requiring large compute, and **academic groups** (Berkeley BAIR, Stanford, CMU, MIT, Oxford VGG, and others), which contribute foundational ideas, train the field's researchers, and often partner with industry.

A defining feature of the 2024–2026 period is the rise of **embodied-AI startups**—Physical Intelligence, Figure AI, 1X, Skild AI, World Labs—often spun out of the academic groups below and absorbing their faculty and students (see [Embodied AI Overview](../06_robotics_and_embodied_ai/00_overview_roadmap.md)). This file maps the major groups, their focus areas, and representative contributions. (Personnel move frequently; affiliations below reflect the field as of mid-2026 and should be verified for current status.)

---

## Industry Research Labs

- **FAIR / Meta AI** — open-research powerhouse; SAM/SAM2, DINO/DINOv2, MAE, ImageBind, V-JEPA/V-JEPA 2, Llama. Strong commitment to open weights and self-supervised learning (see [Self-Supervised Learning](../03_architectures/03_self_supervised_learning.md)). Yann LeCun's world-model agenda (JEPA) is central.
- **Google DeepMind** — merged Google Brain + DeepMind; RT-1/2/X, Gemini, Gemini Robotics, Genie/Genie 2, Imagen, ViT, Flamingo. Spans foundation VLMs, robotics, and world models.
- **OpenAI** — GPT-4V/4o, DALL·E, CLIP (origin), Sora; partnerships with Figure on humanoid VLAs.
- **NVIDIA Research** — Cosmos, GR00T, Isaac Sim/Lab, FoundationPose, StyleGAN lineage, Instant-NGP; the dominant force in robot simulation and physical-AI infrastructure (see [Simulation and Data](../06_robotics_and_embodied_ai/05_simulation_and_data.md)).
- **Microsoft Research** — Swin Transformer, BEiT, LayoutLM, Florence, CvT; strong document AI and backbone work.
- **Apple, Allen Institute for AI (Ai2)** — MobileViT/efficient models (Apple); OLMo, ProcTHOR, Molmo, embodied benchmarks (Ai2).

## Academic Groups (Robotics-CV focus)

- **UC Berkeley (BAIR)** — Sergey Levine, Pieter Abbeel, Trevor Darrell, Alyosha Efros, Jitendra Malik. Robot learning (BridgeData, OpenVLA co-development), generative models, representation learning—arguably the most influential academic CV/robotics group.
- **Stanford** — Fei-Fei Li (ImageNet, spatial intelligence, World Labs), Chelsea Finn (ALOHA, π₀ co-founder, meta-learning), Jiajun Wu, Karen Liu. Vision + robot learning + 3D.
- **CMU** — Robotics Institute; Deepak Pathak, Abhinav Gupta, Shubham Tulsiani, Deva Ramanan. Self-supervised robot learning, 3D, Genesis simulator.
- **MIT** — CSAIL; Russ Tedrake (model-based control, Diffusion Policy), Antonio Torralba, Phillip Isola, Vincent Sitzmann (neural fields), Bill Freeman.
- **Other notable** — Oxford VGG (Andrew Zisserman), University of Toronto/Vector, Princeton, NYU, Tübingen, ETH Zürich, Shanghai AI Lab (InternVL/InternImage), HUST (Vision Mamba).

## Embodied-AI Startups (2024–2026)

- **Physical Intelligence (π)** — Karol Hausman, Sergey Levine, Chelsea Finn, Brian Ichter; π₀/π₀.5 flow-matching VLAs; data-collection at scale.
- **Figure AI** — Figure 02/03 humanoids; Helix dual-system VLA; OpenAI partnership; BMW deployment.
- **1X Technologies** — NEO humanoid; OpenAI-backed; consumer home robots.
- **World Labs** — Fei-Fei Li; spatial intelligence and 3D world models.
- **Skild AI, Covariant (Pieter Abbeel), Wayve** — manipulation foundation models, AV end-to-end driving.

---

## Group Comparison

| Group | Type | Focus | Representative work |
|-------|------|-------|---------------------|
| FAIR / Meta | Industry | SSL, open foundation, world models | SAM, DINOv2, V-JEPA |
| Google DeepMind | Industry | VLMs, robotics, world models | Gemini, RT-2, Genie |
| NVIDIA | Industry | Simulation, physical AI | Cosmos, GR00T, Isaac |
| Berkeley BAIR | Academic | Robot learning, gen models | OpenVLA, BridgeData |
| Stanford | Academic | Vision, robot learning, 3D | ImageNet, ALOHA |
| Physical Intelligence | Startup | Manipulation VLAs | π₀, π₀.5 |

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

---

## Further Reading

- [BAIR Blog](https://bair.berkeley.edu/blog/) — Berkeley AI Research
- [Meta AI Research](https://ai.meta.com/research/) — FAIR publications
- [Google DeepMind](https://deepmind.google/research/) — research index
- [Physical Intelligence](https://www.physicalintelligence.company/) — π₀ and robot foundation models
- [Stanford Vision & Learning Lab (SVL)](https://svl.stanford.edu/) — vision + robotics
