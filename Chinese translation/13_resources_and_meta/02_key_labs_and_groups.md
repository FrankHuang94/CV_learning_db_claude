# 重要实验室与研究组

> **英文原文：** [cv-knowledge-base/13_resources_and_meta/02_key_labs_and_groups.md](../../cv-knowledge-base/13_resources_and_meta/02_key_labs_and_groups.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 基础
> **相关章节：**
> - [具身智能概览与路线图](../06_robotics_and_embodied_ai/00_overview_roadmap.md)
> - [博士学习指南](./04_phd_study_guide.md)
> - [关键论文](./00_key_papers.md)

> **🆕 2026年10月更新：**
> - **Meta** 于 2025 年将其 AI 研究整合进 Meta Superintelligence Labs。其 2025–2026 年发布的视觉成果包括 SAM 3 / SAM 3.1、SAM 3D、DINOv3 和 V-JEPA 2 / 2.1。**Yann LeCun 于 2025 年底离开 Meta**，创立了 **AMI Labs**，这是一家研究 JEPA 式世界模型的初创公司，据报道于 2026 年完成大额融资。
> - **Google DeepMind：** Genie 3、Gemini Robotics 1.5 / ER 1.5、Gemini 3、Veo 3.x，以及 CVPR 2026 最佳论文 D4RT（与 Oxford VGG 合作）。**NVIDIA：** GR00T N1.6 → N1.7、Cosmos Reason 2，以及 Newton 物理引擎（与 Google DeepMind 和 Disney Research 合作，现为 Linux 基金会项目）。
> - 学术界亮点：**Oxford VGG** 连续两年获得 CVPR 最佳论文（2025 年 VGGT、2026 年 D4RT）；**CMU** 获得 ICCV 2025 Marr 奖（BrickGPT）；**帝国理工学院**获得 ECCV 2026 最佳论文（HKTex）；**清华/微软**获得 CVPR 2026 最佳学生论文（TRELLIS.2）。
> - 初创公司变化：π₀.₅ / π\*₀.₆（Physical Intelligence）、Figure 03 以及 Figure 与 OpenAI 合作的终止、1X NEO 面向消费者开放预订、World Labs 推出 Marble、Covariant 创始人加入 Amazon。新增中国产业界实验室（Qwen、字节跳动 Seed）。

---

## 概述

计算机视觉研究由相对集中的一批产业实验室和学术研究组产出，其成果对领域方向的影响远超其数量占比。了解谁在哪里工作，以及每个研究组的研究计划和“风格”，对博士生很有实际价值：它决定了读谁的论文、信任谁的代码、去哪里申请实习，以及如何解读一篇论文的来源和可能的后续工作。整体格局分为两类：**产业研究实验室**（FAIR/Meta、Google DeepMind、OpenAI、NVIDIA、Microsoft Research）日益主导需要大规模算力的前沿工作；**学术研究组**（Berkeley BAIR、Stanford、CMU、MIT、Oxford VGG 等）贡献基础性思想、培养领域人才，并常与产业界合作。

2024–2026 年的一个标志性特征是**具身智能初创公司**的兴起：Physical Intelligence、Figure AI、1X、Skild AI、World Labs 等，它们常由下文所列学术研究组孵化，并吸纳其教师和学生（参见[具身智能概览](../06_robotics_and_embodied_ai/00_overview_roadmap.md)）。本文梳理主要研究组、其关注方向和代表性贡献。（人员流动频繁；下列隶属关系反映截至 2026年10月的情况，请自行核实最新状态。）2025–2026 年还出现了**前沿实验室重组**（Meta Superintelligence Labs）、资深研究者离职创业（LeCun → AMI Labs），以及从产业界到学术界的教职流动（何恺明于 2024 年加入 MIT）。

---

## 产业研究实验室

- **FAIR / Meta AI** — 开放研究的重镇；SAM/SAM 2/SAM 3（概念分割，2025年11月；SAM 3.1 Object Multiplex，2026年3月）、SAM 3D（CVPR 2026 最佳论文荣誉提名）、DINO/DINOv2/DINOv3（2025年8月）、MAE、ImageBind、V-JEPA/V-JEPA 2（2025年6月）/V-JEPA 2.1（2026年3月）、VGGT（与牛津合作）、MapAnything、Llama。坚定致力于开放权重和自监督学习（参见[自监督学习](../03_architectures/03_self_supervised_learning.md)）。2025 年，Meta 将其 AI 工作重组到 **Meta Superintelligence Labs（MSL）** 之下，FAIR 隶属其中；近期的视觉论文（如 SAM 3D）署名为 MSL。JEPA 世界模型议程的设计者 Yann LeCun **于 2025 年底离开 Meta**，创立了 AMI Labs（见下文）。
- **Google DeepMind** — 由 Google Brain 与 DeepMind 合并而成；RT-1/2/X、Gemini（Gemini 3，2025年11月）、Gemini Robotics / Gemini Robotics 1.5 与 ER 1.5（2025年9月）、Genie/Genie 2/Genie 3（2025年8月；据报道是 Waymo World Model 的基础，2026）、Veo 3.x、Imagen、ViT、Flamingo，以及 D4RT（CVPR 2026 最佳论文，与 Oxford VGG/UCL 合作）。Newton 物理引擎的共同发起方，MuJoCo / MuJoCo Warp 的维护者。研究覆盖基础 VLM、机器人和世界模型。
- **OpenAI** — GPT-4V/4o/GPT-5、DALL·E、CLIP（原创）、Sora / Sora 2（2025年9月）。其与 Figure 的人形机器人合作于 2025年2月终止，此后 Figure 转向自研的 Helix VLA。
- **NVIDIA Research** — Cosmos（Predict 2.5、Transfer、Reason / Reason 2）、GR00T N1 → N1.5 → N1.6 → N1.7（Apache-2.0 许可；采用 Cosmos-Reason2 / Qwen3-VL 骨干；使用 2 万小时 EgoScale 人类视频预训练）、Isaac Sim（2026 年为 6.x 版本）/Isaac Lab、Newton（与 Google DeepMind 和 Disney Research 共同发起；现为 Linux 基金会项目）、NitroGen（CVPR 2026 荣誉提名）、FoundationPose、StyleGAN 系列、Instant-NGP；是机器人仿真与物理 AI 基础设施领域的主导力量（参见[仿真与数据](../06_robotics_and_embodied_ai/05_simulation_and_data.md)）。
- **Microsoft Research** — Swin Transformer、BEiT、LayoutLM、Florence、CvT、TRELLIS / TRELLIS.2（CVPR 2026 最佳学生论文，与清华和中国科大合作）；在文档 AI、骨干网络和三维生成方面实力强劲。
- **Apple、艾伦人工智能研究所（Ai2）** — MobileViT/高效模型（Apple）；OLMo、ProcTHOR、Molmo、MolmoAct / MolmoAct2 开放动作推理模型（现已集成到 LeRobot）、具身基准（Ai2）。
- **中国产业界实验室** — 阿里巴巴 **Qwen**（Qwen2.5-VL → Qwen3-VL，被广泛使用的开放 VLM 骨干，NVIDIA 的 Cosmos-Reason2 也基于它构建）；**字节跳动 Seed**（视频生成与机器人 VLA）；上海人工智能实验室（InternVL）。学术界的 VLA 工作如今所用的开放权重 VLM 骨干，很多来自这些实验室。

## 学术研究组（侧重机器人-CV）

- **UC Berkeley (BAIR)** — Sergey Levine、Pieter Abbeel、Trevor Darrell、Alyosha Efros、Jitendra Malik。机器人学习（BridgeData、参与 OpenVLA 开发）、生成模型、表示学习，可以说是最具影响力的学术 CV/机器人研究组。
- **Stanford** — 李飞飞（Fei-Fei Li；ImageNet、空间智能、World Labs）、Chelsea Finn（ALOHA、π₀ 联合创始人、元学习）、吴佳俊（Jiajun Wu）、Karen Liu。视觉 + 机器人学习 + 三维。
- **CMU** — 机器人研究所；Deepak Pathak、Abhinav Gupta、Shubham Tulsiani、Deva Ramanan、朱俊彦（Jun-Yan Zhu）、Changliu Liu。自监督机器人学习、三维、Genesis 仿真器（1.0 于2026年5月发布）；ICCV 2025 Marr 奖（BrickGPT）。
- **MIT** — CSAIL；Russ Tedrake（基于模型的控制、Diffusion Policy）、Antonio Torralba、Phillip Isola、Vincent Sitzmann（神经场）、Bill Freeman、何恺明（Kaiming He；2024 年从 FAIR 加入 MIT EECS；ResNet、Mask R-CNN、MAE）。
- **Oxford VGG** — Andrew Zisserman、Andrea Vedaldi 等。连续两年获得 CVPR 最佳论文：VGGT（2025，与 Meta 合作）和 D4RT（2026，与 Google DeepMind/UCL 合作）。它是前馈式三维/四维重建领域领先的学术研究组。
- **其他值得关注的研究组** — 帝国理工学院（HKTex，ECCV 2026 最佳论文；Zafeiriou、Birdal）、清华大学（TRELLIS.2）、以色列理工学院（Technion；FlowEdit，ICCV 2025 最佳学生论文）、多伦多大学/Vector 研究所、普林斯顿、NYU、图宾根、ETH Zürich、上海人工智能实验室（InternVL/InternImage）、华中科技大学（Vision Mamba）。

## 具身智能初创公司（2024–2026）

- **Physical Intelligence (π)** — Karol Hausman、Sergey Levine、Chelsea Finn、Brian Ichter；π₀/π₀-FAST/π₀.₅ 流匹配（flow matching）VLA（π₀.₅ 权重在 openpi 中开放，自 2025年9月起支持 PyTorch），以及 π\*₀.₆（2025年11月），后者在 VLA 之上加入了基于经验的强化学习；大规模数据采集。
- **Figure AI** — Figure 02 和 Figure 03（2025年10月）人形机器人；Helix 双系统 VLA（自 2025年2月与 OpenAI 合作结束后自研）；在 BMW 部署。
- **1X Technologies** — NEO 人形机器人；获 OpenAI 投资；面向消费者的家用机器人。NEO 于 2025年10月开放预订、计划 2026 年交付，自主能力尚无法完成的任务由遥操作（teleoperation）辅助。
- **World Labs** — 李飞飞、Justin Johnson 及其他联合创始人；空间智能与三维世界模型。推出了 **Marble**（2025年11月），这是一款商业产品，可根据文本、图像或视频生成持久、可导出的三维世界（如高斯泼溅（Gaussian Splatting）表示）。
- **AMI Labs（Advanced Machine Intelligence）** — 由 Yann LeCun 在离开 Meta 后创立（2025年11月宣布；据报道 2026 年完成融资）。它追求 JEPA 式的预测性世界模型，而非生成式 LLM 的规模扩展。预计它将与 Meta、Google DeepMind 和 NVIDIA 争夺同一批自监督视频与世界模型人才（参见[世界模型](../06_robotics_and_embodied_ai/02_world_models.md)）。
- **Skild AI、Covariant（Pieter Abbeel）、Wayve** — 操作基础模型（Skild Brain，2025）、自动驾驶端到端（end-to-end）驾驶。Covariant 的创始人（Abbeel、Peter Chen、Rocky Duan）于 2024 年通过一项授权协议加入 Amazon，这是机器人学习初创公司以“收购式招聘”（acqui-hire）方式并入大型科技公司的一个例子。

---

## 研究组对比

| 研究组 | 类型 | 关注方向 | 代表性工作 |
|-------|------|-------|---------------------|
| FAIR / Meta (MSL) | 产业界 | 自监督学习、开放基础模型、世界模型 | SAM 3, DINOv3, V-JEPA 2 |
| Google DeepMind | 产业界 | VLM、机器人、世界模型 | Gemini 3, Gemini Robotics 1.5, Genie 3 |
| NVIDIA | 产业界 | 仿真、物理 AI | Cosmos, GR00T N1.7, Isaac, Newton |
| Oxford VGG | 学术界 | 三维/四维重建、视频 | VGGT, D4RT |
| Berkeley BAIR | 学术界 | 机器人学习、生成模型 | OpenVLA, BridgeData |
| Stanford | 学术界 | 视觉、机器人学习、三维 | ImageNet, ALOHA |
| Physical Intelligence | 初创公司 | 操作类 VLA | π₀, π₀.₅, π\*₀.₆ |
| World Labs | 初创公司 | 空间智能、三维世界 | Marble |
| AMI Labs | 初创公司 | JEPA 世界模型 | （2026 年成立；早期阶段） |

---

## 优缺点（博士生视角：产业界与学术界）

| 方面 | 产业实验室 | 学术研究组 |
|--------|--------------|-----------------|
| 算力/数据 | 充足；前沿规模 | 有限；倒逼巧思 |
| 自由度 | 受产品/战略约束 | 可自由选择问题 |
| 指导 | 资深研究者，反馈快 | 长期指导、教学 |
| 开放性 | 不一（部分闭源） | 通常公开发表 |

---

## 开放问题与研究空白（元层面）

- **算力鸿沟。** 前沿成果越来越需要产业级资源，压缩了学术界可开展的研究范围。
- **人才流失。** 教师和学生流向初创公司/产业界，重塑了学术界的研究能力。
- **闭源工作的可复现性。** 关键成果（Gemini Robotics、GPT-4o）无法被独立复现。
- **开放与封闭的张力。** 领域的开源传统与商业整合之间存在张力。
- **人才集中。** 少数研究组占据主导，引发对思想多样性的担忧。
- **数据护城河。** 机器人数据采集上的优势可能巩固现有领先者的地位（参见[优缺点与路线图](../06_robotics_and_embodied_ai/07_pros_cons_roadmaps.md)）。
- **实验室不稳定。** 频繁的重组（如 Meta 的 MSL 重组和 2025 年裁员）和创始人离职，使实验室的“风格”和代码维护更难预测。在基于某个代码库开展工作之前，请先确认它是否仍在维护。
- **初创公司“吸纳”。** 收购式招聘和授权交易（Covariant → Amazon）以及数亿到数十亿美元级别的初创公司融资（Physical Intelligence、Skild AI、AMI Labs），把资深研究者拉离了公开发表。

---

## 延伸阅读

- [BAIR Blog](https://bair.berkeley.edu/blog/) — 伯克利人工智能研究院
- [Meta AI Research](https://ai.meta.com/research/) — FAIR 论文发表
- [Google DeepMind](https://deepmind.google/research/) — 研究索引
- [Physical Intelligence](https://www.physicalintelligence.company/) — π₀ 与机器人基础模型
- [Stanford Vision & Learning Lab (SVL)](https://svl.stanford.edu/) — 视觉 + 机器人
- [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/) — VGGT、D4RT
- [World Labs](https://www.worldlabs.ai/) — Marble 与空间智能研究
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — 开放的人形机器人 VLA 发布（N1.5 → N1.7）
- [Newton physics engine](https://github.com/newton-physics/newton) — 由 Disney Research、Google DeepMind 和 NVIDIA 发起的 Linux 基金会项目
