# 3D Vision, Autonomous Driving, and Robotics Datasets

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [Robot Data and Teleoperation](../06_robotics_and_embodied_ai/09_robot_data_and_teleoperation.md) · [Simulation and Data](../07_autonomous_driving/04_simulation_and_data.md) · [Core Datasets](./00_core_datasets.md) · [Evaluation Metrics](./04_evaluation_metrics.md)

> **🆕 October 2026 update:**
> - Added **industrial-scale robot data**: AgiBot World (1M+ trajectories from 100 robots, IROS 2025) and the Galaxea Open-World Dataset (500+ h of mobile manipulation, Sep 2025).
> - Added **human egocentric manipulation data**: EgoDex (Apple, 829 h of Vision Pro video with 3D hand pose). Also added the trend of pretraining VLAs on large human-video corpora (e.g. GR00T N1.7 with 20K h of EgoScale).
> - Added **end-to-end driving benchmarks**: the Waymo Open **End-to-End Driving** dataset with its Rater Feedback Score (2025), **NAVSIM v2** pseudo-simulation (CoRL 2025), and NVIDIA's Physical AI AV dataset that accompanies Alpamayo.
> - Added **3D object data**: uCO3D (~170K turntable videos with per-video 3DGS), and SAM 3D (Nov 2025) with its **SA-3DAO** artist-object benchmark (Jun 2026).
> - Added a new subsection on **robot policy benchmarks and their saturation**: LIBERO (π0.5 96.85%) vs LIBERO-Plus robustness drops, RoboTwin 2.0, RoboArena, and BEHAVIOR-1K.

---

## Overview

Three-dimensional vision datasets and robotics datasets represent the frontier of embodied perception: the systems that must reason about the 3D structure of the physical world in order to navigate, manipulate, or drive within it. Unlike standard 2D image benchmarks where annotation cost scales with image count, 3D datasets require capturing or reconstructing volumetric geometry—through RGB-D sensors, LiDAR point clouds, multi-view stereo, or depth estimation—making collection orders of magnitude more expensive per scene.

The landscape divides into three use-case clusters. First, **indoor 3D scene understanding** (ScanNet, ShapeNet) targets reconstruction, segmentation, and object pose estimation in structured interior environments. These datasets are essential for AR/VR, household robots, and 3D-aware vision-language models. Second, **autonomous driving perception** datasets (KITTI, nuScenes, Waymo Open) provide synchronized multi-modal sensor streams (camera, LiDAR, radar, GPS/IMU) with precise 3D bounding box annotations, forming the backbone of industry-grade 3D object detection and tracking research. Third, **large-scale generative 3D** datasets (ShapeNet, Objaverse, Objaverse-XL) provide textured CAD models and photogrammetry scans enabling neural rendering, novel-view synthesis (NeRF, 3DGS), and 3D generation research.

A more recent and increasingly important cluster is **robot learning datasets**—teleoperated manipulation demonstrations collected across diverse robot platforms, environments, and tasks (Open X-Embodiment, DROID, Ego-Exo4D). These datasets are the training corpora for vision-language-action (VLA) models. Their defining challenge is *embodiment heterogeneity*: data collected on a Franka arm does not transfer trivially to a UR5 or a mobile manipulator. Datasets like Open X-Embodiment provide RLDS-standardized trajectories across 22 robot platforms to directly address this problem.

Between mid-2025 and 2026 the robot-data landscape shifted in three ways. First, **single-organisation industrial collection** overtook academic pooling in scale. AgiBot World alone matches OXE's trajectory count on a uniform, sensor-rich hardware fleet, and companies such as Galaxea release hundreds of hours of single-embodiment mobile-manipulation data. Second, **human egocentric video with 3D hand pose** became a first-class pretraining source. Examples are EgoDex, captured on Apple Vision Pro, and the 20K-hour EgoScale corpus used to pretrain NVIDIA's GR00T N1.7. A shared relative end-effector action space lets human and robot data be mixed. Third, **evaluation moved toward distributed real-world arenas and robustness suites** (RoboArena, LIBERO-Plus), because the standard simulated benchmarks saturated. In autonomous driving, the analogous shift is from 3D-box perception benchmarks to **end-to-end planning benchmarks** with human-preference or pseudo-closed-loop metrics (WOD-E2E, NAVSIM v2).

---

## ScanNet

**Citation:** [Dai2017] Dai, A. et al. (2017). ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. *CVPR 2017*.

ScanNet is an RGB-D video dataset containing 1,513 indoor scene scans across diverse environments (offices, apartments, bathrooms, kitchens, classrooms, libraries), totaling approximately 2.5 million RGB-D frames. Each scan provides: a color video, depth video, 3D mesh reconstruction (via BundleFusion), camera pose estimates, surface normals, semantic instance segmentation at the mesh level, and axis-aligned 3D bounding boxes.

**Key statistics:**
- 1,513 scans in 707 distinct spaces
- 2.5M RGB-D views
- Average scan: 150K mesh vertices, spatial extent ~5.5m × 5.1m × 2.4m
- 40 semantic category labels (20 for evaluation benchmarks)
- Annotation by 500+ crowd workers via Mechanical Turk

**Tasks supported:** 3D semantic segmentation, 3D instance segmentation, 3D object detection, novel-view synthesis (ScanNet is widely used for NeRF and 3DGS experiments), scene reconstruction quality evaluation.

**ScanNet++:** A 2023 extension with higher-resolution RGB-D captures and laser scanner ground-truth meshes for more precise reconstruction benchmarking.

---

## ShapeNet

**Citation:** [Chang2015] Chang, A.X. et al. (2015). ShapeNet: An Information-Rich 3D Model Repository. arXiv:1512.03012.

ShapeNet is a large-scale repository of annotated 3D CAD models indexed under the WordNet taxonomy. The full repository (as of 2015) indexed ~3 million models across 3,135 categories. The curated **ShapeNetCore** subset covers 55 common object categories with ~51,300 unique 3D models (split: 35,764 train / 5,133 val / 10,265 test). **ShapeNetPart** provides part-level segmentation annotations for 16 categories across 16,881 shapes.

**Content:** Models include consistent rigid alignments, bilateral symmetry planes, part annotations, and physical size estimates. Many models are sourced from online CAD repositories (TurboSquid, 3DWarehouse) and vary substantially in mesh quality.

**Primary uses:** Point cloud classification and segmentation (PointNet, DGCNN, PointTransformer), novel view synthesis, 3D shape generation (IM-NET, ShapeFlow), retrieval, and pretraining 3D encoders. ShapeNet's clean CAD geometry provides ideal training data for 3D deep learning, though the sim-to-real gap versus scanned real-world objects is significant.

---

## KITTI

**Citation:** [Geiger2012] Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. *CVPR 2012*.

KITTI is the foundational autonomous driving dataset, captured from a car instrumented with two pairs of stereo cameras (color + grayscale), a Velodyne HDL-64E 3D LiDAR, and a GPS/IMU unit. Total data: 6 hours of recording in Karlsruhe, Germany.

**Benchmark tasks and scales:**
- **Stereo / Optical Flow:** 194 training + 195 test pairs (1242×375 px)
- **3D Object Detection:** 7,481 training / 7,518 test frames; 80K annotated 3D bounding boxes for cars, pedestrians, cyclists
- **Visual Odometry/SLAM:** 22 stereo sequences, 39.2 km total
- **Road estimation, depth completion**

KITTI introduced the BEV (Bird's Eye View) and 3D IoU-based evaluation protocols that remain standard in LiDAR 3D detection. Despite its age, KITTI 3D detection is still reported in nearly all 3D perception papers as a backward-compatibility benchmark.

---

## nuScenes

**Citation:** [Caesar2020] Caesar, H. et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. *CVPR 2020*.

nuScenes is a full-surround autonomous driving dataset containing 1,000 driving scenes (20 seconds each) from Boston and Baxter, Singapore, collected with a full sensor suite: 6 cameras (360° coverage), 5 radars, 1 LiDAR (32-beam, 360°). Total: 1.4M annotated 3D bounding boxes across 23 classes with 8 attributes. Keyframe annotation at 2 Hz (~40 frames per 20s scene). Split: 700 train / 150 val / 150 test.

**Why nuScenes matters:** It was the first public autonomous driving dataset with camera + LiDAR + radar fusion at full 360°, enabling multi-modal 3D detection and tracking research. The nuScenes Detection Score (NDS) captures translation, scale, orientation, velocity, and attribute prediction simultaneously—a more comprehensive metric than KITTI's IoU-based AP alone.

**nuScenes-lidarseg:** Adds dense semantic segmentation labels to LiDAR point clouds. **nuPlan:** A planning-focused extension with logged planner trajectories.

---

## Waymo Open Dataset

**Citation:** [Sun2020] Sun, P. et al. (2020). Scalability in Perception for Autonomous Driving: Waymo Open Dataset. *CVPR 2020*.

The Waymo Open Dataset contains 1,150 twenty-second driving segments captured by Waymo self-driving vehicles (798 train / 202 val / 150 test), each equipped with 5 LiDARs and 5 cameras. Annotations: 12M 3D LiDAR labels, 1.2M 2D camera labels across vehicles, pedestrians, cyclists, and signs. Point cloud data: 177K–300K points per frame (dual-return LiDAR).

**Significance:** Waymo Open is widely considered the hardest public 3D detection benchmark due to sensor quality, annotation density, and environmental diversity (day/night, weather variation, urban/suburban). Its LiDAR quality far exceeds KITTI's 64-line scanner; nuScenes uses a 32-beam scanner. Waymo's sensor head represents near-production autonomous driving hardware.

**Extensions:** Waymo Open Dataset v2 (2023) adds camera-LiDAR correspondence annotations; the Waymo Open Motion Dataset provides future trajectory annotations for motion prediction.

**End-to-End Driving dataset (WOD-E2E, 2025):** This third Waymo dataset targets camera-only, vision-based end-to-end planning in long-tail scenarios. It pairs multi-camera images with high-level routing commands. As of the April 2025 release it has 2,037 training runs, a 479-run validation set with rater-feedback labels and scenario tags, and a 1,505-run test set. Instead of displacement error to a single logged trajectory, it introduces the **Rater Feedback Score (RFS)**. For rated frames, human raters score up to three candidate future trajectories on a 0–10 scale. A predicted trajectory is then scored by how it matches these rated alternatives, so multiple acceptable behaviours are rewarded rather than only the logged one. The scoring code is released (see [Evaluation Metrics](./04_evaluation_metrics.md)).

---

## End-to-End Driving Benchmarks: NAVSIM v2 and Physical AI AV

**Citation:** [Dauner2024] Dauner, D. et al. (2024). NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. *NeurIPS 2024 Datasets & Benchmarks*. [Cao2025] Cao, W., Hallgarten, M., Li, T., Dauner, D., et al. (2025). Pseudo-Simulation for Autonomous Driving. *CoRL 2025*.

**NAVSIM** built on OpenScene/nuPlan logs to sit between brittle open-loop L2 metrics and expensive closed-loop simulation. **NAVSIM v2** introduced *pseudo-simulation*. Real logs are augmented with synthetic observations near the planned trajectory, and a two-stage aggregation then approximates closed-loop behaviour. The authors report strong correlation with closed-loop simulation at about 6× less compute. v2 also introduced the **Extended PDM Score (EPDMS)**, the hard **navhard** split, and the official 2025 challenge (AGC 2025).

**NVIDIA Physical AI Autonomous Vehicles dataset (2025–2026):** NVIDIA released a large multi-camera driving dataset on Hugging Face with a developer kit (`physical_ai_av`). It is the training and evaluation corpus for the open **Alpamayo** reasoning VLA family. Alpamayo-R1 was released in 2025 and renamed Alpamayo 1 at CES 2026, followed by Alpamayo 1.5 in March 2026. It marks a shift toward open, industry-scale end-to-end driving data with reasoning annotations. See [Simulation and Data](../07_autonomous_driving/04_simulation_and_data.md).

---

## Objaverse / Objaverse-XL

**Citation:** [Deitke2023a] Deitke, M. et al. (2023). Objaverse: A Universe of Annotated 3D Objects. *CVPR 2023*. [Deitke2023b] Deitke, M. et al. (2023). Objaverse-XL: A Universe of 10M+ 3D Objects. *NeurIPS 2023*.

**Objaverse 1.0:** 800K+ annotated 3D models created by 100K+ artists on Sketchfab. Models include descriptive captions, tags, animations, and materials.

**Objaverse-XL:** Over 10 million 3D objects from diverse sources: manually designed models, photogrammetry scans of landmarks and everyday items, professional heritage artifact scans. Objaverse-XL is 12× larger than Objaverse 1.0.

**Impact:** Objaverse enabled Zero123 (Liu et al., 2023) which demonstrated large-scale novel-view synthesis pretraining, subsequently unlocking single-image 3D reconstruction (Zero123++, SyncDreamer, Wonder3D). Objaverse-XL's scale enabled training large 3D generative models and serves as the primary pretraining corpus for open-vocabulary 3D understanding models.

**Rendering:** The Objaverse team released multi-view rendered images (100M+ images from Objaverse-XL renders) specifically for pretraining novel-view synthesis and 3D generation models.

---

## uCO3D and SAM 3D / SA-3DAO (Real-World 3D Objects)

**Citation:** [Liu2025] Liu, X. et al. (2025). UnCommon Objects in 3D. arXiv:2501.07574 (Meta). [SAM3D2025] SAM 3D Team, Chen, X., Chu, F.-J., Gleize, P., Liang, K. J., Sax, A., et al. (2025). SAM 3D: 3Dfy Anything in Images. Meta Superintelligence Labs.

**uCO3D** (2025) succeeds CO3Dv2 with ~170,000 full 360° turntable videos of real objects. The objects span ~1,000 LVIS categories grouped into 50 super-categories. Each video carries object masks, camera poses, three types of point cloud, a per-video **3D Gaussian Splat reconstruction**, and long and short VLM captions. Its scale and LVIS-aligned taxonomy make it a real-capture counterweight to synthetic Objaverse assets for training feed-forward reconstruction and 3D generation.

**SAM 3D** (Nov 2025) has two parts. *SAM 3D Objects* reconstructs shape, texture and layout from a single natural image, and *SAM 3D Body* recovers human meshes. Following the SAM recipe, SAM 3D Objects combines progressive training with a **data engine with human feedback** on real, cluttered images. This reduces dependence on synthetic CAD-only training data, and the model outperforms prior 3D generators in human-preference tests on real-world objects and scenes. Meta later released the **SAM 3D Artist Objects (SA-3DAO)** evaluation set and a Hugging Face leaderboard (Jun 2026). It provides artist-created 3D ground truth for objects in real images, a harder target than rendered synthetic views.

---

## Open X-Embodiment

**Citation:** [Open X-Embodiment Collaboration, 2023] Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.

Open X-Embodiment (OXE) consolidates 60+ existing robot datasets from 34 research labs into a unified corpus using RLDS (Robot Learning Dataset Standard) format. Total: 1M+ real robot trajectories across 22 robot embodiments (single arm, bi-manual, quadruped, mobile manipulator), 527 distinct manipulation skills, from institutions including Google, Stanford, UC Berkeley, CMU, MIT, and others.

**Key statistics:**
- 22 robot embodiments
- 527 skills
- 1M+ trajectories
- Average trajectory: ~120 timesteps
- Control frequency: 3–10 Hz depending on robot

**RT-X models:** The OXE paper trained RT-1-X and RT-2-X on this corpus, demonstrating that cross-embodiment training improves generalization beyond single-embodiment datasets. This is the empirical foundation of generalist robot policies.

---

## DROID

**Citation:** [Khazatsky2024] Khazatsky, A. et al. (2024). DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. *RSS 2024*.

DROID (Distributed Robot Interaction Dataset) contains 76,000 teleoperated manipulation demonstration trajectories collected by 18 research institutions across North America, Asia, and Europe over 12 months. Collected using 18 Franka Panda robots across 564 unique real-world scenes and 86 distinct manipulation tasks.

**Diversity emphasis:** DROID was specifically designed to maximize environmental diversity (varied lighting, cluttered scenes, diverse objects) in contrast to laboratory-curated datasets. 3 natural language annotations per trajectory provided for 95% of successful episodes (75K episodes).

**Why it matters:** Training on DROID yields policies with higher performance and better generalization than equivalent-scale homogeneous datasets, validating the importance of in-the-wild diversity for robot learning. DROID has become one of the standard large-scale manipulation pretraining datasets for VLA models.

---

## Ego-Exo4D

**Citation:** [Grauman2023] Grauman, K. et al. (2024). Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives. *CVPR 2024*.

Ego-Exo4D extends Ego4D with synchronized egocentric and exocentric (third-person) camera captures of skilled human activities (sports, music, cooking, mechanics). Approximately 1,422 hours of video from 800+ participants at 13 universities. Provides multi-task benchmarks including: correspondence (ego↔exo), relation, proficiency estimation, and fine-grained action understanding.

**Unique contribution:** The ego-exo correspondence task—matching viewpoints of the same activity across first- and third-person perspectives—is entirely new and has direct applications in skill learning, imitation from demonstration video, and embodied AI.

---

## AgiBot World (Colosseo)

**Citation:** [Bu2025] Bu, Q., Cai, J., Chen, L., Cui, X., Ding, Y., Feng, S., et al. (2025). AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems. *IROS 2025*. arXiv:2503.06669.

AgiBot World, from AgiBot and OpenDriveLab, was released in two stages: Alpha in Dec 2024 and Beta in Mar 2025. It contains **1M+ trajectories collected on a fleet of 100 robots** in **100+ 1:1 replicated real-life scenarios** across five target domains. The hardware includes visuo-tactile sensors, 6-DoF dexterous hands and mobile dual-arm platforms. Unlike OXE, which pools heterogeneous lab data, it is a *single-organisation, standardised-protocol* collection. The accompanying **GO-1** generalist policy (open-sourced Sep 2025) and a follow-up study, *Is Diversity All You Need for Scalable Robotic Manipulation?* (2025), use it to study data-diversity scaling.

---

## Galaxea Open-World Dataset

Released in Sep 2025 by Galaxea AI alongside the G0 VLA (RLDS and LeRobot formats). It contains **500+ hours** of real-world *mobile* manipulation data collected with **one uniform embodiment** (a Galaxea R1-series mobile dual-arm robot) in residential, kitchen, retail and office settings, with fine-grained subtask language annotations. Its single-embodiment, long-horizon, in-the-wild design complements DROID's single-arm tabletop focus.

---

## EgoDex (Human Egocentric Dexterous Manipulation)

**Citation:** [Hoque2025] Hoque, R., Huang, P., Yoon, D. J., Sivapurapu, M., & Zhang, J. (2025). EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video. arXiv:2505.11709 (Apple).

EgoDex contains **829 hours** of 30 Hz, 1080p egocentric video recorded with ARKit on Apple Vision Pro. Each frame is paired with tracked **3D poses of the head, upper body and hands**, plus natural-language annotations. All of it is active tabletop manipulation across **194 tasks**. The split is ~725 h train, ~7 h test and ~97 h of additional data. Because pose is captured natively by the headset rather than estimated afterwards, EgoDex supports direct training of human-hand trajectory prediction and of human-to-robot transfer. It belongs to a broader 2025–2026 trend of scaling *human* data for robot pretraining. NVIDIA's GR00T N1.7, for example, adds **20K hours of EgoScale human video** to its pretraining mix through a relative end-effector action space shared between humans and robots.

---

## Robot Policy Benchmarks and Saturation (2025–2026)

Simulated benchmarks remain the cheapest way to compare VLAs, but the most popular one is now saturated.

- **LIBERO** (2023): four suites (Spatial, Object, Goal, Long). Fine-tuned VLAs are at ceiling. For example, π0.5 fine-tuned in openpi reports 98.8 / 98.2 / 98.0 / 92.4 (**96.85% average**).
- **LIBERO-Plus** [Fei2025] (Oct 2025) re-tests LIBERO policies with 10,030 tasks along seven perturbation dimensions: object layout, camera viewpoint, robot initial state, language rewrites, lighting, background texture and sensor noise. Success rates fall **from ~95% to below 30%** under modest camera or initial-state shifts, and models largely *ignore the language instruction*. This is strong evidence that LIBERO scores measure memorisation rather than generalisation.
- **RoboTwin 2.0** (ICML 2026): a scalable dual-arm simulation benchmark and data generator with strong domain randomisation. It hosted the CVPR 2025 MEIS challenge and was extended by RMBench, a memory-dependent manipulation suite, in Mar 2026.
- **BEHAVIOR-1K** [Li2024]: 1,000 everyday household activities, chosen from time-use surveys, in the OmniGibson simulator. It is used as a long-horizon mobile-manipulation challenge benchmark (2025 BEHAVIOR Challenge).
- **RoboArena** (2025): a *distributed real-world* evaluation in which policies are served remotely and evaluated on the DROID platform at multiple institutions. It is a response to the irreproducibility of single-lab real-robot results.

---

## Key Papers / Datasets

| Name | Authors/Org | Year | Venue | Key Stats / Contribution |
|---|---|---|---|---|
| ScanNet | Dai et al. (Stanford/TUM) | 2017 | CVPR | 1,513 scans, 2.5M RGB-D frames, semantic instance annotations |
| ShapeNet (Core) | Chang et al. (Stanford/Princeton) | 2015 | arXiv | 51.3K models, 55 categories; canonical 3D shape dataset |
| KITTI | Geiger et al. (KIT) | 2012 | CVPR | 6h driving, 80K 3D boxes; pioneered LiDAR detection benchmarks |
| nuScenes | Caesar et al. (nuTonomy) | 2020 | CVPR | 1K scenes, 6 cam+LiDAR+radar, 1.4M 3D boxes, 23 classes |
| Waymo Open | Sun et al. (Waymo) | 2020 | CVPR | 1,150 segments, 12M 3D labels; highest-quality public AV dataset |
| Objaverse | Deitke et al. (AllenAI) | 2023 | CVPR | 800K+ 3D models, artist-created with captions |
| Objaverse-XL | Deitke et al. (AllenAI) | 2023 | NeurIPS | 10M+ 3D objects; enables large-scale 3D generation pretraining |
| Open X-Embodiment | OXE Collaboration | 2023 | arXiv | 1M+ trajectories, 22 robots, 60+ datasets, RLDS-standardized |
| DROID | Khazatsky et al. (Stanford/Berkeley) | 2024 | RSS | 76K trajectories, 18 labs, 564 scenes, 86 tasks, in-the-wild |
| Ego-Exo4D | Grauman et al. (Meta AI) | 2024 | CVPR | 1,422h ego+exo, skilled activities, ego-exo correspondence task |
| NAVSIM v2 (pseudo-simulation) | Cao, Hallgarten, …, Geiger (Tübingen/NVIDIA/Bosch/OpenDriveLab) | 2025 | CoRL | Pseudo-closed-loop AV evaluation; EPDMS; navhard split; ~6× cheaper than closed-loop |
| Waymo Open E2E Driving (WOD-E2E) | Waymo | 2025 | dataset release | 2,037 train / 479 val / 1,505 test runs; camera + routing command; Rater Feedback Score |
| uCO3D | Liu et al. (Meta) | 2025 | arXiv | ~170K 360° object videos, ~1K LVIS categories, per-video 3DGS + captions |
| SAM 3D / SA-3DAO | SAM 3D Team (Meta) | 2025/2026 | Meta tech report | Single-image 3D with human-feedback data engine; artist-object 3D benchmark + leaderboard |
| AgiBot World | Bu et al. (AgiBot / OpenDriveLab) | 2025 | IROS | 1M+ trajectories, 100 robots, 100+ replicated scenes; GO-1 policy |
| Galaxea Open-World | Galaxea AI | 2025 | dataset release | 500+ h single-embodiment mobile manipulation; subtask annotations |
| EgoDex | Hoque et al. (Apple) | 2025 | arXiv | 829h Vision Pro egocentric video, 3D head/body/hand pose, 194 tasks |
| LIBERO-Plus | Fei et al. (Fudan et al.) | 2025 | arXiv | 10,030 perturbed LIBERO tasks, 7 dimensions; exposes VLA fragility |
| RoboTwin 2.0 | RoboTwin Platform team | 2026 | ICML | Dual-arm sim benchmark + data generator with domain randomisation |

---

## Benchmark Comparison (3D Object Detection)

| Dataset | Sensor | 3D Boxes | Classes | Eval Metric | Difficulty Level |
|---|---|---|---|---|---|
| KITTI | 64-line LiDAR + stereo | 80K | 3 (car, ped, cyc) | AP@IoU3D 0.7/0.5 | Medium |
| nuScenes | 32-line LiDAR + 6 cam + 5 radar | 1.4M | 23 | NDS, mAP | High |
| Waymo Open | 5 LiDAR (high-res) + 5 cam | 12M 3D | 4 | mAPH | Very High |

**End-to-end planning and policy benchmarks (2025–2026)**

| Benchmark | Input | Scale | Primary Metric | Status |
|---|---|---|---|---|
| NAVSIM v2 (navhard) | Multi-cam + LiDAR logs (OpenScene/nuPlan) | Curated hard scenarios | EPDMS (pseudo-closed-loop) | Active (AGC 2025 challenge) |
| WOD-E2E | Multi-camera + routing command | ~4K runs (train/val/test) | Rater Feedback Score (0–10 rated trajectories) | Active (Waymo challenge) |
| LIBERO | Sim, single arm | 4 suites × 10 tasks | Success rate | Saturated (~97–99% per suite) |
| LIBERO-Plus | Sim, perturbed LIBERO | 10,030 tasks | Success rate under perturbation | Large drops (<30% under camera/init shifts) |
| RoboArena | Real DROID platform, distributed | Multi-institution | Aggregated real-world evaluations of remotely served policies | Active |

---

## Pros & Cons

| Dataset | Strengths | Limitations |
|---|---|---|
| ScanNet | High-quality 3D reconstructions; semantic + instance labels; widely used for NeRF/3DGS eval | Indoor only; consumer-grade RGB-D; annotation noise at boundaries |
| KITTI | Foundational; small enough for rapid iteration; supports stereo, flow, detection, SLAM | Single-city (Karlsruhe); 64-line LiDAR now outdated; small annotation count |
| nuScenes | Full sensor suite; multi-class with attributes; planning-focused extensions | 32-line LiDAR lower quality; 20s scenes are short |
| Waymo Open | Highest-quality sensors; diverse environments; large-scale annotations | Not freely downloadable without registration; proprietary sensor specs |
| Objaverse-XL | Unprecedented 3D scale; diverse sources; enables generative 3D | Variable model quality; significant noise from artist assets; limited semantic annotations |
| Open X-Embodiment | Cross-embodiment; standardized format; directly enables generalist policies | Quality variance across contributed datasets; action spaces differ |
| DROID | High diversity, in-the-wild; strong policy training results; language annotations | Single robot platform (Franka); predominantly tabletop manipulation |
| AgiBot World | OXE-scale trajectories with uniform hardware/protocol; dexterous + tactile sensing | Single vendor's embodiments; replicated (staged) scenes rather than truly in-the-wild homes |
| EgoDex | Native 3D hand pose at scale; cheap to scale (no robot) | Human-to-robot embodiment gap; tabletop only; requires retargeting |
| WOD-E2E | Long-tail scenarios; human-preference metric tolerates multimodal futures | Rated subset is small; camera-only; rater subjectivity |
| NAVSIM v2 | Closed-loop-correlated scores at open-loop cost | Synthetic observations may carry rendering artefacts; non-reactive agents |
| LIBERO | Cheap, reproducible VLA comparisons | Saturated; LIBERO-Plus shows scores reflect memorisation of fixed viewpoints/initial states |

---

## Open Problems & Research Gaps

- **Closing the sim-to-real gap in 3D:** ShapeNet and Objaverse provide synthetic 3D data; ScanNet provides real scans; but the domain gap between synthetic renders and real RGB-D data remains a major bottleneck for 3D detection and reconstruction.
- **Multi-embodiment robot learning:** DROID and OXE show cross-embodiment benefits, but bridging fundamentally different kinematic structures (arms, mobile bases, dexterous hands) remains unsolved without explicit embodiment-conditional representations.
- **Standardized robot dataset evaluation:** Unlike vision benchmarks, robot manipulation datasets lack consensus evaluation protocols. Each paper uses different environments, objects, and success criteria, making comparison nearly impossible. The 2025–2026 responses each solve only part of the problem. Distributed real-world arenas (RoboArena) improve reproducibility but are slow and costly. Robustness suites (LIBERO-Plus) expose fragility, but only in simulation. Saturated sim benchmarks such as LIBERO (~97%) should no longer be used as headline evidence.
- **Human-to-robot data transfer:** Human egocentric corpora (EgoDex 829 h; EgoScale 20K h in GR00T N1.7) are now one to two orders of magnitude larger than open teleoperation data. How much of this transfers, and through which action representation (relative end-effector, keypoints, latent actions), is not yet established by controlled scaling studies.
- **Human-preference metrics for planning:** WOD-E2E's rater-feedback labels and NAVSIM's pseudo-simulation both try to escape the single-logged-trajectory assumption. Their agreement with each other, and with true closed-loop safety, is unknown.
- **Outdoor 3D scene understanding at scale:** Waymo Open is large by AV standards but tiny compared to vision datasets. Scaling to millions of driving scenes with full annotation is a major data collection challenge.
- **Temporal 4D understanding:** Most 3D datasets are static (ScanNet, ShapeNet) or provide short sequences. Understanding how scenes change over time—object state changes, human motion, deformable objects—is severely underexplored.
- **Long-horizon manipulation datasets:** DROID and OXE contain short task demonstrations (typically <2 minutes). Datasets capturing multi-step, multi-day household tasks are essentially non-existent at scale.
- **World model pretraining data:** Video prediction and world model research (Genie, UniSim) requires large-scale diverse video with rich 3D structure; no dataset is purpose-built for this at adequate scale. Interactive world models such as Genie 3 (Google DeepMind, Aug 2025) are trained on proprietary corpora. Open action-labelled video at comparable scale does not exist, and the community instead stitches together driving logs (WOD-E2E, Physical AI AV), robot data (OXE, AgiBot World) and human egocentric video.

---

## Further Reading

- [ScanNet project page](http://www.scan-net.org/)
- [Waymo Open Dataset documentation](https://waymo.com/open/about/)
- [Open X-Embodiment project](https://robotics-transformer-x.github.io/)
- [DROID dataset](https://droid-dataset.github.io/)
- [Objaverse-XL paper — arXiv:2307.05663](https://arxiv.org/abs/2307.05663)
- [nuScenes dataset and leaderboard](https://www.nuscenes.org/)
- [Waymo Open Dataset code, E2E driving protos and Rater Feedback Score](https://github.com/waymo-research/waymo-open-dataset)
- [NAVSIM v2 / Pseudo-Simulation for Autonomous Driving (arXiv:2506.04218)](https://github.com/autonomousvision/navsim)
- [NVIDIA Physical AI AV dataset devkit](https://github.com/NVlabs/physical_ai_av) · [Alpamayo](https://github.com/NVlabs/alpamayo)
- [AgiBot World Colosseo (arXiv:2503.06669)](https://github.com/OpenDriveLab/AgiBot-World)
- [Galaxea Open-World Dataset and G0/G0.5 VLA](https://github.com/OpenGalaxea/GalaxeaVLA)
- [EgoDex (arXiv:2505.11709)](https://github.com/apple/ml-egodex)
- [uCO3D (arXiv:2501.07574)](https://github.com/facebookresearch/uco3d)
- [SAM 3D Objects and SA-3DAO](https://github.com/facebookresearch/sam-3d-objects)
- [LIBERO-Plus (arXiv:2510.13626)](https://github.com/sylvestf/LIBERO-plus) · [RoboTwin 2.0 (arXiv:2506.18088)](https://github.com/RoboTwin-Platform/RoboTwin) · [RoboArena](https://github.com/robo-arena/roboarena) · [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
