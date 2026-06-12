# 3D Vision, Autonomous Driving, and Robotics Datasets

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [Robot Data and Teleoperation](../06_robotics_and_embodied_ai/09_robot_data_and_teleoperation.md) · [Simulation and Data](../07_autonomous_driving/04_simulation_and_data.md) · [Core Datasets](./00_core_datasets.md) · [Evaluation Metrics](./04_evaluation_metrics.md)

---

## Overview

Three-dimensional vision datasets and robotics datasets represent the frontier of embodied perception: the systems that must reason about the 3D structure of the physical world in order to navigate, manipulate, or drive within it. Unlike standard 2D image benchmarks where annotation cost scales with image count, 3D datasets require capturing or reconstructing volumetric geometry—through RGB-D sensors, LiDAR point clouds, multi-view stereo, or depth estimation—making collection orders of magnitude more expensive per scene.

The landscape divides into three use-case clusters. First, **indoor 3D scene understanding** (ScanNet, ShapeNet) targets reconstruction, segmentation, and object pose estimation in structured interior environments. These datasets are essential for AR/VR, household robots, and 3D-aware vision-language models. Second, **autonomous driving perception** datasets (KITTI, nuScenes, Waymo Open) provide synchronized multi-modal sensor streams (camera, LiDAR, radar, GPS/IMU) with precise 3D bounding box annotations, forming the backbone of industry-grade 3D object detection and tracking research. Third, **large-scale generative 3D** datasets (ShapeNet, Objaverse, Objaverse-XL) provide textured CAD models and photogrammetry scans enabling neural rendering, novel-view synthesis (NeRF, 3DGS), and 3D generation research.

A more recent and increasingly important cluster is **robot learning datasets**—teleoperated manipulation demonstrations collected across diverse robot platforms, environments, and tasks (Open X-Embodiment, DROID, Ego-Exo4D). These datasets are the training corpora for vision-language-action (VLA) models. Their defining challenge is *embodiment heterogeneity*: data collected on a Franka arm does not transfer trivially to a UR5 or a mobile manipulator. Datasets like Open X-Embodiment provide RLDS-standardized trajectories across 22 robot platforms to directly address this problem.

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

---

## Objaverse / Objaverse-XL

**Citation:** [Deitke2023a] Deitke, M. et al. (2023). Objaverse: A Universe of Annotated 3D Objects. *CVPR 2023*. [Deitke2023b] Deitke, M. et al. (2023). Objaverse-XL: A Universe of 10M+ 3D Objects. *NeurIPS 2023*.

**Objaverse 1.0:** 800K+ annotated 3D models created by 100K+ artists on Sketchfab. Models include descriptive captions, tags, animations, and materials.

**Objaverse-XL:** Over 10 million 3D objects from diverse sources: manually designed models, photogrammetry scans of landmarks and everyday items, professional heritage artifact scans. Objaverse-XL is 12× larger than Objaverse 1.0.

**Impact:** Objaverse enabled Zero123 (Liu et al., 2023) which demonstrated large-scale novel-view synthesis pretraining, subsequently unlocking single-image 3D reconstruction (Zero123++, SyncDreamer, Wonder3D). Objaverse-XL's scale enabled training large 3D generative models and serves as the primary pretraining corpus for open-vocabulary 3D understanding models.

**Rendering:** The Objaverse team released multi-view rendered images (100M+ images from Objaverse-XL renders) specifically for pretraining novel-view synthesis and 3D generation models.

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

---

## Benchmark Comparison (3D Object Detection)

| Dataset | Sensor | 3D Boxes | Classes | Eval Metric | Difficulty Level |
|---|---|---|---|---|---|
| KITTI | 64-line LiDAR + stereo | 80K | 3 (car, ped, cyc) | AP@IoU3D 0.7/0.5 | Medium |
| nuScenes | 32-line LiDAR + 6 cam + 5 radar | 1.4M | 23 | NDS, mAP | High |
| Waymo Open | 5 LiDAR (high-res) + 5 cam | 12M 3D | 4 | mAPH | Very High |

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

---

## Open Problems & Research Gaps

- **Closing the sim-to-real gap in 3D:** ShapeNet and Objaverse provide synthetic 3D data; ScanNet provides real scans; but the domain gap between synthetic renders and real RGB-D data remains a major bottleneck for 3D detection and reconstruction.
- **Multi-embodiment robot learning:** DROID and OXE show cross-embodiment benefits, but bridging fundamentally different kinematic structures (arms, mobile bases, dexterous hands) remains unsolved without explicit embodiment-conditional representations.
- **Standardized robot dataset evaluation:** Unlike vision benchmarks, robot manipulation datasets lack consensus evaluation protocols. Each paper uses different environments, objects, and success criteria, making comparison nearly impossible.
- **Outdoor 3D scene understanding at scale:** Waymo Open is large by AV standards but tiny compared to vision datasets. Scaling to millions of driving scenes with full annotation is a major data collection challenge.
- **Temporal 4D understanding:** Most 3D datasets are static (ScanNet, ShapeNet) or provide short sequences. Understanding how scenes change over time—object state changes, human motion, deformable objects—is severely underexplored.
- **Long-horizon manipulation datasets:** DROID and OXE contain short task demonstrations (typically <2 minutes). Datasets capturing multi-step, multi-day household tasks are essentially non-existent at scale.
- **World model pretraining data:** Video prediction and world model research (Genie, UniSim) requires large-scale diverse video with rich 3D structure; no dataset is purpose-built for this at adequate scale.

---

## Further Reading

- [ScanNet project page](http://www.scan-net.org/)
- [Waymo Open Dataset documentation](https://waymo.com/open/about/)
- [Open X-Embodiment project](https://robotics-transformer-x.github.io/)
- [DROID dataset](https://droid-dataset.github.io/)
- [Objaverse-XL paper — arXiv:2307.05663](https://arxiv.org/abs/2307.05663)
- [nuScenes dataset and leaderboard](https://www.nuscenes.org/)
