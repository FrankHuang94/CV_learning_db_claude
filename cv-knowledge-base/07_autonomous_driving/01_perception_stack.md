# Autonomous Driving Perception Stack: BEV Perception, Multi-Camera Fusion, and 3D Detection

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [AV Stack Overview](./00_overview.md) · [Occupancy Prediction](./02_occupancy_prediction.md) · [End-to-End Driving](./03_end_to_end_driving.md) · [Simulation & Data](./04_simulation_and_data.md)

> **🆕 October 2026 update:**
> - New section **"2025–2026: Foundation Backbones, Sparse Queries and Perception inside Driving Foundation Models"**: DINOv3/SAM 3 as backbones and auto-labelers, Point Transformer V3 / Sonata for LiDAR, sparse long-range detectors, and EMMA-style detection as text generation.
> - Sensor debate updated: Waymo's 6th-gen driver (13 cameras, 4 LiDARs, 6 radars), Tesla and XPeng vision-only robotaxis in testing or service, and Waymo's generative LiDAR simulation.
> - Corrected Waymo sensor-suite and mileage figures; new Key Papers, Open Problems and Further Reading.

---

## Overview

The perception stack of a modern autonomous vehicle transforms raw multi-modal sensor inputs—primarily surround-view cameras, spinning LiDARs, and corner radars—into structured representations of the environment suitable for downstream prediction and planning. For over a decade, the dominant paradigm was *perspective-space* object detection: detect 2D bounding boxes in each camera image, lift them to 3D via geometric assumptions, and associate LiDAR points as depth anchors [Chen2016MV3D]. This approach suffered from fundamental limitations: the perspective projection discards metric depth information, multi-camera associations require explicit calibration-aware fusion, and the resulting 3D estimates in ego-frame coordinates are noisy.

The shift to **Bird's-Eye View (BEV)** representations, catalyzed by Lift-Splat-Shoot [Philion2020] in 2020 and matured through BEVFormer [Li2022] and BEVFusion [Liu2022], has largely resolved the multi-camera fusion problem by constructing a unified top-down spatial feature map in a canonical ego-relative coordinate frame. BEV representations are metric by construction—a voxel at $(x, y, z)$ in the feature volume corresponds to a real-world 3D location—making them directly compatible with map coordinates and planning algorithms. Furthermore, the BEV space is a natural fusion target for heterogeneous sensors: camera-derived semantic features and LiDAR geometry can be blended in the same tensor without sensor-specific post-processing.

Temporal reasoning within the BEV paradigm has emerged as a second critical axis of improvement. BEVFormer [Li2022] introduced deformable attention over aligned historical BEV grids, allowing the network to implicitly estimate velocity, resolve short-term occlusions, and accumulate evidence for distant or low-point-count objects. The effective detection range of a camera-only system roughly doubles when leveraging 500 ms of BEV history under typical highway conditions. This temporal fusion mechanism is analogous to classical probabilistic occupancy grids but operates on learned feature representations rather than hand-designed sensor models.

---

## Lift-Splat-Shoot (LSS): Foundational Camera-to-BEV Projection

LSS [Philion2020] introduced the first fully differentiable pipeline for projecting multi-camera image features into BEV without requiring explicit depth supervision. The architecture proceeds in three stages:

**Lift**: For each pixel $(u, v)$ in camera $c$, a depth distribution $\hat{\mathbf{d}} \in \mathbb{R}^D$ over $D$ discrete depth bins is predicted by a shared backbone. The pixel feature $\mathbf{f}_{u,v}$ is broadcast along the depth ray to produce a frustum point cloud:

$$\mathbf{F}_c = \left\{ \mathbf{f}_{u,v} \cdot \hat{d}_k \;\middle|\; u, v \in \text{pixels},\; k \in [D] \right\}$$

**Splat**: Each point in the frustum is projected to 3D via camera intrinsics $K$ and extrinsics $T_{c \to \text{ego}}$, then voxel-pooled into a BEV feature grid using sum-pooling (later upgraded to efficient cumsum-based pillar pooling in BEVDepth [Li2022BEVDepth]).

**Shoot**: The BEV feature map is consumed by a task-specific head (segmentation, detection). The name references the intended application of shooting planning rays through the map.

LSS established that **implicit depth distributions** can be learned purely from 3D detection supervision without LiDAR depth labels. Its key limitation is imprecision in the depth estimate: predicted depth modes are diffuse, causing feature "smearing" in BEV that degrades metric localization. BEVDepth [Li2022BEVDepth] addressed this by adding explicit depth supervision from projected LiDAR points as an auxiliary loss, improving nuScenes NDS by approximately 4–6 points over LSS baselines.

---

## BEVFormer: Spatial-Temporal Transformers for BEV Perception

BEVFormer [Li2022] (ECCV 2022, arXiv:2203.17270) introduced two novel attention mechanisms for learning BEV representations from multi-camera images:

**Spatial Cross-Attention (SCA)**: A set of learnable BEV queries $Q \in \mathbb{R}^{H \times W \times C}$ are defined on a regular grid. Each query samples features from all camera images at 3D reference points obtained by projecting the query's world coordinate through each camera's projection matrix. Deformable attention [Zhu2020] is used so each query attends to a small set of learned offsets around these reference points, enabling efficient multi-scale processing.

**Temporal Self-Attention (TSA)**: The current BEV query is fused with an aligned version of the previous timestep's BEV feature $B_{t-1}$ by warping $B_{t-1}$ using the ego-motion transformation $\Delta T_{t-1 \to t}$:

$$B_t^{\text{tmp}} = \text{Warp}(B_{t-1}, \Delta T_{t-1 \to t})$$

$$B_t = \text{SelfAttn}(Q_t, [B_t^{\text{spa}}, B_t^{\text{tmp}}])$$

This recurrent BEV propagation allows implicit velocity estimation and occlusion recovery without explicit temporal modeling modules.

**Key results**: BEVFormer achieves **56.9% NDS** and **48.1% mAP** on the nuScenes test set (camera-only), 9.0 NDS points above the previous best camera-only method, and on par with many LiDAR-only baselines of the era. BEVFormer v2 [Yang2023BEVFormerV2] extended the architecture with perspective 3D detection auxiliary tasks, further improving performance.

---

## BEVFusion: Unified LiDAR-Camera BEV Fusion

BEVFusion [Liu2022] (MIT, ICRA 2023, arXiv:2205.13542) argued that the key challenge in LiDAR-camera fusion is **feature-level alignment in a unified representation space** rather than result-level fusion (late fusion). The approach:

1. **Camera branch**: Projects multi-camera image features to BEV via LSS-derived voxel lifting.
2. **LiDAR branch**: Processes point clouds with a voxel-based encoder (VoxelNet/PointPillars backbone).
3. **Fusion**: Concatenates camera-BEV and LiDAR-BEV feature maps in the same grid, then applies a convolutional fusion head for joint detection and map segmentation.

The critical insight is that LiDAR-BEV and camera-BEV features are **aligned by construction** in the same metric space, avoiding the geometric inconsistencies of early fusion (raw sensor alignment) or late fusion (independent 3D box merging). The authors report:
- **73.4% NDS on nuScenes validation**
- **+1.3% mAP and NDS over TransFusion** (previous SOTA) on the test split
- **13.6% higher mIoU** for BEV map segmentation
- **1.9× reduction in MACs** vs. TransFusion at comparable accuracy

A concurrent work from Beijing Institute of Technology also named BEVFusion [Liang2022] proposes a similar idea; the MIT version is the more widely cited variant.

---

## LiDAR-Camera Fusion vs. Camera-Only: The Tesla Debate

The production autonomy industry is divided along the sensor axis:

**Camera-only (Tesla)**: Tesla argues that human drivers navigate solely with cameras, and therefore cameras with sufficient scale of training data are sufficient. FSD v12 processes eight cameras through a large transformer backbone, with no LiDAR at any stage. The system has been trained on **10 million video clips** [Tesla2024] and demonstrates emergent generalization. The critical engineering advantage is cost: a production LiDAR adds $200–$2,000+ to BOM. The key scientific weakness is **metric depth ambiguity**: any monocular depth estimate has an inherent scale ambiguity that can only be resolved by multi-frame temporal fusion or geometry priors, introducing failure modes in rapid depth-change scenarios (e.g., cut-ins, sudden stops).

**LiDAR-primary (Waymo)**: Waymo's 5th-generation driver (Jaguar I-PACE) carries 5 LiDARs, 29 cameras and 6 radars. The 6th-generation driver (announced 2024, on Zeekr and Hyundai Ioniq 5 platforms) reduces this to 13 cameras, 4 LiDARs and 6 radars to cut cost while keeping overlapping redundancy. LiDAR provides unambiguous metric 3D coordinates and 360° coverage at 10 Hz; FMCW LiDARs can add per-point radial velocity. Waymo's public safety data grew from 7.1 million rider-only miles [Waymo2023Safety] to ~96 million by June 2025. In 2026 Waymo operated about 500k paid rides per week. The historic disadvantage, LiDAR cost and mechanical reliability, has eroded as automotive LiDAR prices fall, especially with Chinese suppliers such as Hesai and RoboSense.

**2026 status of the debate.** The question has shifted from *"can cameras perceive well enough?"* to *"can a camera-only stack reach driverless reliability at fleet scale?"*. Tesla began unsupervised robotaxi rides in Austin in January 2026, and XPeng's VLA 2.0 robotaxi (no LiDAR, no HD map) entered public-road testing in March 2026. Waymo, Zoox, Baidu, Pony.ai and WeRide all keep LiDAR. No camera-only operator has yet published driverless safety data comparable to Waymo's.

The benchmark evidence on nuScenes clearly shows LiDAR-camera fusion outperforming camera-only by **15–20 NDS points** at comparable architectural complexity, suggesting that metric depth is genuinely useful signal for perception. However, this gap may narrow as camera depth estimation improves with larger training datasets.

---

## 3D Object Detection: PointPillars and CenterPoint

### PointPillars [Lang2019]

PointPillars (CVPR 2019) organizes 3D point clouds into vertical *pillars* (columns with infinite height along $z$) rather than 3D voxels, reducing the 3D convolution to 2D operations:

$$\text{Pillar feature} = \text{PointNet}(\{p_i - \bar{p} \mid p_i \in \text{pillar}\})$$

The pillar pseudo-image is processed by a 2D backbone (similar to SSD/SECOND) for detection. PointPillars achieves real-time performance (~62 Hz on a single GPU) with reasonable accuracy. On the Waymo Open Dataset, PointPillars achieves 63.3 / 62.7 (mAP / mAPH) at Level 1, 55.2 / 54.7 at Level 2. It remains the standard baseline and is deployed in many production systems as the primary detection backbone.

### CenterPoint [Yin2021]

CenterPoint (CVPR 2021, arXiv:2006.11275) replaced anchor-based bounding box regression with a **center heatmap** paradigm inspired by CenterNet. The detector:

1. Encodes point clouds via VoxelNet or PointPillars backbone to a BEV feature map.
2. Detects object centers as a Gaussian heatmap: $\hat{Y} \in \mathbb{R}^{H \times W \times K}$ where $K$ is the number of classes.
3. Regresses offset, height, dimensions, and rotation from the detected centers.
4. Propagates center tracks via a simple velocity-based Kalman filter (CenterPoint-two-stage adds a second-stage refinement on cropped features).

CenterPoint achieves **71.9 mAPH on Waymo** (single model, two-stage), nearly **doubling** the accuracy of PointPillars at Level 2 (66.4 mAPH for pedestrians + vehicles combined). It ranked first among LiDAR-only submissions in multiple Waymo challenges. CenterPoint is now the de facto standard LiDAR detection head, used as the LiDAR branch in BEVFusion and many other multi-modal architectures.

---

## Temporal Fusion and Long-Range Perception

Beyond BEVFormer's recurrent BEV propagation, several strategies have been explored for temporal aggregation:

- **BEV feature stacking**: Concatenate multiple warped BEV frames [BEVDet4D], simple but effective.
- **Transformer temporal attention**: Cross-attend between current and past BEV frames with learned alignment, used in DETR3D and PETRv2.
- **Occupancy flow**: Predict per-voxel velocity vectors to warp future occupancy predictions, enabling joint spatial-temporal forecasting as in UniAD [Hu2023UniAD].

Long-range perception (>100 m camera detection) requires careful handling of perspective compression and quantization artifacts. SparseBEV and StreamPETR demonstrate that sparse query-based methods scale better to longer range than dense BEV grid methods, which require prohibitively fine spatial resolution.

---

## 2025–2026: Foundation Backbones, Sparse Queries and Perception inside Driving Foundation Models

**Vision foundation backbones.** Self-supervised backbones now dominate initialisation. DINOv2 and **DINOv3** (Meta, August 2025; a 7B-parameter teacher with Gram anchoring for dense features, distilled into ViT/ConvNeXt variants) give strong frozen dense features for depth, segmentation and BEV lifting. Driving-specific variants are typically obtained by continued pre-training on fleet video. Promptable open-vocabulary segmenters (**SAM 3**, Meta, November 2025, segmenting all instances of a noun-phrase concept) are mainly used **offline**, as auto-labelers for long-tail categories (debris, construction equipment, animals), not on the vehicle.

**LiDAR backbones.** Point Transformer V3 [Wu2024PTv3] (CVPR 2024) trades exact neighbourhood search for serialised (space-filling-curve) point ordering. It reaches strong results on nuScenes and Waymo segmentation and detection at much higher throughput. **Sonata** [Wu2025Sonata] (CVPR 2025) provides self-supervised point-cloud pre-training on top of PTv3, offering a "DINO for point clouds" for label-efficient LiDAR perception.

**Sparse and long-range camera detection.** The dense-BEV paradigm is giving way to **fully sparse query** detectors (SparseBEV, Sparse4D v3, StreamPETR), which avoid the $O(HW)$ BEV grid. Far3D [Jiang2024Far3D] (AAAI 2024) extends camera detection to ~150 m on Argoverse 2 by generating 3D queries from 2D detections and depth priors. Sparse queries also integrate naturally with E2E planners (SparseDrive, DiffusionDrive).

**Perception as a by-product of driving foundation models.** Waymo's EMMA [Hwang2024EMMA] performs camera-only 3D detection by *generating box coordinates as text* from a Gemini-based MLLM, and is competitive on the Waymo Open Dataset. Co-training on planning, detection and road-graph estimation improves all three. Production VLA and world-action stacks (XPeng VLA 2.0, Huawei ADS 4, Tesla FSD v14) no longer expose a classical box list as the primary interface; explicit perception outputs survive mainly as auxiliary heads and visualisation. On the simulation side, the **Waymo World Model** (built on Genie 3, February 2026) generates *both* camera video and Waymo-specific LiDAR returns. This lets perception be stress-tested on synthetic long-tail scenes (see [Simulation & Data](./04_simulation_and_data.md)).

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Lift, Splat, Shoot | Philion & Fidler | 2020 | ECCV | Differentiable camera-to-BEV via depth distributions; foundational LSS |
| BEVFormer | Li et al. | 2022 | ECCV | Spatial cross-attention + temporal self-attention; 56.9% NDS camera-only |
| BEVFusion (MIT) | Liu et al. | 2022 | ICRA 2023 | Unified LiDAR-camera BEV fusion; 73.4% NDS |
| BEVDepth | Li et al. | 2022 | AAAI 2023 | Explicit depth supervision; significant NDS gain over pure LSS |
| PointPillars | Lang et al. | 2019 | CVPR | Real-time LiDAR detection via pillar pseudo-images |
| CenterPoint | Yin et al. | 2021 | CVPR | Center heatmap 3D detection; 71.9 mAPH on Waymo |
| PETR | Liu et al. | 2022 | ECCV | 3D position encoding for camera-only detection without BEV grid |
| StreamPETR | Wang et al. | 2023 | ICCV | Long-range temporal fusion via sparse propagation queries |
| Far3D | Jiang et al. | 2024 | AAAI | Long-range (~150 m) camera 3D detection via 2D-guided 3D queries |
| Point Transformer V3 | Wu et al. | 2024 | CVPR | Serialized point transformer; fast, scalable LiDAR backbone |
| Sonata | Wu et al. | 2025 | CVPR | Self-supervised point-cloud pre-training on PTv3 |
| EMMA | Hwang et al. (Waymo) | 2024 | arXiv 2410.23262 | MLLM-based driving model; camera-only 3D detection as text generation |
| DINOv3 | Siméoni et al. (Meta) | 2025 | arXiv | 7B SSL vision backbone with strong dense features; widely used for frozen-feature perception |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| BEVFormer | nuScenes test | NDS | 56.9% | Camera-only; ECCV 2022 |
| BEVFormer v2 | nuScenes test | NDS | ~63.4% | Camera-only + perspective head |
| BEVFusion (MIT) | nuScenes test | NDS | 73.9% | LiDAR + Camera |
| BEVDepth | nuScenes val | NDS | ~60.0% | Camera-only + LiDAR depth supervision |
| CenterPoint | Waymo val | mAPH L2 | ~66.4% | LiDAR-only (veh + ped) |
| PointPillars | Waymo val | mAP L1 | 63.3% | LiDAR-only; real-time baseline |

---

## Pros & Cons

| Method | Pros | Cons |
|--------|------|------|
| Camera-only BEV (LSS/BEVFormer) | Low hardware cost; dense semantic features; improving with scale | Metric depth ambiguity; degrades at night/rain |
| LiDAR-only (CenterPoint, PointPillars) | Precise metric 3D; velocity via Doppler; robust to lighting | No semantic texture; expensive hardware; sparse returns at range |
| LiDAR-Camera Fusion (BEVFusion) | Best accuracy; complementary modalities; state of art benchmarks | High system complexity; calibration sensitivity; cost |

---

## Open Problems & Research Gaps

- **Depth estimation quality**: Camera-only NDS lags LiDAR-camera by ~15 NDS points; closing this gap with scale, self-supervision, and radar integration is an active frontier.
- **Real-time BEV resolution trade-off**: High-resolution BEV grids (e.g., 0.1 m/voxel at 100 m range) are computationally intractable; efficient sparse representations are needed.
- **Cross-domain generalization**: Models trained on one geography (e.g., nuScenes: Boston + Singapore) degrade significantly on unseen cities; domain adaptation is critical for deployment.
- **Adverse weather robustness**: Fog, rain, and snow cause simultaneous degradation of all sensor modalities; learned sensor-degradation models are nascent.
- **Rare object categories**: Standard benchmarks focus on vehicles, pedestrians, cyclists; detection of construction equipment, emergency vehicles, and novel obstacle types is poorly benchmarked.
- **4D radar integration**: Emerging 4D imaging radars offer dense, all-weather point clouds; how to best fuse them with cameras and LiDAR in BEV remains an open research question.
- **Long-range detection**: Detecting and classifying objects at 150–300 m with cameras remains challenging; this range is critical for highway merging and high-speed scenarios.
- **Perception observability in VLA stacks**: When perception is implicit in a VLA or world-action model, standard perception metrics (mAP, NDS) no longer measure what the planner "sees". Probing and auxiliary-head methods for auditing implicit perception are immature.
- **Foundation-model auto-labeling quality**: Open-vocabulary auto-labelers (SAM 3, VLMs) scale long-tail annotation, but their systematic errors can propagate into training data. Label-noise auditing at fleet scale is unsolved.

---

## Further Reading

- [arXiv:2203.17270 BEVFormer](https://arxiv.org/abs/2203.17270) — official paper
- [arXiv:2205.13542 BEVFusion MIT](https://arxiv.org/abs/2205.13542) — unified BEV fusion
- [arXiv:2006.11275 CenterPoint](https://arxiv.org/abs/2006.11275) — center-based 3D detection
- [nuScenes 3D Detection Leaderboard](https://nuscenes.org/object-detection) — live benchmark
- [Waymo Open Dataset challenges](https://waymo.com/open/challenges/) — perception challenge results
- [arXiv:2401.06542 Robustness-Aware 3D Detection Review](https://arxiv.org/abs/2401.06542) — survey of robustness in AV detection
- [Pointcept / Point Transformer V3 GitHub](https://github.com/Pointcept/PointTransformerV3) — PTv3 and Sonata code
- [arXiv:2410.23262 EMMA](https://arxiv.org/abs/2410.23262) — detection and planning in a single MLLM
- [Meta DINOv3](https://ai.meta.com/dinov3/) — self-supervised vision backbone (Aug 2025)
- [Waymo 6th-generation Driver blog](https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/) — sensor-suite redesign
