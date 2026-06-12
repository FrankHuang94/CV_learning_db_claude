# 3D Occupancy Prediction: From Bounding Boxes to Volumetric Scene Understanding

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [Perception Stack](./01_perception_stack.md) · [End-to-End Driving](./03_end_to_end_driving.md) · [AV Stack Overview](./00_overview.md) · [Simulation & Data](./04_simulation_and_data.md)

---

## Overview

Classical autonomous driving perception represents the environment as a collection of **labeled 3D bounding boxes**—axis-aligned or oriented cuboids fitted around detected agents (vehicles, pedestrians, cyclists). This representation inherits directly from KITTI-era LiDAR detection benchmarks and encodes strong structural priors: it assumes objects are rigid, convex, and enumerable. For the canonical categories encountered in structured traffic, these assumptions are reasonable, and modern detectors (CenterPoint [Yin2021], BEVFusion [Liu2022]) achieve near-human accuracy on them. However, the bounding-box paradigm suffers from a fundamental **coverage gap**: any obstacle that does not belong to a predefined category—construction barriers, debris, fallen trees, overhanging vegetation, trailers, unusual vehicles—is silently ignored. A self-driving system operating on bounding-box outputs alone is epistemically blind to general obstacles that occupy physical space.

**3D occupancy prediction** addresses this gap by estimating, for every voxel in a discrete volumetric grid around the ego vehicle, whether that voxel is occupied and, optionally, by which semantic class. The output is a dense 3D semantic map $O \in \mathbb{R}^{X \times Y \times Z \times C}$, where $C$ classes include standard driving categories plus a generic *occupied but unknown* class. This representation is **general**: it captures not just agent-class objects but also terrain geometry, road structure, and free-space boundaries. The free-space estimate is itself safety-critical—it determines drivable area directly without requiring an explicit HD map.

The shift from bounding boxes to occupancy grids is philosophically aligned with classical robotics occupancy mapping (Elfes 1989), but the modern incarnation is fundamentally different: it operates from monocular or surround-view cameras without LiDAR at inference time (at least in Tesla's production system), it is learned end-to-end from dense 3D supervision derived from aggregated LiDAR point clouds, and it predicts semantic class membership jointly with geometry. The seminal industrial deployment is Tesla's Occupancy Network, announced at Tesla AI Day 2022 and integrated into FSD v12.

---

## The Occupancy Representation: Formal Definition

Let the ego-vehicle's observable volume be $\mathcal{V} \subset \mathbb{R}^3$, discretized into a grid of $X \times Y \times Z$ voxels with resolution $r \text{ m/voxel}$. The occupancy prediction task is:

$$\hat{O} = f_\theta(I_1, \ldots, I_N)$$

where $I_1, \ldots, I_N$ are $N$ surround-view camera images (or LiDAR sweeps), $f_\theta$ is a learned network, and $\hat{O} \in [0,1]^{X \times Y \times Z \times C}$ assigns per-voxel class probability vectors. Ground-truth $O^*$ is derived by aggregating multi-sweep LiDAR point clouds, projecting into the voxel grid, and assigning semantic labels via a fusion of human annotation and automatic labeling pipelines.

The **geometric semantic occupancy completion** (GSOC) challenge—predicting occupied voxels that were not observed by any LiDAR beam (occluded regions)—is what distinguishes modern occupancy prediction from simple LiDAR voxelization. The network must use contextual reasoning to infer that if the bottom of a vehicle is observed, the interior volume is also occupied, or that a wall segment continues behind an occlusion.

The standard evaluation metric is **mean Intersection over Union (mIoU)** over all $C$ semantic classes:

$$\text{mIoU} = \frac{1}{C}\sum_{c=1}^C \frac{|\hat{O}_c \cap O_c^*|}{|\hat{O}_c \cup O_c^*|}$$

where voxels are thresholded at $p > 0.5$ for binary occupancy evaluation. The Occ3D-nuScenes benchmark [Tian2023Occ3D] uses an $X \times Y \times Z = 200 \times 200 \times 16$ grid at 0.4 m resolution covering $[-40, 40] \times [-40, 40] \times [-1, 5.4]$ meters, with 17 semantic classes + 1 free class.

---

## Why Occupancy Beats Bounding Boxes for General Obstacles

Five concrete failure modes of bounding-box perception motivate occupancy:

1. **Open-set obstacle blindness**: An unknown object (e.g., a sofa fallen from a truck) has non-zero physical extent but zero detection probability from a closed-set detector. The occupancy grid encodes its physical presence regardless of semantic category.

2. **Non-convex geometry**: A bicyclist with outstretched arms, a truck with an open trailer door, or road construction with irregular shapes cannot be faithfully represented by a single oriented bounding box. Occupancy natively handles arbitrary geometry.

3. **Drivable-surface estimation**: The free-space boundary (where the vehicle can safely drive) is implicit in the occupancy grid but requires a separate map module in bounding-box architectures. Unified occupancy eliminates this architectural seam.

4. **Occlusion reasoning**: The bounding-box approach provides no signal about occupied space behind detected objects. Occupancy prediction, trained with aggregated LiDAR from multiple viewpoints, learns to reason about occluded regions probabilistically.

5. **Interaction with planning**: Planning algorithms (e.g., model-predictive control, lattice planners) operate directly on cost maps derived from occupancy; converting bounding boxes to cost maps introduces additional heuristics and approximation.

---

## Tesla Occupancy Network

Tesla's Occupancy Network was publicly described at AI Day 2022 and constitutes the most significant production deployment of camera-based 3D occupancy prediction as of June 2026. Key architectural details (from public presentations; detailed architecture not formally published):

- **Input**: Eight surround-view cameras at up to 1280 × 960 resolution, producing 360° coverage.
- **Backbone**: Multi-scale feature extraction per camera (HydraNet-style shared trunk with task-specific heads).
- **BEV/voxel lifting**: Camera features are lifted to a 3D voxel grid via a learned view transformation (analogous to LSS but with proprietary modifications).
- **Temporal fusion**: Recurrent BEV propagation using ego-motion-compensated alignment across frames.
- **Output volume**: A dense 3D occupancy grid at multiple resolutions; Tesla refers to a "vector space" representation that also encodes object motion.
- **Training data**: 1.4 billion frames from Tesla's global fleet; auto-labeled using multi-sweep LiDAR reconstruction during internal data collection phases, then trained camera-only at inference.
- **Training infrastructure**: Trained on Dojo, Tesla's custom AI supercomputer operational since July 2023 at Gigafactory Texas.

The critical design choice is **camera-only inference with LiDAR-supervised training**: Tesla collects LiDAR data internally for label generation but does not install LiDAR on production vehicles. This allows dense geometric supervision without production LiDAR cost. The training pipeline uses multi-view stereo reconstruction plus fleet-scale weak supervision from physics-consistent re-observation.

---

## Academic Occupancy Methods

### Occ3D [Tian2023]

Occ3D (arXiv:2304.14365) introduced a large-scale occupancy benchmark and baseline. The method uses a BEVFormer-style camera backbone to produce BEV features, then expands them along the $z$-axis via a learned voxel decoder. Key annotation contribution: Occ3D-nuScenes provides dense 3D labels at 0.4 m resolution derived from accumulated LiDAR sweeps with dynamic-object filtering.

### TPVFormer [Huang2023]

TPVFormer (CVPR 2023, arXiv:2302.07817) introduced **Tri-Perspective View (TPV)** as an alternative to dense voxel grids. Rather than a single $X \times Y \times Z$ volume, TPVFormer maintains three orthogonal 2D feature planes:

$$\text{TPV} = \{F_{XY} \in \mathbb{R}^{H_1 \times W_1 \times C},\; F_{YZ} \in \mathbb{R}^{H_2 \times W_2 \times C},\; F_{XZ} \in \mathbb{R}^{H_3 \times W_3 \times C}\}$$

A point's feature is obtained by projecting onto all three planes and summing. This reduces memory from $O(XYZ)$ to $O(XY + YZ + XZ)$, a significant saving at practical resolutions. TPVFormer achieves **7.8 mIoU on Occ3D-nuScenes** as a camera-only baseline—lower in absolute terms than later methods but notable given its computational efficiency. TPVFormer was designed as "an academic alternative to Tesla's occupancy network."

### SurroundOcc [Wei2023]

SurroundOcc (ICCV 2023) constructs a high-resolution 3D occupancy volume from multi-camera inputs via a coarse-to-fine spatial cross-attention mechanism. It uses a 3D deformable attention to hierarchically fill in voxel features from image queries. SurroundOcc achieves **20.3 mIoU on Occ3D-nuScenes** using camera-only inputs, significantly outperforming TPVFormer and BEVFormer baselines.

### OccNet [Tong2023]

OccNet (ICCV 2023) proposes occupancy as a **general scene representation** replacing the perception module in end-to-end stacks, demonstrating that occupancy features provide a better planning interface than bounding boxes. The paper evaluates on nuScenes planning metrics (L2 error, collision rate) with occupancy vs. bounding-box intermediate representations.

### MonoScene [Cao2022]

MonoScene (CVPR 2022) was an early camera-only semantic scene completion method using 2D-3D feature projection and a UNet3D decoder. It achieves only 6.06 mIoU on Occ3D-nuScenes but established the task framing for camera-based methods without LiDAR.

---

## Occupancy Flow: Dynamic Occupancy Prediction

Static occupancy prediction does not distinguish between static geometry (buildings, road markings) and dynamic agents (vehicles, pedestrians). **Occupancy flow** extends the prediction to include a per-voxel velocity field $V \in \mathbb{R}^{X \times Y \times Z \times 3}$, enabling the system to propagate occupancy forward in time:

$$O_{t+\Delta t} \approx \text{Warp}(O_t, V_t \cdot \Delta t)$$

UniAD [Hu2023] uses occupancy flow prediction as an intermediate stage between perception and motion planning, demonstrating that explicit future occupancy substantially reduces planning collision rates vs. bounding-box motion forecasting. The Waymo Occupancy Flow Challenge benchmark evaluates predicted occupancy at $t + 1s, 2s, \ldots, 8s$ future horizons using AUC and EPE metrics.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Tesla Occupancy Network | (AI Day presentation, no formal paper) | 2022 | AI Day | Production camera-only 3D occupancy; LiDAR-supervised training |
| Occ3D | Tian et al. | 2023 | NeurIPS | Occ3D-nuScenes/Waymo benchmark; 0.4 m resolution 18-class labeling |
| TPVFormer | Huang et al. | 2023 | CVPR | Tri-perspective view; memory-efficient alternative to dense voxels; 7.8 mIoU |
| SurroundOcc | Wei et al. | 2023 | ICCV | Coarse-to-fine 3D deformable attention; 20.3 mIoU camera-only |
| OccNet | Tong et al. | 2023 | ICCV | Occupancy as unified scene representation for E2E planning |
| MonoScene | Cao & de Charette | 2022 | CVPR | First camera-only semantic scene completion; 6.06 mIoU baseline |
| UniAD | Hu et al. | 2023 | CVPR (Best Paper) | Occupancy flow integrated into E2E planning; reduces collision rate 28% |

---

## Benchmark Performance: Occ3D-nuScenes

| Model | Modality | mIoU | Notes |
|-------|----------|------|-------|
| MonoScene | Camera-only | 6.06 | CVPR 2022; early monocular baseline |
| TPVFormer | Camera-only | 7.8 | CVPR 2023; efficient tri-plane representation |
| BEVFormer | Camera-only | 16.75 | Adapted from detection architecture |
| SurroundOcc | Camera-only | 20.3 | ICCV 2023; best published camera-only (2023) |
| CTF-Occ | Camera-only | 28.53 | Coarse-to-fine; top camera-only at time of survey |
| OccGen (LiDAR-cam) | LiDAR + Camera | ~37–42 | Fusion methods substantially outperform camera-only |

*Note: Benchmarks evolve rapidly; verify current SOTA at the [Occ3D-nuScenes leaderboard](https://github.com/Tsinghua-MARS-Lab/Occ3D).*

---

## Pros & Cons

| Aspect | Occupancy Prediction | Bounding-Box Detection |
|--------|---------------------|----------------------|
| Open-set coverage | Yes — any physical obstacle occupies voxels | No — silently ignores unknown classes |
| Computational cost | High — $O(XYZ \cdot C)$ per timestep | Low — sparse detections only |
| Planning interface | Direct cost map derivation | Requires box-to-map conversion heuristics |
| Training annotation | Dense LiDAR aggregation; expensive to label | Per-frame 3D box annotations |
| Dynamic agent modeling | Requires occupancy flow extension | Naturally encodes per-agent dynamics |
| Long-range accuracy | Degrades at range due to sparse LiDAR supervision | Comparable degradation in detector recall |

---

## Open Problems & Research Gaps

- **Annotation scalability**: Dense 3D occupancy labels require multi-sweep LiDAR accumulation and dynamic-object filtering; semi-automatic pipelines for annotation at billion-frame scale are not yet mature.
- **Camera-only depth accuracy**: The gap between camera-only occupancy (mIoU ~20–28) and LiDAR-aided occupancy (~37–42+) is large; closing it requires either better depth estimation or multi-modal training.
- **Semantic completeness**: Occupancy networks predict volumetric geometry but often lack fine-grained semantic classes for rare objects; the tail distribution of obstacle types is underrepresented in training.
- **Temporal consistency**: Per-frame occupancy predictions often flicker; enforcing temporal consistency via recurrent architectures or test-time smoothing without accumulating dynamic-object ghosting is unsolved.
- **Scalable evaluation**: mIoU averages over all voxels, giving excessive weight to free-space (the dominant class); safety-relevant metrics that weight rare obstacle categories appropriately are needed.
- **Occluded region prediction**: Learning to hallucinate the geometry behind visible surfaces from camera inputs alone requires strong learned priors; this is qualitatively different from observed-region occupancy.
- **Integration with downstream planning**: The formal interface between occupancy outputs and planning cost maps (potential fields, lattice planners, neural planners) is architecture-specific; a standard API is absent.

---

## Further Reading

- [Occ3D GitHub and benchmark](https://github.com/Tsinghua-MARS-Lab/Occ3D) — labels, code, leaderboard
- [arXiv:2302.07817 TPVFormer](https://arxiv.org/abs/2302.07817) — CVPR 2023 tri-perspective view
- [Tesla AI Day 2022 Occupancy Network segment](https://www.youtube.com/watch?v=ODSJsviD_SU&t=2000s) — production system overview
- [Waymo Occupancy Flow Challenge](https://waymo.com/open/challenges/occupancy-flow/) — benchmark with future prediction evaluation
- [arXiv:2405.05173 Survey on Occupancy Perception](https://arxiv.org/abs/2405.05173) — comprehensive 2024 survey
- [arXiv:2304.14365 Occ3D paper](https://arxiv.org/abs/2304.14365) — benchmark construction methodology
