# 3D 视觉、自动驾驶与机器人数据集

> **英文原文：** [cv-knowledge-base/11_datasets_and_benchmarks/02_3d_and_robotics_datasets.md](../../cv-knowledge-base/11_datasets_and_benchmarks/02_3d_and_robotics_datasets.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：** [机器人数据与遥操作](../06_robotics_and_embodied_ai/09_robot_data_and_teleoperation.md) · [仿真与数据](../07_autonomous_driving/04_simulation_and_data.md) · [核心数据集](./00_core_datasets.md) · [评估指标](./04_evaluation_metrics.md)

> **🆕 2026年10月更新：**
> - 新增**工业规模机器人数据**：AgiBot World（来自 100 台机器人的 100 万+ 条轨迹，IROS 2025）和 Galaxea Open-World Dataset（500+ 小时移动操作数据，2025年9月）。
> - 新增**人类第一人称操作数据**：EgoDex（Apple，829 小时 Vision Pro 视频，含 3D 手部姿态）。并补充了在大规模人类视频语料上预训练 VLA 的趋势（例如 GR00T N1.7 使用了 2 万小时的 EgoScale）。
> - 新增**端到端（end-to-end）驾驶基准**：带评分者反馈分数（Rater Feedback Score）的 Waymo Open **端到端驾驶**数据集（2025）、**NAVSIM v2** 伪仿真（CoRL 2025），以及配合 Alpamayo 发布的 NVIDIA Physical AI AV 数据集。
> - 新增 **3D 物体数据**：uCO3D（约 17 万个转台视频，每个视频附 3DGS 重建），以及 SAM 3D（2025年11月）及其 **SA-3DAO** 艺术家物体基准（2026年6月）。
> - 新增关于**机器人策略基准及其饱和**的小节：LIBERO（π0.5 达 96.85%）与 LIBERO-Plus 下的鲁棒性骤降、RoboTwin 2.0、RoboArena 和 BEHAVIOR-1K。

---

## 概述

三维视觉数据集与机器人数据集代表了具身感知的前沿：这些系统必须对物理世界的 3D 结构进行推理，才能在其中导航、操作或驾驶。与标注成本随图像数量增长的标准 2D 图像基准不同，3D 数据集需要通过 RGB-D 传感器、激光雷达点云、多视图立体或深度估计来采集或重建体积几何，这使每个场景的采集成本高出几个数量级。

这一领域可划分为三个用例集群。第一，**室内 3D 场景理解**（ScanNet、ShapeNet）面向结构化室内环境中的重建、分割和物体位姿估计。这些数据集对 AR/VR、家用机器人以及具备 3D 感知的视觉-语言模型至关重要。第二，**自动驾驶感知**数据集（KITTI、nuScenes、Waymo Open）提供同步的多模态传感器流（相机、激光雷达、毫米波雷达、GPS/IMU）以及精确的 3D 边界框标注，构成工业级 3D 目标检测与跟踪研究的基石。第三，**大规模生成式 3D** 数据集（ShapeNet、Objaverse、Objaverse-XL）提供带纹理的 CAD 模型和摄影测量扫描，支撑神经渲染、新视角合成（NeRF、3DGS）和 3D 生成研究。

一个较新且日益重要的集群是**机器人学习数据集**：在多种机器人平台、环境和任务上采集的遥操作（teleoperation）操作示范（Open X-Embodiment、DROID、Ego-Exo4D）。这些数据集是视觉-语言-动作模型（Vision-Language-Action, VLA）的训练语料。其核心挑战是*具身异构性*：在 Franka 机械臂上采集的数据无法简单迁移到 UR5 或移动操作机器人上。Open X-Embodiment 等数据集提供了覆盖 22 种机器人平台、按 RLDS 标准化的轨迹，正是为了直接解决这一问题。

2025年中至 2026年，机器人数据格局在三个方面发生了变化。第一，**单一机构的工业化采集**在规模上超过了学术界的数据汇集。仅 AgiBot World 一家就在统一的、传感器丰富的硬件机队上达到了与 OXE 相当的轨迹数量，Galaxea 等公司也发布了数百小时的单一具身移动操作数据。第二，**带 3D 手部姿态的人类第一人称视频**成为一等的预训练数据来源，例如在 Apple Vision Pro 上采集的 EgoDex，以及用于预训练 NVIDIA GR00T N1.7 的 2 万小时 EgoScale 语料。共享的相对末端执行器动作空间使人类数据与机器人数据可以混合使用。第三，由于标准仿真基准已经饱和，**评估转向分布式真实世界竞技场和鲁棒性套件**（RoboArena、LIBERO-Plus）。在自动驾驶领域，对应的转变是从 3D 框感知基准转向使用人类偏好或伪闭环指标的**端到端规划基准**（WOD-E2E、NAVSIM v2）。

---

## ScanNet

**引用：** [Dai2017] Dai, A. et al. (2017). ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. *CVPR 2017*.

ScanNet 是一个 RGB-D 视频数据集，包含 1,513 个室内场景扫描，涵盖多样环境（办公室、公寓、浴室、厨房、教室、图书馆），共约 250 万帧 RGB-D 图像。每次扫描提供：彩色视频、深度视频、3D 网格重建（通过 BundleFusion）、相机位姿估计、表面法向、网格级语义实例分割，以及轴对齐 3D 边界框。

**关键统计：**
- 707 个不同空间中的 1,513 次扫描
- 250 万个 RGB-D 视图
- 平均每次扫描：15 万个网格顶点，空间范围约 5.5m × 5.1m × 2.4m
- 40 个语义类别标签（评估基准使用 20 个）
- 由 500 多名 Mechanical Turk 众包工作者标注

**支持的任务：** 3D 语义分割、3D 实例分割、3D 目标检测、新视角合成（ScanNet 被广泛用于 NeRF 和 3DGS 实验）、场景重建质量评估。

**ScanNet++：** 2023 年的扩展版本，提供更高分辨率的 RGB-D 采集和激光扫描仪真值网格，用于更精确的重建基准评测。

---

## ShapeNet

**引用：** [Chang2015] Chang, A.X. et al. (2015). ShapeNet: An Information-Rich 3D Model Repository. arXiv:1512.03012.

ShapeNet 是一个按 WordNet 类别体系索引的大规模带标注 3D CAD 模型库。完整模型库（截至 2015 年）索引了 3,135 个类别中约 300 万个模型。精选的 **ShapeNetCore** 子集涵盖 55 个常见物体类别，约 51,300 个唯一 3D 模型（划分：35,764 训练 / 5,133 验证 / 10,265 测试）。**ShapeNetPart** 为 16 个类别的 16,881 个形状提供部件级分割标注。

**内容：** 模型包括一致的刚性对齐、双侧对称平面、部件标注和物理尺寸估计。许多模型来自在线 CAD 库（TurboSquid、3DWarehouse），网格质量差异很大。

**主要用途：** 点云分类与分割（PointNet、DGCNN、PointTransformer）、新视角合成、3D 形状生成（IM-NET、ShapeFlow）、检索以及 3D 编码器预训练。ShapeNet 干净的 CAD 几何为 3D 深度学习提供了理想的训练数据，但与扫描得到的真实物体之间存在显著的仿真到现实（sim-to-real）差距。

---

## KITTI

**引用：** [Geiger2012] Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. *CVPR 2012*.

KITTI 是奠基性的自动驾驶数据集，采集车辆配备两对立体相机（彩色 + 灰度）、一台 Velodyne HDL-64E 3D 激光雷达和一套 GPS/IMU 单元。总数据量：在德国卡尔斯鲁厄录制的 6 小时数据。

**基准任务与规模：**
- **立体匹配 / 光流：** 194 对训练 + 195 对测试（1242×375 像素）
- **3D 目标检测：** 7,481 训练帧 / 7,518 测试帧；汽车、行人、骑行者共 8 万个标注 3D 边界框
- **视觉里程计/SLAM：** 22 个立体序列，共 39.2 km
- **道路估计、深度补全**

KITTI 引入了鸟瞰图（BEV）和基于 3D IoU 的评估协议，这些协议至今仍是激光雷达 3D 检测的标准。尽管年代久远，KITTI 3D 检测几乎仍在所有 3D 感知论文中作为向后兼容的基准被报告。

---

## nuScenes

**引用：** [Caesar2020] Caesar, H. et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. *CVPR 2020*.

nuScenes 是一个全环视自动驾驶数据集，包含来自波士顿和新加坡的 1,000 个驾驶场景（每个 20 秒），采用完整传感器套件采集：6 个相机（360° 覆盖）、5 个毫米波雷达、1 个激光雷达（32 线，360°）。共有 23 个类别、8 种属性的 140 万个标注 3D 边界框。关键帧标注频率为 2 Hz（每个 20 秒场景约 40 帧）。划分：700 训练 / 150 验证 / 150 测试。

**nuScenes 为何重要：** 它是第一个提供全 360° 相机 + 激光雷达 + 毫米波雷达融合的公开自动驾驶数据集，推动了多模态 3D 检测与跟踪研究。nuScenes 检测分数（NDS）同时衡量平移、尺度、朝向、速度和属性预测，是比 KITTI 仅基于 IoU 的 AP 更全面的指标。

**nuScenes-lidarseg：** 为激光雷达点云增加了密集的语义分割标签。**nuPlan：** 面向规划的扩展，带有记录的规划器轨迹。

---

## Waymo Open Dataset

**引用：** [Sun2020] Sun, P. et al. (2020). Scalability in Perception for Autonomous Driving: Waymo Open Dataset. *CVPR 2020*.

Waymo Open Dataset 包含 Waymo 自动驾驶车辆采集的 1,150 个 20 秒驾驶片段（798 训练 / 202 验证 / 150 测试），每辆车配备 5 个激光雷达和 5 个相机。标注：车辆、行人、骑行者和标志共 1200 万个 3D 激光雷达标签和 120 万个 2D 相机标签。点云数据：每帧 17.7 万–30 万个点（双回波激光雷达）。

**意义：** 由于传感器质量、标注密度和环境多样性（昼/夜、天气变化、城市/郊区），Waymo Open 被普遍认为是最难的公开 3D 检测基准。其激光雷达质量远超 KITTI 的 64 线扫描仪；nuScenes 使用的是 32 线扫描仪。Waymo 的传感器头代表了接近量产水平的自动驾驶硬件。

**扩展：** Waymo Open Dataset v2（2023）增加了相机-激光雷达对应标注；Waymo Open Motion Dataset 提供用于运动预测的未来轨迹标注。

**端到端驾驶数据集（WOD-E2E，2025）：** 这第三个 Waymo 数据集面向长尾场景中仅依赖相机、基于视觉的端到端规划，将多相机图像与高层路线指令配对。截至 2025年4月的发布，它包含 2,037 个训练片段（runs）、一个带评分者反馈标签和场景标签的 479 个片段的验证集，以及 1,505 个片段的测试集。它不再使用相对单条记录轨迹的位移误差，而是引入了**评分者反馈分数（Rater Feedback Score, RFS）**。对于被评分的帧，人类评分者会对最多三条候选未来轨迹给出 0–10 分。预测轨迹按其与这些已评分备选轨迹的匹配程度计分，因此多种可接受的行为都能得到奖励，而不仅是记录下来的那一条。评分代码已公开（参见[评估指标](./04_evaluation_metrics.md)）。

---

## 端到端驾驶基准：NAVSIM v2 与 Physical AI AV

**引用：** [Dauner2024] Dauner, D. et al. (2024). NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. *NeurIPS 2024 Datasets & Benchmarks*. [Cao2025] Cao, W., Hallgarten, M., Li, T., Dauner, D., et al. (2025). Pseudo-Simulation for Autonomous Driving. *CoRL 2025*.

**NAVSIM** 基于 OpenScene/nuPlan 日志构建，介于脆弱的开环 L2 指标和昂贵的闭环仿真之间。**NAVSIM v2** 引入了*伪仿真（pseudo-simulation）*：在规划轨迹附近用合成观测增强真实日志，再通过两阶段聚合来近似闭环行为。作者报告其与闭环仿真高度相关，而计算量约为后者的 1/6。v2 还引入了**扩展 PDM 分数（Extended PDM Score, EPDMS）**、困难的 **navhard** 划分以及官方 2025 挑战赛（AGC 2025）。

**NVIDIA Physical AI Autonomous Vehicles 数据集（2025–2026）：** NVIDIA 在 Hugging Face 上发布了一个大型多相机驾驶数据集，并配有开发工具包（`physical_ai_av`）。它是开放的 **Alpamayo** 推理型 VLA 系列的训练与评估语料。Alpamayo-R1 于 2025 年发布，在 CES 2026 上更名为 Alpamayo 1，随后 Alpamayo 1.5 于 2026年3月发布。这标志着向带推理标注的开放、工业规模端到端驾驶数据的转变。参见[仿真与数据](../07_autonomous_driving/04_simulation_and_data.md)。

---

## Objaverse / Objaverse-XL

**引用：** [Deitke2023a] Deitke, M. et al. (2023). Objaverse: A Universe of Annotated 3D Objects. *CVPR 2023*. [Deitke2023b] Deitke, M. et al. (2023). Objaverse-XL: A Universe of 10M+ 3D Objects. *NeurIPS 2023*.

**Objaverse 1.0：** 由 Sketchfab 上 10 万多名艺术家创作的 80 万+ 个带标注 3D 模型。模型包含描述文本、标签、动画和材质。

**Objaverse-XL：** 来自多种来源的 1000 万+ 个 3D 物体：人工设计的模型、地标与日常物品的摄影测量扫描、专业文物扫描。Objaverse-XL 的规模是 Objaverse 1.0 的 12 倍。

**影响：** Objaverse 促成了 Zero123（Liu et al., 2023），后者展示了大规模新视角合成预训练，进而开启了单图像 3D 重建（Zero123++、SyncDreamer、Wonder3D）。Objaverse-XL 的规模使训练大型 3D 生成模型成为可能，并成为开放词汇 3D 理解模型的主要预训练语料。

**渲染：** Objaverse 团队发布了多视图渲染图像（来自 Objaverse-XL 渲染的 1 亿+ 张图像），专门用于预训练新视角合成和 3D 生成模型。

---

## uCO3D 与 SAM 3D / SA-3DAO（真实世界 3D 物体）

**引用：** [Liu2025] Liu, X. et al. (2025). UnCommon Objects in 3D. arXiv:2501.07574 (Meta). [SAM3D2025] SAM 3D Team, Chen, X., Chu, F.-J., Gleize, P., Liang, K. J., Sax, A., et al. (2025). SAM 3D: 3Dfy Anything in Images. Meta Superintelligence Labs.

**uCO3D**（2025）是 CO3Dv2 的后继者，包含约 170,000 个真实物体的完整 360° 转台视频。物体涵盖约 1,000 个 LVIS 类别，归为 50 个超类。每个视频都带有物体掩码、相机位姿、三种点云、一个逐视频的 **3D 高斯泼溅（Gaussian Splatting）重建**，以及长短两种 VLM 描述。其规模和与 LVIS 对齐的类别体系，使它在训练前馈式重建和 3D 生成时成为合成 Objaverse 资产的真实采集对照。

**SAM 3D**（2025年11月）分为两部分：*SAM 3D Objects* 从单张自然图像重建形状、纹理和布局，*SAM 3D Body* 恢复人体网格。延续 SAM 的方法，SAM 3D Objects 在真实、杂乱的图像上将渐进式训练与**带人类反馈的数据引擎**相结合。这降低了对纯合成 CAD 训练数据的依赖，且该模型在真实物体与场景的人类偏好测试中优于此前的 3D 生成模型。Meta 随后发布了 **SAM 3D Artist Objects（SA-3DAO）**评估集和 Hugging Face 排行榜（2026年6月）。它为真实图像中的物体提供艺术家制作的 3D 真值，是比渲染的合成视图更难的目标。

---

## Open X-Embodiment

**引用：** [Open X-Embodiment Collaboration, 2023] Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.

Open X-Embodiment（OXE）使用 RLDS（Robot Learning Dataset Standard）格式，将来自 34 个研究实验室的 60 多个现有机器人数据集整合为统一语料。总计：覆盖 22 种机器人具身形态（单臂、双臂、四足、移动操作机器人）的 100 万+ 条真实机器人轨迹，527 种不同的操作技能，参与机构包括 Google、Stanford、UC Berkeley、CMU、MIT 等。

**关键统计：**
- 22 种机器人具身形态
- 527 种技能
- 100 万+ 条轨迹
- 平均轨迹长度：约 120 个时间步
- 控制频率：3–10 Hz，取决于机器人

**RT-X 模型：** OXE 论文在该语料上训练了 RT-1-X 和 RT-2-X，证明跨具身（cross-embodiment）训练比单一具身数据集能带来更好的泛化。这是通用机器人策略的经验基础。

---

## DROID

**引用：** [Khazatsky2024] Khazatsky, A. et al. (2024). DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. *RSS 2024*.

DROID（Distributed Robot Interaction Dataset）包含 76,000 条遥操作操作示范轨迹，由北美、亚洲和欧洲的 18 家研究机构历时 12 个月采集。采集使用 18 台 Franka Panda 机器人，覆盖 564 个独特的真实世界场景和 86 种不同的操作任务。

**强调多样性：** 与实验室精心布置的数据集不同，DROID 专门为最大化环境多样性而设计（光照多变、场景杂乱、物体多样）。95% 的成功回合（7.5 万个回合）每条轨迹配有 3 条自然语言标注。

**为何重要：** 在 DROID 上训练得到的策略，比在同等规模同质数据集上训练的策略性能更高、泛化更好，验证了真实场景多样性对机器人学习的重要性。DROID 已成为 VLA 模型的标准大规模操作预训练数据集之一。

---

## Ego-Exo4D

**引用：** [Grauman2023] Grauman, K. et al. (2024). Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives. *CVPR 2024*.

Ego-Exo4D 扩展了 Ego4D，对熟练的人类活动（体育、音乐、烹饪、机械维修）进行同步的第一人称和外部（第三人称）相机采集。数据约 1,422 小时，来自 13 所大学的 800 多名参与者。提供的多任务基准包括：对应关系（ego↔exo）、关系、熟练度估计和细粒度动作理解。

**独特贡献：** 第一人称-第三人称对应任务（在第一人称和第三人称视角之间匹配同一活动的视角）是全新的，在技能学习、从示范视频模仿以及具身智能中有直接应用。

---

## AgiBot World（Colosseo）

**引用：** [Bu2025] Bu, Q., Cai, J., Chen, L., Cui, X., Ding, Y., Feng, S., et al. (2025). AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems. *IROS 2025*. arXiv:2503.06669.

AgiBot World 由智元机器人（AgiBot）与 OpenDriveLab 推出，分两个阶段发布：Alpha 版于 2024年12月发布，Beta 版于 2025年3月发布。它包含**在 100 台机器人组成的机队上采集的 100 万+ 条轨迹**，覆盖五个目标领域中的 **100+ 个 1:1 复刻的真实生活场景**。硬件包括视触觉传感器、6 自由度灵巧手和移动双臂平台。与汇集异构实验室数据的 OXE 不同，它是*单一机构、协议标准化*的数据采集。配套的 **GO-1** 通用策略（2025年9月开源）以及后续研究 *Is Diversity All You Need for Scalable Robotic Manipulation?*（2025）都利用它来研究数据多样性的扩展规律。

---

## Galaxea Open-World Dataset

由星海图（Galaxea AI）于 2025年9月随 G0 VLA 一同发布（RLDS 和 LeRobot 格式）。它包含 **500+ 小时**的真实世界*移动*操作数据，使用**单一统一的具身形态**（一款 Galaxea R1 系列移动双臂机器人）在住宅、厨房、零售和办公场景中采集，并带有细粒度子任务语言标注。其单一具身、长时程、真实场景的设计，与 DROID 以单臂桌面操作为主的定位形成互补。

---

## EgoDex（人类第一人称灵巧操作）

**引用：** [Hoque2025] Hoque, R., Huang, P., Yoon, D. J., Sivapurapu, M., & Zhang, J. (2025). EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video. arXiv:2505.11709 (Apple).

EgoDex 包含在 Apple Vision Pro 上用 ARKit 录制的 **829 小时** 30 Hz、1080p 第一人称视频。每一帧都配有跟踪得到的**头部、上半身和双手的 3D 姿态**，以及自然语言标注。全部内容都是主动的桌面操作，覆盖 **194 项任务**。划分为约 725 小时训练、约 7 小时测试和约 97 小时附加数据。由于姿态由头显原生采集而非事后估计，EgoDex 支持直接训练人手轨迹预测以及人到机器人的迁移。它属于 2025–2026 年扩大*人类*数据规模用于机器人预训练这一更广泛的趋势。例如 NVIDIA 的 GR00T N1.7 通过人类与机器人共享的相对末端执行器动作空间，在预训练数据中加入了 **2 万小时的 EgoScale 人类视频**。

---

## 机器人策略基准及其饱和（2025–2026）

仿真基准仍是比较 VLA 最便宜的方式，但最流行的那个如今已经饱和。

- **LIBERO**（2023）：四个套件（Spatial、Object、Goal、Long）。微调后的 VLA 已达到上限。例如在 openpi 中微调的 π0.5 报告了 98.8 / 98.2 / 98.0 / 92.4（**平均 96.85%**）。
- **LIBERO-Plus** [Fei2025]（2025年10月）用 10,030 个任务，沿七个扰动维度重新测试 LIBERO 策略：物体布局、相机视角、机器人初始状态、语言改写、光照、背景纹理和传感器噪声。在适度的相机或初始状态偏移下，成功率**从约 95% 跌至 30% 以下**，且模型在很大程度上*忽略了语言指令*。这有力地表明 LIBERO 分数衡量的是记忆而非泛化。
- **RoboTwin 2.0**（ICML 2026）：一个可扩展的双臂仿真基准与数据生成器，具有强领域随机化。它承办了 CVPR 2025 MEIS 挑战赛，并于 2026年3月扩展出依赖记忆的操作套件 RMBench。
- **BEHAVIOR-1K** [Li2024]：在 OmniGibson 仿真器中，从时间使用调查中选出的 1,000 项日常家务活动。它被用作长时程移动操作挑战基准（2025 BEHAVIOR Challenge）。
- **RoboArena**（2025）：一种*分布式真实世界*评估，策略以远程服务的方式部署，并在多个机构的 DROID 平台上接受评估。它是对单一实验室真实机器人结果难以复现这一问题的回应。

---

## 关键论文 / 数据集

| 名称 | 作者/机构 | 年份 | 会议/期刊 | 关键统计 / 贡献 |
|---|---|---|---|---|
| ScanNet | Dai et al. (Stanford/TUM) | 2017 | CVPR | 1,513 次扫描、250 万帧 RGB-D、语义实例标注 |
| ShapeNet (Core) | Chang et al. (Stanford/Princeton) | 2015 | arXiv | 5.13 万个模型、55 个类别；经典 3D 形状数据集 |
| KITTI | Geiger et al. (KIT) | 2012 | CVPR | 6 小时驾驶数据、8 万个 3D 框；开创激光雷达检测基准 |
| nuScenes | Caesar et al. (nuTonomy) | 2020 | CVPR | 1K 个场景、6 相机+激光雷达+毫米波雷达、140 万个 3D 框、23 个类别 |
| Waymo Open | Sun et al. (Waymo) | 2020 | CVPR | 1,150 个片段、1200 万个 3D 标签；质量最高的公开自动驾驶数据集 |
| Objaverse | Deitke et al. (AllenAI) | 2023 | CVPR | 80 万+ 个 3D 模型，艺术家创作并附描述 |
| Objaverse-XL | Deitke et al. (AllenAI) | 2023 | NeurIPS | 1000 万+ 个 3D 物体；支撑大规模 3D 生成预训练 |
| Open X-Embodiment | OXE Collaboration | 2023 | arXiv | 100 万+ 条轨迹、22 种机器人、60+ 个数据集，RLDS 标准化 |
| DROID | Khazatsky et al. (Stanford/Berkeley) | 2024 | RSS | 7.6 万条轨迹、18 个实验室、564 个场景、86 项任务，真实场景 |
| Ego-Exo4D | Grauman et al. (Meta AI) | 2024 | CVPR | 1,422 小时第一人称+第三人称视频、熟练活动、ego-exo 对应任务 |
| NAVSIM v2 (pseudo-simulation) | Cao, Hallgarten, …, Geiger (Tübingen/NVIDIA/Bosch/OpenDriveLab) | 2025 | CoRL | 伪闭环自动驾驶评估；EPDMS；navhard 划分；成本约为闭环的 1/6 |
| Waymo Open E2E Driving (WOD-E2E) | Waymo | 2025 | 数据集发布 | 2,037 训练 / 479 验证 / 1,505 测试片段；相机 + 路线指令；评分者反馈分数 |
| uCO3D | Liu et al. (Meta) | 2025 | arXiv | 约 17 万个 360° 物体视频、约 1K 个 LVIS 类别、逐视频 3DGS + 描述 |
| SAM 3D / SA-3DAO | SAM 3D Team (Meta) | 2025/2026 | Meta 技术报告 | 基于人类反馈数据引擎的单图像 3D；艺术家物体 3D 基准 + 排行榜 |
| AgiBot World | Bu et al. (AgiBot / OpenDriveLab) | 2025 | IROS | 100 万+ 条轨迹、100 台机器人、100+ 个复刻场景；GO-1 策略 |
| Galaxea Open-World | Galaxea AI | 2025 | 数据集发布 | 500+ 小时单一具身移动操作；子任务标注 |
| EgoDex | Hoque et al. (Apple) | 2025 | arXiv | 829 小时 Vision Pro 第一人称视频、3D 头部/身体/手部姿态、194 项任务 |
| LIBERO-Plus | Fei et al. (Fudan et al.) | 2025 | arXiv | 10,030 个扰动 LIBERO 任务、7 个维度；暴露 VLA 的脆弱性 |
| RoboTwin 2.0 | RoboTwin Platform team | 2026 | ICML | 带领域随机化的双臂仿真基准 + 数据生成器 |

---

## 基准对比（3D 目标检测）

| 数据集 | 传感器 | 3D 框 | 类别 | 评估指标 | 难度 |
|---|---|---|---|---|---|
| KITTI | 64 线激光雷达 + 立体相机 | 80K | 3（汽车、行人、骑行者） | AP@IoU3D 0.7/0.5 | 中 |
| nuScenes | 32 线激光雷达 + 6 相机 + 5 毫米波雷达 | 1.4M | 23 | NDS, mAP | 高 |
| Waymo Open | 5 个激光雷达（高分辨率）+ 5 相机 | 12M 3D | 4 | mAPH | 非常高 |

**端到端规划与策略基准（2025–2026）**

| 基准 | 输入 | 规模 | 主要指标 | 状态 |
|---|---|---|---|---|
| NAVSIM v2 (navhard) | 多相机 + 激光雷达日志（OpenScene/nuPlan） | 精选困难场景 | EPDMS（伪闭环） | 活跃（AGC 2025 挑战赛） |
| WOD-E2E | 多相机 + 路线指令 | 约 4K 个片段（训练/验证/测试） | 评分者反馈分数（0–10 分评分轨迹） | 活跃（Waymo 挑战赛） |
| LIBERO | 仿真，单臂 | 4 个套件 × 10 项任务 | 成功率 | 已饱和（每个套件约 97–99%） |
| LIBERO-Plus | 仿真，扰动版 LIBERO | 10,030 项任务 | 扰动下的成功率 | 大幅下降（相机/初始状态偏移下 <30%） |
| RoboArena | 真实 DROID 平台，分布式 | 多机构 | 远程部署策略的真实世界评估汇总 | 活跃 |

---

## 优缺点

| 数据集 | 优势 | 局限 |
|---|---|---|
| ScanNet | 高质量 3D 重建；语义 + 实例标签；被广泛用于 NeRF/3DGS 评估 | 仅室内；消费级 RGB-D；边界处标注有噪声 |
| KITTI | 奠基性；规模小、便于快速迭代；支持立体、光流、检测、SLAM | 单一城市（卡尔斯鲁厄）；64 线激光雷达已过时；标注数量少 |
| nuScenes | 完整传感器套件；带属性的多类别；面向规划的扩展 | 32 线激光雷达质量较低；20 秒场景较短 |
| Waymo Open | 传感器质量最高；环境多样；大规模标注 | 需注册才能下载；传感器规格为专有 |
| Objaverse-XL | 3D 规模空前；来源多样；支撑生成式 3D | 模型质量参差不齐；艺术家资产带来大量噪声；语义标注有限 |
| Open X-Embodiment | 跨具身；格式标准化；直接支撑通用策略 | 各贡献数据集质量差异大；动作空间不同 |
| DROID | 多样性高、真实场景；策略训练效果好；有语言标注 | 单一机器人平台（Franka）；以桌面操作为主 |
| AgiBot World | 在统一硬件/协议下达到 OXE 规模的轨迹量；灵巧操作 + 触觉感知 | 单一厂商的具身形态；复刻（布景）场景而非真正的真实家庭 |
| EgoDex | 大规模原生 3D 手部姿态；扩展成本低（无需机器人） | 人到机器人的具身差距；仅限桌面；需要重定向（retargeting） |
| WOD-E2E | 长尾场景；人类偏好指标可容纳多模态的未来 | 已评分子集较小；仅相机；评分者主观性 |
| NAVSIM v2 | 以开环成本得到与闭环相关的分数 | 合成观测可能带有渲染伪影；智能体非反应式 |
| LIBERO | 低成本、可复现的 VLA 比较 | 已饱和；LIBERO-Plus 表明分数反映的是对固定视角/初始状态的记忆 |

---

## 开放问题与研究空白

- **缩小 3D 的仿真到现实差距：** ShapeNet 和 Objaverse 提供合成 3D 数据，ScanNet 提供真实扫描，但合成渲染与真实 RGB-D 数据之间的领域差距仍是 3D 检测与重建的主要瓶颈。
- **多具身机器人学习：** DROID 和 OXE 展示了跨具身的收益，但在没有显式具身条件表征的情况下，弥合根本不同的运动学结构（机械臂、移动底盘、灵巧手）仍未解决。
- **标准化的机器人数据集评估：** 与视觉基准不同，机器人操作数据集缺乏公认的评估协议。每篇论文使用不同的环境、物体和成功标准，使比较几乎不可能。2025–2026 年的应对措施都只解决了部分问题：分布式真实世界竞技场（RoboArena）提升了可复现性，但缓慢且昂贵；鲁棒性套件（LIBERO-Plus）暴露了脆弱性，但仅限于仿真。LIBERO（约 97%）这类已饱和的仿真基准不应再被用作主要证据。
- **人到机器人的数据迁移：** 人类第一人称语料（EgoDex 829 小时；GR00T N1.7 中的 EgoScale 2 万小时）如今比开放遥操作数据大一到两个数量级。其中有多少能够迁移、通过哪种动作表征（相对末端执行器、关键点、潜在动作）迁移，尚未有受控的扩展研究给出结论。
- **面向规划的人类偏好指标：** WOD-E2E 的评分者反馈标签和 NAVSIM 的伪仿真都试图摆脱"单一记录轨迹"的假设。二者之间的一致性，以及它们与真实闭环安全性的一致性，目前未知。
- **大规模室外 3D 场景理解：** 按自动驾驶标准，Waymo Open 规模很大，但与视觉数据集相比很小。扩展到数百万个完整标注的驾驶场景是一项重大的数据采集挑战。
- **时序 4D 理解：** 大多数 3D 数据集是静态的（ScanNet、ShapeNet）或只提供短序列。理解场景如何随时间变化（物体状态变化、人体运动、可变形物体）严重缺乏研究。
- **长时程操作数据集：** DROID 和 OXE 包含的是短任务示范（通常 <2 分钟）。大规模记录多步骤、跨多天家务任务的数据集基本不存在。
- **世界模型预训练数据：** 视频预测和世界模型研究（Genie、UniSim）需要具有丰富 3D 结构的大规模多样化视频，但目前没有专为此构建、规模足够的数据集。Genie 3（Google DeepMind，2025年8月）等交互式世界模型使用的是专有语料。目前不存在规模相当的开放带动作标签视频，社区只能拼接驾驶日志（WOD-E2E、Physical AI AV）、机器人数据（OXE、AgiBot World）和人类第一人称视频。

---

## 延伸阅读

- [ScanNet 项目页面](http://www.scan-net.org/)
- [Waymo Open Dataset 文档](https://waymo.com/open/about/)
- [Open X-Embodiment 项目](https://robotics-transformer-x.github.io/)
- [DROID 数据集](https://droid-dataset.github.io/)
- [Objaverse-XL 论文 — arXiv:2307.05663](https://arxiv.org/abs/2307.05663)
- [nuScenes 数据集与排行榜](https://www.nuscenes.org/)
- [Waymo Open Dataset 代码、E2E 驾驶 proto 定义与评分者反馈分数](https://github.com/waymo-research/waymo-open-dataset)
- [NAVSIM v2 / Pseudo-Simulation for Autonomous Driving (arXiv:2506.04218)](https://github.com/autonomousvision/navsim)
- [NVIDIA Physical AI AV 数据集开发工具包](https://github.com/NVlabs/physical_ai_av) · [Alpamayo](https://github.com/NVlabs/alpamayo)
- [AgiBot World Colosseo (arXiv:2503.06669)](https://github.com/OpenDriveLab/AgiBot-World)
- [Galaxea Open-World Dataset 与 G0/G0.5 VLA](https://github.com/OpenGalaxea/GalaxeaVLA)
- [EgoDex (arXiv:2505.11709)](https://github.com/apple/ml-egodex)
- [uCO3D (arXiv:2501.07574)](https://github.com/facebookresearch/uco3d)
- [SAM 3D Objects 与 SA-3DAO](https://github.com/facebookresearch/sam-3d-objects)
- [LIBERO-Plus (arXiv:2510.13626)](https://github.com/sylvestf/LIBERO-plus) · [RoboTwin 2.0 (arXiv:2506.18088)](https://github.com/RoboTwin-Platform/RoboTwin) · [RoboArena](https://github.com/robo-arena/roboarena) · [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
