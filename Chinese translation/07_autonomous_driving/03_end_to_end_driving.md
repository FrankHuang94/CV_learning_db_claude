# 端到端自动驾驶：从 UniAD 到神经世界模型

> **英文原文：** [cv-knowledge-base/07_autonomous_driving/03_end_to_end_driving.md](../../cv-knowledge-base/07_autonomous_driving/03_end_to_end_driving.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：** [自动驾驶技术栈概览](./00_overview.md) · [感知栈](./01_perception_stack.md) · [占用预测](./02_occupancy_prediction.md) · [世界模型](../06_robotics_and_embodied_ai/02_world_models.md) · [VLA 模型](../06_robotics_and_embodied_ai/01_vla_models.md) · [仿真与数据](./04_simulation_and_data.md)

> **🆕 2026年10月更新：**
> - 新增 **“2025–2026：驾驶基础模型与推理型 VLA”** 一节：Waymo EMMA、NVIDIA Alpamayo-R1 / 开放的 Alpamayo 1（100 亿参数）、SimLingo、潜在推理趋势，以及量产 VLA / 世界行为技术栈（小鹏 VLA 2.0、理想、华为 ADS 4 WEWA）。
> - 新增 **“扩散与多模态规划器”** 一节：Hydra-MDP 与 DiffusionDrive（NAVSIM 上 88.1 PDMS）。
> - 新增 **“超越 nuScenes：NAVSIM（v1/v2）与 Bench2Drive”** 一节，介绍 PDM Score、v2 的伪仿真 / EPDMS，以及 Bench2Drive 闭环结果；新增基准表格行。
> - GAIA 一节扩展了 **GAIA-2**（2025 年 3 月）和 **GAIA-3**（2025 年 12 月）。Tesla 一节更新了 **FSD v14** 和无安全员 Robotaxi（Austin，2026 年 1 月）。
> - 新增开放问题（推理忠实性、推理延迟、世界模型中的强化学习）和延伸阅读。

---

## 概述

端到端（end-to-end, E2E）自动驾驶指的是这样一类架构：它们学习从原始传感器观测到车辆控制输出（转向、油门和制动）的直接映射，而不需要人工设计的中间表征，例如 3D 边界框、高精地图多边形或显式运动预测模块。这一概念可追溯到 ALVINN [Pomerleau1989]（在卡内基梅隆大学校园中训练了一个从相机到转向的浅层网络），以及 DAVE-2 [Bojarski2016]（在 NVIDIA 用深度 CNN 实现了同样的思路）。现代端到端驾驶与这些先驱的不同之处在于规模（数十亿参数、数百万小时训练数据）、表征丰富度（多相机 Transformer 编码器、BEV 提升、语言条件化）和任务范围（包括复杂交互在内的完整城市驾驶，而不只是高速公路上的车道保持）。

2022–2025 年端到端驾驶的架构演进可从两个维度刻画：**结构粒度**与**监督丰富度**。早期端到端系统（CILRS [Codevilla2019]、Transfuser [Chitta2022]）使用单流主干网络，将感知与规划压缩到一次前向传播中，几乎没有中间监督。这使它们在训练分布之外很脆弱，也难以调试。2023 年的一代——以 **UniAD** [Hu2023]（CVPR 2023 最佳论文奖）、**VAD** [Jiang2023] 和 **SparseDrive** 为代表——保留了通过交叉注意力连接的任务专用查询流（检测查询、地图查询、运动查询、规划查询），在实现端到端梯度流动的同时保留了可解释的中间输出。2024–2025 年的一代引入了**语言条件推理**（DriveVLM [Tian2024]、DriveLM [Sima2024]）和**生成式世界模型**（GAIA-1 [Hu2023GAIA]、DriveDreamer、MAGICDRIVE），后者可以在给定自车动作的条件下模拟未来传感器帧，从而无需真实世界数据即可进行闭环自博弈训练。**2025–2026 年的一代**在两个方向上更进一步。其一，它建立在*预训练多模态基础模型*之上：原生多模态大模型规划器（如 EMMA [Hwang2024EMMA]）和推理型 VLA（如 Alpamayo-R1 [Wang2025Alpamayo]），后者先在推理轨迹上进行监督微调，再进行强化学习（RL）后训练。其二，它用能够表示多模态未来的*生成式轨迹解码器*（扩散 / 流匹配（flow matching））取代回归头。与此同时，评估已从 nuScenes 开环 L2 转向 NAVSIM 和 Bench2Drive。

从模块化到端到端系统的量产转变，最直观地体现在 **Tesla FSD v12**（2024）上：它用一个通过模仿学习在约 1,000 万个带标注驾驶视频片段上训练的大型神经网络，取代了约 30 万行基于规则的 C++ 代码 [Tesla2024]。其工程意义在于：FSD v12 无需模块级规格说明，无需手写的边缘情况处理程序，并随着数据增加而迅速改进——这证明了端到端架构的数据可扩展性优势，学术界早有此理论，但此前量产团队从未在如此规模上验证过。

---

## 从模块化到端到端的转变：结构分析

模块化技术栈将驾驶策略分解为若干独立优化组件的乘积。设 $x_t$ 为时刻 $t$ 的传感器观测，$s_t$ 为潜在场景状态，$a_t$ 为控制动作。模块化分解为：

$$p(a_t \mid x_t) = \int p(a_t \mid s_t) \cdot p(s_t \mid x_t)\; ds_t$$

其中 $p(s_t \mid x_t)$ 为感知模块，$p(a_t \mid s_t)$ 为规划模块，二者分别用独立的损失 $\mathcal{L}_\text{perc}$ 和 $\mathcal{L}_\text{plan}$ 优化。

这种分解的病灶在于**信息瓶颈**：$s_t$ 必须具有足够的表达力以完全决定 $a_t$，但 $x_t$ 中任何未被所选表征 $s_t$ 捕获的信息都会被永久丢弃。超出 $s_t$ 规格范围的软失效——模糊的观测、不寻常的几何、部分遮挡的智能体——会产生在下游悄然累积的误差。

端到端系统联合优化 $p(a_t \mid x_t)$：

$$\theta^* = \arg\min_\theta \mathbb{E}_{(x, a^*) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), a^*) \right]$$

梯度 $\partial \mathcal{L} / \partial x$ 同时流经所有层，使感知表征能够专门化于与规划相关的特征。其实际结果是，端到端系统会隐式学习设计者未指定的场景特征——例如影响制动距离的路面湿滑程度，或某一类行人的行为模式。

---

## UniAD：面向规划的端到端驾驶（CVPR 2023 最佳论文）

**UniAD**（*Unified Autonomous Driving*）[Hu2023UniAD, arXiv:2212.10156] 来自 OpenDriveLab（商汤科技 + 上海人工智能实验室），荣获 CVPR 2023 最佳论文奖。它是对**结构化端到端**范式最清晰的架构表述：一系列 Transformer 解码器模块，每个模块解决一个标准驾驶子任务，并通过共享的查询接口相连。

### 架构

```
相机图像（环视）
        ↓
BEV 编码器（BEVFormer 主干）
        ↓
TrackFormer  →  MapFormer  →  MotionFormer  →  OccFormer  →  PlanFormer
（检测+跟踪）   （在线高精     （多智能体        （占用流）      （自车轨迹
                 建图）         运动预测）                       规划）
```

每个模块输出一组带类型的查询嵌入并向前传递。关键在于，五个模块共享同一个 BEV 特征表征，并在每个中间输出上使用任务专用损失进行**端到端联合训练**：

$$\mathcal{L}_\text{total} = \lambda_1 \mathcal{L}_\text{track} + \lambda_2 \mathcal{L}_\text{map} + \lambda_3 \mathcal{L}_\text{motion} + \lambda_4 \mathcal{L}_\text{occ} + \lambda_5 \mathcal{L}_\text{plan}$$

### 在 nuScenes 上的性能

与此前的 SOTA（BEV-Planner 及各任务的专用模型）相比：
- 多目标跟踪 **MOTA 提升 20%**
- 在线车道建图（基于 IoU）**精度提升 30%**
- 运动预测 **minFDE 降低 38%**
- 3 s 时域的**规划 L2 误差降低 28%**
- **3s 规划 L2**：0.71 m（非端到端规划基线约为 1.0 m）

UniAD 证明了**联合优化能同时提升每个子任务**，而不只是规划——验证了“共享表征有益于所有模块”的理论论点。

---

## VAD：向量化自动驾驶（ICCV 2023）

**VAD**（*Vectorized Autonomous Driving*）[Jiang2023VAD, arXiv:2303.12077] 为端到端驾驶提出了完全向量化的场景表征，用稀疏的实例级向量表征取代栅格化的 BEV 特征图：

- **智能体实例**：表示为折线 token（采样的轨迹点）。
- **地图元素**：表示为折线 token（车道中心线、道路边界、人行横道）。
- **自车查询**：单个可学习 token，通过注意力与所有智能体和地图向量交互。

向量化方法消除了稠密 BEV 网格的高计算成本，同时通过相对位置编码保留几何关系。VAD 达到：
- 1s/2s/3s 的 **L2 误差**：0.41 / 0.70 / 1.05 m
- **平均碰撞率**：0.22%
- 在精度相当时**推理速度比 UniAD 快 4 倍**

VAD 的核心洞见在于：规划并不需要像素级 BEV 特征；智能体交互的结构拓扑才是相关信息，而向量比网格更高效地编码拓扑。

---

## DriveVLM：用于自动驾驶的视觉-语言模型

**DriveVLM** [Tian2024DriveVLM, arXiv:2402.12289] 来自清华大学 MARS Lab，将大型视觉-语言模型（VLM）集成到驾驶流水线中，以处理标准端到端网络无法显式推理的**复杂与罕见场景**。其核心架构洞见是**双系统设计**：

- **DriveVLM-Dual**：慢速的 VLM 推理系统负责场景理解和元动作生成（例如“人行横道附近有一个孩子在追球；应提前制动”）；快速的空间端到端模块负责标准的轨迹规划。
- **SUP 任务**：DriveVLM 正式将*场景理解与规划*（Scene Understanding and Planning, SUP）定义为新的基准任务，并为描述正确性和规划轨迹质量都设计了评估指标。

在 nuScenes 验证集上，DriveVLM-Dual 达到：
- **1s/2s/3s 的 L2**：0.18 / 0.34 / 0.68 m
- **平均碰撞率**：0.27%

VLM 组件对不寻常场景提供**思维链（chain-of-thought）推理**，提升了对训练分布之外新场景的少样本泛化能力——这正是纯模仿学习无法解决的长尾问题。

---

## GAIA-1：面向自动驾驶的生成式世界模型（Wayve）

**GAIA-1**（*Generative AI for Autonomy*）[Hu2023GAIA, arXiv:2309.17080] 来自 Wayve，代表了一次根本性的架构转变：GAIA-1 并不直接训练端到端策略，而是训练一个**生成式世界模型（World Model）**，可以在自车动作以及文本/图像条件下预测未来视频帧。

### 架构

GAIA-1 通过对所有输入进行 token 化，将世界建模表述为无监督序列预测：
- **视频 token**：用 VQ-VAE 将每个相机帧 token 化 → 离散 token。
- **动作 token**：自车状态（速度、转向、加速度）→ 离散化 token。
- **文本 token**：场景描述（天气、地点类型、危险）→ BPE token。

一个大型自回归 Transformer（GPT 式）在视频、动作和文本上联合预测下一个 token：

$$p(v_{t+1}, a_{t+1} \mid v_{1:t}, a_{1:t}, \text{text})$$

**主要特性**：
- **涌现的场景动力学**：尽管没有显式的物理监督，GAIA-1 仍学会了连贯的车辆运动、遮挡处理和跨帧场景一致性。
- **可控生成**：通过以不同的动作序列为条件，用户可以合成罕见场景（险情、对抗性行人、边缘光照条件）。
- **训练数据**：2019–2023 年在伦敦采集的约 4,700 小时专有驾驶视频，25 Hz；约 4.2 亿个唯一帧。
- **规模**：基础模型；后来扩展到 **90 亿参数** [Wayve2024Scaling]。

GAIA-1 本身并不是驾驶策略；它是一个**仿真器**，可用于扩充训练数据、在无需真实部署的情况下进行闭环策略评估，并可能作为基于 RL 的策略训练的奖励模型。这使它成为 CARLA 等基于物理的仿真器的神经替代方案。

### GAIA-2 与 GAIA-3（2025）

**GAIA-2** [Russell2025GAIA2]（arXiv 2503.20523，2025 年 3 月）用**潜在扩散**世界模型取代了 GAIA-1 的离散 token 自回归世界模型。它可生成最多五路空间和时间上一致的相机视频流（448×960）。生成以结构化输入为条件：自车动力学，其他智能体的位置、朝向和尺寸，道路语义，地域（英国/美国/德国），天气和时段。它还可以以外部潜变量为条件，例如 CLIP 嵌入或专有的驾驶嵌入。**GAIA-3**（2025 年 12 月）被 Wayve 介绍为约 150 亿参数的模型，训练数据量约为 GAIA-2 的 10 倍，视频 tokenizer 规模是后者的 2 倍。它被明确定位为**评估基础设施**：Wayve 报告称，在 GAIA-3 中计算的策略排名与实车道路测试得到的排名高度相关（这是公司声明，尚未经独立验证）。从 GAIA-1 到 GAIA-3 的演进反映了该领域从“世界模型作为数据生成器”到“世界模型作为验证器”的转变。

---

## 2025–2026：驾驶基础模型与推理型 VLA

### EMMA（Waymo）

**EMMA** [Hwang2024EMMA]（arXiv 2410.23262）在基于 Gemini 的多模态大语言模型之上构建端到端驾驶模型。输入是相机图像；所有其他输入（导航指令、自车历史）和输出（规划器轨迹航点、3D 框、道路图折线）都被序列化为**纯文本**。任务选择通过提示词完成。EMMA 报告了 nuScenes 上最先进的运动规划结果，以及在 Waymo Open Motion Dataset 上具有竞争力的结果和 WOD 上的纯视觉 3D 检测结果。对规划、检测和道路图任务的联合训练提升了三者，在多模态大模型内部复现了 UniAD 的多任务发现。其自述局限（帧数少、不使用激光雷达/毫米波雷达、推理成本高）定义了量产级 VLA 的研究议程。

### Alpamayo-R1 / Alpamayo 1（NVIDIA）

**Alpamayo-R1** [Wang2025Alpamayo]（arXiv 2511.00088）通过显式因果推理来应对长尾问题：

- **因果链（Chain of Causation, CoC）数据集**：由混合的自动标注 + 人在回路流水线生成的、以决策为依据且因果关联的推理轨迹。
- **架构**：由 **Cosmos-Reason** VLM 主干（NVIDIA 面向物理 AI 的推理 VLM）产生推理，再由**基于扩散的轨迹解码器**实时生成动力学可行的规划。
- **训练**：先针对推理进行 SFT，再进行 **RL 后训练**，对推理质量和推理-动作一致性给予奖励。
- **报告结果**：在困难案例上的规划精度比仅轨迹基线最多提高 12%；闭环仿真中的近距离接触减少 35%；RL 使推理质量提升 45%，推理-动作一致性提升 37%；车端道路测试延迟为 99 ms。

在 CES（2026 年 1 月 5 日）上，NVIDIA 以 **Alpamayo 1** 之名开放发布了该系列：这是一个 100 亿参数的 VLA，权重托管在 Hugging Face 上。它被定位为可微调并蒸馏到车端技术栈的*教师*模型，并与 **AlpaSim** 闭环仿真器以及超过 1,700 小时的开放驾驶数据一同发布。

### 语言引导的闭环驾驶：SimLingo

**SimLingo** [Renz2025SimLingo]（CVPR 2025）是一个用于 CARLA 的纯视觉 VLA。它同时执行驾驶、视觉问答（VQA）和指令跟随，并引入“Action Dreaming”以强制实现*语言指令与动作之间的对齐*。其前身 CarLLaVA 赢得了 2024 CARLA Autonomous Driving Challenge，这是基于 VLM 的驾驶模型首次登顶官方闭环排行榜。

### 潜在推理与量产 VLA

在 10–30 Hz 频率下，显式文本推理代价高昂。2026 年的一个趋势是**潜在推理或单步推理**：模型在紧凑的潜在空间中推理，仅在需要解释时才解码出语言。例子包括小米的 OneVL（arXiv 2604.18486）和 TacitVLA（arXiv 2610.17540），以及加速 Alpamayo 1 轨迹生成的工作（arXiv 2605.08975）。在量产方面，小鹏的 **VLA 2.0** 在单一模型中整合感知、推理和动作，并去掉了中间语言层以降低延迟。它以纯视觉方式运行在自研图灵芯片上（Robotaxi 自 2026 年 3 月起进行道路测试；计划于 2027 年面向全球交付）。**理想汽车**推出了 VLA 司机大模型（MindVLA 体系）。**华为** ADS 4 则采用 **世界引擎 + 世界行为（WEWA）** 设计，直接将传感器映射为动作，不存在语言瓶颈。云端扩散“世界引擎”生成困难场景用于训练；华为声称端到端延迟降低约 50%、通行效率提升 20%、急刹事件减少 30%（公司数据）。

---

## 扩散与多模态规划器

使用 L2/L1 回归头的模仿学习会对多个模态取平均（例如“从左侧通过”和“从右侧通过”平均后导致碰撞）。两类方法解决了这一问题：

- **对轨迹词表打分。** **Hydra-MDP** [Li2024HydraMDP]（NVIDIA）用多个头对一个大型固定候选轨迹词表打分。每个头都从基于规则的教师（PDM 子分数：碰撞、可行驶区域、TTC、进度、舒适性）以及人类示范中蒸馏得到。它赢得了 CVPR 2024 NAVSIM 挑战赛。
- **截断扩散。** **DiffusionDrive** [Liao2025DiffusionDrive]（CVPR 2025）从以聚类轨迹锚点为中心的*锚定*高斯混合分布（而非纯噪声）开始去噪，只需 **2 步去噪**。它在 NAVSIM navtest 上达到 **88.1 PDMS**，速度约 45 FPS（RTX 4090），高于 TransFuser 基线（84.0）。

同样的生成式解码器如今也出现在 VLA 内部（Alpamayo-R1 的扩散解码器；以及借鉴自机器人领域的流匹配动作头，参见 [VLA 模型](../06_robotics_and_embodied_ai/01_vla_models.md)）。

```
DiffusionDrive（截断扩散）：
  锚点 {a_k} ──► 加入少量噪声 ──► x_T  (T ≪ 1000)
  x_T ──► 去噪器(BEV/场景特征, 自车状态) ──► x_{T-1} ──► … ──► x_0
  x_0 + 每个锚点的置信度 ──► 选择轨迹
```

---

## Tesla FSD v12：量产端到端神经驾驶

截至 2026 年 10 月，Tesla FSD v12（2024 年初发布）及其后续版本仍是最受关注的量产端到端自动驾驶系统。主要技术细节：

- **输入**：八个相机（360° 环视，120–250 m 范围），推理时不使用激光雷达。
- **输出**：以约 10–20 Hz 直接输出车辆控制（转向角、油门、制动力矩）。
- **架构**：一个大型 Transformer 网络；内部架构细节未公开。该系统包含 48 个不同的神经网络组件 [FredPope2024]，但这些组件是真正相互独立，还是共享架构中的多个头，并未公开说明。
- **训练**：基于约 1,000 万个人类驾驶片段的模仿学习；人工接管作为负样本；车队数据通过离线标注提供伪标签。在 Dojo 超级计算机上训练。
- **能力提升**：与基于规则的 FSD v11 相比，FSD v12 在“自然驾驶行为”上表现出质的改进——变道更平顺、让行更好、突然制动减少。这些难以从公开数据中量化；Tesla 不公布逐个驾驶动作的评估指标。

**更新：FSD v13–v14 与 Robotaxi（2025–2026）。** FSD v14 于 2025 年 10 月开始推送，新增了避让紧急车辆、绕行受阻道路等功能。马斯克最初承诺 v14 的网络规模将扩大约 10 倍。后续报道指出，实际推送的 v14.x 系列仍运行经过改进的“小”模型（据报道约 10 亿参数），大模型被推迟到 v15；这些数字来自爱好者媒体转述的马斯克言论，而非技术文档。Tesla 的 Robotaxi 服务于 2025 年 6 月 22 日在 Austin 推出，车内配有安全监督员。2026 年 1 月 22 日，Tesla 使用专门的“unsupervised”FSD v14 分支开始提供**无安全员的公共出行**（车内无人，远程监控），并报告自 2026 年 7 月起在 Miami、Orlando 和 Tampa 提供无安全员出行。这是纯视觉端到端策略的首批无人驾驶部署，但车队规模远小于 Waymo。

**关键局限**：Tesla 不公布能够支撑学术或监管评估的正式安全统计数据（每次脱离的里程、单位里程事故率、与人类基准的比较）。其安全声明基于向 NHTSA 提交的季度事故报告和公司自愿披露。

---

## nuScenes 规划指标

nuScenes 规划基准在包含 150 个真实驾驶场景的保留集上评估端到端驾驶规划器，使用两个基于多时域预测计算的主要指标：

1. **L2 位移误差**：在 $t \in \{1, 2, 3\}$ 秒时，预测自车轨迹与专家轨迹之间的平均欧氏距离。

$$\text{L2}(t) = \frac{1}{N}\sum_{i=1}^N \|\hat{p}_i^t - p_i^{t*}\|_2$$

2. **碰撞率**：在任一预测时间步与任一标注智能体边界框相交的预测轨迹所占比例。

| 模型 | L2 @1s | L2 @2s | L2 @3s | 碰撞率（平均） |
|-------|--------|--------|--------|----------------------|
| IL baseline | ~0.80 | ~1.20 | ~1.60 | ~0.60% |
| ST-P3 [Hu2022] | 1.33 | 2.11 | 3.24 | 0.71% |
| UniAD [Hu2023] | 0.36 | 0.58 | 0.71 | 0.12% |
| VAD [Jiang2023] | 0.41 | 0.70 | 1.05 | 0.22% |
| DriveVLM [Tian2024] | 0.18 | 0.34 | 0.68 | 0.27% |

*注：nuScenes 开环规划指标因不能反映闭环驾驶质量而受到批评；仅跟随专家轨迹而不进行安全推理也能获得较低的 L2。AD-MLP [Zhai2023] 和 Li 等人 [Li2024EgoStatus]（CVPR 2024）表明，仅使用**自车状态**（速度、加速度、历史）、完全不使用相机输入的模型，在这些指标上就能达到或超过 UniAD/VAD。应将这些数字视为相对比较，而非绝对的安全指标。*

---

## 超越 nuScenes：NAVSIM（v1/v2）与 Bench2Drive

### NAVSIM

**NAVSIM** [Dauner2024NAVSIM]（NeurIPS 2024 Datasets & Benchmarks）介于开环评估与完整闭环评估之间。它构建于 OpenScene（nuPlan 的再发布版本）之上，使用 LQR 控制器和运动学自行车模型，对规划的 4 s 轨迹进行**非反应式仿真**。结果由 **PDM Score（PDMS）**评分，该分数聚合了基于规则的子指标：

$$\text{PDMS} = \underbrace{\text{NC} \times \text{DAC}}_{\text{multiplicative penalties}} \times \frac{5\,\text{TTC} + 2\,\text{C} + 5\,\text{EP}}{12}$$

其中 NC 为无责任碰撞，DAC 为可行驶区域合规性，TTC 为碰撞时间，C 为舒适性，EP 为自车进度。`navtest` 划分过滤掉了平凡场景（例如用匀速模型即可解决的场景），从而消除了大部分“自车状态捷径”。**NAVSIM v2**（“伪仿真”，CoRL 2025）加入了合成的后续观测——这些观测通过 3D 高斯泼溅从扰动后的初始状态渲染得到——从而根据规划器从自身偏离中*恢复*的能力进行评分。它还引入了**扩展 PDM 分数（Extended PDM Score, EPDMS）**，新增车道保持、行驶方向合规、交通信号灯合规和扩展舒适性等项。自 2024 年起，NAVSIM 一直作为 CVPR Autonomous Grand Challenge 端到端驾驶赛道的基准。

### Bench2Drive

**Bench2Drive** [Jia2024Bench2Drive]（NeurIPS 2024 Datasets & Benchmarks）是一个基于 CARLA v2 的闭环基准。它提供约 200 万帧带标注的官方训练集，评估覆盖 **44 种交互场景**（加塞、超车、绕行、紧急制动、让行）中的 **220 条短路线**。指标包括 Driving Score、Success Rate、Efficiency、Comfortness 以及按能力划分的“Multi-Ability”分数。其基线结果揭示了 2023 年在 nuScenes 上训练的端到端技术栈在闭环中有多么薄弱：UniAD-Base 的 Driving Score 为 45.81，VAD 为 42.35，均低于 ThinkTwice 和 DriveAdapter 等专家蒸馏方法。此后，基于 VLM 和基于扩散的规划器（SimLingo 以及许多 2025 年的方法）大幅提升了 Bench2Drive 分数。该基准现已成为学术端到端论文默认的闭环检验。

---

## 关键论文

| 论文 | 作者 | 年份 | 发表平台 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| UniAD | Hu et al. | 2023 | CVPR (Best Paper) | 面向规划的端到端；完整自动驾驶任务栈的联合优化 |
| VAD | Jiang et al. | 2023 | ICCV | 向量化场景表征；比 UniAD 快 4 倍 |
| DriveVLM | Tian et al. | 2024 | CoRL | VLM + 端到端双系统；少样本泛化能力更强 |
| GAIA-1 | Hu et al. | 2023 | arXiv | 生成式世界模型；90 亿参数；4,700 小时训练数据 |
| Transfuser | Chitta et al. | 2022 | TPAMI | 传感器融合 Transformer；强大的 CARLA 基准基线 |
| CILRS | Codevilla et al. | 2019 | ICCV | 条件模仿学习；确立了端到端基准 |
| ST-P3 | Hu et al. | 2022 | ECCV | nuScenes 上首个纯视觉端到端规划；BEV + 安全代价 |
| DriveLM | Sima et al. | 2024 | ECCV | 面向驾驶的图式视觉问答；以语言为基础的端到端规划 |
| EMMA | Hwang et al. (Waymo) | 2024 | arXiv 2410.23262 | 基于 Gemini 的多模态大模型驾驶；以文本输出轨迹、边界框和道路图 |
| Hydra-MDP | Li et al. (NVIDIA) | 2024 | arXiv / CVPR 2024 challenge | 从基于规则的评分器进行多目标蒸馏；NAVSIM 挑战赛冠军 |
| NAVSIM | Dauner et al. | 2024 | NeurIPS (D&B) | 非反应式仿真基准；PDM Score |
| Bench2Drive | Jia et al. | 2024 | NeurIPS (D&B) | 基于 CARLA v2 的多能力闭环端到端基准 |
| GAIA-2 | Russell et al. (Wayve) | 2025 | arXiv 2503.20523 | 多相机可控潜在扩散世界模型 |
| DiffusionDrive | Liao et al. | 2025 | CVPR | 截断扩散规划器；以实时速度在 NAVSIM 上达到 88.1 PDMS |
| SimLingo | Renz et al. | 2025 | CVPR | 具备语言-动作对齐的纯视觉 VLA；CARLA 挑战赛谱系 |
| Alpamayo-R1 | Wang et al. (NVIDIA) | 2025 | arXiv 2511.00088 | 推理型 VLA（因果链 + RL）；以 Alpamayo 1（100 亿参数）开放发布 |

---

## 基准性能

| 模型 | 数据集 | 指标 | 分数 | 备注 |
|-------|---------|--------|-------|-------|
| UniAD | nuScenes val | 规划 L2 @3s | 0.71 m | CVPR 2023 最佳论文 |
| VAD | nuScenes val | 规划 L2 @3s | 1.05 m | 向量化；推理更快 |
| DriveVLM | nuScenes val | 规划 L2 @3s | 0.68 m | VLM 推理；标准指标上最佳 |
| Transfuser | CARLA Town05 | 路线完成率 | 54.5% | 相机 + 激光雷达融合 |
| UniAD | nuScenes val | 平均碰撞率 | 0.12% | 2023 年端到端方法中最低；需注意自车状态问题 |
| TransFuser | NAVSIM navtest | PDMS | 84.0 | NAVSIM 参考基线 |
| DiffusionDrive | NAVSIM navtest | PDMS | 88.1 | 2 步截断扩散；约 45 FPS |
| UniAD-Base | Bench2Drive | Driving Score | 45.81 | 闭环；Bench2Drive 论文基线 |
| VAD | Bench2Drive | Driving Score | 42.35 | 闭环；Bench2Drive 论文基线 |

---

## 优缺点

| 方面 | 端到端神经网络 | 模块化技术栈 |
|--------|-----------------|---------------|
| 联合优化 | 完整梯度贯穿所有任务 | 各模块独立优化 |
| 数据可扩展性 | 随原始驾驶视频扩展 | 需要每个模块的标注 |
| 可解释性 | 中间表征不透明 | 每个阶段都有显式输出 |
| 失效模式诊断 | 困难——是哪个阶段出错？ | 模块级日志记录与回放 |
| 监管验证 | 安全论证方法尚不成熟 | 已有成熟的独立验证 |
| 对新场景的泛化 | 结合 VLM 后更好 | 受限于手写规则 |

---

## 开放问题与研究空白

- **开环与闭环之间的差距**：nuScenes 的 L2/碰撞指标采用开环评估；一个模仿专家轨迹、却无法避开新插入障碍物的模型能得到高分，但会在部署中失败。闭环评估（nuPlan、CARLA、Waymax）是必要的，但代价高昂。
- **长尾泛化**：基于模仿学习训练的端到端系统受限于训练分布；对抗性或新颖场景（黑天鹅事件）要么需要海量数据集覆盖，要么需要有原则的分布外检测。
- **模仿学习中的因果混淆**：智能体会从示范数据中学到虚假相关（例如总是在学校附近减速，只因为专家这样做，而非理解原因）。因果干预式训练是一个开放的研究方向。
- **用于实时控制的 VLM 延迟**：大型 VLM（70 亿参数以上）无法在车载硬件上以实时速率推理；DriveVLM 的双系统方法只是权宜之计，而非解决方案。需要面向自动驾驶的硬件高效 VLM 架构。
- **神经策略的安全认证**：目前尚无公认的方法能形式化地界定大型神经网络策略发生灾难性失效的概率。这是在无人监管下部署 L4 的核心障碍。
- **多智能体交互建模**：当前端到端系统将其他智能体建模为需要避开的障碍物，而非有目标的理性智能体；显式的心智理论和多智能体博弈论规划尚处于起步阶段。
- **基于 RL 的端到端的奖励信号**：模仿学习收敛到平均行为；RL 可以针对安全关键场景进行优化，但需要可靠的奖励信号。将 GAIA-1 等世界模型用作 RL 环境是一个有前景但尚不成熟的方向。到 2026 年，VLA 的 RL 后训练（Alpamayo-R1）和基于世界模型的评估（GAIA-3、AlpaSim）已使其成为主流，但针对不完美仿真器的奖励投机（reward hacking）是一个现实的担忧。
- **推理的忠实性**：推理型 VLA 会输出人类可读的因果解释，但 2026 年的研究开始探测这些解释是否真正决定了轨迹。例子包括 *“Is VLA Reasoning Faithful? Probing Safety of Chain-of-Causation”*（arXiv 2605.17268），以及发现传感器扰动会暴露推理脆弱性的 *“Lost in Fog”*（arXiv 2605.21446）。对安全论证而言，不忠实的解释比没有解释更糟。
- **推理延迟与收益**：显式思维链会为每次决策增加数百个 token。潜在推理或单步推理（OneVL、TacitVLA）能否在不增加延迟的情况下保留长尾收益，仍未解决。
- **基准饱和与有效性**：NAVSIM navtest 和 Bench2Drive 的分数在 2025 年迅速上升。伪仿真（NAVSIM v2）和基于生成式世界模型的评估旨在缩小与反应式闭环测试之间的剩余差距，但目前都尚未针对真实世界的安全结果得到验证。

---

## 延伸阅读

- [arXiv:2212.10156 UniAD](https://arxiv.org/abs/2212.10156) — CVPR 2023 最佳论文
- [arXiv:2309.17080 GAIA-1](https://arxiv.org/abs/2309.17080) — Wayve 世界模型
- [arXiv:2402.12289 DriveVLM](https://arxiv.org/abs/2402.12289) — VLM 集成
- [OpenDriveLab/UniAD GitHub](https://github.com/OpenDriveLab/UniAD) — 官方代码
- [arXiv:2401.08658 2022-2023 端到端规划综述](https://arxiv.org/abs/2401.08658) — 全面综述
- [Wayve 扩展 GAIA-1 的博客文章](https://wayve.ai/thinking/scaling-gaia-1/) — 90 亿参数结果
- [arXiv:2410.23262 EMMA](https://arxiv.org/abs/2410.23262) — Waymo 基于多模态大模型的端到端驾驶
- [arXiv:2511.00088 Alpamayo-R1](https://arxiv.org/abs/2511.00088) — 基于因果链的推理型 VLA
- [arXiv:2503.20523 GAIA-2](https://arxiv.org/abs/2503.20523) — 多相机可控世界模型
- [NAVSIM GitHub](https://github.com/autonomousvision/navsim) — 基准代码、v1/v2 与排行榜
- [Bench2Drive GitHub](https://github.com/Thinklab-SJTU/Bench2Drive) — 闭环多能力基准
- [DiffusionDrive GitHub](https://github.com/hustvl/DiffusionDrive) — 截断扩散规划器
