# 评估指标

> **英文原文：** [cv-knowledge-base/11_datasets_and_benchmarks/04_evaluation_metrics.md](../../cv-knowledge-base/11_datasets_and_benchmarks/04_evaluation_metrics.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 中级
> **相关章节：**
> - [生成式评估与安全](../10_generative_vision/04_evaluation_and_safety.md)
> - [核心数据集](./00_core_datasets.md)
> - [多模态数据集](./03_multimodal_datasets.md)
> - [目标检测](../02_core_tasks/01_object_detection.md)

> **🆕 2026年10月更新：**
> - 新增 **cgF1**（分类门控 F1，classification-gated F1），这是随 SAM 3 / SA-Co（2025年11月）引入的开放词汇概念分割指标。它将图像级存在性校准（IL_MCC）与正样本 micro-F1 相乘。
> - 新增**端到端驾驶指标**：NAVSIM 的 PDMS → **EPDMS**（NAVSIM v2，CoRL 2025），以及 Waymo 的**评分者反馈分数（Rater Feedback Score）**（WOD-E2E，2025）。
> - 新增 **LMM 评估协议**：循环评估、**pass@k 与 pass^k**（ZeroBench）、MMStar 的多模态增益/泄漏，以及用于数值型空间答案的 MRA（VSI-Bench）。
> - 更新生成式指标：将 **FD<sub>DINOv2</sub>** 作为 Inception 特征的推荐替代，并加入面向视频生成的 **VBench++ / VBench-2.0**。
> - 新增**具身策略评估**：扰动下的成功率（LIBERO-Plus）和分布式真实世界竞技场（RoboArena）。

---

## 概述

评估指标是计算机视觉衡量进展的量化工具，其精确定义比新手通常意识到的更重要：IoU 阈值、平均方式或匹配规则上的细微差异都可能改变排名、误导比较。一名合格的研究者不仅要知道基准使用*哪个*指标，还要知道它*究竟如何*计算，因为只有在完全相同的协议下，报告的数字才具有可比性。本文按主要任务类别（分类、检测、分割、生成、检索和深度）整理标准指标，给出其数学定义，并说明使简单比较变得危险的陷阱。

一个贯穿始终的主题是：**每个指标都是一个代理**，它捕捉了质量的某些方面而忽略了其他方面，优化指标并不等同于解决任务。mAP 奖励排序但不奖励校准；FID 衡量的是特征空间中的分布相似性，而非语义正确性或组合性；mIoU 对类别取平均，却掩盖了单个类别的失效。最重要的指标也常常通过测试时技巧、集成或对基准过拟合而被*钻空子*，因此审慎的读者会把排行榜数字视为必要但不充分的证据（生成任务的情况尤其棘手，参见[生成式评估与安全](../10_generative_vision/04_evaluation_and_safety.md)）。

2025–2026 年有两项进展改变了指标格局。第一，随着开放词汇系统和具身系统的成熟，指标开始**惩罚校准错误、奖励可接受的替代答案**，而不再只对照单一真值。cgF1 根据模型是否正确判断某个概念存在来门控定位质量。Waymo 的评分者反馈分数和 NAVSIM 的 EPDMS 依据人类评分或基于规则的可接受性来为规划轨迹打分，而不是对照单条记录路径。第二，对于生成式多模态模型，**采样与协议成为指标的一部分**。pass@k 与 pass^k、选项的循环置换、帧预算和"思考"预算对核心数字的影响可能超过一代模型的差距。省略这些信息的报告不具可比性。

---

## 分类与检索

**Top-1 / Top-5 准确率** — 真实标签是最高预测（或位于前 5 个预测之内）的图像所占比例。对于类别不平衡的设定使用**精确率/召回率/F1**；**平均精度（Average Precision, AP）** = 精确率-召回率曲线下的面积。对于检索，**Recall@K**（R@1、R@5、R@10）衡量正确项是否出现在前 K 个结果中；**mAP** 对所有查询的 AP 取平均。

## 检测：mAP

检测的核心指标是在某个交并比（Intersection-over-Union, IoU）阈值下的**平均精度均值（mean Average Precision）**：

```
IoU(A, B) = |A ∩ B| / |A ∪ B|
# 若 IoU(pred, gt) ≥ 阈值且类别匹配，则该预测为真正例
AP = ∫₀¹ precision(recall) d(recall)     # 每个类别的 PR 曲线下面积
mAP = mean over classes of AP
# COCO mAP 在 IoU 阈值 0.50:0.05:0.95 上取平均（即 AP@[.5:.95]）
```

COCO 的主要指标在十个 IoU 阈值（0.50–0.95）上对 AP 取平均，因此比 PASCAL VOC 的 AP@0.5 更严格（参见[目标检测](../02_core_tasks/01_object_detection.md)）。

## 分割：mIoU、Dice、PQ

**平均交并比（mean Intersection-over-Union, mIoU）**对各类别 IoU 取平均，是语义分割的标准指标。**Dice 系数**（= 像素级 F1）在医学影像中更受青睐。**全景质量（Panoptic Quality, PQ）**统一了语义评估与实例评估：

```
PQ = (Σ_{(p,g)∈TP} IoU(p,g)) / (|TP| + ½|FP| + ½|FN|)
   = SQ × RQ      # 分割质量（Segmentation Quality）× 识别质量（Recognition Quality）
```

## 开放词汇概念分割：cgF1

AP 并不适合*可提示概念分割*：一个文本提示在一张图像中可能没有、只有一个或有多个匹配实例。AP 在置信度阈值上取平均，并跨图像对预测排序，因此不会惩罚一个对并不存在的概念自信地分割出某物的模型。为此，SAM 3 / SA-Co [Carion2025] 引入了**分类门控 F1（classification-gated F1, cgF1）**：

```
pmF1   = 实例掩码的正样本 micro-F1，仅在概念存在的（图像, 短语）对上计算
         （固定工作点下的定位质量）
IL_MCC = 图像级二元判断"该概念是否存在？"在所有正负样本对上的
         Matthews 相关系数（校准）
cgF1   = 100 × pmF1 × IL_MCC
# 在 SA-Co/Gold 上，真值来自 3 名独立标注者；
# 对每个数据点采用最有利的那份标注（oracle 设定）。
```

SA-Co/Gold 上的参考值：人类 72.8，SAM 3 54.1，OWLv2* 24.6，Gemini 2.5 13.0。对于视频，SA-Co 同时报告 cgF1 和一个 HOTA 变体（**pHOTA**）。

## 生成：FID、IS、CLIPScore、FVD

**Fréchet Inception 距离（Fréchet Inception Distance, FID）**比较真实图像与生成图像的 Inception 特征分布：

```
FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2})
# 越低越好；假设特征服从高斯分布
```

**Inception Score（IS）**通过标签熵衡量质量与多样性；**CLIPScore** 衡量文生图中的图文对齐程度；**FVD** 借助视频特征网络将 FID 扩展到视频。这些指标因与人类判断相关性差而受到批评，从而催生了学习式/人类偏好指标（GenEval、ImageReward；参见[生成式评估与安全](../10_generative_vision/04_evaluation_and_safety.md)）。

**特征提取器的选择（FD<sub>DINOv2</sub>）：** 一项针对 41 个生成模型的研究 [Stein2023] 发现，Inception-V3 特征无法反映人类判断下扩散模型的真实感。该研究建议在 Fréchet 距离及相关指标（精确率/召回率、密度/覆盖率、核距离）中用 **DINOv2** 特征替换 Inception。**FD<sub>DINOv2</sub>** 如今在 ImageNet 生成任务中常与 FID 一同报告（例如 EDM2 时代的论文）。同一研究还发现，现有的记忆检测指标在复杂数据集上会失效。

**视频生成评测套件：** VBench 将视频质量分解为层级化的维度，并对每个维度做人类一致性检验。**VBench++**（TPAMI 2025）增加了图生视频和可信度评估。**VBench-2.0**（2025）针对*内在忠实度*（物理、常识、人体保真度、可控性、创造力），而非表层像素质量。基于人类两两投票的视频竞技场（包括 VBench-2.0 和 VBench-I2V 竞技场）对这些自动评分形成补充。

## 深度、光流、姿态

**深度**：AbsRel、RMSE 以及阈值准确率 δ<1.25。**光流**：端点误差（End-Point Error, EPE）。**姿态**：PCK、基于 OKS 的 AP（COCO 关键点）、MPJPE（3D，毫米）。**跟踪**：MOTA、IDF1、HOTA。

## 驾驶与具身策略指标

**PDMS / EPDMS（NAVSIM）。** NAVSIM 的 PDM 分数将乘性安全门控与加权的舒适性和行进进度项相结合。NAVSIM v2 的**扩展 PDMS** [Cao2025] 增加了行驶方向和交通信号灯合规门控、车道保持和扩展舒适性项，以及一个*人类过滤器*：当人类驾驶员也犯了同样的违规时，便不施加该项惩罚：

```
EPDMS = Π_{m ∈ {NC, DAC, DDC, TLC}} filter_m(agent, human)
        × ( Σ_{m ∈ {TTC, EP, HC, LK, EC}} w_m · filter_m(agent, human) ) / Σ w_m
# 权重：EP = 5，TTC = 5，LK = 2，HC = 2，EC = 2
# filter_m(agent, human) = 1.0（若 m(human) = 0），否则为 m(agent)
# NC：无责碰撞，DAC：可行驶区域合规，DDC：行驶方向
# 合规，TLC：交通信号灯合规，EP：自车进度，TTC：碰撞时间
```

在 v2 中，分数在原始场景和合成扰动的后续场景上分两阶段聚合（*伪仿真*）。这在保持开环的同时近似了闭环评估。

**评分者反馈分数（Waymo WOD-E2E）。** 对于被评分的帧，人类评分者为最多三条候选未来轨迹给出 0–10 分。预测轨迹根据其与这些已评分备选轨迹的匹配程度计分，因此规划器选择一种不同但可接受的机动时不会被扣分。位移误差（ADE/FDE）仍作为次要诊断指标。

**机器人策略。** 主流指标仍是**任务成功率**（长时程任务常给予部分进度分）。2025–2026 年有两个限定条件成为标准。其一是*受控扰动下的成功率*：LIBERO-Plus 报告了沿七个扰动维度的成功率，揭示出从约 95% 到不足 30% 的下降。其二是*分布式真实世界比较*：RoboArena 在多个机构共享的真实机器人平台上评估远程部署的策略，而不是在单一实验室的设置上评估。

## 多模态大模型评估协议

- **循环评估（MMBench、OpenCompass 空间排行榜）：** 只有当模型在选项的每一种循环置换下都回答正确时，该多项选择题才算正确。这消除了位置偏差，并减少了侥幸猜中。
- **pass@k 与 pass^k（ZeroBench）：** pass@k 表示 k 次采样中*任意*一次正确即算解出该题；pass^k 要求 k 次采样*全部*正确。在 ZeroBench 上，2026 年的最佳模型 pass@5 为 30.0，而 pass^5 为 13.0，因此两者分别衡量能力与可靠性。
- **多模态增益 / 泄漏（MMStar）：** MG = LVLM 带图像时的分数 − 同一 LVLM 不带图像时的分数。ML = LVLM 在*不带*图像时超出其基础 LLM 的幅度，表明基准样本泄漏进了多模态训练。两者合起来量化了一个"视觉"分数究竟在多大程度上依赖视觉。
- **平均相对准确率（Mean Relative Accuracy, MRA，VSI-Bench）：** 对于数值型答案（距离、尺寸、计数），不采用精确匹配，而是在一系列相对误差容忍阈值上对准确率取平均，从而为近似正确的空间估计给出部分得分。
- **必须报告的协议变量：** 视频的帧数（或 fps）与分辨率、字幕/音频访问、推理/"思考"预算、工具使用、采样次数以及答案抽取方法。由于这些选择的不同，第三方的重新评估经常与模型卡中的自报数字相差数个点。

---

## 关键参考文献

| 指标 | 提出/标准化者 | 年份 | 领域 | 备注 |
|--------|---------------------------|------|--------|------|
| mAP (COCO) | Lin et al. (MS-COCO) | 2014 | 检测 | AP@[.5:.95] |
| mIoU | PASCAL VOC / Cityscapes | 2008+ | 分割 | 各类别 IoU 的均值 |
| Panoptic Quality | Kirillov et al. | 2019 | 全景分割 | SQ × RQ |
| FID | Heusel et al. | 2017 | 生成 | Inception 特征的 Fréchet 距离 |
| FVD | Unterthiner et al. | 2018 | 视频生成 | 基于视频特征的 FID |
| HOTA | Luiten et al. | 2021 | 跟踪 | 平衡检测与关联 |
| FD<sub>DINOv2</sub> | Stein et al. | 2023 | 生成 | 以 DINOv2 特征替代 Inception |
| MG / ML | Chen et al. (MMStar) | 2024 | LMM 评估 | 多模态增益 / 泄漏 |
| PDMS → EPDMS | Dauner et al. → Cao et al. (NAVSIM) | 2024 → 2025 | 自动驾驶规划 | 带门控与人类过滤的伪闭环分数 |
| Rater Feedback Score | Waymo (WOD-E2E) | 2025 | 自动驾驶规划 | 人类评分的备选轨迹（0–10） |
| VBench-2.0 | Zheng et al. | 2025 | 视频生成 | 内在忠实度维度 |
| cgF1 / pHOTA | Carion et al. (SAM 3 / SA-Co) | 2025 | 概念分割 | IL_MCC × 正样本 micro-F1 |
| pass^k | Roberts et al. (ZeroBench) 等 | 2025 | LMM 评估 | 全部 k 次的可靠性 vs pass@k 能力 |

---

## 优缺点（指标设计）

| 方面 | 优点 | 缺点 |
|--------|------|------|
| mAP | 考虑排序，对阈值稳健（COCO） | 忽略校准；解释复杂 |
| mIoU/PQ | 标准化，可洞察各类别 | 掩盖稀有类别的失效；对边界敏感 |
| FID/FVD | 分布层面的质量 | 与人类判断相关性差；特征提取器偏差 |
| 人类偏好 | 与感知质量一致 | 成本高、噪声大、难以复现 |
| cgF1 | 惩罚为不存在的概念生成的幻觉掩码；单一工作点 | 与 AP 不可比；依赖所选阈值；较新，尚未广泛采用 |
| EPDMS / RFS | 奖励可接受的行为，而非模仿单一日志 | 规则权重（EPDMS）和评分者主观性（RFS）包含设计选择；与真实世界安全性的相关性尚未证实 |
| pass@k / pass^k | 区分能力与可靠性 | 需要 k 倍推理；为保证可比性必须报告 k |

---

## 开放问题与研究空白

- **与人类判断相关性的差距。** FID/IS/CLIPScore 与人类判断相关性较弱；更好的自动生成指标仍未解决（参见[生成式评估与安全](../10_generative_vision/04_evaluation_and_safety.md)）。换用 DINOv2 特征有所帮助，但并未解决组合正确性或物理正确性问题。VBench-2.0 式的分解套件以多个只经过部分验证的维度取代了单一标量。
- **基准钻空子。** 测试时增强、集成和过拟合会在没有真实进展的情况下抬高指标。在 LMM 时代，对应的手段是思考预算、工具使用、best-of-k 采样以及针对基准的提示设计。
- **组合性评估。** 指标很少检验属性绑定、计数或关系正确性。
- **饱和。** 许多指标在标准数据集上已接近上限，丧失了区分能力。
- **协议不一致。** 细微的实现差异（裁剪、阈值）破坏了论文之间的可比性。对于视频大模型，帧采样和字幕访问扮演着同样的角色。
- **考虑校准的指标。** cgF1 展示了如何将存在性校准纳入开放词汇感知的定位指标。在 VLM 视觉定位、VQA 弃答以及具身系统的"我做不到"决策方面，类似的考虑校准的指标基本缺失。
- **整体性/智能体评估。** 目前没有衡量 VLA/世界模型实用性（物理合理性、任务成功）的标准指标（参见[优缺点与路线图](../06_robotics_and_embodied_ai/07_pros_cons_roadmaps.md)）。已有进展包括：扰动下的成功率（LIBERO-Plus）、分布式真实世界竞技场（RoboArena）和基于人类偏好的规划分数（RFS）。一个能从离线评估预测真实世界部署可靠性的指标尚不存在。

---

## 延伸阅读

- [MS-COCO (arXiv:1405.0312)](https://arxiv.org/abs/1405.0312) — 检测/分割指标
- [Panoptic Segmentation (arXiv:1801.00868)](https://arxiv.org/abs/1801.00868) — PQ 指标
- [FID / GANs Trained by TTUR (arXiv:1706.08500)](https://arxiv.org/abs/1706.08500) — Fréchet Inception 距离
- [HOTA (arXiv:2009.07736)](https://arxiv.org/abs/2009.07736) — 高阶跟踪准确率
- [GenEval (arXiv:2310.11513)](https://arxiv.org/abs/2310.11513) — 组合式文生图评估
- [Stein et al. (2023) — Exposing flaws of generative model evaluation metrics (arXiv:2306.04675)](https://github.com/layer6ai-labs/dgm-eval) — FD<sub>DINOv2</sub> 以及 15 种指标 × 8 种编码器
- [VBench / VBench++ / VBench-2.0 (arXiv:2503.21755)](https://github.com/Vchitect/VBench) — 视频生成评测套件
- [SAM 3 SA-Co/Gold 评估器（cgF1）](https://github.com/facebookresearch/sam3/tree/main/scripts/eval/gold)
- [NAVSIM EPDMS 定义](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md) — 以及 [Pseudo-Simulation (arXiv:2506.04218)](https://arxiv.org/abs/2506.04218)
- [Waymo Open Dataset — E2E 驾驶与评分者反馈分数代码](https://github.com/waymo-research/waymo-open-dataset)
- [MMStar (arXiv:2403.20330)](https://github.com/MMStar-Benchmark/MMStar) — 多模态增益/泄漏
- [ZeroBench (arXiv:2502.09696)](https://github.com/jonathan-roberts1/zerobench) — pass@k 与 pass^k 的报告
