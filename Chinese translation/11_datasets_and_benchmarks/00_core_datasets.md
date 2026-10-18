# 核心视觉数据集：分类、检测、分割与场景理解

> **英文原文：** [cv-knowledge-base/11_datasets_and_benchmarks/00_core_datasets.md](../../cv-knowledge-base/11_datasets_and_benchmarks/00_core_datasets.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 中级 / 高级
> **相关章节：** [图像分类](../02_core_tasks/00_image_classification.md) · [目标检测](../02_core_tasks/01_object_detection.md) · [语义分割](../02_core_tasks/02_semantic_segmentation.md) · [评估指标](./04_evaluation_metrics.md)

> **🆕 2026年10月更新：**
> - 新增 **SA-1B** 以及 Meta 的 **SA-Co**（Segment Anything with Concepts，2025年11月）。SA-Co 是由 SAM 3 数据引擎构建的开放词汇（open-vocabulary）概念分割基准，包含约 270K 个唯一概念。SAM 3 在 SA-Co/Gold 上达到 54.1 cgF1，人类参考水平为 72.8。
> - 新增 DINOv3 所用的网络规模精选预训练语料 **LVD-1689M** 与 **SAT-493M**（2025年8月）。它们延续了从带标注的 ImageNet-21K 预训练转向精选、无标注图像池的趋势。
> - 新增 **RF100-VL**（Roboflow + CMU，2025），这是一个由 100 个数据集组成的多领域检测基准，实时检测器如今会将其与 COCO 一并报告（例如 RF-DETR-2XL 报告以 17.2 ms 延迟达到 60.1 COCO AP）。
> - 更新了 COCO 与 LVIS 的 SOTA 表述：Co-DETR 在 COCO test-dev 上达到 66.0 AP，SAM 3 在 LVIS 上零样本 mask AP 达到 48.5，而 DINO-X 为 38.5。同时注明 Papers With Code 排行榜现已归档。

---

## 概述

现代计算机视觉研究生态围绕少数几个经典数据集展开。这些数据集同时充当训练语料、评估基准，也在隐含地定义哪些任务"重要"。理解它们的设计选择（标注方法、类别体系、数据划分与评估协议）对任何博士层次的研究者都至关重要，因为数据集的特性往往比模型架构本身更能解释已发表的数字。

本文涵盖的数据集跨越近二十年的数据集设计演进：从 ImageNet 基于 WordNet 同义词集（synset）的类别体系（Deng et al., 2009），到 COCO 的实例级标注（Lin et al., 2014）、LVIS 的大词表长尾分布（Gupta et al., 2019），以及 ADE20K 中的全景统一（Zhou et al., 2017）。每一个设计决策（标注什么、以何种粒度、采用何种共识机制、覆盖何种图像分布）都嵌入了会传导为模型偏差的假设。例如，仅依赖 ImageNet top-1 准确率的研究者，会系统性地忽视模型在稀有物体类别、纹理偏移图像和分布外场景上的失效模式。

第二个主题是从单任务标注向多任务标注的转变。COCO 的开创性在于在同一批图像上同时提供边界框、分割掩码、关键点和描述文本。Open Images V7 与 LVIS 通过关系标注和密集的长尾覆盖进一步推进了这一点。现代预训练工作利用这些多重标注数据集，对检测、分割和描述头进行联合监督，这一范式是 Florence-2、InternImage 等系统的核心。理解哪些数据集支持哪些标注类型，是设计任何多任务学习流程的前提知识。

第三个主题主导了 2025–2026 年：**模型在环（model-in-the-loop）数据引擎**与**无标注精选图像池**的兴起。SA-1B（2023）和 SA-Co（2025）主要由历代 SAM 模型标注、再经人工核验，其规模与概念覆盖如今已超过任何人工众包所能产出的数据。DINOv3 的 LVD-1689M 同样表明，精心筛选的无标注图像池可以取代 ImageNet-21K，成为骨干网络预训练的默认数据集。经典基准（ImageNet、COCO、ADE20K）越来越成为*仅用于评估*的工件，常常以冻结骨干网络的方式运行，而前沿模型背后的训练数据则是内部数据或机器标注数据。

---

## ImageNet

**全称：** ImageNet Large Scale Visual Recognition Challenge（ILSVRC）
**引用：** [Deng2009] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*. [Russakovsky2015] Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. *IJCV*.

完整的 ImageNet 数据库包含超过 1400 万张图像，按 WordNet 名词层级组织，覆盖 20,000 多个同义词集。ILSVRC 基准使用精选的 1,000 类子集：128 万训练图像 / 5 万验证图像 / 10 万测试图像。每个训练类别约有 1,300 张图像；验证集和测试集每类各 50 张图像。

**标注：** ILSVRC-CLS 中每张图像只有一个图像级标签；ILSVRC-DET 为 200 个物体类别增加了边界框。ImageNet-21K（完整超集）常用于预训练。

**主要划分：** ILSVRC-1K 为训练集（约 128 万）、验证集（5 万）、测试集（10 万）。ImageNet-21K 有约 1420 万张图像，覆盖 21,841 个类别。

**基准意义：** 2012 年 AlexNet 的结果（top-5 错误率 15.3%，此前最佳为 26.2%）引爆了深度学习革命。截至 2024 年，基于 ViT 的模型在 ImageNet-1K 上的 top-1 准确率超过 90%，该数据集在粗粒度分类上被视为"已解决"，促使社区转向检验鲁棒性和分布外泛化能力的更难变体（ImageNet-A、-R、-Sketch、ObjectNet）。

**2025–2026 年现状：** 微调后的 ImageNet-1K top-1 准确率自 2022 年以来几乎没有变化（约 91%）。它的角色已经转变：如今主要作为冻结表征的*诊断*工具（对 DINOv3 [Simeoni2025] 等骨干网络做线性或 k-NN 探测），以及鲁棒性变体的来源，而不再用于对架构排名。验证集中的标注错误（由 [Beyer2020] 记录）使有意义的上限远低于 100%。

---

## MS-COCO

**全称：** Microsoft COCO: Common Objects in Context
**引用：** [Lin2014] Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*.

COCO 包含 32.8 万张图像，标注了 80 个物体类别、250 万个实例，每张图像配有 5 条描述文本。经典的 COCO 2017 划分为 11.8 万训练 / 5 千验证 / 4.1 万测试图像。物体实例同时标注了边界框和高质量多边形分割掩码（实例分割）。该数据集还为人物实例提供 17 关键点骨架标注（COCO-Keypoints），以及覆盖 80 个"thing"类别和 53 个"stuff"类别的全景分割标注。

**评估协议：** 检测采用 COCO AP（在 IoU 阈值 0.5:0.05:0.95 上取平均）评估，比 Pascal VOC 的 mAP@0.5 更严格。自 2015 年以来，COCO 排行榜推动了大部分检测与实例分割研究。

**众包设计：** 标注通过分阶段的 Amazon Mechanical Turk 流程收集，实例分割由标注员点击边界多边形完成。标注员间一致性与分层质量控制在 [Lin2014] 中有描述。

**描述文本标注：** 32.8 万张图像中每张都有 5 条独立撰写的描述，共计约 150 万条参考描述，用于图像描述评估（CIDEr、BLEU-4、METEOR、SPICE）。

**排行榜现状（2026）：** 闭集 COCO 检测已接近上限。Co-DETR 搭配 3.04 亿参数的 ViT-L 于 2023 年在 test-dev 上报告了 66.0 box AP，此后的系统只带来了边际提升。研究重心已转向两种设定。其一是 (i) **实时检测**，DETR 系列检测器如今已胜过 YOLO 系列检测器。例如 RF-DETR-2XL 报告以 17.2 ms 达到 60.1 AP<sub>50:95</sub>，RF-DETR-M 以 4.4 ms 达到 54.7 AP，所有数字均由 Roboflow 内部重新测量。其二是 (ii) **零样本 / 开放词汇迁移**，SAM 3 在以类别名称提示、未做 COCO 专门训练的情况下报告了 53.6 的 COCO box AP [Carion2025]。

---

## PASCAL VOC

**全称：** PASCAL Visual Object Classes Challenge
**引用：** [Everingham2010] Everingham, M. et al. (2010). The Pascal Visual Object Classes (VOC) Challenge. *IJCV*.

PASCAL VOC 于 2005–2012 年举办，定义了早期的目标检测基准。VOC 2007 包含 9,963 张图像（5,011 训练/验证 + 4,952 测试），20 个类别共 24,640 个标注物体。VOC 2012 包含 22,531 张图像（11,540 训练/验证 + 10,991 测试）。VOC07+12 联合训练集（约 1.65 万张图像）是 COCO 出现之前的标准检测预训练集。

**标注：** 边界框 + 类别标签；后续版本增加了分割掩码。20 个类别涵盖人、动物（鸟、猫、牛、狗、马、羊）、交通工具和室内物品。

**遗产：** VOC mAP@0.5 仍为历史对比而被报告。20 类设定刻意保持较小规模，因此稀有类别的失效模式无法显现，这一局限正是 LVIS 直接要解决的问题。

---

## Open Images

**全称：** Open Images V7（Google）
**引用：** [Kuznetsova2020] Kuznetsova, A. et al. (2020). The Open Images Dataset V4. *IJCV*. V7 于 2022 年发布。

Open Images V7 覆盖约 900 万张图像，具有多层次标注：
- 190 万张图像上 600 个物体类别的 1600 万个边界框
- 94.4 万张图像上 350 个类别的 270 万个实例分割掩码
- 1,466 种关系三元组的 330 万条视觉关系标注
- 140 万张图像上涵盖 5,827 个类别的 6640 万个点级标签
- 20,638 个类别的 6140 万个图像级标签

**层级类别体系：** 类别按层级组织，允许在不同具体程度上进行评估。Open Images 还提供"Localized Narratives"标注（与鼠标轨迹同步的口述描述），支持密集指代表达和密集描述研究。

**训练/验证/测试划分：** 共 900 万张图像，检测任务的标注子集划分为训练集（174 万）、验证集（4.1 万）和测试集（12.5 万）。

---

## ADE20K

**全称：** ADE20K Scene Parsing Dataset（MIT CSAIL）
**引用：** [Zhou2017] Zhou, B. et al. (2017). Scene Parsing through ADE20K Dataset. *CVPR 2017*.

ADE20K 在 25,562 张图像（20,210 训练、2,000 验证、3,352 测试）上提供 150 个类别（stuff 与 thing 统一）的密集像素级语义标注。图像来自 SUN 和 Places 数据库。平均每张图像包含 19.5 个物体实例和 10.5 个不同的语义类别。直到 2021 年左右，该基准一直是语义分割的主要评估场所；Swin-T、SegFormer 和 Mask2Former 等模型都将 ADE20K mIoU 作为主要指标。

**标注深度：** 与 COCO（二值前景掩码）不同，ADE20K 采用层级化的物体-部件标注：每个物体都可以有标注的子部件（例如作为"汽车"一部分的"车轮"）。这使超越扁平分割的场景解析研究成为可能。

---

## Cityscapes

**全称：** The Cityscapes Dataset for Semantic Urban Scene Understanding
**引用：** [Cordts2016] Cordts, M. et al. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. *CVPR 2016*.

Cityscapes 提供 5,000 张精细标注图像（2,975 训练 / 500 验证 / 1,525 测试）和 20,000 张粗标注图像，均从 50 座城市中行驶的车辆上拍摄。图像分辨率为 2048×1024，标注 30 个类别（评估 19 个）。精细标注使用多边形工具，平均每张图像约需 90 分钟。可评估语义分割（mIoU）、实例分割（AP）和全景分割。

**领域特定性：** 作为城市驾驶数据集，Cityscapes 是街景理解的经典基准。模型必须应对尺度变化（远处行人与近处公交车）、运动模糊以及天气/光照变化。还提供视频序列供时序方法使用。

**粗标注：** 额外的 2 万张粗标注图像旨在支持半监督和自监督预训练研究。

---

## LVIS

**全称：** Large Vocabulary Instance Segmentation
**引用：** [Gupta2019] Gupta, A., Dollar, P., & Girshick, R. (2019). LVIS: A Dataset for Large Vocabulary Instance Segmentation. *CVPR 2019*.

LVIS 为 16.4 万张图像（来自 COCO）标注了 1,203 个类别的高质量实例分割掩码。它刻意保留了自然图像中物体的长尾频率分布：337 个"稀有"类别（1–10 张训练图像）、461 个"常见"类别（11–100 张）和 405 个"频繁"类别（>100 张）。总标注量：220 万个实例。训练划分：10 万张图像 / 120 万个标注。

**对长尾学习的意义：** 在 COCO 上训练的标准检测器在 LVIS 稀有类别上会崩溃。LVIS 推动了大量关于重复因子采样 [Gupta2019]、均衡损失 [Tan2020] 和联邦损失的研究，与任何处理开放词汇检测的生产系统直接相关。

**评估：** AP_r、AP_c、AP_f 按频率分层评估性能，揭示模型在代表性不足类别上的行为。

**开放词汇用法（2025–2026）：** LVIS 已成为概念提示式检测器和分割器的标准*零样本*测试平台。SAM 3 在未经 LVIS 训练的情况下报告了 48.5 的 LVIS mask AP，而 DINO-X 为 38.5。OWLv2 达到 43.4，但其部分训练数据来自 LVIS。在 LVIS 上微调的有监督 Co-DETR 在 LVIS val 上达到 68.0 box AP 和 56.0 mask AP（2023）。

---

## Places

**全称：** Places365 / Places205
**引用：** [Zhou2018] Zhou, B. et al. (2018). Places: A 10 Million Image Database for Scene Recognition. *TPAMI*.

Places365-Standard：365 个场景类别，共 180 万张训练图像，另有每类 50 张验证图像和 900 张测试图像。Places365-Challenge 将训练集扩展到约 800 万张图像。完整的 Places 数据库超过 1000 万张图像。该数据库按语义场景类别（如"修道院"、"厨房"、"高速公路"）而非物体类别组织，因此在场景级任务的迁移学习研究中与 ImageNet 形成互补。

**跨数据集迁移：** 一些研究表明，在 Places 上预训练的模型迁移到卫星图像分类和医学场景标注时，比 ImageNet 预训练模型效果更好，这推动了迁移学习中的数据集选择分析。

---

## SA-1B 与 SA-Co（Segment Anything 数据引擎）

**引用：** [Kirillov2023] Kirillov, A. et al. (2023). Segment Anything. *ICCV 2023*. [Carion2025] Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). SAM 3: Segment Anything with Concepts. Meta Superintelligence Labs.

**SA-1B**（2023）包含约 1100 万张经授权、做过隐私保护的图像和约 11 亿个类别无关（class-agnostic）掩码，由三阶段模型在环数据引擎（辅助人工 → 半自动 → 全自动）生成。它定义了*可提示分割（promptable segmentation）*任务，其掩码不带语义标签。

**SA-Co**（Segment Anything with Concepts，2025年11月）是 SAM 3 背后的数据与基准发布，它加入了*语义*：每张图像或每段视频都配有简短名词短语，与短语匹配的每个实例都被穷尽地标注掩码并赋予唯一物体 ID。没有匹配物体的短语充当困难的*负提示（negative prompts）*。官方发布中的关键事实：
- 该基准涵盖**约 270K 个唯一概念**，Meta 称这一数量是此前开放词汇分割基准的 50 倍以上。它有两个图像划分（**SA-Co/Gold**，三重标注；**SA-Co/Silver**）和一个视频划分（**SA-Co/VEval**）。
- 训练数据引擎结合 SAM/MLLM 提议与 AI 和人工核验器，自动标注了**超过 400 万个唯一概念**。
- 主要指标是 **cgF1**，即一种分类门控 F1（classification-gated F1），它将图像级存在性校准（IL_MCC）与定位上的正样本 micro-F1 相乘（参见[评估指标](./04_evaluation_metrics.md)）。
- SA-Co/Gold 上的参考结果（实例分割，cgF1）：人类 72.8；**SAM 3 54.1**；OWLv2* 24.6；DINO-X 21.3；Gemini 2.5 13.0。SAM 3 是一个 8.48 亿参数的模型，由共享同一视觉编码器的检测器和跟踪器组成。2026年3月又发布了多目标跟踪更快的 SAM 3.1 检查点。

**为何重要：** SA-Co 把开放词汇分割从 1,203 类的 LVIS 问题变成了带显式负样本、约 25 万概念的问题。在这种变化下，在 COCO 风格基准上看起来很强的基于 VLM 的检测器会丢失大部分准确率。"模型在环引擎 + 仅对困难样本做人工核验"的做法已成为新的大规模标注工作的模板。SAM 3D（2025年11月）将其扩展到 3D（参见[3D 与机器人数据集](./02_3d_and_robotics_datasets.md)）。

---

## LVD-1689M 与 SAT-493M（精选预训练图像池）

**引用：** [Simeoni2025] Siméoni, O., Vo, H. V., Seitzer, M., et al. (2025). DINOv3. arXiv:2508.10104 (Meta FAIR).

DINOv3（2025年8月）在 **LVD-1689M**（约 17 亿张精选网络图像）上进行无标注训练，另一个变体则在 **SAT-493M**（约 4.93 亿张卫星图像）上训练。其 70 亿参数的 ViT 教师模型被蒸馏为 ViT-S/B/L/H+ 和 ConvNeXt 学生模型。这些数据集本身并未公开发布。它们的意义在于证明：*围绕种子分布进行检索/聚类式的精选*可以把无标注预训练扩展到 ImageNet-21K 之外。据报告，冻结的 DINOv3 特征无需微调即可在众多密集任务上超越专用的最先进模型。这使 ADE20K、COCO 和 NYUv2 越来越成为针对冻结骨干网络的*探测*基准。下游发布包括 COCO2017 检测头、ADE20K 分割头以及 Canopy Height Maps v2 模型（2026年3月），均构建在冻结的 7B 骨干网络之上。

---

## RF100-VL（面向 VLM 的多领域检测基准）

**引用：** [Robicheaux2025] Robicheaux, P., Popov, M., Madan, A., Robinson, I., Nelson, J., Ramanan, D., & Peri, N. (2025). Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models. Roboflow / CMU.

RF100-VL 是托管在 Roboflow Universe 上、以 COCO-JSON 格式分发的 100 个真实世界检测数据集的精选集合，其中大多数领域在网络预训练数据中代表性不足。它有一个少样本变体（**RF100-VL-FSOD**）和一个 20 数据集子集（**RF20-VL**），因此既能评估常规微调检测器，也能评估 VLM 风格的零样本和少样本检测。它迅速成为实时检测器在 COCO 之外报告的*第二个维度*。例如 RF-DETR 同时报告 COCO AP 和 RF100-VL AP，因为 COCO 上的提升越来越难以迁移到分布外领域。

---

## 关键论文 / 数据集

| 名称 | 作者/机构 | 年份 | 会议/期刊 | 关键统计 / 贡献 |
|---|---|---|---|---|
| ImageNet ILSVRC | Deng, Dong, Socher, Li, Fei-Fei (Stanford) | 2009 | CVPR | 1420 万张图像、2.1 万个同义词集（ILSVRC：128 万张 / 1K 类）；引爆深度学习革命 |
| MS-COCO | Lin, Maire, et al. (Microsoft Research) | 2014 | ECCV | 32.8 万张图像、250 万个实例、80 个类别；引入 AP@0.5:0.95 |
| PASCAL VOC 2012 | Everingham et al. | 2012 | IJCV | 2.25 万张图像、20 个类别；COCO 之前的标准检测基准 |
| Open Images V7 | Kuznetsova et al. (Google) | 2022 | IJCV | 900 万张图像、600 个边界框类别、350 个掩码类别、视觉关系 |
| ADE20K | Zhou et al. (MIT CSAIL) | 2017 | CVPR | 2.55 万张图像、150 个语义类别、层级部件标注 |
| Cityscapes | Cordts et al. (Daimler et al.) | 2016 | CVPR | 5K 精细 + 20K 粗标注图像；2048×1024 城市场景，30 个类别 |
| LVIS | Gupta, Dollar, Girshick (FAIR) | 2019 | CVPR | 16.4 万张图像、1,203 个类别、220 万个实例；长尾分布 |
| Places365 | Zhou, Lapedriza et al. (MIT) | 2018 | TPAMI | 180 万张训练图像、365 个场景类别；最大的场景识别数据库 |
| ImageNet-21K | Deng et al. / Ridnik et al. (2021) | 2009/2021 | CVPR/arXiv | 1420 万张图像、2.18 万个类别；ViT 类模型偏好的预训练集 |
| SA-1B | Kirillov et al. (Meta AI) | 2023 | ICCV | 约 1100 万张图像、约 11 亿个类别无关掩码；模型在环数据引擎 |
| SA-Co (SAM 3) | Carion, Gustafson, Hu et al. (Meta) | 2025 | Meta 技术报告 | 约 270K 概念的开放词汇分割基准（Gold/Silver/VEval）；400 万+ 自动标注概念；cgF1 指标 |
| LVD-1689M / SAT-493M (DINOv3) | Siméoni et al. (Meta FAIR) | 2025 | arXiv | 约 17 亿张精选网络图像 / 约 4.93 亿张卫星图像，用于无标注 7B ViT 预训练（未公开） |
| RF100-VL | Robicheaux, Popov et al. (Roboflow/CMU) | 2025 | arXiv | 100 个多领域检测数据集 + 少样本（FSOD）变体；COCO 的分布外补充 |

---

## 基准对比表

| 数据集 | 图像数（训练） | 类别 | 标注类型 | 主要任务 | 评估指标 |
|---|---|---|---|---|---|
| ImageNet-1K | 1.28M | 1,000 | 图像级类别 | 分类 | Top-1 / Top-5 Acc |
| MS-COCO 2017 | 118K | 80 things + 53 stuff | 边界框 + 掩码 + 描述 | 检测 / 分割 / 描述 | AP@0.5:0.95 |
| PASCAL VOC 2012 | 11.5K | 20 | 边界框 + 分割掩码 | 检测 | mAP@0.5 |
| Open Images V7 | 1.74M（已标注） | 600（检测） | 边界框 + 掩码 + 关系 | 检测 / 分割 / VRD | mAP |
| ADE20K | 20.2K | 150 | 密集像素 | 语义分割 | mIoU |
| Cityscapes | 2.975K 精细 | 19（评估） | 密集像素 + 实例 | 语义 / 实例分割 | mIoU / AP |
| LVIS v1 | 100K | 1,203 | 实例掩码 | 实例分割 | AP, AP_r/c/f |
| Places365 | 1.8M | 365 | 图像级类别 | 场景分类 | Top-1 / Top-5 |
| SA-1B | ~11M | 无（类别无关） | 掩码 | 可提示分割 | mIoU（提示式） |
| SA-Co/Gold | 仅评估 | ~270K 概念（完整基准） | 短语 → 全部实例掩码 + 负样本 | 概念分割 / 检测 | cgF1 (IL_MCC × pmF1) |
| RF100-VL | 100 个数据集（各异） | 按数据集而定 | 边界框 | 多领域检测 | AP<sub>50:95</sub>（平均） |

---

## 优缺点

| 数据集 | 优势 | 局限 |
|---|---|---|
| ImageNet-1K | 经典；预训练模型生态庞大；偏差已被充分理解 | 纹理偏差；每张图像单一标签；不代表真实世界分布；接近饱和 |
| MS-COCO | 多任务标注；生态丰富；严格的 AP 指标能暴露定位质量 | 80 个类别遗漏稀有物体；COCO 图像经过筛选，并非自然分布 |
| LVIS | 真实的长尾分布；高质量掩码 | 稀有类别的每类训练集很小；难以从零训练 |
| Open Images | 规模巨大；包括关系在内的多类型标注 | 众包标签噪声更大；核验方法与 COCO 不同 |
| ADE20K | 深度层级标注；场景 + 物体 + 部件 | 绝对规模小（2 万训练）；室内/室外场景覆盖不均 |
| Cityscapes | 高分辨率、高质量精细标注；真实城市场景 | 单一领域（欧洲城市）；5K 精细图像限制模型容量 |
| Places365 | 最大的场景数据库；365 个多样类别 | 仅图像级标注；无空间标注 |
| SA-Co | 概念词表巨大；显式负提示；图像 + 视频划分 | 机器在环标签继承了 SAM/MLLM 的偏差；仅支持名词短语提示（不含复杂指代表达） |
| RF100-VL | 领域多样；暴露对 COCO 的过拟合；有少样本协议 | 100 个社区数据集的标注质量参差不齐；平均 AP 掩盖了各领域的失效 |

---

## 开放问题与研究空白

- **长尾与开放词汇评估：** LVIS 揭示了模型在 1,203 个类别上的失效，而真实世界系统面对的是数以百万计的类别。SA-Co（2025）把词表提升到约 270K 个概念并加入负提示。即便如此，最佳模型（SAM 3，在 SA-Co/Gold 上 54.1 cgF1）仍远低于人类一致性水平（72.8），通用 VLM 的得分则低得多（Gemini 2.5：13.0）。长指代表达、组合属性和关系仍在该基准的名词短语范围之外。
- **机器标注的真值：** 随着数据引擎（SA-1B、SA-Co）取代人工标注，基准可能衡量的是与*上一代模型*的一致性。审计模型在环标签的协议（例如 SA-Co/Gold 中的三重人工标注和"在多位标注者中取最优"的评分）仍然是临时性的。
- **数据集偏差与伪相关：** ImageNet 模型依赖纹理线索 [Geirhos2019]。测量与纠正分布偏移的系统性工具仍是开放领域，尤其是在合成数据越来越多地进入训练流程的情况下。
- **标注噪声量化：** 众包数据集中的真值标签包含系统性错误（例如 COCO 实例边界）。用于衡量标注噪声对模型性能影响的基准尚不成熟。
- **多标签与组合式真值：** 大多数基准只分配一个主导类别，而真实场景具有组合语义。GQA 和 Visual Genome 指向了更丰富的真值，但组合式场景图的可扩展标注仍未解决。
- **时序与视频扩展：** 静态图像基准无法衡量时序一致性。视频目标分割基准（DAVIS、YouTube-VOS）虽已存在，但规模不及图像领域的同类基准。
- **地理与人口多样性：** ImageNet、COCO 和 Places 的数据不成比例地来自西方语境。近期工作（GeoDE、Dollar-Sign 数据集）对此进行了量化，但大规模修正仍未解决。
- **基准饱和与古德哈特定律：** 一旦 ImageNet-1K 这样的基准实际上饱和（top-1 >90%），继续优化就可能是在过拟合评估，而不是提升真实世界能力。新的评估范式（如模型汤、多样化测试套件）是活跃的研究方向。多领域套件（RF100-VL）和冻结骨干探测协议（DINOv3 风格）是当前的应对方式。
- **排行榜基础设施：** Papers With Code 曾是这些数据集事实上的公共 SOTA 追踪器，如今已不再维护。其评估表只以静态归档形式保存（Hugging Face 上的 `pwc-archive`）。排行榜现分散在基准所有者网站、Hugging Face Spaces 和自行报告的模型卡中，因此"当前 SOTA"的说法更难核实，协议差异也更不易察觉。

---

## 延伸阅读

- [ImageNet 网站与 ILSVRC 结果存档](https://www.image-net.org/)
- [COCO 基准排行榜](https://cocodataset.org/#detection-leaderboard)
- [Papers With Code — ImageNet 基准](https://paperswithcode.com/sota/image-classification-on-imagenet)（已归档；数据转储见 [paperswithcode-data](https://github.com/paperswithcode/paperswithcode-data)）
- [Beyer et al. (2020) — Are we done with ImageNet? arXiv:2006.07159](https://arxiv.org/abs/2006.07159)
- [Gupta et al. (2019) — LVIS 论文 arXiv:1908.03195](https://arxiv.org/abs/1908.03195)
- [Open Images V7 数据概况](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [SAM 3 与 SA-Co 基准 — 官方仓库](https://github.com/facebookresearch/sam3)
- [Kirillov et al. (2023) — Segment Anything (SA-1B) arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- [DINOv3（LVD-1689M / SAT-493M）— arXiv:2508.10104 及代码](https://github.com/facebookresearch/dinov3)
- [RF100-VL 基准仓库](https://github.com/roboflow/rf100-vl)
- [RF-DETR — 在 COCO 与 RF100-VL 上评测的实时检测器](https://github.com/roboflow/rf-detr)
