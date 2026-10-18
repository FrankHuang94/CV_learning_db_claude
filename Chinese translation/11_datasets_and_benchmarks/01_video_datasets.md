# 面向动作识别、时序推理与第一人称感知的视频数据集

> **英文原文：** [cv-knowledge-base/11_datasets_and_benchmarks/01_video_datasets.md](../../cv-knowledge-base/11_datasets_and_benchmarks/01_video_datasets.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 中级 / 高级
> **相关章节：** [视频理解](../02_core_tasks/08_video_understanding.md) · [核心数据集](./00_core_datasets.md) · [3D 与机器人数据集](./02_3d_and_robotics_datasets.md) · [评估指标](./04_evaluation_metrics.md)

> **🆕 2026年10月更新：**
> - 新增前沿实验室如今会报告的 **MLLM 时代视频基准**：Video-MME（CVPR 2025）、Video-MMMU、MotionBench（CVPR 2025）、LongVideoBench 和 LVBench。前沿模型的自报分数目前在 Video-MME 上约为 85–88%，在 Video-MMMU 上约为 87.6%（Gemini 3 Pro，2025年11月）。
> - 新增**空间视频基准与数据集**：VSI-Bench（CVPR 2025 Oral），以及 Cambrian-S（2025年11月）发布的 VSI-590K 和 VSI-SUPER。
> - 新增**第一人称（egocentric）数据集**：HD-EPIC（CVPR 2025）和 EgoLife（CVPR 2025）。EgoDex 见 3D/机器人文件。
> - 新增**大规模视频生成与视频-语言语料**：Panda-70M、OpenVid-1M（ICLR 2025）、Koala-36M（CVPR 2025），以及 Meta 的 PE-Video 和 PLM-Video-Human（2025）。
> - 更新 SSv2 的最先进水平：V-JEPA 2 以冻结注意力探测（frozen attentive probe）达到 77.3%（2025年6月），V-JEPA 2.1 于 2026年3月发布。另补充了关于基准饱和与帧采样不可比性的说明。

---

## 概述

视频理解基准大致经历了三代演进。第一代（UCF-101、HMDB-51、Sports-1M，2011–2014）确立了基本的动作识别范式，但使用的是多样性有限的短剪辑片段。第二代（Kinetics-400/600/700、Something-Something v2、ActivityNet，2016–2019）大幅扩展了片段数量和语义丰富度，并揭示出模型可以通过利用静态外观线索而非运动在 Kinetics 上取得高准确率，这促成了 Something-Something 偏向时序推理的设计。第三代（Ego4D、EPIC-KITCHENS-100、AVA、Charades，2018–2022）将重点转向未剪辑视频、第一人称感知、密集时空标注和长视频理解，这些挑战直接源于具身智能（Embodied AI）和 AR/VR 应用的需求。

一个反复出现的方法论张力在于外观偏向型基准与运动偏向型基准之间的区别。在 Kinetics 上，将最先进的图像模型逐帧应用即可达到约 65% 的 top-1 准确率，与许多视频专用基线相当，因为场景上下文（厨房 → 烹饪，球场 → 篮球）对动作具有很强的预测性。Something-Something v2 通过使用模板化的动作标签（"Moving [object] closer to [object]"）明确消除了这一捷径，这些标签要求理解运动的方向和效果，而不仅仅是场景。这一设计选择使两个基准从根本上互补，也解释了为何单一架构在不加修改的情况下很少能在两者上同时取得最佳成绩。

第三个关键主题是长视频与未剪辑视频理解。ActivityNet、Charades 和 Ego4D 都包含分钟级到小时级的视频，任务要求在很大的上下文窗口中对动作进行时序定位。这仍是方法开发的活跃领域，内存高效的 Transformer、层级架构以及视频-语言模型（Video-LLaMA、TimeChat）都在竞相处理长序列。

**第四代**（2024–2026）改变了"视频基准"的含义。被评估的系统如今通常是多模态大语言模型而非分类器，基准则是*问答套件*而非带标签的片段集合，例如 Video-MME、Video-MMMU、MotionBench、LongVideoBench 和 LVBench。这些套件规模较小（数百到数千个问答对）、由人工撰写，旨在规避单帧捷径和语言先验。同样的转变还催生了建立在 3D 扫描场景之上的*空间*视频基准（VSI-Bench、VSI-SUPER），用于检验模型能否从一段漫游视频中建立持久的心理地图。与此同时，*训练*侧分化为两个分支：(i) 面向文生视频的带描述片段语料（Panda-70M、OpenVid-1M、Koala-36M），以及 (ii) 面向视频大模型的人工标注细粒度视频问答与描述数据（PLM-Video-Human）。经典识别数据集（Kinetics、SSv2、EPIC-KITCHENS）如今主要作为 V-JEPA 2 等自监督视频模型的冻结编码器探测基准而存在。

---

## Kinetics-400 / 600 / 700

**引用：** [Kay2017] Kay, W. et al. (2017). The Kinetics Human Action Video Dataset. arXiv:1705.06950 (Kinetics-400). [Smaira2020] Smaira, L. et al. (2020). A Short Note on the Kinetics-700-2020 Human Action Dataset. arXiv:2010.10864.

由 DeepMind 开发的 Kinetics 系列是主流的大规模动作识别基准：

- **Kinetics-400：** 24 万–30 万个 10 秒 YouTube 片段，400 个人类动作类别，每类 ≥400 个片段（平均 683 个）。划分：约 24 万训练 / 2 万验证 / 4 万测试。
- **Kinetics-600：** 扩展至 48 万个片段、600 个类别，每类 ≥600 个片段（平均 762 个）。新增类别包括细粒度的体育动作。
- **Kinetics-700：** 约 65 万个片段、700 个类别，每类 ≥700 个片段（平均 906 个）。包含更多细粒度区分。
- **Kinetics-700-2020：** 更新版本，刷新了划分以应对 URL 链接失效问题。

所有版本都使用分辨率不一的 10 秒片段。类别涵盖人-物交互（演奏乐器、烹饪）、人-人交互（握手、拥抱）以及细粒度体育动作。一个关键局限是约 10–20% 的 URL 会随时间失效，这意味着由于下载不完整，不同实验室发表的数字未必可比。

**主要用途：** 为迁移到下游任务而预训练骨干网络（I3D、SlowFast、TimeSformer、Video Swin）。Kinetics-400 top-1 准确率是视频分类模型的标准报告指标。

---

## Something-Something v2

**引用：** [Goyal2017] Goyal, R. et al. (2017). The "something something" video database for learning and evaluating visual common sense. *ICCV 2017*.

Something-Something v2（SSv2）包含 220,847 个通过众包收集的短视频片段（2–6 秒，平均 4.0 秒），涵盖 174 个按模板组织的细粒度物理交互类别（例如 "Dropping [something] into [something]"、"Pretending to pick [something] up"）。该数据集由 1,300 名众包工作者创建。

**主要划分：** 168,913 训练 / 24,777 验证 / 27,157 测试。

**时序推理要求：** 由于类别标签取决于运动的*方向*和*结果*（而不仅是出现的物体），图像级特征的泛化效果很差。SSv2 是评估时序动态理解的主要基准。约 70–77% 的 top-1 准确率（例如 MTV、VideoMAE-v2）与 Kinetics-400 上接近 90% 的成绩形成鲜明对比，体现了时序推理能力上的差距。

**2025–2026 年更新：** 在互联网规模视频上预训练的自监督视频编码器 V-JEPA 2 [Assran2025] 报告了**冻结注意力探测下 77.3% 的 SSv2 top-1**，此前冻结探测的最佳结果为 69.7%（InternVideo2-1B）。V-JEPA 2 还在另一个以运动为中心的基准 Diving48 上报告了 90.2%。该模型于 2025年6月发布，V-JEPA 2.1（2026年3月）加入了密集特征目标。SSv2 如今主要用作运动理解的冻结表征探测，而非完整微调的排行榜。

---

## AVA（Atomic Visual Actions，原子视觉动作）

**引用：** [Gu2018] Gu, C. et al. (2018). AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions. *CVPR 2018*.

AVA 对 437 段 15 分钟的好莱坞电影片段进行了密集标注，包含 80 个原子视觉动作类别。动作在时空上被定位（以 1 Hz 的关键帧边界框标注），共得到 159 万个动作标签，每个人物实例经常同时有多个标签。人物在各片段间被跟踪，以支持时序关联。

**关键设计：** "原子"动作被定义为单一概念的基本动作（"站立"、"与……交谈"、"吃"），而非复合活动标签。这检验了对同时发生的并发活动进行推理的能力，这是活动识别系统已知的弱点。

**AVA-Kinetics：** 后续扩展将 AVA 风格的标注关联到 Kinetics 片段上，提供了更大规模的时空定位训练数据。

---

## ActivityNet

**引用：** [Heilbron2015] Heilbron, F.C. et al. (2015). ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding. *CVPR 2015*.

ActivityNet v1.3 包含 19,994 个未剪辑的 YouTube 视频（总计约 849 小时），覆盖 200 个活动类别，平均每类 137 个视频。每个视频平均有 1.41 个活动实例。划分：10,024 训练 / 4,926 验证 / 5,044 测试，平均视频时长 117 秒。

**支持的任务：** 时序活动检测（在未剪辑视频中定位动作片段）、活动分类和密集视频描述。ActivityNet Captions 为该数据集补充了密集的时序描述，在 2 万个视频上共有 10 万条自然语言描述。

**局限：** 活动标签相对粗粒度（与 Something-Something 相比）；许多类别（如"打排球"）可从静态帧识别，降低了对运动理解的要求。

---

## Charades

**引用：** [Sigurdsson2016] Sigurdsson, G.A. et al. (2016). Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding. *ECCV 2016*.

Charades 包含 9,848 个众包录制的日常室内活动视频（平均 30 秒），有 157 个动作类别和 66,500 个带时序标注的动作区间，另有 46 个物体类别的 41,104 个物体标签。视频由来自 3 个大洲的 267 名参与者按类似剧本的指示录制，因此较为自然但略带脚本化。还包含 27,847 条视频文本描述。

**Charades-STA：** 一个增加时序定位查询的扩展，将自然语言句子与精确的视频时间区间配对，支持时序语句定位研究。

---

## Ego4D

**引用：** [Grauman2022] Grauman, K. et al. (2022). Ego4D: Around the World in 3,000 Hours of Egocentric Video. *CVPR 2022*.

Ego4D 是规模最大、最多样的第一人称视频数据集，由 13 所大学组成的国际联盟在 9 个国家的 74 个地点采集。它包含 3,670 小时的第一人称视频，由 855 名以上的佩戴者在多样的日常生活场景（家庭、户外、工作场所、休闲）中拍摄。

**基准套件：** Ego4D 附带五项基准任务：
1. **情景记忆（Episodic Memory）**（自然语言查询、Moments in Time、物体状态变化）
2. **预测（Forecasting）**（下一个交互物体、预判）
3. **手-物交互（Hand & Object Interactions）**
4. **视听说话人分离（Audio-Visual Diarization）**
5. **社交互动（Social Interaction）**

**规模优势：** Ego4D 比此前任何第一人称数据集大 20 倍以上（EPIC-KITCHENS 约为 100 小时）。环境、文化背景和活动的多样性使其成为可泛化第一人称表征学习的主要资源。

---

## EPIC-KITCHENS

**引用：** [Damen2018] Damen, D. et al. (2018). Scaling Egocentric Vision: The EPIC-KITCHENS Dataset. *ECCV 2018*. [Damen2020] Damen, D. et al. (2020). Rescaling Egocentric Vision. arXiv:2006.13256 (EPIC-KITCHENS-100).

EPIC-KITCHENS-55（2018）包含 55 小时第一人称烹饪视频，32 名参与者，1150 万帧，3.96 万个动作片段，45.42 万个物体边界框。EPIC-KITCHENS-100（2020）扩展到 4 个国家 45 名参与者、97 个厨房的 100 小时视频，并提供动词与名词的多实例动作识别标签（例如 "wash pan"、"cut cucumber"）。

**独特标注：** 动作被分解为动词 + 名词对而非复合标签，从而能够评估组合泛化能力。EPIC-KITCHENS 是烹饪领域中细粒度第一人称动作识别、动作预判和动作检测的主要基准。

**预判任务 SOTA（2025）：** V-JEPA 2 在 EK-100 动作预判上报告了 39.7% 的 recall@5，此前最佳为 27.6%（PlausiVL）。预判问题仍远未解决。

**HD-EPIC（CVPR 2025）：** [Perrett2025] 由布里斯托大学牵头的 EPIC 团队的后续工作。它录制了无脚本的第一人称厨房活动，标注更加密集、细致，并配有一个前沿视频大模型也难以应对的 VQA 基准。它被定位为 EPIC-KITCHENS-100 的细粒度、长尾补充。

**EgoLife（CVPR 2025）：** [Yang2025] 六名参与者在一周的共同生活中佩戴 Meta Aria 眼镜，同时配有同步的第三人称摄像头和毫米波（mmWave）传感器。它面向*超长上下文*的第一人称助理（周级记忆与检索），并发布了 EgoIT-99K 指令数据以及 EgoGPT/EgoRAG 基线。

---

## HowTo100M

**引用：** [Miech2019] Miech, A. et al. (2019). HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. *ICCV 2019*.

HowTo100M 包含 122 万个教学类 YouTube 视频，总时长约 15 年，通过与视频片段对齐的自动语音识别（ASR）转录文本，被拆分为 1.36 亿个 4 秒的片段-描述对。视频涵盖 12 个领域（烹饪、手工、个人护理、园艺等）的 2.3 万个任务类别。

**意义：** HowTo100M 确立了将带噪的语音转录用作视频-文本预训练免费监督的范式。它促成了 MIL-NCE [Miech2020]，后者在零样本文本-视频检索上取得了很强的迁移效果，并影响了后续的大规模预训练数据集（WebVid、InternVid）。ASR 描述中的高噪声（错位、无关解说）推动了后来基于过滤的方法。

---

## WebVid

**引用：** [Bain2021] Bain, M. et al. (2021). Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval. *ICCV 2021*.

WebVid-2.5M（初版）和 WebVid-10M 包含从素材视频网站抓取、带描述性替代文本描述的视频片段。WebVid-2.5M：约 250 万个片段，平均时长 18 秒。WebVid-10M：1070 万个片段，总计约 52,000 小时，平均每段 18 秒。由于描述由专业素材描述人员撰写，其质量高于 ASR 转录文本。

**可用性说明：** WebVid-10M 曾被广泛用于视频-语言预训练（例如 InstructBLIP-Video、Video-ChatGPT），但由于版权问题，该数据集的公开获取在 2023 年受到限制。InternVid（Wang et al., 2023）和 Panda-70M 作为后继者出现。

---

## 视频生成与视频大模型训练语料（2024–2026）

后 WebVid 时代将视频-文本数据分为两类：*面向生成*的语料需要高审美质量、干净的镜头边界和较长的密集描述；*面向理解*的则是指令数据。

- **Panda-70M** [Chen2024]（CVPR 2024，Snap）：7000 万个片段，由多个跨模态教师模型生成描述，再由微调过的检索模型为每个片段选出最佳描述。它曾是文生视频预训练的主要开放 WebVid 后继者。
- **OpenVid-1M** [Nan2025]（ICLR 2025）：100 万个精选的高审美片段，配有表现力强的描述。其 1080p 子集 **OpenVidHD-0.4M** 包含 40 万个片段。它可直接使用或作为质量微调的补充，并被 Pyramid Flow、Allegro 等开源视频生成模型引用。
- **Koala-36M** [Wang2025]（CVPR 2025，快手/可灵）：3600 万个片段的语料，具有改进的时序切分、平均超过 200 词的结构化描述，以及用于过滤的学习式**视频训练适用性评分（Video Training Suitability Score, VTSS）**。它认为切分方式、描述细节和适用性过滤比原始规模更能决定数据集质量。
- **PE-Video（PVD）** 与 **PLM-Video-Human** [Cho2025]（Meta，2025年4月）：PVD 包含 100 万个以运动为中心的视频，其中 12 万个带有人工核验的标注。PLM-Video-Human 提供人工标注的细粒度问答（FGQA）、区域-时序定位和区域（密集）描述数据。两者均与完全开放的 PerceptionLM 和 **PLM-VideoBench** 评估套件一同发布，以减少对专有模型蒸馏的依赖。
- **VSI-590K** [Yang2025b]（2025年11月）：约 59 万个为*空间感知*而整理的空间定位视频问答样本，随 Cambrian-S 一同发布，同时还发布了一个 300 万样本的开放视频指令集合（Cambrian-S-3M，2025年12月）。

---

## MLLM 时代的视频基准（2024–2026）

这些是规模较小、精心构建的问答套件，用于衡量多模态大模型处理视频的能力。其核心数字在很大程度上取决于*帧采样预算*、是否使用字幕/音频以及"思考"预算，因此只有在协议一致时才可比较。

- **Video-MME** [Fu2025]（CVPR 2025）：900 个新采集的视频（254 小时，11 秒至 1 小时），配有 2,700 个人工撰写的多项选择问答对，覆盖 6 个领域和 30 个子领域。它分为短（<2 分钟）、中（4–15 分钟）、长（30–60 分钟）三档，每档都在有字幕和无字幕两种设定下评估。它已成为*业界标准*的长上下文视频指标，在 GPT-4.1、Gemini 2.5 Pro（84.8%）、GPT-5 和 Gemini 3 Pro 的发布中都被引用。在官方提交排行榜上，排名最高的开放模型（video-SALMONN 2+，72B，2025年9月）在无字幕时为 79.7%，有字幕时为 81.6%。Moonshot 的 Kimi K2.5 报告（2026年1月）给出 Kimi K2.5 为 87.4%，并对 Gemini 3 Pro 重新评估得到 88.4%。因此该基准对前沿系统而言已接近饱和。
- **Video-MMMU** [Hu2025]：300 个大学水平的讲座视频，覆盖 6 个学科，每个视频 3 道问答（共 900 道）。问题从感知、理解和应用三个层次考察*知识获取*能力。Gemini 3 Pro 报告 87.6%（2025年11月），Kimi K2.5 报告 86.6%。
- **MotionBench** [Hong2025]（CVPR 2025，智谱/清华）：涵盖六类能力的细粒度运动级问答，使用网络视频、公开数据集视频和 Unity 合成视频。它还发布了 5,000 个带人工细粒度运动描述的视频。前沿模型得分约 70%（据 Moonshot 报告，Kimi K2.5 为 70.4，Gemini 3 Pro 为 70.3），表明运动感知落后于语义层面的视频问答。
- **LongVideoBench** [Wu2024] 与 **LVBench** [Wang2024b]：长上下文交错式视频-语言问答，LVBench 的视频长达两小时。2026 年初报告的前沿分数在 75 分上下至约 80% 之间。
- **空间视频：VSI-Bench 与 VSI-SUPER。** **VSI-Bench** [Yang2025a]（CVPR 2025 Oral，NYU/Yale/Stanford）包含 5,000 多个问答对，来自 ScanNet、ScanNet++ 和 ARKitScenes 验证集场景的 288 段第一人称漫游视频。其八项任务涵盖构型推理、测量估计和时空推理。**VSI-SUPER** [Yang2025b]（2025年11月）将其扩展到任意长度的视频流，包含空间回忆与持续计数任务，其设计使得单纯扩展上下文的蛮力方法会失效。它倡导*预测式感知*（基于"惊奇度"的记忆），而非更长的上下文窗口。基于图像的空间套件（MMSI-Bench、ERQA）见[多模态数据集](./03_multimodal_datasets.md)。

---

## 关键论文 / 数据集

| 名称 | 作者/机构 | 年份 | 会议/期刊 | 关键统计 / 贡献 |
|---|---|---|---|---|
| Kinetics-400 | Kay et al. (DeepMind) | 2017 | arXiv | 约 30 万个片段、400 个类别、10 秒片段；主流预训练基准 |
| Kinetics-700 | Smaira et al. (DeepMind) | 2020 | arXiv | 约 65 万个片段、700 个类别；扩展的类别体系 |
| Something-Something v2 | Goyal et al. (TwentyBN) | 2017 | ICCV | 22 万个片段、174 个类别；需要时序推理 |
| AVA | Gu et al. (Google) | 2018 | CVPR | 437 段电影片段、80 个原子动作、159 万个时空标签 |
| ActivityNet v1.3 | Heilbron et al. | 2015 | CVPR | 2 万个视频、200 个类别、849 小时；未剪辑视频的时序检测 |
| Charades | Sigurdsson et al. (CMU/Allen AI) | 2016 | ECCV | 9.8K 个视频、157 个动作、6.65 万个时序区间；室内脚本化 |
| Ego4D | Grauman et al. (Meta AI + consortium) | 2022 | CVPR | 3,670 小时第一人称视频、74 个地点、5 个基准套件 |
| EPIC-KITCHENS-100 | Damen et al. (Univ. Bristol) | 2020 | arXiv | 100 小时烹饪视频、45 名参与者、动词+名词分解 |
| HowTo100M | Miech et al. (INRIA) | 2019 | ICCV | 122 万个视频、1.36 亿个片段-描述对，基于 ASR；视频-文本预训练 |
| WebVid-10M | Bain et al. (Oxford) | 2021 | ICCV | 1070 万个片段、5.2 万小时；高质量素材描述 |
| Panda-70M | Chen et al. (Snap) | 2024 | CVPR | 7000 万个片段，多教师模型生成描述；文生视频领域的 WebVid 后继者 |
| OpenVid-1M | Nan et al. (NJU) | 2025 | ICLR | 100 万个高审美片段；40 万个 1080p（OpenVidHD） |
| Koala-36M | Wang et al. (Kuaishou/Kling) | 2025 | CVPR | 3600 万个片段、>200 词的结构化描述、VTSS 适用性过滤 |
| PE-Video / PLM-Video-Human | Cho et al. (Meta FAIR) | 2025 | arXiv | 100 万个视频（12 万个人工核验）；人工标注的 FGQA / 区域-时序数据；PLM-VideoBench |
| Video-MME | Fu et al. | 2025 | CVPR | 900 个视频（254 小时）、2,700 道问答；短/中/长三档 ± 字幕；业界标准的视频 MLLM 指标 |
| Video-MMMU | Hu et al. (NTU S-Lab / LMMs-Lab) | 2025 | arXiv | 300 个讲座视频、900 道问答；知识获取（感知 → 理解 → 应用） |
| MotionBench | Hong et al. (Zhipu/Tsinghua) | 2025 | CVPR | 细粒度运动问答，6 类能力；另含 5K 个运动描述视频 |
| VSI-Bench | Yang, Yang, Gupta, Han, Fei-Fei, Xie (NYU) | 2025 | CVPR (Oral) | 基于 288 段 3D 扫描场景视频的 5K+ 问答；视觉-空间智能 |
| VSI-590K / VSI-SUPER (Cambrian-S) | Yang, Yang et al. (NYU/Stanford) | 2025 | arXiv | 59 万条空间问答训练集；流式空间回忆/计数基准 |
| HD-EPIC | Perrett et al. (Bristol et al.) | 2025 | CVPR | 高度细致的无脚本第一人称厨房视频 + VQA 基准 |
| EgoLife | Yang et al. (NTU et al.) | 2025 | CVPR | 为期一周的多人 Aria 眼镜生活日志；EgoIT-99K；超长上下文问答 |

---

## 基准对比

| 数据集 | 视频数 | 时长/片段 | 任务 | 关键指标 | 是否需要时序推理 |
|---|---|---|---|---|---|
| Kinetics-400 | ~240K | 10 秒（剪辑） | 动作识别 | Top-1 Acc | 低（外观偏向） |
| Something-Something v2 | 220K | 2–6 秒（剪辑） | 动作识别 | Top-1 Acc | 高 |
| AVA | 437 部电影 | 15 分钟（未剪辑） | 时空检测 | mAP | 中 |
| ActivityNet | 20K | 平均约 2 分钟（未剪辑） | 时序检测 | mAP@tIoU | 中 |
| Charades | 9.8K | 平均 30 秒（未剪辑） | 动作定位 | mAP | 中 |
| Ego4D | 共 3,670 小时 | 连续（未剪辑） | 多任务第一人称 | 按任务而定 | 高 |
| EPIC-KITCHENS-100 | 700 个会话 | 平均约 8.5 分钟 | 识别 + 检测 | Top-1 Acc, mAP | 高 |
| HowTo100M | 1.22M | 约 15 秒片段（含噪） | 检索预训练 | R@1（下游） | 低（预训练） |
| Video-MME | 900 | 11 秒–1 小时（3 档） | MLLM 多项选择问答 | 准确率（± 字幕） | 中–高 |
| Video-MMMU | 300 | 讲座长度 | 知识获取问答 | 准确率, Δknowledge | 中 |
| MotionBench | — | 短片段 | 细粒度运动问答 | 准确率 | 高 |
| VSI-Bench | 288 | 场景漫游 | 空间推理问答 | 准确率 / MRA（数值题） | 高（3D 记忆） |

---

## 优缺点

| 数据集 | 优势 | 局限 |
|---|---|---|
| Kinetics-400/700 | 规模巨大；标准预训练资源；标签干净 | 外观偏向；URL 失效降低可复现性；top-1 接近饱和 |
| Something-Something v2 | 检验真正的时序推理；众包带来多样性 | 仅 174 个类别；片段短；以物体操作为主 |
| Ego4D | 第一人称视频规模空前；多基准套件 | 基准设置复杂；第一人称领域差距限制迁移 |
| EPIC-KITCHENS-100 | 细粒度组合式标签；精心整理 | 单一领域（厨房）；仅限烹饪活动 |
| ActivityNet | 长时未剪辑视频；提供密集描述 | 活动标签粗粒度；许多类别仅凭外观即可识别 |
| HowTo100M | 大规模免费 ASR 监督 | 对齐噪声大；话题漂移；需要仔细过滤 |
| AVA | 密集原子标签；空间定位 | 片段数少（437 部电影）；标注仅限 1 fps 关键帧 |
| Video-MME | 新采集数据（发布时泄漏风险低）；时长分档；被广泛采用 | 对前沿模型已接近饱和（约 85–88%）；分数强烈依赖帧预算与字幕 |
| VSI-Bench / VSI-SUPER | 以 3D 扫描为依据 → 答案有客观度量；考察空间记忆 | 仅室内场景；扫描场景分布狭窄（ScanNet/ScanNet++/ARKitScenes） |
| Koala-36M / OpenVid-1M | 达到生成级质量的过滤；长描述 | 描述由模型生成；源视频的授权情况仍不明确 |

---

## 开放问题与研究空白

- **长上下文视频理解：** 当前基于 Transformer 的视频模型能处理几十到几百帧的序列，而 Ego4D 类任务需要在数千帧上推理。高效注意力机制、压缩记忆和检索增强架构是活跃的研究方向。
- **跨领域第一人称泛化：** 在 EPIC-KITCHENS 烹饪视频上训练的模型在 Ego4D 的施工或户外任务上会严重失效。目前还没有统一的第一人称预训练基准。
- **基于自然语言的时序定位：** 时序语句定位（在长视频中定位文本所描述事件发生的时间）远未解决，对于复杂的组合式描述尤其如此。
- **细粒度运动理解：** 区分"轻轻拿起"与"快速拿起"或区分方向性运动仍然困难，因为当前的视频表征以外观为主导。
- **数据集 URL 失效：** 随着 YouTube 删除内容，Kinetics 和 HowTo100M 每年都会丢失相当比例的视频。数据集托管方案（例如将 Kinetics 放入云存储桶）可以缓解但无法消除这一问题。
- **多摄像头与多模态视频：** 大多数基准都是单目视频，而真实世界系统使用多摄像头装置（来自电影的 AVA 是个例外）。将视频与 IMU、深度、视线（如 Ego4D 中）相结合的数据集在预训练中尚未得到充分探索。
- **动作预判与预测：** 从部分观察中预测未来动作是具身智能的一项关键能力，已在 EPIC-KITCHENS 和 Ego4D 上评估，但可达到的准确率仍远低于人类水平，需要根本性进展。即使是 V-JEPA 2 在 2025 年的大幅跃升，在 EK-100 上也只达到 39.7% 的 recall@5。
- **视频大模型基准的协议可比性：** Video-MME、LVBench 和 Video-MMMU 的分数取决于采样帧数（16 到 1,000+ 帧，或按 fps 采样）、分辨率、字幕/音频访问以及推理预算。自报的模型卡数字与独立重新评估之间常相差数个点（例如 Kimi K2.5 报告中带星号的重新评估结果）。目前仍缺少标准的"帧预算 × token 预算"报告惯例。
- **第一波视频大模型基准的饱和：** 前沿系统在发布仅约 18 个月后，就在 Video-MME 上达到约 85–88%、在 Video-MMMU 上达到约 87%。细粒度运动（MotionBench 约 70%）和流式空间记忆（VSI-SUPER）仍明显未解决，很可能成为下一批目标。
- **空间超感知（spatial supersensing）：** VSI-Bench 和 VSI-SUPER 表明，MLLM 无法在长时第一人称视频流中维持持久的 3D 地图。预测式/世界模型（World Model）记忆（Cambrian-S、V-JEPA 风格的目标）与显式 3D 重建，哪一种才是正确的归纳偏置，仍是开放问题。

---

## 延伸阅读

- [Ego4D 项目网站](https://ego4d-data.org/)
- [EPIC-KITCHENS 数据集与挑战赛](https://epic-kitchens.github.io/)
- [Papers With Code — 视频分类基准](https://paperswithcode.com/task/video-classification)
- [Zhu & Yang (2020) — ActBERT: Learning Global-Local Video-Text Representations. CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.pdf)
- [Tran et al. (2018) — A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR 2018](https://arxiv.org/abs/1711.11248)
- [DeepMind Kinetics 数据集文档](https://deepmind.com/research/open-source/kinetics)
- [Video-MME 基准与评估流程](https://github.com/MME-Benchmarks/Video-MME) — [arXiv:2405.21075](https://arxiv.org/abs/2405.21075)
- [Video-MMMU (arXiv:2501.13826)](https://github.com/EvolvingLMMs-Lab/VideoMMMU)
- [MotionBench (CVPR 2025)](https://github.com/zai-org/MotionBench)
- [Thinking in Space / VSI-Bench (arXiv:2412.14171)](https://github.com/vision-x-nyu/thinking-in-space)
- [Cambrian-S：VSI-590K 与 VSI-SUPER (arXiv:2511.04670)](https://github.com/cambrian-mllm/cambrian-s)
- [V-JEPA 2 / 2.1 代码与结果 (arXiv:2506.09985)](https://github.com/facebookresearch/vjepa2)
- [HD-EPIC 标注 (CVPR 2025)](https://github.com/hd-epic/hd-epic-annotations)
- [EgoLife 项目 (arXiv:2503.03803)](https://github.com/EgoLife-AI/EgoLife)
- [Koala-36M](https://github.com/KlingTeam/Koala-36M) · [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M) · [Panda-70M](https://github.com/snap-research/Panda-70M)
- [Perception Models：PE-Video、PLM-Video-Human、PLM-VideoBench](https://github.com/facebookresearch/perception_models)
