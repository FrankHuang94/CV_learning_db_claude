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
