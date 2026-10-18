# 生成式视觉模型的评估与安全

> **英文原文：** [cv-knowledge-base/10_generative_vision/04_evaluation_and_safety.md](../../cv-knowledge-base/10_generative_vision/04_evaluation_and_safety.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：** [扩散模型](./01_diffusion_models.md) | [GAN 与 VAE](./00_gans_and_vaes.md) | [评估指标](../11_datasets_and_benchmarks/04_evaluation_metrics.md) | [视频生成](./02_video_generation.md)

> **🆕 2026年10月更新：**
> - 新增**监管与标识要求**一节：欧盟《人工智能法》（EU AI Act）第 50 条透明度义务（自 2026 年 8 月 2 日起适用）及配套的 AI 生成内容标记与标识行为准则；中国《人工智能生成合成内容标识办法》（2025 年 9 月 1 日施行）；美国 TAKE IT DOWN 法案（2025 年 5 月）。
> - 更新水印/来源证明内容：SynthID Detector 与应用内 SynthID 验证，Nano Banana 2、Sora 2 和 Pixel 10 相机中的 C2PA 内容凭证（Content Credentials）。
> - 新增**人类偏好竞技场与新一代基准**一节（LMArena/Artificial Analysis 竞技场、强化学习导致的 GenEval 饱和、OneIG-Bench、VBench-2.0、Physics-IQ）。
> - 新增关于肖像/同意控制以及 2025–2026 年版权诉讼的内容；扩充了关键论文、开放问题与延伸阅读。

---

## 概述

高质量生成式视觉系统——文本到图像、文本到视频和 3D 生成——的迅速普及，迫切需要超越感知质量分数的评估方法，以衡量组合正确性、与人类意图的对齐、安全性和真实性。本文综述了主要的自动评估与人工评估基准、最常用指标的数学基础，以及为应对深度伪造（deepfake）和合成媒体危机而兴起的安全与来源证明（provenance）工具生态。

生成图像质量的自动指标分为两类：**基于参考**的指标，比较生成图像与真实图像分布的统计量（FID、IS、FVD）；以及**基于对齐**的指标，衡量生成图像与其条件信号的匹配程度（CLIPScore、ImageReward、DINO score）。基于参考的指标需要一个留出的参考集（通常为来自训练分布的 3 万–5 万张图像），并对参考集和特征提取器（Inception V3、ViT-g）中的偏差敏感。基于对齐的指标无需参考，但依赖于底层文本-视觉编码器的质量。

生成式视觉的安全格局包含三个相互交叠的关切：(1) **深度伪造**——刻意歪曲真实人物或事件的合成媒体，可被用于欺诈、虚假信息和未经同意的私密影像（NCII）；(2) **记忆化与数据隐私**——生成模型倾向于近乎逐字地复现训练数据，使私密或受版权保护的材料可被重建；(3) **来源与归属**——下游使用者无法判断一段媒体是否由 AI 生成。针对这些关切的应对措施包括隐形水印（SynthID [DeepMind2023]、Stable Signature [Fernandez2023]）、内容来源标准（C2PA）以及针对同意与授权的数据集过滤。自 2025 年以来，这些技术措施已在主要司法辖区成为**法律义务**：中国的标识办法于 2025 年 9 月 1 日施行，欧盟《人工智能法》第 50 条的透明度义务——对合成输出进行机器可读标记并披露深度伪造——自 2026 年 8 月 2 日起适用。随着照片级逼真的带音频视频模型（Veo 3、Sora 2、Seedance 2.0）和对话式图像编辑器的出现，第四个关切急剧上升：**肖像与同意**，即控制谁可以被描绘。

---

## Fréchet Inception 距离（FID）

FID [Heusel2017] 通过比较 Inception V3 pool3 层激活的统计量，衡量真实图像与生成图像分布之间的距离：

```
FID = ||mu_r - mu_g||^2
    + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^{1/2})

其中：
  mu_r, Sigma_r = 真实图像特征的均值与协方差
  mu_g, Sigma_g = 生成图像特征的均值与协方差
  特征取自 Inception V3 pool3 层（2048 维）
  Tr = 矩阵的迹
  (Sigma_r * Sigma_g)^{1/2} = 矩阵平方根（通过特征分解计算）

性质：
  - FID 越低 = 质量/多样性与真实数据越匹配
  - FID(P, P) = 0；FID(P, Q) = FID(Q, P)（对称）
  - 对样本数量敏感：强烈建议 N >= 10,000
  - 对 Inception V3 预训练和分辨率预处理敏感

常见失效模式：
  - 受参考数据集影响（如 FFHQ 与 CelebA 给出不同基线）
  - 对多样性的惩罚不足：记忆训练数据的模型也能获得低 FID
  - Inception 特征偏向 ImageNet 物体类别
```

**sFID**（空间 FID）使用 Inception 的中间层激活来捕捉空间结构而非全局统计量，能更好地惩罚空间不连贯和物体摆放伪影。

---

## Inception 分数（IS）

Inception 分数 [Salimans2016] 同时衡量质量（条件标签分布 $p(y|x)$ 应当尖锐）和多样性（边缘分布 $p(y)$ 应当平坦）：

```
IS(G) = exp(E_{x~p_g} [KL(p(y|x) || p(y))])

其中：
  p(y|x) = 生成图像 x 的 Inception V3 类别概率
  p(y) = 边缘分布 = (1/N) * sum_x p(y|x)
  KL = Kullback-Leibler 散度

展开形式：
  IS = exp( H(p(y)) - E_x[H(p(y|x))] )
  = exp( 高边缘熵 - 低条件熵 )

解释：
  好的生成器：图像质量高（H(y|x) 低）且输出多样（H(y) 高）

局限：
  - 只衡量与 ImageNet 类别的对齐；对非 ImageNet 内容失效
  - 对类内多样性不敏感
  - 每类一张清晰图像即可"骗过"该指标
  - 未归一化：依赖于 Inception V3 的版本
```

---

## CLIPScore 与 DINO 分数

**CLIPScore** [Hessel2021] 使用 CLIP 联合嵌入空间中的余弦相似度来衡量文本-图像对齐：

```
CLIPScore(image, text) = w * max(cos(CLIP_I(image), CLIP_T(text)), 0)

其中：
  CLIP_I = CLIP 图像编码器输出（已归一化）
  CLIP_T = CLIP 文本编码器输出（已归一化）
  w = 2.5（原论文中的缩放常数）
  cos = 余弦相似度

RefCLIPScore（带参考图像）：
  RefCLIPScore = harmonic_mean(CLIPScore(gen, text), CLIPScore(gen, ref))

性质：
  - 无需参考；可针对提示评估任意生成图像
  - 在文本-图像对齐上与人类判断相关
  - 在细粒度属性绑定和计数上较弱
  - 在高质量区间趋于饱和；对细微的文本变化不敏感
```

**DINO 分数**在个性化实验（DreamBooth 评估）中使用 ViT-DINO 特征衡量类内一致性：DINO 分数高 = 生成图像看起来像参考照片中的主体。

---

## Fréchet 视频距离（FVD）

FVD [Unterthiner2019] 将 FID 扩展到视频，用在 Kinetics 上训练的 I3D 网络取代 Inception V3：

```
FVD = ||mu_r - mu_g||^2
    + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^{1/2})

其中特征为视频片段上的 I3D 激活。

实践注意事项：
  - 建议 N >= 2048 个视频片段以获得稳定估计
  - 对时间分辨率和帧采样高度敏感
  - I3D 偏向 Kinetics 动作类别
  - 单独的 FVD 无法反映物理合理性或文本-视频对齐
```

VBench [Huang2024] 提供了更全面的多维视频评估框架，涵盖 16 个维度，包括主体一致性、背景一致性、时间闪烁、运动平滑度和文本-视频对齐，弥补了标量 FVD 的局限。

---

## GenEval

GenEval [Ghosh2023] 是一个评估特定生成能力的组合式文本到图像基准：

```
类别：
  1. 单个物体（基线；应当容易）
  2. 两个物体（共现）
  3. 计数（1-4 个物体）
  4. 颜色（属性绑定）
  5. 颜色归属（哪个物体是哪种颜色）
  6. 空间关系（左/右/上/下）
  7. 非空间关系（拿着、穿着）

评估：对生成图像进行目标检测 + 属性验证
每个类别得分 + 总分（0-1 范围）

代表性分数（发表时）：
  SDXL：总分 0.55
  DALL-E 3：总分 0.67
  FLUX.1-dev：总分 0.66
  Ideogram 2.0：总分约 0.70

GenEval 2（2025）：扩充了更多类别和对抗性提示。
```

到 2025 年，GenEval 已接近**饱和**：以检测器为奖励的强化学习后训练（Flow-GRPO）将 SD3.5-Medium 从 0.63 提升到 0.95，因此高 GenEval 分数已无法区分前沿系统，并且可能反映的是针对检测器的奖励投机（reward hacking），而非通用的组合能力。

---

## T2I-CompBench 与 ImageReward

**T2I-CompBench** [Huang2023] 使用基于 VQA 的自动评估，从 8 个维度（属性绑定、空间关系、非空间关系、复杂组合）评估组合式文本到图像生成：

```
评估流程：
  1. 根据 8,000 条组合式提示生成图像
  2. 使用 VQA 模型（BLIP-VQA 或 UniDet）验证属性是否存在
  3. 计算各维度得分 + 加权总分

维度：
  - 颜色绑定（"a red cube"）
  - 形状绑定（"a round table"）
  - 纹理绑定（"a striped shirt"）
  - 空间关系（左/右/上）
  - 非空间关系（骑、拿）
  - 复杂组合（综合以上全部）
```

**ImageReward** [Xu2023] 在人类偏好数据（收集自真实的文本到图像生成用户会话）上训练奖励模型，以预测人类对给定文本提示下生成图像的评分。它提供一个标量的"人类对齐"分数，在对齐敏感的任务上比 FID 或 CLIPScore 与人类判断的相关性更好。它被用作扩散模型 RLHF 微调（DPOK、DDPO）中的奖励。

---

## 人类偏好竞技场与新一代基准（2025–2026）

随着自动化组合式基准趋于饱和，前沿图像与视频模型事实上的排行榜变成了**两两对比的人类偏好竞技场**：LMArena 的文本到图像（Text-to-Image）与图像编辑（Image Edit）竞技场，以及 Artificial Analysis 的图像/视频竞技场，它们根据盲测 A/B 投票计算 Elo / Bradley–Terry 评分。厂商如今以竞技场排名宣布新品（例如 OpenAI 报告 gpt-image-2 在 2026 年 4 月发布后数小时内即登上 Image Arena 第一；Seedance 1.0 在 2025 年年中领跑 Artificial Analysis 视频竞技场）。

```
Bradley-Terry 竞技场评分：
  P(模型 i 胜过模型 j) = sigma(s_i - s_j) = 1 / (1 + exp(-(s_i - s_j)))
  s = argmax_s  sum_{投票 (i>j)} log sigma(s_i - s_j)  - lambda * ||s||^2
  Elo 式显示：R_i = 400 * s_i / ln(10) + offset
注意事项：提示分布由志愿用户决定；存在风格/长度等混杂因素；
         平局以及厂商针对竞技场的专门调优；评分会随模型池变化而漂移。
```

较新的自动化基准针对竞技场和 GenEval 遗漏的方面：

| 基准 | 年份 | 关注点 |
|-----------|------|-------|
| OneIG-Bench | 2025 | 全维度文生图评估：对齐、文字渲染、推理/知识、风格、多样性 |
| VBench-2.0 [Zheng2025VBench2] | 2025 | 视频的"内在忠实度"：人物保真度、可控性、创造力、物理、常识 |
| Physics-IQ [Motamed2025] | 2025 | 实拍的物理实验；检验视频模型能否预测物理上正确的后续画面 |
| 视频零样本推理测试集 [Wiedemer2025] | 2025 | 通过提示视频模型完成的感知、建模、操控与推理任务 |

Physics-IQ 和 VBench-2.0 的一致发现是：**视觉真实感与物理理解之间仅有弱相关**——被人类评为最逼真的模型仍可能违反基本的动力学规律。

---

## 深度伪造与检测

深度伪造——歪曲真实人物的 AI 生成或 AI 篡改媒体——构成重大的社会风险。检测领域是一场对抗性的军备竞赛：

```
检测方法：
  1. 频域伪影：GAN 生成的图像存在系统性的频谱伪影
     （网格状模式），可通过 FFT 分析检测
  2. 生理信号：眨眼不一致、通过 rPPG 检测脉搏；
     在现代模型中已被抑制
  3. 语义不一致：违反物理规律的眼睛反光、手指数量、
     首饰；在前沿模型中越来越少见
  4. 神经指纹：基于重建的检测器，利用模型特有伪影
     训练"是否来自分布 X？"的判别
  5. 通用检测器：CNNDetect、UnivFD，跨 GAN/扩散模型训练；
     随着新模型出现准确率下降（泛化差距）

关键挑战：
  扩散模型（DDPM、LDM）不具有 GAN 那样的频谱网格伪影，
  导致基于频域的检测器失效。
  现代检测器必须在多样化的合成数据上训练。
```

**肖像与同意控制（2025–2026）。** 由于音视频生成器能够生成真实人物以假乱真的说话视频，服务商在下游检测之外增加了*上游*控制：Sora 2 的自愿加入式"客串"功能让经过验证的用户决定谁可以生成其形象并可随时撤销授权；在收到投诉后，对公众人物和已故人物的描绘受到了限制；图像编辑器会拒绝或加水印处理对真实人物的编辑。这些控制在 2025–2026 年的若干事件中明显失效（例如 2026 年初被广泛报道的、利用集成于聊天机器人的图像编辑功能生成真实人物性化图像的滥用事件，引发了监管调查；以及制片厂对 Seedance 2.0 生成的名人和受版权保护角色片段的反对），这凸显出政策过滤器无法替代稳健的来源证明。

---

## 水印：SynthID 与 Stable Signature

**SynthID** [DeepMind2023，2024 年更新为 SynthID-Image] 是 Google DeepMind 用于 AI 生成图像的生产级水印系统：
- 在图像生成过程中（而非事后）嵌入隐形水印，方法是与生成模型一同微调一个小型检测器。
- 水印人眼不可察觉，但可由配对的分类器检测。
- 对 JPEG 压缩、缩放、裁剪和颜色变换具有鲁棒性。
- 已部署于 Imagen、Veo 以及所有 Gemini 图像模型（Nano Banana、Nano Banana Pro、Nano Banana 2）；SynthID-Text 已于 2024 年开源。
- 图像/视频水印模型未开源。Google 推出了 **SynthID Detector** 门户（2025 年 5 月），并随 Nano Banana Pro（2025 年 11 月）让用户可在 Gemini 应用中询问某张图像是否由 Google AI 生成或编辑。Nano Banana 2（2026 年 2 月）的输出同时带有 SynthID 和可互操作的 C2PA 内容凭证——体现了监管机构所期望的、正在兴起的"水印 + 签名清单"双层方案。

**Stable Signature** [Fernandez2023] 将二进制消息（48 位）嵌入潜在扩散解码器的权重中，使解码器生成的每张图像都自动携带水印：

```
训练：
  1. 训练水印解码器 W：图像 -> 48 位消息
  2. 微调 VAE 解码器 D，使其生成的图像 x = D(z) 满足：
     W(x) ~= 目标消息  （消息恢复损失）
     ||x - D_orig(z)||_lpips < epsilon  （感知保真度约束）

性质：
  - 水印固化在解码器中；无推理开销
  - 对 JPEG（Q=30）、高斯噪声、模糊具有鲁棒性
  - 可追溯：不同模型可嵌入不同消息
  - 局限：水印固定于解码器，不随输入而变化
```

---

## C2PA 内容来源证明

内容来源与真实性联盟（Coalition for Content Provenance and Authenticity, C2PA）定义了一个开放标准，用于记录创作和编辑历史的、经过加密签名的内容清单（manifest）：

```
附加到媒体文件的 C2PA 清单：
  {
    "claim_generator": "StabilityAI/StableDiffusion3",
    "assertions": [
      {"label": "c2pa.training_and_data_mining", "data": {...}},
      {"label": "c2pa.ai_generative_info", "data": {...}}
    ],
    "signature": {
      "alg": "ES256",
      "cert": "<签名机构的 X.509 证书>",
      "issuer": "StabilityAI"
    }
  }

验证：
  - 证书链验证（PKI）
  - 清单中嵌入媒体内容的哈希
  - 防篡改：任何像素改动都会使哈希失效

采用情况：
  - Adobe（Firefly）、Microsoft、Sony、OpenAI（DALL-E 3）、Leica、Qualcomm
  - Adobe 软件中可见"内容凭证"（Content Credentials）徽章
  - CAI（内容真实性倡议，Content Authenticity Initiative）是行业实施机构
  - 2025-2026：OpenAI GPT Image 与 Sora 2 的输出；Google Nano Banana 2；
    Google Pixel 10 相机原生为照片签名（2025 年 8 月）
```

C2PA 提供的是来源证明而非真实性检测：恶意行为者可以生成不带 C2PA 凭证的媒体，或干脆将其去除。它最适用于为已签名的媒体建立信任，而不是检测未签名的合成媒体。因此业界逐渐形成了**分层来源证明**的共识：(1) 签名的 C2PA 清单（信息丰富，但可被剥离）；(2) 能在元数据被剥离和适度编辑后留存的隐形水印（SynthID、Stable Signature 类方法）；(3) 可选的可见标签（如 Sora 2 的移动水印）；(4) 当前两者都被去除时用于恢复的指纹/哈希查询。相机端签名（Pixel 10、Leica、Sony）则从另一端着手，使*真实*拍摄可被验证。

---

## 监管与标识要求（2025–2026）

| 司法辖区 | 法律文件 | 生效时间 | 与生成式视觉相关的要求 |
|--------------|-----------|----------|-------------------------------------------|
| 欧盟 | 《人工智能法》第 50 条 | 2026 年 8 月 2 日 | 50(2)：生成合成图像/视频/音频/文本的系统的提供者，须以机器可读格式标记输出，使其可被检测为人工生成，并"在技术可行范围内"采用有效、可互操作、稳健且可靠的技术；50(4)：部署者须披露深度伪造（对明显属于艺术/讽刺性质的作品义务减轻） |
| 欧盟 | AI 生成内容标记与标识行为准则（Code of Practice） | 自 2025 年末起起草；旨在支持第 50 条合规 | 自愿性准则，涵盖水印、元数据、指纹/日志记录以及深度伪造的通用标签/图标 |
| 中国 | 《人工智能生成合成内容标识办法》+ 强制性国家标准 GB 45438-2025 | 2025 年 9 月 1 日 | 显式标识（可见/可听）加文件元数据中的隐式标识；平台须进行检测和标识 |
| 美国（联邦） | TAKE IT DOWN 法案 | 2025 年 5 月 19 日签署；平台流程须于 2026 年 5 月前落实 | 将发布未经同意的私密影像（包括 AI 生成的"数字伪造品"）定为刑事犯罪；收到有效请求后 48 小时内删除 |
| 美国（加利福尼亚州） | 《AI 透明度法》（SB 942，经修订） | 2026 年施行（据报道） | 大型生成式 AI 提供者须提供免费检测工具并嵌入潜在披露信息 |

实施层面仍有问题待解：对于用户可移除水印模块的开放权重模型，何为"稳健"的标记；鉴于元数据极易被剥离，仅有元数据的 C2PA 能否满足第 50(2) 条；以及如何标识部分经 AI 编辑的媒体。欧盟委员会 2025 年 11 月的"数字综合法案"（Digital Omnibus）提案还提出为已上市系统提供第 50(2) 条的过渡性宽限；从业者应查阅最终通过的文本及委员会的指南。

**版权与训练数据诉讼。** 权利人诉讼对数据实践的影响不亚于监管：Getty Images 诉 Stability AI 案（英国高等法院 2025 年 11 月判决）基本驳回了 Getty 关于模型权重本身构成"侵权复制品"的间接侵权主张，仅就复现的水印作出了有限的商标侵权认定；迪士尼与环球影业于 2025 年 6 月起诉 Midjourney，指其生成了旗下角色的形象；制片厂也在转向授权合作，例如迪士尼 2025 年 12 月与 OpenAI 达成协议，在进行股权投资的同时授权其角色用于 Sora。这些案件使记忆化研究（下一节）具有了直接的法律意义。

---

## 扩散模型中的记忆化

Carlini 等人 [Carlini2023] 证明扩散模型会以可测量的比率记忆并逐字复现训练图像：

```
记忆化类型：
  1. 逐字记忆：在像素级精确复现训练图像
  2. 模板记忆：以不同内容复现结构模板
  3. 概念记忆：复现特定概念（如某个人的面孔）

测量协议 [Carlini2023]：
  - 使用训练集中的描述作为提示
  - 针对重复次数最多的前 500 条提示，每条生成 500 张图像
  - 在像素空间中对训练集做 L2 最近邻搜索
  - 阈值 L2/分辨率 < 0.1 -> 判定为"已记忆"

发现：
  - 在分析的前 50 万张训练图像中，约 350 张被记忆
  - 逐字记忆率约为被查询提示的 0.03%
  - 训练集中重复度高的图像更容易被记忆
  - DDPM 目标不提供任何形式化的隐私保证

缓解措施：
  - 训练数据去重（记忆化降低约 5 倍）
  - 差分隐私训练（epsilon 较大；有质量损失）
  - 训练中进行成员推断测试；移除重复项
  - 推理时的噪声增强（并不完全有效）
```

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 核心贡献 |
|-------|---------|------|-------|-----------------|
| GANs Trained by a Two Time-Scale Update Rule (FID) | Heusel et al. | 2017 | NeurIPS | Fréchet Inception 距离；成为标准质量指标 |
| Improved Techniques for Training GANs (IS) | Salimans et al. | 2016 | NeurIPS | Inception 分数；质量 + 多样性指标 |
| CLIPScore | Hessel et al. | 2021 | EMNLP | 基于 CLIP 的无参考图像-描述对齐 |
| Towards Video Understanding (FVD) | Unterthiner et al. | 2019 | arXiv | Fréchet 视频距离；使用 I3D 特征评估视频 |
| GenEval | Ghosh et al. | 2023 | NeurIPS | 组合式文生图评估；物体/属性/空间 |
| T2I-CompBench | Huang et al. | 2023 | NeurIPS | 多维组合式文生图基准；8,000 条提示 |
| ImageReward | Xu et al. | 2023 | NeurIPS | 文生图的人类偏好奖励模型；RLHF 信号 |
| Extracting Training Data from Diffusion Models | Carlini et al. | 2023 | USENIX Security | 量化 DDPM/LDM 中的记忆化；逐字复现 |
| SynthID | Lim et al. (Google DeepMind) | 2024 | arXiv | 生产规模的隐形水印；鲁棒性分析 |
| Stable Signature | Fernandez et al. | 2023 | ICCV | 固化在 LDM 解码器中的逐解码器二进制水印 |
| VBench | Huang et al. | 2024 | CVPR | 16 维视频生成基准；全面评估 |
| WAVES: Benchmarking the Robustness of Image Watermarks | An et al. | 2024 | ICML | 在失真、再生成和对抗去除攻击下对图像水印进行标准化压力测试 |
| Do generative video models understand physical principles? (Physics-IQ) | Motamed et al. | 2025 | arXiv | 真实世界物理基准；表明真实感 ≠ 物理理解 |
| VBench-2.0 | Zheng et al. | 2025 | arXiv | 内在忠实度视频基准（物理、常识、可控性、人物保真度） |
| Flow-GRPO | Liu et al. | 2025 | NeurIPS | 使 GenEval 饱和（0.63 → 0.95）的强化学习后训练，暴露出基准的局限 |

---

## 基准性能

| 模型 | 基准 | 指标 | 分数 | 备注 |
|-------|-----------|--------|-------|-------|
| DALL-E 3 | GenEval | 总体准确率 | 0.67 | 论文发表时最佳 |
| FLUX.1-dev | GenEval | 总体准确率 | 0.66 | 开源竞争者 |
| SDXL 1.0 | GenEval | 总体准确率 | 0.55 | 双编码器；1024px |
| SD 2.1 | MS-COCO 30K | FID | 11.7 | 标准参考 |
| SDXL | MS-COCO 30K | FID | ~11 | 含精修器 |
| HunyuanVideo | VBench（视频） | 总分 | 85.09 | 发布时最佳开源模型 |
| SynthID | COCO 512px | AUROC（水印检测） | 98.7% | 经 JPEG Q=80 压缩后 |
| Stable Signature | COCO 512px | 比特准确率（48 位） | 96.2% | 经 JPEG Q=30 后 |
| UnivFD | 新生成器 | AUROC | ~63% | 跨生成器深度伪造检测 |
| SD3.5-M + Flow-GRPO | GenEval | 总体准确率 | 0.95 | 经强化学习调优；GenEval 接近饱和 |
| gpt-image-2 | Image Arena (LMArena) | 排名 | 发布时第一（2026 年 4 月） | 厂商自报；两两对比的人类偏好 |

---

## 优缺点

| 方面 | 优点 | 缺点 |
|--------|------|------|
| **FID/IS** | 使用广泛；便于跨论文比较；同时反映质量与多样性 | 受 Inception 预训练影响；对参考集敏感；不衡量文本对齐；容易被低多样性模型"刷分" |
| **CLIPScore** | 无需参考；直接衡量提示对齐；计算快 | 在高质量区间饱和；对计数、空间和细粒度属性不敏感 |
| **GenEval/T2I-CompBench** | 组合式；衡量特定失败模式；可复现 | 评估模型（检测器/VQA）本身可能出错；与真实使用存在分布差距 |
| **SynthID/水印** | 隐形；对常见变换鲁棒；可追溯 | 需要模型方配合；可被强对抗攻击去除；与模型绑定 |
| **C2PA** | 开放标准；密码学安全；支持硬件相机 | 自愿采用；无法检测未签名的合成媒体；元数据可被剥离 |
| **人类偏好竞技场** | 追踪真实用户偏好；难以用固定提示过拟合；覆盖编辑与视频 | 志愿者决定提示分布；存在风格混杂因素；厂商发布前测试不透明；无法离线复现 |
| **监管标识（欧盟第 50 条、中国标识办法）** | 建立统一底线，并为来源证明提供市场激励 | 对开放权重模型的技术可行性不明；各司法辖区规则碎片化；执法才刚刚开始 |

---

## 开放问题与研究空白

- **统一的质量-对齐-安全基准**：没有任何单一基准能同时衡量感知质量（FID）、文本对齐（CLIPScore）、组合准确性（GenEval）、安全内容（NSFW 比率）和记忆化。研究者往往孤立地优化各个指标。
- **深度伪造检测器的泛化**：在一个生成模型家族上训练的检测器在另一个家族上表现极差；泛化差距是一个没有根本解决方案的军备竞赛问题。
- **对抗性去除下的稳健水印**：对抗性图像变换（JPEG、裁剪、噪声、利用 IP2P 的生成式对抗去除）可将水印检测率降至阈值以下；面对能力无界的攻击者，无法实现嵌入层面的鲁棒性保证。
- **法律与监管状况**：欧盟《人工智能法》第 50 条（自 2026 年 8 月起）和中国标识办法（自 2025 年 9 月起）如今已*强制要求*机器可读标记，但二者都未规定具体技术；协调标准、各家专有水印（SynthID 与其他方案）之间的互操作性以及对开放权重模型的处理仍未解决，而美国的要求则是州法律加 TAKE IT DOWN 法案拼凑而成。
- **基准饱和与奖励投机**：GenEval 类基于检测器的指标可通过强化学习后训练被推至接近 1.0；竞技场更难被操纵但无法复现。对推理密集型输出（信息图、图表、多语言文字、事实正确性）的评估缺乏公认的自动指标。
- **开放权重生成器的水印**：解码器级水印（Stable Signature）可通过微调去除，事后水印则可直接跳过；当用户掌控权重时，没有任何方案能提供稳健的标记。
- **超越记忆化的成员推断**：判定某张特定图像（而非完全相同的副本）是否在模型训练集中——这对版权执法很重要——在逐字记忆之外的情形下仍未解决。
- **人工评估的可扩展性**：ImageReward 及类似奖励模型基于有限的人类偏好数据训练，无法在全球范围内捕捉文化多样性、语境适当性和不断演变的审美标准。
- **视频真实性**：FVD 和 VBench 衡量的是分布质量；目前没有成熟的基准来衡量视频级的深度伪造真实性、合成人物的时间一致性或音视频口型同步的保真度。

---

## 延伸阅读

- [Heusel et al. (2017), "GANs Trained by a Two Time-Scale Update Rule," NeurIPS 2017](https://arxiv.org/abs/1706.08500)
- [Hessel et al. (2021), "CLIPScore: A Reference-free Evaluation Metric for Image Captioning," EMNLP 2021](https://arxiv.org/abs/2104.08718)
- [Carlini et al. (2023), "Extracting Training Data from Diffusion Models," USENIX Security 2023](https://arxiv.org/abs/2301.13188)
- [Fernandez et al. (2023), "The Stable Signature: Rooting Watermarks in Latent Diffusion Models," ICCV 2023](https://arxiv.org/abs/2303.15435)
- [Ghosh et al. (2023), "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment," NeurIPS 2023](https://arxiv.org/abs/2310.11513)
- [Huang et al. (2024), "VBench: Comprehensive Benchmark Suite for Video Generative Models," CVPR 2024](https://arxiv.org/abs/2311.17982)
- [Zheng et al. (2025), "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness"](https://arxiv.org/abs/2503.21755)
- [Motamed et al. (2025), "Do generative video models understand physical principles?"](https://arxiv.org/abs/2501.09038)
- [欧盟《人工智能法》第 50 条：特定 AI 系统提供者与部署者的透明度义务](https://artificialintelligenceact.eu/article/50/)
- [C2PA 技术规范](https://c2pa.org/specifications/)
- [TAKE IT DOWN 法案（S.146，第 119 届国会）](https://www.congress.gov/bill/119th-congress/senate-bill/146)
- [OpenAI (2026), "System Card: ChatGPT Images 2.0 and Thinking mode"](https://deploymentsafety.openai.com/chatgpt-images-2-0/chatgpt-images-2-0.pdf)
- [Google (2026)，"Nano Banana 2" 发布公告（SynthID + C2PA）](https://blog.google/innovation-and-ai/technology/ai/nano-banana-2/)
