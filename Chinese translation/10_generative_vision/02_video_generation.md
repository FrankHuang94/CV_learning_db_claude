# 视频生成：从 Sora 到开源世界模拟器

> **英文原文：** [cv-knowledge-base/10_generative_vision/02_video_generation.md](../../cv-knowledge-base/10_generative_vision/02_video_generation.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：** [扩散模型](./01_diffusion_models.md) | [GAN 与 VAE](./00_gans_and_vaes.md) | [世界模型](../06_robotics_and_embodied_ai/02_world_models.md) | [评估与安全](./04_evaluation_and_safety.md)

> **🆕 2026年10月更新：**
> - 更正了 Sora 2 的时间线（发布于 2025 年 9 月 30 日，而非 2026 年），并新增 Veo 3.1（2025 年 10 月）与 Google 的 Gemini Omni（I/O 大会，2026 年 5 月；Omni Flash 开发者预览版于 2026 年 6 月开放）。
> - 新增**原生音视频联合生成**一节（Veo 3/3.1、Sora 2、Kling 2.6、Wan 2.5、LTX-2）以及 **Seedance**（字节跳动）一节。
> - 新增**自回归、少步与实时视频生成**一节（CausVid、Self Forcing、MAGI-1、Seaweed-APT），以及与交互式世界模型（Genie 3）的联系。
> - 新增**视频模型作为零样本视觉推理器**一节（Veo 3）；更新了对比表、关键论文、开放问题与延伸阅读。

---

## 概述

视频生成与图像合成处于质上不同的难度层级：时间连贯性、物理合理性、相机运动、长程依赖以及数秒的时长，都提出了无法通过独立生成各帧来满足的要求。2024 年之前的主流范式——循环网络、视频 GAN 或基于帧序列的自回归模型——难以生成超过几秒且视觉一致的画面。OpenAI 于 2024 年初发布的 Sora 表明，单一架构（作用于时空图块的扩散 Transformer）在大规模训练后，可根据文本或图像提示生成长达 60 秒、照片级逼真且物理上合理的视频，从根本上改变了该领域的预期。

现代模型背后的核心架构洞见，是把空间压缩与时间压缩统一到共享的 3D 潜空间中（通过 3D 因果 VAE），再由基于 Transformer 的扩散过程作用于"时空图块"（spacetime patches）——跨越若干帧和一个空间区块的连续 3D token。由于图块可以从任意分辨率、时长和宽高比的视频中提取而无需修改架构，同一个模型即可处理多种多样的生成任务。Peebles 与 Xie [Peebles2023] 首先证明，用 Transformer（DiT，扩散 Transformer）取代 U-Net 骨干在图像生成上的扩展性优于 U-Net；Sora 及其后继者将 DiT 扩展到了时空域。

截至 2026 年 10 月，视频生成格局分为闭源专有模型（Sora 2、Veo 3.1 与 Google 的 Gemini Omni、Runway Gen-4/4.5、Kling 2.6/3.0、Seedance 1.x/2.0）和日趋成熟的开放权重生态（Wan 2.1/2.2、HunyuanVideo 1.5、LTX-Video/LTX-2、CogVideoX）。2025–2026 年的决定性转变是**原生同步音频**——对白、音效和音乐与画面联合生成——由 Veo 3（2025 年 5 月）和 Sora 2（2025 年 9 月）率先大规模实现。第二个转变是**实时自回归生成**（Self Forcing、Genie 3），它模糊了视频生成器与交互式世界模型（World Model）之间的界限。开源模型在运动真实感、物理和音频方面仍落后于闭源前沿，但差距以月而非以年计。几乎所有主流系统都采用以流匹配（flow matching）而非 DDPM 训练的时空图块 DiT，体现了 SD3/FLUX 的经验对视频领域的影响。

---

## 核心架构：时空图块 DiT

Sora 风格的架构通过 3D 卷积将视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$ 分解为时空图块：

```
视频 V (T x H x W x 3)
  --> 3D 因果 VAE 编码器
  --> 潜变量 L (T/t_c x H/s x W/s x C)    [典型取值 t_c=4, s=8]
  --> 图块化（p_t x p_h x p_w 个 token）
  --> 展平为长度为 N 的 token 序列
  --> Transformer 层（对全部 N 个 token 做自注意力）
  --> 反图块化
  --> 3D 因果 VAE 解码器
  --> 输出视频

N = (T/t_c / p_t) * (H/s / p_h) * (W/s / p_w)

对于 480p、5 秒 @ 24fps：
  T=120, H=480, W=854
  VAE 之后：30 x 60 x 107 x C
  图块大小 (1,2,2)：N = 30*30*54 = 48,600 个 token
```

Transformer 采用带 3D RoPE 位置编码的完整时空自注意力（HunyuanVideo、Wan），或高效近似（长视频与蒸馏实时变体中的稀疏 / 滑动窗口注意力）。文本条件通过对编码器嵌入的交叉注意力注入（大多数现代模型使用 T5-XXL）。3D 因果 VAE 至关重要："因果"结构意味着在编码时未来帧不能关注未来的潜变量，从而支持流式生成和时间一致性。

---

## Sora（OpenAI，2024）

OpenAI 的 Sora [OpenAI2024] 使用作用于视频潜变量时空图块的 DiT。关键特性：

- **架构**：采用可变分辨率、可变时长时空图块的扩散 Transformer（DiT）；通过 DALL-E 风格的重新标注描述进行文本条件化。
- **时长**：2024 年研究预览版最长 60 秒。**Sora 2**（2025 年 9 月 30 日发布，同时推出社交类 iOS 应用"Sora"）可生成同步的对白与音效，能遵循多镜头指令，并引入了基于同意的"客串"（cameos）功能（用户可插入经验证的本人形象）；片段最初为 10 秒，后延长至 15 秒（Pro 用户的故事板模式为 25 秒）。Sora 2 的输出带有可见的移动水印以及 C2PA 元数据（参见[评估与安全](./04_evaluation_and_safety.md)）。
- **时空图块**：图块是同时跨越多帧和多个空间区域的 3D token，使模型对宽高比、分辨率和帧率不敏感——无需为不同格式设置单独的编码器。
- **物理与 3D 一致性**：Sora 在没有显式物理仿真的情况下，展现出涌现的 3D 一致性、物体恒存性以及一定的物理合理性（流体动力学、碰撞），这被归因于训练数据的规模。
- **训练数据**：未公开披露；估计涉及数千万个视频片段，并通过合成重标注实现文本对齐。

OpenAI 发布的是技术报告而非完整的研究论文；许多架构细节仍未公开。

---

## Stable Video Diffusion（SVD）

Stable Video Diffusion [Blattmann2023] 通过以下方式将 Stable Diffusion XL 适配到视频：
1. 训练一个视频专用的 3D U-Net（在空间 U-Net 中插入时间注意力层）。
2. 图像到视频条件化：给定参考帧作为条件，生成 512×512 至 1024×576 分辨率的 14–25 帧。
3. 三阶段课程：图像预训练 → 视频预训练 → 高质量视频微调。

SVD 基础版不支持文本提示；运动幅度通过 `motion_bucket_id` 和 `augmentation_level` 控制。SVD-XT 将输出扩展到 25 帧。

---

## Runway Gen-3 Alpha 与 Kling

**Runway Gen-3 Alpha**（2024）是一个专有模型，支持根据文本或图像提示生成最长 10 秒、1280×768 的片段，强调电影级相机运动以及通过自然语言实现的导演式控制。

**Kling（可灵）**（快手，2024）使用带有专有时空联合注意力机制的扩散 Transformer，可生成长达 5 分钟的 1080p 视频。Kling 1.5/2.0 增加了"高一致性"模式，用于在多个镜头间保持主体身份。到 Kling 3.0（2026），通过 Omni One 架构支持原生 4K 输出及 60fps 上采样。

---

## Veo 2 与 Veo 3

**Veo 2**（Google DeepMind，2024）：VideoPoet 的后继者，采用带视频专用 Transformer 骨干的潜在扩散模型。支持最长 60 秒的 1080p 输出，并可用自然语言指定相机轨迹控制（平移、缩放、俯仰）。发布时在 UCF-101 和 Kinetics 上取得了最先进的 FVD。

**Veo 3**（Google I/O，2025 年 5 月）：新增与视频同步的原生音频生成（带口型同步的对白、环境声、音效）；架构细节未公开。支持最长 8 秒、带音频的 1080p 片段；扩展功能允许拼接成数分钟的视频。

**Veo 3.1**（2025 年 10 月）：音频更丰富、提示遵循更强，支持参考图像（"素材"，ingredients）条件、首尾帧插值和场景延展，通过 Google 的 Flow 电影制作工具和 Gemini API 提供（包括更便宜的 Veo 3.1 Fast 档位）。

**Gemini Omni**（Google I/O，2026 年 5 月）：Google 的"任意模态到视频"模型，可将文本、图像和音频的组合转换为视频，并支持对话式多轮视频编辑；**Gemini Omni Flash** 开发者预览版于 2026 年 6 月 30 日开放（API 初期支持 10 秒生成）。这标志着视频生成被吸收进通用的多模态 Gemini 产品线，类似于 Nano Banana 吸收了图像生成（参见[扩散模型](./01_diffusion_models.md)）。

---

## CogVideoX（清华 / 智谱 AI）

CogVideoX [Yang2024] 是一个完全开源（Apache 2.0）的视频生成模型，基于 3D 因果 VAE + 专家 Transformer（Expert Transformer）架构：
- **CogVideoX-5B**：5B 参数；生成 720×480、6 秒、8fps 的片段；英文提示最长 226 个 token；最低需 12GB 显存。
- **架构**：带 3D RoPE 的专家 Transformer；全帧（时间 + 空间）注意力。
- 该模型在 UCF-101 上取得了有竞争力的 FVD 分数，并在 VBench 上展现出很强的文本-视频对齐能力。

---

## Movie Gen（Meta，2024）

Meta 的 Movie Gen [Polyak2024] 是一个 30B 参数的视频与音频生成基础模型（Foundation Model）：
- 文本到视频：1080p、16 秒 @ 16fps；音频协同生成模型（13B 参数）。
- 在约 10 亿个视频-文本对上训练；通过图像条件实现个性化生成。
- 架构：基于 Transformer 的时间自编码器 + 流匹配扩散；发布时在人类偏好评估中优于 Kling 和 Runway Gen-3。
- 未开源；发布了详细的技术报告。

---

## Wan 2.x（阿里巴巴）

Wan（万相）[WanTeam2025] 是阿里巴巴推出的开源视频生成模型系列：
- **Wan 2.1**（2025 年 2 月）：首个在多个基准上达到专有模型质量的开源模型；支持 720p 文本到视频和图像到视频。
- **Wan 2.2**（2025 年 7 月）：混合专家（Mixture-of-Experts, MoE）架构——A14B 模型将*高噪声*去噪步骤（全局布局）和*低噪声*步骤（细节）分别路由到两个约 14B 的独立专家，因此每步计算量和显存占用保持在 2.1 的水平；在大幅扩充的图像 + 视频数据上训练，并带有电影美学标签。稠密的 **TI2V-5B** 模型使用 16×16×4 的高压缩 VAE，可在单张消费级 GPU 上生成 720p@24fps 视频。在人物主体的照片级真实感（面部细节、皮肤纹理、头发）上领先于其他开源模型。
- **Wan 2.5**（2025 年 9 月，预览版）：新增原生音视频生成（人声、音效、口型同步）；最初通过阿里云提供，而非以开放权重形式发布。

---

## HunyuanVideo（腾讯）

HunyuanVideo（混元视频）[KongTeam2024] 在首次发布时是参数规模最大的公开开源视频模型：
- 基础模型 **13B 参数**；3D 因果 VAE 在空间上压缩 16 倍、时间上压缩 4 倍。
- 架构：带 3D RoPE 的全 Transformer；双流文本条件（CLIP + LLM）。
- **HunyuanVideo 1.5**（2025）：精简到 8.3B 参数；借助卸载（offloading）可在 14GB 显存上运行；擅长物理效果（流体动力学、布料模拟）。
- 人类偏好评估显示其运动质量可与闭源模型竞争。

---

## 原生音视频联合生成（2025–2026）

2025–2026 年的前沿把一个片段视为联合的视听信号。大多数系统是闭源的，但已公开的开源设计具有共同模式：一个视频 DiT 流和一个音频 DiT 流（作用于频谱图或音频编解码器潜变量）通过交叉注意力或共享的联合注意力模块交换信息，并采用时间对齐的位置编码，使口型动作与撞击声等声音对齐。

```
音视频联合流匹配（示意）：
  z_v ~ 视频潜变量 (T_v x H x W x C_v),  z_a ~ 音频潜变量 (T_a x C_a)
  x_t = (z_v,t , z_a,t) = (1 - t) * (z_v, z_a) + t * (eps_v, eps_a)
  L = E || v_theta^V(x_t, t, c) - (eps_v - z_v) ||^2
    + lambda_a * E || v_theta^A(x_t, t, c) - (eps_a - z_a) ||^2
  跨模态同步：共享时间对齐的 RoPE，使第 k 帧 <-> 音频窗口 [k*T_a/T_v, (k+1)*T_a/T_v]
```

| 系统 | 发布时间 | 音频能力 | 备注 |
|--------|----------|------------------|-------|
| Veo 3 / 3.1 | 2025 年 5 月 / 10 月 | 对白、音效、环境声 | 首个大规模部署的音视频联合模型 |
| Sora 2 | 2025 年 9 月 | 对白、音效 | 客串功能（经同意的本人形象）；通过社交应用分发 |
| Wan 2.5 | 2025 年 9 月（预览） | 人声、音效、口型同步 | 阿里云服务 |
| Kling 2.6 | 2025 年 12 月（据报道） | 原生音频 | 快手首个原生音频版 Kling |
| LTX-2 | 2025 年末（开放权重） | 同步音频 + 视频 | Lightricks；面向消费级 GPU 的开源 DiT 音视频模型 |
| Seedance 1.5 / 2.0 | 2025 年末 / 2026 年 2 月（据报道） | 音视频联合 | 字节跳动；多模态参考 |
| Gemini Omni | 2026 年 5 月 | 音频可作输入与输出 | 任意模态到视频，支持对话式编辑 |

---

## Seedance（字节跳动）

**Seedance 1.0** [Seedance2025]（2025 年 6 月）是字节跳动 Seed 团队的视频基础模型。其技术报告强调：(i) 多来源数据整理及精确的稠密视频描述；(ii) 一个空间层与时间层解耦、并采用交错多模态位置编码的 DiT，原生支持文本到视频和图像到视频的**多镜头**生成（跨剪辑保持主体一致）；(iii) 使用多个奖励模型的视频专用 RLHF；(iv) 多阶段蒸馏加系统级优化，报告约 10 倍推理加速（在 NVIDIA L20 上约 41 秒生成一段 5 秒 1080p 视频）。发布时 Seedance 1.0 在第三方文本到视频和图像到视频竞技场中位居榜首。**Seedance 2.0**（2026 年 2 月，据报道）新增联合音频以及基于图像、视频和音频的多参考条件；涉及名人和受版权保护角色的病毒式传播片段引发了好莱坞制片厂和行业机构的公开反对，这正说明了[评估与安全](./04_evaluation_and_safety.md)中讨论的肖像权与版权问题。

---

## 自回归、少步与实时视频生成

双向 DiT 对整个片段进行联合去噪，因此延迟随片段长度增长，而且在结束之前无法显示任何内容。2025 年的一系列工作将其转换为**因果的流式生成器**：

- **CausVid** [Yin2025CausVid]（CVPR 2025）用 DMD 将双向教师模型蒸馏为块因果的自回归学生模型，并借助 KV 缓存实现逐帧流式输出。
- **Self Forcing** [Huang2025SelfForcing]（NeurIPS 2025）在学生模型*自身*推演出的上下文（而非真实帧）上训练自回归学生模型，并使用整体的视频级分布匹配损失，弥合了训练-测试之间的暴露偏差（exposure bias），在单张 GPU 上实现实时流式生成。
- **MAGI-1**（Sand AI，2025）是一个开源的、按块自回归的视频扩散模型，专为流式生成和无限延展而设计。
- **Seaweed-APT** [Lin2025APT] 表明对抗后训练可以实现*单步* 720p 视频生成（参见 [GAN 与 VAE](./00_gans_and_vaes.md)）。

```
Self Forcing（示意）：
  for 块 k = 1..K:
      x^k ~ G_theta( . | x^{<k}_generated , c )   # 少步去噪，KV 缓存保存自身过去的输出
  L = D( p_theta(x^{1:K}) || p_data(x^{1:K}) )     # 视频级 DMD / GAN / SiD 目标
  => 不使用教师强制（teacher forcing）；训练中遇到的误差与推理时一致
```

同样的要素也支撑着**交互式世界模型**：DeepMind 的 **Genie 3**（2025 年 8 月）可实时生成 24 fps、720p 的可导航环境，具备分钟级一致性和可通过提示触发的世界事件；2026 年初又开放了范围更广的"Project Genie"访问。这些系统在[世界模型](../06_robotics_and_embodied_ai/02_world_models.md)中有详细介绍。

---

## 视频模型作为零样本视觉推理器

Wiedemer 等人 [Wiedemer2025]（"Video models are zero-shot learners and reasoners"）以一张输入图像和一条文本指令提示 Veo 3，并从生成的视频中读出答案。在没有任务特定训练的情况下，Veo 3 能完成分割、边缘检测、物理属性推理、图像编辑，以及走迷宫、对称补全等简单视觉推理任务——相比 Veo 2 有大幅提升。作者认为，视频模型可能会像 LLM 之于语言那样，成为通用的视觉基础模型（以"帧链"，chain-of-frames，类比思维链）。然而，Physics-IQ 等以物理为重点的基准表明，视觉真实感与物理理解之间的相关性仍然很弱（参见[评估与安全](./04_evaluation_and_safety.md)）。

---

## 模型对比表

| 模型 | 机构 | 开源 | 最大时长 | 最大分辨率 | 架构 | 音频 | 主要优势 |
|-------|-------------|------|-----------|---------|-------------|-------|-------------|
| Sora 2 | OpenAI | 否 | 10–25s | 1080p | DiT 时空图块 | 是 | 照片级真实感、物理、客串 |
| Veo 3.1 | Google DeepMind | 否 | 8s（可延展） | 1080p | LDM Transformer | 是 | 相机控制、音频同步、参考图像 |
| Gemini Omni | Google DeepMind | 否 | 约 10s（Flash API） | 未披露 | 多模态 Gemini（未公开） | 是 | 任意模态到视频、对话式编辑 |
| Runway Gen-4 | Runway | 否 | 10s | 1280×768 | 专有 DiT | 否 | 电影级运动 |
| Kling 3.0 | 快手 | 否 | 10s | 4K（厂商自报） | Omni One DiT（厂商自报） | 是（自 2.6 起） | 4K、多镜头 |
| Seedance 1.0 Pro | 字节跳动 | 否 | 10s | 1080p | 多镜头 DiT + RLHF | 否（1.5/2.0 支持音频） | 多镜头叙事、速度 |
| LTX-2 | Lightricks | 是 | 约 10s（据报道） | 最高 4K（厂商自报） | 音视频 DiT | 是 | 开源音视频、高效 |
| Movie Gen | Meta | 否 | 16s | 1080p | 30B 流 Transformer | 是 | 最长的高分辨率、音频 |
| HunyuanVideo 1.5 | 腾讯 | 是 | 13s | 720p | 8.3B DiT + 3D VAE | 否 | 物理、开放权重 |
| Wan 2.2 | 阿里巴巴 | 是 | 10s | 720p | MoE DiT | 否 | 照片级真实的人物 |
| CogVideoX-5B | 智谱 AI | 是 | 6s | 720×480 | 5B 专家 Transformer | 否 | 轻量、易用 |

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 核心贡献 |
|-------|---------|------|-------|-----------------|
| Scalable Diffusion Models with Transformers (DiT) | Peebles & Xie | 2023 | ICCV | 用 Transformer 取代 U-Net；表明 ViT 的扩展规律适用于扩散模型 |
| Video generation models as world simulators (Sora) | OpenAI | 2024 | Technical Report | 时空图块；可变分辨率/时长生成；60 秒视频 |
| Stable Video Diffusion (SVD) | Blattmann et al. | 2023 | arXiv | 基于时间 U-Net 的图像到视频；三阶段训练课程 |
| CogVideo | Hong et al. | 2022 | ICLR | 首个大规模预训练的文本到视频 Transformer |
| CogVideoX | Yang et al. | 2024 | arXiv | 开源专家 Transformer；3D 因果 VAE；有竞争力的 FVD |
| Movie Gen | Polyak et al. (Meta) | 2024 | arXiv | 30B 模型；视频 + 音频联合；1080p 16 秒 |
| HunyuanVideo | Kong et al. (Tencent) | 2024 | arXiv | 13B 开源模型；双流条件；3D RoPE |
| Wan | Wan Team (Alibaba) | 2025 | arXiv | MoE 开源模型；图像 + 视频预训练；人类偏好最优 |
| Emu Video | Girdhar et al. (Meta) | 2023 | CVPR 2024 | 分解式文本 + 图像到视频；时间一致性 |
| From Slow Bidirectional to Fast Autoregressive Video Diffusion (CausVid) | Yin et al. | 2025 | CVPR | 通过 DMD 蒸馏为块因果的流式生成器 |
| Wan: Open and Advanced Large-Scale Video Generative Models | Wan Team (Alibaba) | 2025 | arXiv | Wan 2.1（14B/1.3B）技术报告；Wan-VAE；开放权重 |
| Seedance 1.0 | ByteDance Seed | 2025 | arXiv | 原生多镜头文生视频/图生视频；视频 RLHF；约 10 倍蒸馏推理加速 |
| Self Forcing | Huang et al. | 2025 | NeurIPS | 在自身推演上训练自回归视频扩散；实时流式生成 |
| Diffusion Adversarial Post-Training (Seaweed-APT) | Lin et al. (ByteDance) | 2025 | arXiv | 通过对抗后训练实现单步视频生成 |
| Video models are zero-shot learners and reasoners | Wiedemer et al. (Google DeepMind) | 2025 | arXiv | Veo 3 零样本完成感知/操控/推理任务；"帧链" |

---

## 基准性能

| 模型 | 数据集 | 指标 | 分数 | 备注 |
|-------|---------|--------|-------|-------|
| Sora | UCF-101 | FVD | 未公开报告 | 技术报告；未披露 FVD |
| SVD | UCF-101 | FVD | 242 | 14 帧，256px |
| CogVideoX-5B | VBench Total | 分数 | 81.97 | 全面的 16 维基准 |
| HunyuanVideo | VBench Total | 分数 | 85.09 | 发布时最佳开源模型 |
| Wan 2.1 | VBench Total | 分数 | 83.2 | 开源模型；人物主体质量 |
| Movie Gen | 人类偏好 vs Kling | 胜率 | 73% | Meta 内部评估 |
| Veo 2 | EvalCrafter | Overall | 未公开报告 | Google 在专有基准上的评估 |
| Seedance 1.0 | Artificial Analysis Video Arena（T2V、I2V） | 排名 | 发布时第一（2025 年 6 月） | 第三方人类偏好竞技场；排名变化频繁 |
| Self Forcing（基于 Wan-1.3B） | 流式延迟 | 吞吐量 | 实时（单张 H100 约 17 fps，据报道） | 首帧延迟低于 1 秒 |

---

## 优缺点

| 方面 | 优点 | 缺点 |
|--------|------|------|
| **时空 DiT** | 统一架构可处理任意分辨率/时长；时间一致性强；随算力扩展良好 | 长序列上的二次注意力代价高；长视频需要窗口注意力近似 |
| **开源模型** | HunyuanVideo/Wan 可与闭源模型竞争；社区庞大；可微调 | 落后于专有前沿 6–12 个月；全质量运行需 40GB 以上显存 |
| **视频中的流匹配** | 比 DDPM 采样更快、一致性更好；SD3/FLUX 的经验可迁移 | 极长序列下训练不稳定；运动 ODE 的积分误差随时间累积 |
| **文本条件化** | 自然语言控制；对新概念零样本泛化 | 通过文本精确控制运动、轨迹或时间并不可靠；"镜头缓慢推近"之类的语义不明确 |
| **专有前沿** | Sora/Veo/Movie Gen 实现接近照片的真实感并带音频 | 权重闭源；无法进行社区微调；API 访问成本高；失败模式不透明 |

---

## 开放问题与研究空白

- **长视频连贯性**：对于电影级生成器，在 30 秒以上保持身份、光照和场景布局的一致性仍未解决。交互式世界模型（Genie 3）在 720p 下达到分钟级一致性，但依靠的是仍会漂移的自回归记忆；显式场景记忆（3D 缓存、检索过去的帧）与隐式上下文之间如何取舍，仍是开放的设计问题。
- **自回归视频中的暴露偏差**：Self Forcing 式训练减少但并未消除长推演中的误差累积；如何把实时因果生成与双向教师模型的质量结合起来仍是开放问题。
- **物理准确性**：当前模型展现出涌现的物理合理性，但在刚体碰撞、流体飞溅和可变形体动力学上会可预见地失败。整合可微物理仿真器或物理信息损失是一个活跃方向。
- **相机轨迹控制**：自然语言的相机指令（"缓慢推近"、"180° 环绕"）的落地效果很差。视频模型中正在探索显式的相机条件（外参矩阵序列），类似于图像领域的 ControlNet。
- **高效长视频生成**：生成 5 分钟以上的 1080p 视频需要分层潜表征或流式架构，而目前任何开源模型都尚不具备。在当前 token 密度下，内存需求随时长线性增长。
- **音视频同步**：Movie Gen、Veo 3/3.1、Sora 2 和 LTX-2 表明音视频联合生成可行且具有商业价值，但精确的音素-视素对齐、多说话人对白归属以及画内声源绑定仍不够精确，且尚无衡量音视频同步质量的标准基准。
- **肖像、同意与版权**：经同意的客串功能（Sora 2）以及制片厂对 Seedance 2.0 输出的反对表明，控制对真实人物和受保护角色的生成已成为首要的产品约束；相应的技术手段（身份过滤器、肖像登记、来源证明）尚不成熟。
- **评估**：VBench 和 EvalCrafter 捕捉了质量的部分维度，但没有基准能充分衡量物理真实感、3D 一致性或细粒度的时间忠实度。关于将视频生成视为世界建模的讨论，参见[世界模型](../06_robotics_and_embodied_ai/02_world_models.md)。
- **个性化与一致性**：在不微调的情况下跨镜头保持主体身份（特定人物、物体），需要比 IP-Adapter 类方法泛化更好的稳健参考图像条件化。

---

## 延伸阅读

- [OpenAI Sora Technical Report (2024)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Peebles & Xie (2023), "Scalable Diffusion Models with Transformers," ICCV 2023](https://arxiv.org/abs/2212.09748)
- [Kong et al. (2024), "HunyuanVideo: A Systematic Framework For Large Video Generation Model"](https://arxiv.org/abs/2412.03603)
- [Polyak et al. (2024), "Movie Gen: A Cast of Media Foundation Models," Meta arXiv](https://arxiv.org/abs/2410.13720)
- [Yang et al. (2024), "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"](https://arxiv.org/abs/2408.06072)
- [Blattmann et al. (2023), "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets"](https://arxiv.org/abs/2311.15127)
- [OpenAI (2025), "Sora 2 is here"](https://openai.com/index/sora-2/)
- [Google DeepMind，Veo 模型页面](https://deepmind.google/models/veo/)
- [TechCrunch (2026), "Google's Gemini Omni turns images, audio, and text into video"](https://techcrunch.com/2026/05/19/googles-gemini-omni-turns-images-audio-and-text-into-video-and-thats-just-the-start/) 以及 [Google (2026), "Start building with Nano Banana 2 Lite and Gemini Omni Flash"](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-omni-flash-nano-banana-2-lite/)
- [Wan Team (2025), "Wan: Open and Advanced Large-Scale Video Generative Models"](https://arxiv.org/abs/2503.20314) 以及 [Wan 2.2 代码库](https://github.com/Wan-Video/Wan2.2)
- [ByteDance Seed (2025), "Seedance 1.0: Exploring the Boundaries of Video Generation Models"](https://arxiv.org/abs/2506.09113)
- [Yin et al. (2025), "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models"（CausVid），CVPR 2025](https://arxiv.org/abs/2412.07772)
- [Huang et al. (2025), "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"](https://arxiv.org/abs/2506.08009)
- [Wiedemer et al. (2025), "Video models are zero-shot learners and reasoners"](https://arxiv.org/abs/2509.20328)
- [Google DeepMind (2025), "Genie 3: A new frontier for world models"](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
