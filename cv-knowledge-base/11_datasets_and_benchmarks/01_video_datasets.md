# Video Datasets for Action Recognition, Temporal Reasoning, and Egocentric Perception

> **Last Updated:** October 2026
> **Level:** Intermediate / Advanced
> **Related Sections:** [Video Understanding](../02_core_tasks/08_video_understanding.md) · [Core Datasets](./00_core_datasets.md) · [3D and Robotics Datasets](./02_3d_and_robotics_datasets.md) · [Evaluation Metrics](./04_evaluation_metrics.md)

> **🆕 October 2026 update:**
> - Added the **MLLM-era video benchmarks** that frontier labs now report: Video-MME (CVPR 2025), Video-MMMU, MotionBench (CVPR 2025), LongVideoBench and LVBench. Self-reported frontier scores are now ~85–88% on Video-MME and ~87.6% on Video-MMMU (Gemini 3 Pro, Nov 2025).
> - Added **spatial video benchmarks and datasets**: VSI-Bench (CVPR 2025 Oral), plus VSI-590K and VSI-SUPER from Cambrian-S (Nov 2025).
> - Added new **egocentric datasets**: HD-EPIC (CVPR 2025) and EgoLife (CVPR 2025). See the 3D/robotics file for EgoDex.
> - Added **large video-generation and video-language corpora**: Panda-70M, OpenVid-1M (ICLR 2025), Koala-36M (CVPR 2025), and Meta's PE-Video and PLM-Video-Human (2025).
> - Updated the SSv2 state of the art: V-JEPA 2 reaches 77.3% with a frozen attentive probe (Jun 2025), and V-JEPA 2.1 followed in Mar 2026. Also added a note on benchmark saturation and frame-sampling non-comparability.

---

## Overview

Video understanding benchmarks have evolved through three broad generations. The first generation (UCF-101, HMDB-51, Sports-1M, 2011–2014) established the basic action recognition paradigm but used short, trimmed clips with limited diversity. The second generation (Kinetics-400/600/700, Something-Something v2, ActivityNet, 2016–2019) dramatically scaled both clip counts and semantic richness, revealing that models could achieve high accuracy on Kinetics by exploiting static appearance cues rather than motion, motivating Something-Something's bias-toward-temporal-reasoning design. The third generation (Ego4D, EPIC-KITCHENS-100, AVA, Charades, 2018–2022) shifted focus toward untrimmed video, egocentric perception, dense spatio-temporal annotation, and long-form understanding—challenges directly motivated by embodied AI and AR/VR applications.

A recurring methodological tension concerns the distinction between appearance-biased and motion-biased benchmarks. On Kinetics, a state-of-the-art image model applied to single frames achieves ~65% top-1 accuracy, close to many video-specific baselines, because scene context (kitchen → cooking, court → basketball) is highly predictive of the action. Something-Something v2 explicitly removes this shortcut by using template-structured action labels ("Moving [object] closer to [object]") that require understanding the direction and effect of motion, not just the scene. This design choice makes the two benchmarks fundamentally complementary and explains why a single architecture rarely achieves top performance on both without modification.

A third critical theme is long-form and untrimmed video understanding. ActivityNet, Charades, and Ego4D all contain minute- to hour-long videos where the task requires temporal localization of actions within a large context window. This remains an area of active method development, with memory-efficient transformers, hierarchical architectures, and video-language models (Video-LLaMA, TimeChat) competing to handle extended sequences.

A **fourth generation** (2024–2026) has changed what "a video benchmark" means. The evaluated system is now usually a multimodal LLM rather than a classifier, and the benchmark is a *question-answering suite* rather than a labelled clip collection. Examples are Video-MME, Video-MMMU, MotionBench, LongVideoBench and LVBench. These suites are small (hundreds to a few thousand QA pairs), human-written, and designed to defeat single-frame shortcuts and language priors. The same shift has produced *spatial* video benchmarks built on 3D-scanned scenes (VSI-Bench, VSI-SUPER), which probe whether a model can build a persistent mental map from a walkthrough. Meanwhile the *training* side has split into two branches: (i) captioned clip corpora for text-to-video generation (Panda-70M, OpenVid-1M, Koala-36M), and (ii) human-annotated fine-grained video QA and captioning data for video-LLMs (PLM-Video-Human). Classical recognition datasets (Kinetics, SSv2, EPIC-KITCHENS) survive mainly as frozen-encoder probes for self-supervised video models such as V-JEPA 2.

---

## Kinetics-400 / 600 / 700

**Citation:** [Kay2017] Kay, W. et al. (2017). The Kinetics Human Action Video Dataset. arXiv:1705.06950 (Kinetics-400). [Smaira2020] Smaira, L. et al. (2020). A Short Note on the Kinetics-700-2020 Human Action Dataset. arXiv:2010.10864.

The Kinetics series, developed by DeepMind, is the dominant large-scale action recognition benchmark:

- **Kinetics-400:** 240K–300K 10-second YouTube clips, 400 human action classes, ≥400 clips/class (mean 683). Split: ~240K train / 20K val / 40K test.
- **Kinetics-600:** Extended to 480K clips, 600 classes, ≥600 clips/class (mean 762). Additional classes include fine-grained sports actions.
- **Kinetics-700:** ~650K clips, 700 classes, ≥700 clips/class (mean 906). Includes more fine-grained distinctions.
- **Kinetics-700-2020:** Updated version with refreshed splits to address URL link decay.

All versions use 10-second clips at varying resolutions. Classes span human-object interactions (playing instruments, cooking), human-human interactions (handshakes, hugs), and fine-grained sports actions. A critical limitation is that ~10–20% of URLs become unavailable over time, meaning published numbers are not always comparable across labs due to incomplete downloads.

**Primary use:** Pretraining backbone networks (I3D, SlowFast, TimeSformer, Video Swin) for transfer to downstream tasks. Kinetics-400 top-1 accuracy is the standard reported metric for video classification models.

---

## Something-Something v2

**Citation:** [Goyal2017] Goyal, R. et al. (2017). The "something something" video database for learning and evaluating visual common sense. *ICCV 2017*.

Something-Something v2 (SSv2) contains 220,847 short video clips (2–6 seconds, mean 4.0s) collected via crowdsourcing, covering 174 fine-grained physical interaction classes organized as templates (e.g., "Dropping [something] into [something]", "Pretending to pick [something] up"). The dataset was created by 1,300 crowd workers.

**Key splits:** 168,913 train / 24,777 val / 27,157 test.

**Temporal reasoning requirement:** Because the category label depends on the *direction* and *result* of motion (not just the objects present), image-level features generalize poorly. SSv2 is the primary benchmark for evaluating temporal dynamics understanding. Top-1 accuracy of ~70–77% (e.g., MTV, VideoMAE-v2) contrasts sharply with near-90% on Kinetics-400, illustrating the gap in temporal reasoning capability.

**2025–2026 update:** V-JEPA 2 [Assran2025], a self-supervised video encoder pretrained on internet-scale video, reports **77.3% SSv2 top-1 with a frozen attentive probe**. The previous best frozen-probe result was 69.7% (InternVideo2-1B). V-JEPA 2 also reports 90.2% on Diving48, another motion-centric benchmark. The model was released in June 2025, and V-JEPA 2.1 (Mar 2026) added dense-feature objectives. SSv2 is now used mostly as a frozen-representation probe of motion understanding rather than a full fine-tuning leaderboard.

---

## AVA (Atomic Visual Actions)

**Citation:** [Gu2018] Gu, C. et al. (2018). AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions. *CVPR 2018*.

AVA densely annotates 437 fifteen-minute Hollywood movie clips with 80 atomic visual action classes. Actions are spatio-temporally localized (keyframe bounding boxes at 1 Hz), resulting in 1.59M action labels with frequent multi-label co-occurrence per person instance. People are tracked across segments to enable temporal association.

**Key design:** "Atomic" actions are defined as single-concept primitives ("stand", "talk to", "eat") rather than composite activity labels. This tests the ability to reason about simultaneous concurrent activities—a known weakness of activity recognition systems.

**AVA-Kinetics:** A later extension links AVA-style annotations to Kinetics clips, providing larger-scale spatio-temporal localization training data.

---

## ActivityNet

**Citation:** [Heilbron2015] Heilbron, F.C. et al. (2015). ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding. *CVPR 2015*.

ActivityNet v1.3 contains 19,994 untrimmed YouTube videos (total ~849 hours) spanning 200 activity classes, averaged at 137 videos per class. Each video has 1.41 activity instances on average. Split: 10,024 train / 4,926 val / 5,044 test, average video duration 117 seconds.

**Tasks supported:** Temporal activity detection (localization of action segments within untrimmed video), activity classification, and dense video captioning. ActivityNet Captions extends the dataset with dense temporal captions covering 100K natural language descriptions over 20K videos.

**Limitations:** Activity labels are relatively coarse (compared to Something-Something); many classes (e.g., "playing volleyball") are recognizable from static frames, reducing the motion understanding requirement.

---

## Charades

**Citation:** [Sigurdsson2016] Sigurdsson, G.A. et al. (2016). Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding. *ECCV 2016*.

Charades contains 9,848 crowd-sourced videos of daily indoor activities (average 30 seconds), with 157 action classes and 66,500 temporally annotated action intervals, plus 41,104 object class labels across 46 object categories. Videos were collected from 267 participants across 3 continents following script-like instructions, making it naturalistic but slightly scripted. Also includes 27,847 textual video descriptions.

**Charades-STA:** An extension adding temporal grounding queries, pairing natural language sentences with precise video time intervals, enabling temporal sentence grounding research.

---

## Ego4D

**Citation:** [Grauman2022] Grauman, K. et al. (2022). Ego4D: Around the World in 3,000 Hours of Egocentric Video. *CVPR 2022*.

Ego4D is the largest and most diverse egocentric video dataset, collected by an international consortium of 13 universities across 74 worldwide locations in 9 countries. It contains 3,670 hours of egocentric (first-person) video captured by 855+ unique wearers across diverse daily-life scenarios (household, outdoor, workplace, leisure).

**Benchmark suite:** Ego4D ships with five benchmark tasks:
1. **Episodic Memory** (Natural Language Queries, Moments in Time, Object State Change)
2. **Forecasting** (next active object, anticipation)
3. **Hand & Object Interactions**
4. **Audio-Visual Diarization**
5. **Social Interaction**

**Scale advantage:** Ego4D is more than 20× larger than any prior egocentric dataset (EPIC-KITCHENS was ~100 hours). The diversity of environments, cultural contexts, and activities makes it the primary resource for generalizable egocentric representation learning.

---

## EPIC-KITCHENS

**Citation:** [Damen2018] Damen, D. et al. (2018). Scaling Egocentric Vision: The EPIC-KITCHENS Dataset. *ECCV 2018*. [Damen2020] Damen, D. et al. (2020). Rescaling Egocentric Vision. arXiv:2006.13256 (EPIC-KITCHENS-100).

EPIC-KITCHENS-55 (2018) contains 55 hours of egocentric cooking video, 32 participants, 11.5M frames, 39.6K action segments, 454.2K object bounding boxes. EPIC-KITCHENS-100 (2020) extends to 100 hours across 45 participants and 97 kitchens in 4 countries, with multi-instance action recognition labels for verb and noun (e.g., "wash pan", "cut cucumber").

**Unique annotation:** Actions are decomposed into verb + noun pairs rather than composite labels, enabling evaluation of compositional generalization. EPIC-KITCHENS is the primary benchmark for fine-grained egocentric action recognition, action anticipation, and action detection in cooking domains.

**Anticipation SOTA (2025):** V-JEPA 2 reports 39.7% recall@5 on EK-100 action anticipation, up from 27.6% (PlausiVL). Anticipation remains far from solved.

**HD-EPIC (CVPR 2025):** [Perrett2025] A follow-up from the Bristol-led EPIC team. It records unscripted egocentric kitchen activity with much denser, highly detailed annotations and an accompanying VQA benchmark that frontier video-LLMs find hard. It is positioned as a fine-grained, long-tail complement to EPIC-KITCHENS-100.

**EgoLife (CVPR 2025):** [Yang2025] Six participants wore Meta Aria glasses during a week of shared living, with synchronized third-person cameras and mmWave sensors. It targets *ultra-long-context* egocentric assistance (week-scale memory and retrieval) and ships the EgoIT-99K instruction data with EgoGPT/EgoRAG baselines.

---

## HowTo100M

**Citation:** [Miech2019] Miech, A. et al. (2019). HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. *ICCV 2019*.

HowTo100M contains 1.22 million instructional YouTube videos totaling approximately 15 years of video, decomposed into 136 million 4-second clip-caption pairs obtained via automatic speech recognition (ASR) transcripts aligned to video segments. Videos span 23K task categories across 12 domains (cooking, crafts, personal care, gardening, etc.).

**Significance:** HowTo100M established the paradigm of using noisy speech transcripts as free supervision for video-text pretraining. It enabled MIL-NCE [Miech2020], which achieved strong transfer to zero-shot text-video retrieval, and influenced subsequent large-scale pretraining datasets (WebVid, InternVid). The high noise level in ASR captions (misalignment, irrelevant narration) motivated later filtering-based approaches.

---

## WebVid

**Citation:** [Bain2021] Bain, M. et al. (2021). Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval. *ICCV 2021*.

WebVid-2.5M (initial) and WebVid-10M contain video clips scraped from stock footage websites with descriptive alt-text captions. WebVid-2.5M: ~2.5M clips, mean duration 18 seconds. WebVid-10M: 10.7M clips, ~52,000 total hours, mean 18 seconds per clip. Captions are higher quality than ASR transcripts because they were written by professional stock footage describers.

**Note on availability:** WebVid-10M was widely used in video-language pretraining (e.g., for InstructBLIP-Video, Video-ChatGPT) but the dataset's public availability became restricted in 2023 due to copyright concerns. InternVid (Wang et al., 2023) and Panda-70M emerged as successors.

---

## Video Generation and Video-LLM Training Corpora (2024–2026)

The post-WebVid era split video-text data into *generation-oriented* corpora, which need high aesthetic quality, clean shot boundaries and long dense captions, and *understanding-oriented* instruction data.

- **Panda-70M** [Chen2024] (CVPR 2024, Snap): 70M clips captioned by multiple cross-modality teacher models, with a fine-tuned retrieval model selecting the best caption per clip. It was the main open WebVid successor for text-to-video pretraining.
- **OpenVid-1M** [Nan2025] (ICLR 2025): 1M curated high-aesthetic clips with expressive captions. The 1080p subset, **OpenVidHD-0.4M**, has 0.4M clips. It is used directly or as a quality-tuning complement and is cited by open video generators such as Pyramid Flow and Allegro.
- **Koala-36M** [Wang2025] (CVPR 2025, Kuaishou/Kling): a 36M-clip corpus with improved temporal splitting, structured captions averaging over 200 words, and a learned **Video Training Suitability Score (VTSS)** for filtering. It argues that splitting, caption detail and suitability filtering determine dataset quality more than raw scale.
- **PE-Video (PVD)** and **PLM-Video-Human** [Cho2025] (Meta, Apr 2025): PVD has 1M motion-centred videos, 120K of them with human-verified annotations. PLM-Video-Human provides human-annotated fine-grained QA (FGQA), region-temporal localisation and region (dense) captioning data. Both were released with the fully open PerceptionLM and the **PLM-VideoBench** evaluation suite to counter reliance on distillation from proprietary models.
- **VSI-590K** [Yang2025b] (Nov 2025): ~590K spatially grounded video QA samples curated for *spatial sensing*, released with Cambrian-S together with a 3M-sample open video instruction collection (Cambrian-S-3M, Dec 2025).

---

## MLLM-Era Video Benchmarks (2024–2026)

These are small, carefully constructed QA suites that measure what multimodal LLMs can do with video. Their headline numbers depend heavily on the *frame-sampling budget*, the use of subtitles/audio, and the "thinking" budget. Scores are therefore comparable only under matched protocols.

- **Video-MME** [Fu2025] (CVPR 2025): 900 newly collected videos (254 h, 11 s to 1 h) with 2,700 human-written multiple-choice QA pairs across 6 domains and 30 subfields. It has short (<2 min), medium (4–15 min) and long (30–60 min) tiers, each evaluated with and without subtitles. It became the *industry-standard* long-context video metric, cited in the GPT-4.1, Gemini 2.5 Pro (84.8%), GPT-5 and Gemini 3 Pro launches. On the official submission leaderboard the top open entry (video-SALMONN 2+, 72B, Sep 2025) scores 79.7% without subtitles and 81.6% with them. Moonshot's Kimi K2.5 report (Jan 2026) gives 87.4% for Kimi K2.5 and a re-evaluated 88.4% for Gemini 3 Pro. The benchmark is therefore approaching saturation for frontier systems.
- **Video-MMMU** [Hu2025]: 300 college-level lecture videos across 6 disciplines × 3 QA each (900 QA). The questions probe *knowledge acquisition* at three levels: perception, comprehension and adaptation. Gemini 3 Pro reports 87.6% (Nov 2025), and Kimi K2.5 reports 86.6%.
- **MotionBench** [Hong2025] (CVPR 2025, Zhipu/Tsinghua): fine-grained motion-level QA across six capability categories, using web, public-dataset and Unity-synthesised videos. It also releases 5,000 videos with manual fine-grained motion descriptions. Frontier models score about 70% (Kimi K2.5 70.4, Gemini 3 Pro 70.3 as reported by Moonshot), which shows that motion perception lags behind semantic video QA.
- **LongVideoBench** [Wu2024] and **LVBench** [Wang2024b]: long-context interleaved video-language QA, with LVBench videos up to two hours. Reported frontier scores in early 2026 sit in the mid-70s to around 80%.
- **Spatial video: VSI-Bench and VSI-SUPER.** **VSI-Bench** [Yang2025a] (CVPR 2025 Oral, NYU/Yale/Stanford) has over 5,000 QA pairs from 288 egocentric walkthrough videos of ScanNet, ScanNet++ and ARKitScenes validation scenes. Its eight tasks cover configurational, measurement-estimation and spatio-temporal reasoning. **VSI-SUPER** [Yang2025b] (Nov 2025) extends this to arbitrarily long streams, with spatial recall and continual counting tasks designed so that brute-force context extension fails. It motivates *predictive sensing* (surprise-driven memory) rather than longer context windows. See [Multimodal Datasets](./03_multimodal_datasets.md) for image-based spatial suites (MMSI-Bench, ERQA).

---

## Key Papers / Datasets

| Name | Authors/Org | Year | Venue | Key Stats / Contribution |
|---|---|---|---|---|
| Kinetics-400 | Kay et al. (DeepMind) | 2017 | arXiv | ~300K clips, 400 classes, 10s clips; dominant pretraining benchmark |
| Kinetics-700 | Smaira et al. (DeepMind) | 2020 | arXiv | ~650K clips, 700 classes; extended taxonomy |
| Something-Something v2 | Goyal et al. (TwentyBN) | 2017 | ICCV | 220K clips, 174 classes; requires temporal reasoning |
| AVA | Gu et al. (Google) | 2018 | CVPR | 437 movie clips, 80 atomic actions, 1.59M spatio-temporal labels |
| ActivityNet v1.3 | Heilbron et al. | 2015 | CVPR | 20K videos, 200 classes, 849h; untrimmed temporal detection |
| Charades | Sigurdsson et al. (CMU/Allen AI) | 2016 | ECCV | 9.8K videos, 157 actions, 66.5K temporal intervals; indoor scripted |
| Ego4D | Grauman et al. (Meta AI + consortium) | 2022 | CVPR | 3,670h egocentric, 74 locations, 5 benchmark suites |
| EPIC-KITCHENS-100 | Damen et al. (Univ. Bristol) | 2020 | arXiv | 100h cooking, 45 participants, verb+noun decomposition |
| HowTo100M | Miech et al. (INRIA) | 2019 | ICCV | 1.22M videos, 136M clip-caption pairs, ASR-based; video-text pretraining |
| WebVid-10M | Bain et al. (Oxford) | 2021 | ICCV | 10.7M clips, 52Kh; high-quality stock captions |
| Panda-70M | Chen et al. (Snap) | 2024 | CVPR | 70M clips with multi-teacher captions; WebVid successor for T2V |
| OpenVid-1M | Nan et al. (NJU) | 2025 | ICLR | 1M high-aesthetic clips; 0.4M 1080p (OpenVidHD) |
| Koala-36M | Wang et al. (Kuaishou/Kling) | 2025 | CVPR | 36M clips, >200-word structured captions, VTSS suitability filtering |
| PE-Video / PLM-Video-Human | Cho et al. (Meta FAIR) | 2025 | arXiv | 1M videos (120K human-verified); human-annotated FGQA / region-temporal data; PLM-VideoBench |
| Video-MME | Fu et al. | 2025 | CVPR | 900 videos (254h), 2,700 QA; short/medium/long tiers ± subtitles; industry-standard video MLLM metric |
| Video-MMMU | Hu et al. (NTU S-Lab / LMMs-Lab) | 2025 | arXiv | 300 lecture videos, 900 QA; knowledge acquisition (perception → comprehension → adaptation) |
| MotionBench | Hong et al. (Zhipu/Tsinghua) | 2025 | CVPR | Fine-grained motion QA, 6 capability types; +5K motion-caption videos |
| VSI-Bench | Yang, Yang, Gupta, Han, Fei-Fei, Xie (NYU) | 2025 | CVPR (Oral) | 5K+ QA on 288 3D-scanned scene videos; visual-spatial intelligence |
| VSI-590K / VSI-SUPER (Cambrian-S) | Yang, Yang et al. (NYU/Stanford) | 2025 | arXiv | 590K spatial QA training set; streaming spatial recall/counting benchmark |
| HD-EPIC | Perrett et al. (Bristol et al.) | 2025 | CVPR | Highly detailed unscripted egocentric kitchen video + VQA benchmark |
| EgoLife | Yang et al. (NTU et al.) | 2025 | CVPR | Week-long multi-person Aria-glasses lifelog; EgoIT-99K; ultra-long-context QA |

---

## Benchmark Comparison

| Dataset | Videos | Duration/clip | Task | Key Metric | Temporal Reasoning Required |
|---|---|---|---|---|---|
| Kinetics-400 | ~240K | 10s (trimmed) | Action recognition | Top-1 Acc | Low (appearance-biased) |
| Something-Something v2 | 220K | 2–6s (trimmed) | Action recognition | Top-1 Acc | High |
| AVA | 437 movies | 15min (untrimmed) | Spatio-temporal detection | mAP | Medium |
| ActivityNet | 20K | ~2min avg (untrimmed) | Temporal detection | mAP@tIoU | Medium |
| Charades | 9.8K | 30s avg (untrimmed) | Action localization | mAP | Medium |
| Ego4D | 3,670h total | Continuous (untrimmed) | Multi-task egocentric | Task-specific | High |
| EPIC-KITCHENS-100 | 700 sessions | ~8.5min avg | Recognition + detection | Top-1 Acc, mAP | High |
| HowTo100M | 1.22M | ~15s clips (noisy) | Retrieval pretraining | R@1 (downstream) | Low (pretraining) |
| Video-MME | 900 | 11s–1h (3 tiers) | MLLM multiple-choice QA | Accuracy (± subtitles) | Medium–High |
| Video-MMMU | 300 | Lecture-length | Knowledge acquisition QA | Accuracy, Δknowledge | Medium |
| MotionBench | — | Short clips | Fine-grained motion QA | Accuracy | High |
| VSI-Bench | 288 | Scene walkthroughs | Spatial reasoning QA | Accuracy / MRA (numeric) | High (3D memory) |

---

## Pros & Cons

| Dataset | Strengths | Limitations |
|---|---|---|
| Kinetics-400/700 | Massive scale; standard pretraining resource; clean labels | Appearance-biased; URL link decay reduces reproducibility; near-saturated at top-1 |
| Something-Something v2 | Tests true temporal reasoning; crowdsourced diversity | Only 174 classes; short clips; primarily object manipulation |
| Ego4D | Unprecedented scale of egocentric video; multi-benchmark suite | Complex benchmark setup; egocentric domain gap limits transfer |
| EPIC-KITCHENS-100 | Fine-grained compositional labels; well-curated | Single domain (kitchens); limited to cooking activities |
| ActivityNet | Long-form untrimmed video; dense captions available | Coarse activity labels; appearance-sufficient for many classes |
| HowTo100M | Free ASR supervision at massive scale | Noisy alignment; topic drift; requires careful filtering |
| AVA | Dense atomic labels; spatial localization | Small clip count (437 movies); annotation limited to 1 fps keyframes |
| Video-MME | Newly collected (low leakage at release); duration tiers; widely adopted | Near-saturated for frontier models (~85–88%); scores strongly depend on frame budget and subtitles |
| VSI-Bench / VSI-SUPER | Grounded in 3D scans → objective metric answers; probes spatial memory | Indoor scenes only; narrow scanned-scene distribution (ScanNet/ScanNet++/ARKitScenes) |
| Koala-36M / OpenVid-1M | Generation-grade quality filtering; long captions | Captions are model-generated; licensing of source videos remains unclear |

---

## Open Problems & Research Gaps

- **Long-context video understanding:** Current transformer-based video models handle sequences of dozens to hundreds of frames; Ego4D-style tasks require reasoning over thousands. Efficient attention mechanisms, compressed memory, and retrieval-augmented architectures are active research areas.
- **Cross-domain egocentric generalization:** Models trained on EPIC-KITCHENS cooking videos fail catastrophically on Ego4D construction or outdoor tasks. No unified egocentric pretraining benchmark exists.
- **Temporal grounding with natural language:** Temporal sentence grounding (localizing when an event described in text occurs in a long video) is far from solved, especially for complex compositional descriptions.
- **Fine-grained motion understanding:** Distinguishing "picking up gently" from "picking up quickly" or directional motions remains difficult because current video representations are dominated by appearance.
- **Dataset URL decay:** Kinetics and HowTo100M lose a substantial fraction of videos annually as YouTube removes content. Dataset hosting solutions (e.g., Kinetics in cloud buckets) alleviate but do not eliminate this problem.
- **Multi-camera and multi-modal video:** Most benchmarks are monocular video. Real-world systems use multi-camera rigs (AVA from movies is an exception). Datasets combining video with IMU, depth, gaze (as in Ego4D) are underexplored for pretraining.
- **Action anticipation and forecasting:** Predicting future actions from partial observation, a key capability for embodied AI, is evaluated on EPIC-KITCHENS and Ego4D but achievable accuracy is still far from human-level—fundamental progress is needed. Even V-JEPA 2's large 2025 jump reaches only 39.7% recall@5 on EK-100.
- **Protocol comparability of video-MLLM benchmarks:** Video-MME, LVBench and Video-MMMU scores depend on frames sampled (16 to 1,000+, or fps-based), resolution, subtitle/audio access and reasoning budget. Self-reported model-card numbers and independent re-evaluations often differ by several points (e.g. starred re-evaluations in the Kimi K2.5 report). A standard "frame budget × token budget" reporting convention is still missing.
- **Saturation of first-wave video-LLM benchmarks:** Frontier systems now score ~85–88% on Video-MME and ~87% on Video-MMMU only ~18 months after release. Fine-grained motion (MotionBench ~70%) and streaming spatial memory (VSI-SUPER) remain clearly unsolved and are the likely next targets.
- **Spatial supersensing:** VSI-Bench and VSI-SUPER show that MLLMs fail to maintain persistent 3D maps over long egocentric streams. Whether predictive/world-model memory (Cambrian-S, V-JEPA-style objectives) or explicit 3D reconstruction is the right inductive bias is open.

---

## Further Reading

- [Ego4D project website](https://ego4d-data.org/)
- [EPIC-KITCHENS dataset and challenges](https://epic-kitchens.github.io/)
- [Papers With Code — Video Classification benchmark](https://paperswithcode.com/task/video-classification)
- [Zhu & Yang (2020) — ActBERT: Learning Global-Local Video-Text Representations. CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.pdf)
- [Tran et al. (2018) — A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR 2018](https://arxiv.org/abs/1711.11248)
- [DeepMind Kinetics dataset documentation](https://deepmind.com/research/open-source/kinetics)
- [Video-MME benchmark and evaluation pipeline](https://github.com/MME-Benchmarks/Video-MME) — [arXiv:2405.21075](https://arxiv.org/abs/2405.21075)
- [Video-MMMU (arXiv:2501.13826)](https://github.com/EvolvingLMMs-Lab/VideoMMMU)
- [MotionBench (CVPR 2025)](https://github.com/zai-org/MotionBench)
- [Thinking in Space / VSI-Bench (arXiv:2412.14171)](https://github.com/vision-x-nyu/thinking-in-space)
- [Cambrian-S: VSI-590K and VSI-SUPER (arXiv:2511.04670)](https://github.com/cambrian-mllm/cambrian-s)
- [V-JEPA 2 / 2.1 code and results (arXiv:2506.09985)](https://github.com/facebookresearch/vjepa2)
- [HD-EPIC annotations (CVPR 2025)](https://github.com/hd-epic/hd-epic-annotations)
- [EgoLife project (arXiv:2503.03803)](https://github.com/EgoLife-AI/EgoLife)
- [Koala-36M](https://github.com/KlingTeam/Koala-36M) · [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M) · [Panda-70M](https://github.com/snap-research/Panda-70M)
- [Perception Models: PE-Video, PLM-Video-Human, PLM-VideoBench](https://github.com/facebookresearch/perception_models)
