# Video Datasets for Action Recognition, Temporal Reasoning, and Egocentric Perception

> **Last Updated:** June 2026
> **Level:** Intermediate / Advanced
> **Related Sections:** [Video Understanding](../02_core_tasks/08_video_understanding.md) · [Core Datasets](./00_core_datasets.md) · [3D and Robotics Datasets](./02_3d_and_robotics_datasets.md) · [Evaluation Metrics](./04_evaluation_metrics.md)

---

## Overview

Video understanding benchmarks have evolved through three broad generations. The first generation (UCF-101, HMDB-51, Sports-1M, 2011–2014) established the basic action recognition paradigm but used short, trimmed clips with limited diversity. The second generation (Kinetics-400/600/700, Something-Something v2, ActivityNet, 2016–2019) dramatically scaled both clip counts and semantic richness, revealing that models could achieve high accuracy on Kinetics by exploiting static appearance cues rather than motion, motivating Something-Something's bias-toward-temporal-reasoning design. The third generation (Ego4D, EPIC-KITCHENS-100, AVA, Charades, 2018–2022) shifted focus toward untrimmed video, egocentric perception, dense spatio-temporal annotation, and long-form understanding—challenges directly motivated by embodied AI and AR/VR applications.

A recurring methodological tension concerns the distinction between appearance-biased and motion-biased benchmarks. On Kinetics, a state-of-the-art image model applied to single frames achieves ~65% top-1 accuracy, close to many video-specific baselines, because scene context (kitchen → cooking, court → basketball) is highly predictive of the action. Something-Something v2 explicitly removes this shortcut by using template-structured action labels ("Moving [object] closer to [object]") that require understanding the direction and effect of motion, not just the scene. This design choice makes the two benchmarks fundamentally complementary and explains why a single architecture rarely achieves top performance on both without modification.

A third critical theme is long-form and untrimmed video understanding. ActivityNet, Charades, and Ego4D all contain minute- to hour-long videos where the task requires temporal localization of actions within a large context window. This remains an area of active method development, with memory-efficient transformers, hierarchical architectures, and video-language models (Video-LLaMA, TimeChat) competing to handle extended sequences.

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

---

## Open Problems & Research Gaps

- **Long-context video understanding:** Current transformer-based video models handle sequences of dozens to hundreds of frames; Ego4D-style tasks require reasoning over thousands. Efficient attention mechanisms, compressed memory, and retrieval-augmented architectures are active research areas.
- **Cross-domain egocentric generalization:** Models trained on EPIC-KITCHENS cooking videos fail catastrophically on Ego4D construction or outdoor tasks. No unified egocentric pretraining benchmark exists.
- **Temporal grounding with natural language:** Temporal sentence grounding (localizing when an event described in text occurs in a long video) is far from solved, especially for complex compositional descriptions.
- **Fine-grained motion understanding:** Distinguishing "picking up gently" from "picking up quickly" or directional motions remains difficult because current video representations are dominated by appearance.
- **Dataset URL decay:** Kinetics and HowTo100M lose a substantial fraction of videos annually as YouTube removes content. Dataset hosting solutions (e.g., Kinetics in cloud buckets) alleviate but do not eliminate this problem.
- **Multi-camera and multi-modal video:** Most benchmarks are monocular video. Real-world systems use multi-camera rigs (AVA from movies is an exception). Datasets combining video with IMU, depth, gaze (as in Ego4D) are underexplored for pretraining.
- **Action anticipation and forecasting:** Predicting future actions from partial observation, a key capability for embodied AI, is evaluated on EPIC-KITCHENS and Ego4D but achievable accuracy is still far from human-level—fundamental progress is needed.

---

## Further Reading

- [Ego4D project website](https://ego4d-data.org/)
- [EPIC-KITCHENS dataset and challenges](https://epic-kitchens.github.io/)
- [Papers With Code — Video Classification benchmark](https://paperswithcode.com/task/video-classification)
- [Zhu & Yang (2020) — ActBERT: Learning Global-Local Video-Text Representations. CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.pdf)
- [Tran et al. (2018) — A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR 2018](https://arxiv.org/abs/1711.11248)
- [DeepMind Kinetics dataset documentation](https://deepmind.com/research/open-source/kinetics)
