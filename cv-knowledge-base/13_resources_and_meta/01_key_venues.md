# Key Venues in Computer Vision Research

> **Last Updated:** October 2026
> **Level:** Foundational
> **Related Sections:**
> - [PhD Study Guide](./04_phd_study_guide.md)
> - [Key Papers](./00_key_papers.md)
> - [Key Labs & Groups](./02_key_labs_and_groups.md)

> **🆕 October 2026 update:**
> - **CVPR 2026** (Denver, 3–7 June) set records: 16,092 submissions (+24% year on year), 4,089 accepted (~25.4%), and ~12,200 registrants. Best Paper went to D4RT and Best Student Paper to TRELLIS.2.
> - Recent results added: **ICCV 2025** (Honolulu; ~11.2k submissions, ~2.7k accepted; Marr Prize to BrickGPT) and **ECCV 2026** (Malmö; reported ~10.5k submissions, ~2.9k accepted; Best Paper to HKTex, Imperial).
> - New **upcoming-venue calendar**: CVPR 2027 Seattle (paper deadline mid-Nov 2026), ICLR 2027 San Francisco, ICCV 2027 Hong Kong, NeurIPS 2026 Sydney (with Atlanta and Paris satellites), CoRL 2026 Austin, RSS 2027 Athens (reported).
> - Venue comparison table refreshed with the latest acceptance figures. Papers With Code entry updated (sunset in July 2025; data archived on Hugging Face).

---

## Overview

Computer vision research is published across a tight cluster of highly competitive conferences and a small set of archival journals, with the field's center of gravity firmly on **conferences** rather than journals—a cultural inheritance from the fast-moving nature of the field where a 6–12 month journal cycle is often too slow. Understanding the venue landscape is essential for a PhD student: it determines where to submit, what to read, how to calibrate the novelty bar, and how to interpret a paper's pedigree. The three "vision-native" conferences (CVPR, ICCV, ECCV) anchor the field, the "learning" conferences (NeurIPS, ICLR, ICML) host the more theoretical and methodological work, and the robotics venues (CoRL, RSS, ICRA, IROS) have become essential as embodied AI rose to prominence (see [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md)).

Two structural facts shape the landscape. First, **acceptance rates are low and falling**—typically 20–27% for the top conferences amid exploding submission volumes—making publication highly competitive and reviewing quality variable. Second, **venue does not equal impact**: many of the most cited works (CLIP, SAM, Stable Diffusion) had outsized influence regardless of venue, and arXiv preprints increasingly set the agenda before formal publication. This file maps the major venues, their scope, selectivity, and the kind of work each rewards, to guide submission and reading strategy (see [PhD Study Guide](./04_phd_study_guide.md)).

---

## Vision-Native Conferences

- **CVPR** (IEEE/CVF Conference on Computer Vision and Pattern Recognition) — the flagship, held annually in June. The largest and most influential CV venue; broad scope; ~22–27% acceptance with thousands of papers. Hosts the most workshops, tutorials, and challenges. CVPR 2026 (Denver) received 16,092 submissions, more than double the volume of five years earlier, and accepted 4,089 (~25.4%). CVPR 2027 is in Seattle (June 2027).
- **ICCV** (International Conference on Computer Vision) — biennial (odd years), the premier international CV conference alongside CVPR; comparable selectivity and prestige. Home of the Marr Prize (best paper). ICCV 2025 (Honolulu, Oct 2025) accepted ~2.7k of ~11.2k submissions (~24%). ICCV 2027 is listed by the CVF for Hong Kong, 2–8 Oct 2027.
- **ECCV** (European Conference on Computer Vision) — biennial (even years), the European counterpart; slightly more theory-friendly reputation; Springer LNCS proceedings. ECCV 2026 was held in Malmö, Sweden (8–12 Sept 2026; main conference 10–12 Sept), with a reported ~10.5k submissions and ~2.9k acceptances.

Together CVPR/ICCV/ECCV form the "big three" and publish the bulk of canonical CV work.

## Machine-Learning Conferences

- **NeurIPS** — the largest ML venue; hosts learning-theoretic, generative, and foundational work, plus a major Datasets & Benchmarks track. Note: NeurIPS reviewing/format conventions evolve year to year. NeurIPS has become **multi-site**: 2025 ran in San Diego and Mexico City, and 2026 is listed with Sydney as the main venue (December) plus satellite events in Atlanta and Paris.
- **ICLR** — focused on representation learning and deep learning; fully open-review (OpenReview); fast-moving, favors methods and empirical insight. ICLR 2026 was held in Rio de Janeiro (April 2026). ICLR 2027 is in San Francisco (26–30 April 2027); its submission deadline (late Sept 2026) has passed.
- **ICML** — broad ML theory and methods; strong for optimization, generative models, and learning theory. ICML 2026 was held in Seoul (July 2026).

Vision work with a strong learning/methodology contribution (ViT, MAE, diffusion theory) often targets these.

## Robotics & Graphics Venues

- **CoRL** (Conference on Robot Learning) — the premier robot-learning venue; home of RT-1/2, Diffusion Policy, OpenVLA (see [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md)). CoRL 2025 was in Seoul (Sept 2025); CoRL 2026 is in Austin, Texas (9–12 Nov 2026).
- **RSS** (Robotics: Science and Systems) — single-track, highly selective; systems + theory; home of ALOHA, UMI, DROID. RSS 2026 was in Sydney (13–17 July 2026). RSS 2027 is listed for Athens, with a two-stage submission whose first stage is due in early Dec 2026 (per community deadline trackers; confirm on roboticsconference.org).
- **ICRA / IROS** — large IEEE robotics conferences; broad robotics scope including perception and manipulation.
- **SIGGRAPH / SIGGRAPH Asia** — premier graphics venues; home of 3D Gaussian Splatting and rendering/generation work (see [Gaussian Splatting](../04_3d_vision_and_scene/02_gaussian_splatting.md)).

## Archival Journals

- **TPAMI** (IEEE Transactions on Pattern Analysis and Machine Intelligence) — the most prestigious CV journal; extended/definitive versions, surveys, rigorous evaluation.
- **IJCV** (International Journal of Computer Vision) — Springer; comparable prestige; longer-form contributions.
- **JMLR**, **TMLR** — machine-learning journals; TMLR (Transactions on Machine Learning Research) is increasingly used for solid contributions without the novelty-bar pressure of conferences (DINOv2, CoCa appeared in TMLR).

---

## Venue Comparison

| Venue | Type | Cadence | Approx. acceptance | Best for |
|-------|------|---------|--------------------|----------|
| CVPR | Conf | Annual (Jun) | ~22–27% (2026: 25.4% of 16,092) | Broad CV, flagship |
| ICCV | Conf | Biennial (odd) | ~24–26% (2025: ~2.7k of ~11.2k) | Broad CV, international |
| ECCV | Conf | Biennial (even) | ~25–28% (2026: reported ~2.9k of ~10.5k) | CV, theory-leaning |
| NeurIPS | Conf | Annual (Dec; multi-site since 2025) | ~25% | Learning, generative, datasets |
| ICLR | Conf | Annual (Apr/May) | ~30% | Representation/deep learning |
| ICML | Conf | Annual (Jul) | ~26% | ML theory & methods |
| CoRL | Conf | Annual | ~30% | Robot learning |
| RSS | Conf | Annual | ~25–30% | Robotics systems, single-track |
| TPAMI | Journal | Rolling | n/a (rigorous) | Definitive/survey |

*Acceptance rates are approximate and vary year to year.*

---

## Recent Awards (2025–2026)

| Venue | Award | Paper | Affiliation |
|-------|-------|-------|-------------|
| CVPR 2025 (Nashville) | Best Paper | VGGT: Visual Geometry Grounded Transformer | Oxford VGG / Meta |
| ICCV 2025 (Honolulu) | Best Paper (Marr Prize) | Generating Physically Stable and Buildable Brick Structures from Text (BrickGPT) | CMU |
| ICCV 2025 | Best Student Paper | FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models | Technion |
| ICCV 2025 | Best Paper Hon. Mentions | Spatially-Varying Autofocus; RayZer: A Self-supervised Large View Synthesis Model | CMU (autofocus); multi-institution (RayZer) |
| NeurIPS 2025 | Test of Time | Faster R-CNN (Ren, He, Girshick, Sun, 2015) | Microsoft Research (at the time) |
| CVPR 2026 (Denver) | Best Paper | Efficiently Reconstructing Dynamic Scenes One D4RT at a Time | Google DeepMind / Oxford VGG / UCL |
| CVPR 2026 | Best Student Paper | Native and Compact Structured Latents for 3D Generation (TRELLIS.2) | Tsinghua / Microsoft / USTC |
| CVPR 2026 | Best Paper Hon. Mentions | NitroGen (open gaming-agent foundation model); SAM 3D: 3Dfy Anything in Images | NVIDIA et al.; Meta |
| ECCV 2026 (Malmö) | Best Paper | Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat | Imperial College London |

Two trends are visible in the 2025–2026 awards: **feed-forward 3D/4D reconstruction** (VGGT, D4RT) and **3D generation** (TRELLIS.2, SAM 3D, BrickGPT). See [Key Papers](./00_key_papers.md) for the watch list.

## Upcoming Venues (as of October 2026)

| Venue | Dates | Location | Key deadline (verify on the official site) |
|-------|-------|----------|--------------------------------------------|
| CoRL 2026 | 9–12 Nov 2026 | Austin, Texas, USA | Passed (May 2026) |
| NeurIPS 2026 | Dec 2026 | Sydney (main) + Atlanta and Paris satellites | Passed (May 2026) |
| ICLR 2027 | 26–30 Apr 2027 | San Francisco, USA | Passed (25 Sept 2026); rebuttal in Nov 2026 |
| CVPR 2027 | ~20–25 Jun 2027 | Seattle, USA | Paper deadline ~mid-Nov 2026 (abstract a few days earlier) |
| RSS 2027 | Jul 2027 (reported) | Athens, Greece (reported) | Stage 1 ~early Dec 2026 (reported) |
| ICCV 2027 | 2–8 Oct 2027 | Hong Kong | ~March 2027 (TBA) |

*Dates come from the CVF upcoming-conference list and community deadline trackers (ccf-deadlines, Hugging Face ai-deadlines). Official calls for papers take precedence.*

---

## Pros & Cons (Conference vs. Journal)

| Aspect | Conference | Journal |
|--------|-----------|---------|
| Speed | Fast (months); fits CV's pace | Slow (often >1 year) |
| Prestige | Primary currency in CV | High for TPAMI/IJCV; secondary overall |
| Review | Variable quality; rebuttal phase | More thorough, multiple revision rounds |
| Length/depth | Page-limited | Room for full detail, ablations, surveys |

---

## Open Problems & Research Gaps (meta)

- **Reviewing scalability.** Exploding submission volumes strain reviewer quality and consistency. CVPR submissions more than doubled in five years (16,092 in 2026), and venues now require authors to serve as reviewers and police LLM-written reviews.
- **Review confidentiality.** A November 2025 OpenReview API bug briefly exposed reviewer identities for ICLR 2026 submissions, highlighting the fragility of double-blind infrastructure.
- **Geographic access.** Multi-site conferences (NeurIPS 2025 in San Diego and Mexico City; NeurIPS 2026 in Sydney, Atlanta and Paris) respond to visa and travel barriers, but they fragment the community.
- **arXiv vs. peer review.** Preprints increasingly set the agenda before/without formal review, raising questions about quality control.
- **Reproducibility.** Inconsistent code/data release despite improving norms (see [Software Tools](./03_software_tools.md)).
- **Novelty bias.** Conferences reward novelty over rigor/replication; negative results are under-published.
- **Venue inflation.** Pressure to publish at "big three" distorts incentives and timelines.
- **Benchmark saturation.** Leaderboard chasing on saturated datasets crowds out problem-finding.

---

## Further Reading

- [CVF Open Access](https://openaccess.thecvf.com/) — free CVPR/ICCV/ECCV proceedings
- [OpenReview](https://openreview.net/) — ICLR/NeurIPS reviews and submissions
- [Papers With Code archive](https://huggingface.co/datasets/pwc-archive/evaluation-tables) — the original site was sunset in July 2025; its data dumps (papers, code links, evaluation tables) are archived on Hugging Face
- [PMLR](https://proceedings.mlr.press/) — ICML/CoRL/AISTATS proceedings
- [Robotics Proceedings (RSS)](https://www.roboticsproceedings.org/) — RSS archive
- [CVPR 2026 closing release](https://cvpr.thecvf.com/Conferences/2026/News/Closing) — 2026 statistics and awards
- [CVF upcoming conferences](https://www.thecvf.com/?cat=2) — official CVPR/ICCV/WACV dates and locations
- [ccf-deadlines](https://github.com/ccfddl/ccf-deadlines) and [Hugging Face AI Deadlines](https://huggingface.co/spaces/huggingface/ai-deadlines) — community-maintained deadline trackers
