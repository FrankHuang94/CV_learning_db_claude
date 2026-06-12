# Key Venues in Computer Vision Research

> **Last Updated:** June 2026
> **Level:** Foundational
> **Related Sections:**
> - [PhD Study Guide](./04_phd_study_guide.md)
> - [Key Papers](./00_key_papers.md)
> - [Key Labs & Groups](./02_key_labs_and_groups.md)

---

## Overview

Computer vision research is published across a tight cluster of highly competitive conferences and a small set of archival journals, with the field's center of gravity firmly on **conferences** rather than journals—a cultural inheritance from the fast-moving nature of the field where a 6–12 month journal cycle is often too slow. Understanding the venue landscape is essential for a PhD student: it determines where to submit, what to read, how to calibrate the novelty bar, and how to interpret a paper's pedigree. The three "vision-native" conferences (CVPR, ICCV, ECCV) anchor the field, the "learning" conferences (NeurIPS, ICLR, ICML) host the more theoretical and methodological work, and the robotics venues (CoRL, RSS, ICRA, IROS) have become essential as embodied AI rose to prominence (see [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md)).

Two structural facts shape the landscape. First, **acceptance rates are low and falling**—typically 20–27% for the top conferences amid exploding submission volumes—making publication highly competitive and reviewing quality variable. Second, **venue does not equal impact**: many of the most cited works (CLIP, SAM, Stable Diffusion) had outsized influence regardless of venue, and arXiv preprints increasingly set the agenda before formal publication. This file maps the major venues, their scope, selectivity, and the kind of work each rewards, to guide submission and reading strategy (see [PhD Study Guide](./04_phd_study_guide.md)).

---

## Vision-Native Conferences

- **CVPR** (IEEE/CVF Conference on Computer Vision and Pattern Recognition) — the flagship, held annually in June. The largest and most influential CV venue; broad scope; ~22–27% acceptance with thousands of papers. Hosts the most workshops, tutorials, and challenges.
- **ICCV** (International Conference on Computer Vision) — biennial (odd years), the premier international CV conference alongside CVPR; comparable selectivity and prestige. Home of the Marr Prize (best paper).
- **ECCV** (European Conference on Computer Vision) — biennial (even years), the European counterpart; slightly more theory-friendly reputation; springer LNCS proceedings.

Together CVPR/ICCV/ECCV form the "big three" and publish the bulk of canonical CV work.

## Machine-Learning Conferences

- **NeurIPS** — the largest ML venue; hosts learning-theoretic, generative, and foundational work, plus a major Datasets & Benchmarks track. Note: NeurIPS reviewing/format conventions evolve year to year.
- **ICLR** — focused on representation learning and deep learning; fully open-review (OpenReview); fast-moving, favors methods and empirical insight.
- **ICML** — broad ML theory and methods; strong for optimization, generative models, and learning theory.

Vision work with a strong learning/methodology contribution (ViT, MAE, diffusion theory) often targets these.

## Robotics & Graphics Venues

- **CoRL** (Conference on Robot Learning) — the premier robot-learning venue; home of RT-1/2, Diffusion Policy, OpenVLA (see [VLA Models](../06_robotics_and_embodied_ai/01_vla_models.md)).
- **RSS** (Robotics: Science and Systems) — single-track, highly selective; systems + theory; home of ALOHA, UMI, DROID.
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
| CVPR | Conf | Annual (Jun) | ~22–27% | Broad CV, flagship |
| ICCV | Conf | Biennial (odd) | ~26% | Broad CV, international |
| ECCV | Conf | Biennial (even) | ~25–28% | CV, theory-leaning |
| NeurIPS | Conf | Annual (Dec) | ~25% | Learning, generative, datasets |
| ICLR | Conf | Annual (Apr/May) | ~30% | Representation/deep learning |
| ICML | Conf | Annual (Jul) | ~26% | ML theory & methods |
| CoRL | Conf | Annual | ~30% | Robot learning |
| RSS | Conf | Annual | ~25–30% | Robotics systems, single-track |
| TPAMI | Journal | Rolling | n/a (rigorous) | Definitive/survey |

*Acceptance rates are approximate and vary year to year.*

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

- **Reviewing scalability.** Exploding submission volumes strain reviewer quality and consistency.
- **arXiv vs. peer review.** Preprints increasingly set the agenda before/without formal review, raising questions about quality control.
- **Reproducibility.** Inconsistent code/data release despite improving norms (see [Software Tools](./03_software_tools.md)).
- **Novelty bias.** Conferences reward novelty over rigor/replication; negative results are under-published.
- **Venue inflation.** Pressure to publish at "big three" distorts incentives and timelines.
- **Benchmark saturation.** Leaderboard chasing on saturated datasets crowds out problem-finding.

---

## Further Reading

- [CVF Open Access](https://openaccess.thecvf.com/) — free CVPR/ICCV/ECCV proceedings
- [OpenReview](https://openreview.net/) — ICLR/NeurIPS reviews and submissions
- [Papers With Code](https://paperswithcode.com/) — venue-tagged SOTA tracking (note: maintenance status changed in 2025)
- [PMLR](https://proceedings.mlr.press/) — ICML/CoRL/AISTATS proceedings
- [Robotics Proceedings (RSS)](https://www.roboticsproceedings.org/) — RSS archive
