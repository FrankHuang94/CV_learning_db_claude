# PhD Study Guide for Computer Vision Research

> **Last Updated:** June 2026
> **Level:** Foundational
> **Related Sections:** [Key Venues](./01_key_venues.md) · [Key Papers](./00_key_papers.md) · [Software Tools](./03_software_tools.md) · [Research Frontier 2024–2026](../12_research_frontier_2024_2026/00_overview_latest.md)

---

## Overview

A PhD in computer vision is not a straight line from courses to papers to degree. It is an iterative process of building taste — the ability to distinguish between work that matters and work that merely executes — and then developing the technical fluency to pursue the problems you care about. Most students arrive capable of implementing ideas from papers but uncertain how to generate their own. The gap between consumer and producer of research is the central challenge of a PhD, and closing it requires deliberate habits around reading, writing, problem selection, and community engagement that are rarely taught explicitly.

This guide is a practical reference for that transition. It covers the mechanics of reading and critiquing papers, the harder craft of identifying good research directions, and the logistics of publishing in a field where the state of the art can shift within a month. It also addresses the community infrastructure of computer vision: which venues matter and why, who to follow, where to go to learn, and how to build a presence as the field becomes increasingly aware of open-source as a form of scientific communication.

The guide is written for computer vision specifically but draws on broader research methodology literature — Keshav's three-pass reading method, Hamming's "You and Your Research" talk on problem selection, Andrej Karpathy's PhD survival notes, and Chris Olah's writing on research taste. These sources are referenced throughout rather than summarized in isolation, because the goal is to show how they apply to the specific rhythms of CV research in 2024–2026: a moment when the field is advancing faster than annual conference cycles, when the most important papers sometimes appear on arXiv weeks before any review, and when the clearest signal of a research group's impact is often a GitHub repository rather than a citation count.

---

## 1. How to Read Papers

### The Three-Pass Method

S. Keshav's paper "How to Read a Paper" (ACM SIGCOMM Computer Communication Review, 37(3):83–84, 2007; freely available at [dl.acm.org/doi/10.1145/1273445.1273458](https://dl.acm.org/doi/10.1145/1273445.1273458)) describes a three-pass method that remains the best practical framework for the volume and density of CV literature.

**First pass (5–10 minutes):** Read the title, abstract, introduction, section headings, and conclusion. Ignore figures, proofs, and implementation details. The goal is to answer five questions: (1) category — is this a new architecture, a new dataset, a new task formulation, a systems paper? (2) context — which prior work does it build on? (3) correctness — do the assumptions seem plausible? (4) contributions — what does the paper claim to add? (5) clarity — is it written well enough to warrant a second pass? In a fast-moving field, the first pass alone is sufficient for most papers on your reading list. Do it every morning with the previous day's relevant arXiv submissions.

**Second pass (up to one hour):** Read the paper carefully, but skip dense derivations. Pay attention to figures — in computer vision, a good figure conveys the key insight in five seconds; a bad figure signals that the authors have not yet understood their own contribution. Annotate: underline claims you cannot verify, circle undefined notation, note wherever a design choice seems arbitrary. By the end you should be able to summarize the paper's one central claim and its experimental support in two sentences.

**Third pass (several hours):** Attempt to virtually re-implement the paper from scratch: what data would you collect, what baseline would you start from, what ablation would you run first? This is the pass that builds genuine technical understanding. For the papers that will form the core of your related work — typically 10–20 per project — the third pass is non-negotiable.

### Building Intuition vs. Technical Understanding

Many PhD students spend too much time in second-pass reading and not enough time in third-pass doing. The mental model you need for research is not the ability to reproduce a paper's proof, but the ability to predict what will happen when you change a design choice. That predictive intuition only comes from implementing. Within your first year, replicate at least one significant result — not using a released checkpoint, but training from scratch (or near-scratch) on the same data. The experience of watching validation loss curves, diagnosing training instabilities, and finding that the paper omitted a crucial implementation detail is worth more than reading fifty papers in second-pass mode.

Good CV papers have two layers: the surface claim (e.g., "our method achieves X% on benchmark Y") and the structural insight (e.g., "spatial attention at high resolution requires local windows or the memory cost is prohibitive"). The surface claim becomes stale within a year; the structural insight survives. Train yourself to extract the insight by asking: "If I forget the numbers, what do I still know that I did not know before?"

### Organizing What You Read

Use [Zotero](https://www.zotero.org/) (free, open-source) for bibliography management. The Zotero browser extension saves PDF metadata in one click. For the note layer, write one paragraph per paper after your second pass: the contribution, the key insight, the limitations, and one question the paper leaves open. That paragraph is what you will actually use when writing your related work section — not the paper itself.

---

## 2. Identifying Good Research Problems

### Derivative vs. Paradigm-Shifting Work

Richard Hamming's 1986 Bell Labs talk "You and Your Research" (transcript at [cs.virginia.edu/~robins/YouAndYourResearch.html](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html)) poses the question that should haunt every PhD student: "What are the most important problems in your field, and why aren't you working on them?" Most research is derivative: it applies an existing technique to a new domain, swaps one component of a known architecture, or benchmarks a method on a new dataset. Derivative work is necessary — science progresses incrementally — but it rarely produces the papers that define a career or a field. Paradigm-shifting work changes the vocabulary of the field: it introduces a new task framing, a new architecture family, or a new training regime that makes a class of prior methods obsolete.

The distinction between the two is not always obvious in advance. Residual connections looked like a minor implementation trick; they turned out to restructure how depth could be traded for expressivity. Vision Transformers were initially dismissed as "not better than CNNs for vision"; they became the dominant paradigm. The clearest retrospective marker of paradigm-shifting work is that it enables things that were literally impossible before, not merely better on existing benchmarks.

### The "Taste" Question

Chris Olah's essay "Research Taste Exercises" ([colah.github.io/notes/taste/](https://colah.github.io/notes/taste/)) frames taste as a trainable skill: the ability to tell whether a research problem is interesting versus merely tractable. Tractable problems are everywhere — the literature is full of gaps that could be filled. Interesting problems are those where the answer, whatever it turns out to be, would force a conceptual update on how we think about vision, learning, or intelligence.

Practical test: imagine writing the paper. If the paper could be titled "X Method Applied to Y Problem" and the main contribution is the application, it is likely tractable but not interesting. If the paper would require inventing a new section heading in a survey to place it, it is more likely interesting. Another test due to Andrej Karpathy (in his 2016 PhD survival guide at [karpathy.github.io/2016/09/07/phd/](http://karpathy.github.io/2016/09/07/phd/)): the best research problems are ones your advisor has not thought of yet. If your contribution is implementing your advisor's idea better than their previous student, you are executing, not researching.

### Finding Gaps

In computer vision, the most productive way to find gaps is to look for *inconsistencies between what a model claims to understand and what it demonstrably fails at*. Every new benchmark reveals a gap: MME, MMMU, and RealWorldQA all exposed that strong zero-shot CLIP performance on ImageNet classification does not transfer to spatial reasoning, object counting, or fine-grained attribute discrimination. Each such inconsistency is a candidate research problem.

A second productive approach is to follow what the best robotics and autonomy groups are actually deploying: when practitioners abandon an approach that the academic community still treats as unsolved, that abandonment contains information. Conversely, when practitioners report that a problem thought to be solved in the lab fails in deployment, that is a gap. The gap between lab SOTA and real-world deployment is arguably the defining gap of 2024–2026 vision research (see [Research Frontier Overview](../12_research_frontier_2024_2026/00_overview_latest.md)).

---

## 3. Research Taste: Interesting vs. Incremental

A useful heuristic from ICLR and NeurIPS area chair commentary (see the 2025 ICLR rebuttal analysis at [arxiv.org/abs/2511.15462](https://arxiv.org/abs/2511.15462)): reviewers and area chairs consistently reward papers that (a) identify a problem that was not previously recognized, (b) provide an explanation — not just an empirical demonstration — for why a method works, and (c) suggest what to do next. Incremental papers typically score high on (a) and low on (b) and (c).

In CV specifically, "incremental" almost always means one of: (1) a new backbone swap (replacing ResNet with ViT in a two-stage detector and claiming it as a contribution), (2) a dataset that is essentially a harder version of an existing dataset with no new annotations beyond difficulty, or (3) an ablation published as a full paper. By contrast, papers that *explain* a phenomenon — why dense prediction benefits from register tokens in ViT (Darcet et al., 2023, NeurIPS), why training on synthetic data from a world model does not transfer the way you expect (a still-open question in 2025–2026) — have lasting value.

The conference program committees have become more explicit about this since 2024. CVPR 2025 introduced a "reproducibility" meta-review criterion and ICLR 2026 added a structured rebuttal format that forces authors to distinguish new experimental results from clarifications of existing results. Understanding these norms is part of doing good work, not separate from it.

---

## 4. Conference vs. Journal Strategy

### The Main Venues

For CV PhD students, the primary publishing venues form a clear hierarchy:

**Top CV conferences (CVPR / ICCV / ECCV):** CVPR is the highest-volume, highest-impact venue; Google Scholar ranks it second overall across all academic fields by h5-index. ICCV (odd years) and ECCV (even years; ECCV 2026 will be in Malmö, Sweden) are comparably prestigious with lower acceptance rates. Oral presentations at CVPR run under 4% of submissions; posters are roughly 20%. These venues have 3–4 month cycles from submission to acceptance notification, with an author rebuttal period (typically one week) in between. During rebuttal, you may clarify reviewer misunderstandings and add new experimental results — but the bar for flipping a rejection is high. Spend rebuttal time addressing the single most substantive objection with a direct experiment, not defending every peripheral comment.

**Learning-heavy work (NeurIPS / ICLR / ICML):** If your CV contribution is architecturally or theoretically heavy — new training objectives, new attention mechanisms, scaling analyses, theoretical convergence guarantees — NeurIPS, ICLR, and ICML are appropriate co-targets. The review culture is somewhat more theoretical and the PC is more comfortable with negative results. ICLR in particular has a fully open review system on OpenReview, which means reviews are public after the decision. NeurIPS 2025 modified its rebuttal process to an author survey format, reducing the back-and-forth; check the current author FAQ before each cycle.

**Robotics-first work (CoRL / RSS):** For work in embodied AI, VLA models, or robot manipulation, Conference on Robot Learning (CoRL 2026 is in Austin, Texas) and Robotics: Science and Systems (RSS 2026 is in Sydney) are the right primary targets. These communities evaluate contributions differently from pure CV — they care heavily about real robot experiments, sim-to-real transfer, and deployment practicality. A paper accepted at RSS with compelling robot results will often be more influential in the robotics community than the same paper published at CVPR.

**Journals (TPAMI / IJCV):** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI, impact factor approximately 20.8) and International Journal of Computer Vision (IJCV, impact factor approximately 11.6) are the prestige journals for CV. In practice, journal submissions are used for: (1) extended versions of conference papers with substantial new content (typically +30% new material), (2) survey papers, and (3) work that requires more pages than a conference format allows. Review cycles are 3–6 months per round. Do not plan your PhD timeline around TPAMI acceptances as primary outputs; use conferences for cadence and journals for consolidation.

### The Rebuttal

Write your rebuttal assuming the reviewer will spend ten minutes on it. Lead with what you did, not with what they misunderstood. "We ran the ablation the reviewer requested; results in Table R1 show that removing component X drops performance by 3.2 pp, confirming its importance" is actionable. "We believe the reviewer misread our method" is not. For borderline papers, the area chair reads the rebuttal to decide whether to upgrade a weak accept to accept; give them a clear reason to do so.

---

## 5. Open Source as Research Currency

Releasing code, models, and datasets is no longer optional for CV research with empirical claims. Since roughly 2021, the community has converged on an expectation that any paper claiming a new SOTA result will have an associated public repository, and that any new dataset paper will include a download link. Papers Without Code — the inverse of [Papers With Code](https://paperswithcode.com) (operated independently of the Meta-maintained original after Meta discontinued hosting in July 2025) — are increasingly penalized at review.

Practically: open-source your code before submission, not after acceptance. CVPR 2025 and later ICCV/ECCV cycles have encouraged authors to include a (anonymized) code link in supplementary. Releasing a clean, documented repository consistently generates more long-term citation traffic than the paper alone.

**What to release:**
- Model weights (on [Hugging Face Hub](https://huggingface.co/) — the standard distribution channel for CV model weights as of 2024–2026).
- Training code with a reproducibility script that achieves within 0.5% of your reported number.
- Evaluation code that runs on standard benchmarks without manual setup.
- A model card describing training data, intended use, known failure modes, and compute requirements.

For dataset releases, the community standard has converged on Hugging Face Datasets for small-to-medium datasets and AWS/GCS public buckets for large ones. Include a datasheet (following Gebru et al., 2018) as part of the release.

---

## 6. Tools

### Experiment Tracking

[Weights & Biases (W&B)](https://wandb.ai) is the dominant experiment tracking tool for CV research as of 2026. It provides run logging, hyperparameter sweep management (Sweeps), artifact versioning, and collaborative dashboards. The free tier is sufficient for individual PhD students. For teams or large sweeps, the academic plan is available. Hugging Face released [Trackio](https://huggingface.co/docs/trackio) in 2025 as a lightweight, open-source alternative native to the Hub — useful for small experiments where W&B's overhead is not warranted.

### Paper Discovery

- **[arXiv](https://arxiv.org/) (cs.CV, cs.RO, cs.LG, cs.AI):** The primary distribution channel for CV preprints. Subscribe to the daily email digest for cs.CV. The typical submission deadline is Sunday at 23:59 EST; new submissions appear Tuesday morning.
- **[arXiv-sanity-lite](https://arxiv-sanity-lite.com/):** Andrej Karpathy's SVM+TF-IDF recommendation tool, rewritten as a lightweight personal recommender. Tag papers you like; it surfaces similar new papers. Run locally or use the hosted instance.
- **[Semantic Scholar](https://www.semanticscholar.org/):** AI-assisted literature search with citation graph, author disambiguation, and an API for programmatic access to 200M+ papers. Better than Google Scholar for finding all versions of a preprint and for citation context.
- **[Connected Papers](https://www.connectedpapers.com/):** Builds a force-directed visual graph of papers related to a seed paper, using Semantic Scholar as its backend. Excellent for mapping an unfamiliar subfield quickly. Free tier allows two graphs per month; paid tier is unrestricted.
- **[Hugging Face Daily Papers](https://huggingface.co/papers/trending):** Community-curated arXiv highlights, updated daily. Better signal-to-noise than raw arXiv for identifying which preprints are drawing community attention.

### Reference Management

[Zotero](https://www.zotero.org/) (free, open-source) with the Better BibTeX plugin for clean `.bib` export is the standard choice. The browser extension saves PDFs and metadata in one click. Use collections for per-project organization and tags for cross-project themes (e.g., "3DGS", "VLA", "benchmark"). Sync via Zotero's free 300MB cloud or self-host WebDAV for larger libraries. Avoid Mendeley (owned by Elsevier; privacy concerns) and Note that the Better Notes plugin adds structured annotation workflows inside Zotero, useful for building per-paper summaries.

### Code and Compute

- **[Hugging Face Hub](https://huggingface.co/):** Model weights, datasets, and Spaces (demo hosting). The standard distribution channel for open CV models.
- **GitHub + Git LFS:** For code versioning and large file storage. Tag your code at submission time to produce a permanent snapshot that matches the paper.
- **[W&B Artifacts](https://docs.wandb.ai/guides/artifacts):** For versioning intermediate training checkpoints and evaluation results.

---

## 7. Writing

### Structure That Works

The best CV papers have a single, well-stated problem in the abstract and introduction, and every subsequent section exists to support the answer. Write the abstract last; write the introduction second-to-last. The introduction's job is to (1) convince the reader the problem is real and important, (2) explain why prior work does not solve it, and (3) preview your contribution and its evidence. Every sentence in the introduction should be doing one of those three things.

**Related work:** Do not write related work as a catalog of who did what. Write it as a narrative that places your contribution in context. For each cluster of prior work, explain what they collectively got right and what gap they collectively left open — and make the gap your paper's contribution. Use [Semantic Scholar](https://www.semanticscholar.org/) and [Connected Papers](https://www.connectedpapers.com/) to ensure completeness.

**Ablation studies:** Design ablations before you run the main experiments, not after. An ablation answers: "What is the evidence that each component of my method is necessary?" Good ablations are minimal — remove one thing at a time — and the components they test are the same ones claimed in the contribution. Reviewers check whether the ablation table actually supports the claims in the abstract; an ablation that was added post-hoc to pass review is usually detectable.

**Figures:** In CV papers, the first figure is critical. It should convey your problem setup and the key result in one glance — ideally a side-by-side showing what prior work produces versus what your method produces on a representative example. Spend disproportionate time on this figure. Use LaTeX/TikZ or a vector graphics tool (Inkscape, Adobe Illustrator) rather than rasterized screenshots; reviewers on high-DPI monitors will notice.

### The Rebuttal (Writing Side)

See Section 4 for strategy. On mechanics: stay within the word/character limit, use a table for new experimental results (compact, easy to scan), and address all reviewers, not just the most critical one. Do not argue about taste ("we believe our contribution is more significant than the reviewer thinks"); argue about facts ("the reviewer claims X; experiment R2 in the rebuttal tests this directly").

---

## 8. Community

### Researchers to Follow

The following researchers are active on X (formerly Twitter) and post regularly about CV research directions and paper commentary. This is a representative list, not exhaustive:

- **Andrej Karpathy** (@karpathy) — architecture thinking, scaling laws, training insights
- **Kaiming He** (@kaiming0328) — foundational architecture work, current at Meta FAIR
- **Fei-Fei Li** (@drfeifei) — spatial intelligence, World Labs, embodied AI
- **Ross Girshick** (@rbg) — object detection, large-scale vision systems
- **Ilya Sutskever** (@ilyasut) — representation learning, language-vision intersection
- **Aditya Ramesh** (@adityaramesh99) — generative vision
- **Alexei Efros** (@aefros) — image synthesis, visual correspondence, research taste
- **Deva Ramanan** (@devaramanan) — video understanding, pose, efficient inference
- **Justin Johnson** (@justinljohnson) — ViTs, multimodal, computational efficiency

For robotics-CV: **Sergey Levine** (@svlevine), **Chelsea Finn** (@chelseabfinn), **Pieter Abbeel** (@pabbeel), and **Russ Tedrake** (@russelltedrake) post heavily about VLA and embodied AI directions.

### Discord, Slack, and Reading Groups

The [Hugging Face Discord](https://discord.gg/hugging-face) has active #papers and #computer-vision channels. Many top CV labs maintain public arXiv reading groups; Mila, Stanford SAIL, CMU Robotics, and UC Berkeley BAIR run seminar series with public streams and YouTube archives.

Within your institution, starting or joining a weekly paper reading group is probably the highest-ROI time investment in your first year. The norm for productive reading groups: one presenter responsible for a third-pass understanding of one paper per week, followed by open discussion. Rotate presenters.

### Summer Schools

- **[MLSS (Machine Learning Summer School)](https://mlss2026.is.tuebingen.mpg.de/):** The flagship series, run by the Max Planck Institute for Intelligent Systems and rotating hosts. MLSS 2026 is at MPI Tübingen (June) and Columbia University (New York, also June 2026). Highly selective; apply in January. The program combines lectures, tutorials, and labs on current ML frontiers including vision and embodied AI.
- **[Awesome MLSS](https://github.com/awesome-mlss/awesome-mlss):** Community-maintained list of all upcoming MLSS-style summer schools globally.
- **CVPR and ICCV Tutorials:** Half-day and full-day tutorials at CVPR and ICCV are taught by active researchers on current-frontier topics (in 2025: world models, VLAs, 3D Gaussian Splatting extensions, video generation). These are the fastest way to get up to speed on a new subfield from practitioners who are building it. Tutorial slide decks and recordings are typically posted on the conference website within two weeks.
- **[DeepMind Research Retreat / Google Brain Residency programs](https://research.google/programs-and-events/):** Competitive but worth applying to in years 2–4 of a PhD for exposure to large-scale compute and collaborative research culture.

---

## 9. Staying Current in a Fast Field

### The Pace Problem

CV is unusual among technical fields in that the state of the art in many subfields turns over faster than the 12-month conference review cycle. From early 2024 through June 2026, the field has seen: the consolidation of VLMs around the SigLIP+LLM architecture, the emergence of diffusion-based robot policies (π₀, GR00T N1) as the dominant VLA paradigm, the mainstreaming of 3D Gaussian Splatting for scene representation, and the emergence of video generation models (Sora, Cosmos, Wan) as proto-world models. None of these were predictable 18 months earlier. Managing the pace requires a structured reading practice, not just willingness to work hard.

### Practical Workflow

1. **Daily (10 minutes):** Scan the cs.CV arXiv feed (email digest or [arXiv-sanity-lite](https://arxiv-sanity-lite.com/)). Do a first pass on 3–5 papers per day. Flag 1–2 for second pass.
2. **Weekly (2 hours):** Second-pass the flagged papers. Add notes to Zotero. Check [Hugging Face Daily Papers](https://huggingface.co/papers/trending) for community-highlighted work you may have missed.
3. **Monthly:** Do a third-pass deep read of one paper central to your research direction. Attempt a partial re-implementation.
4. **Per conference cycle:** Read all accepted papers in your core subfield from CVPR/ICCV/ECCV (typically 50–100 papers per subfield per year). Most conference papers are posted on arXiv before the camera-ready deadline, so you rarely need to wait for the proceedings.

### Semantic Scholar Alerts

Semantic Scholar supports email alerts for new papers citing a given paper or by a given author. Set alerts for the 10–15 papers that are closest to your research direction. When a new paper cites all of them, it is likely directly relevant.

### The 2024–2026 VLA and World Model Frontier

If your research touches embodied AI, robotics, or autonomous systems, the frontier to track as of June 2026 is the intersection of three questions: (1) Can world models generate physically plausible synthetic training data that transfers to real robot policies? (NVIDIA Cosmos, Genie 2, V-JEPA 2); (2) Can VLA models generalize across embodiments without per-robot fine-tuning? (Octo, OpenVLA, GR00T N1, π₀); (3) Can spatial intelligence models (see [Spatial Intelligence](../12_research_frontier_2024_2026/06_spatial_intelligence.md)) provide the 3D scene understanding that both VLAs and autonomous driving systems currently lack in open-world settings?

These three questions are converging: the answer to all three likely involves a world model that provides both the training data (synthetic diverse trajectories) and the planning substrate (latent-space rollouts) for a universal embodied policy. The key open problems are data coverage (how to cover the long tail of real-world situations), evaluation (how to benchmark generalization without deploying real robots at scale), and architecture (whether autoregressive discrete-action VLAs or continuous diffusion-policy VLAs will dominate at scale). For a deeper map of this frontier, see [Research Frontier Overview](../12_research_frontier_2024_2026/00_overview_latest.md) and [VLA/Embodied 2025–2026](../12_research_frontier_2024_2026/04_vla_embodied_2025_2026.md).

---

## Further Reading

1. **S. Keshav, "How to Read a Paper"** — ACM SIGCOMM CCR, 2007. The canonical three-pass method. [dl.acm.org/doi/10.1145/1273445.1273458](https://dl.acm.org/doi/10.1145/1273445.1273458)

2. **Richard Hamming, "You and Your Research"** — 1986 Bell Labs talk, transcript maintained at UVA. The foundational text on why scientists choose the problems they do and how to choose better ones. [cs.virginia.edu/~robins/YouAndYourResearch.html](https://www.cs.virginia.edu/~robins/YouAndYourResearch.html)

3. **Andrej Karpathy, "A Survival Guide to a PhD"** — 2016 blog post. Specific, honest, opinionated advice on research productivity, advisor relationships, and the psychology of the PhD. [karpathy.github.io/2016/09/07/phd/](http://karpathy.github.io/2016/09/07/phd/)

4. **Chris Olah, "Research Taste Exercises"** — Rough notes on how to train the ability to evaluate whether a research direction is worth pursuing. [colah.github.io/notes/taste/](https://colah.github.io/notes/taste/)

5. **ICLR 2026 Author Guide** — Current rebuttal norms, formatting requirements, and review criteria. Check before each submission cycle. [iclr.cc/Conferences/2026/AuthorGuide](https://iclr.cc/Conferences/2026/AuthorGuide)

6. **NeurIPS 2025 Reproducibility Program** — Community standards for code and data release, including the checklist that has become a de facto standard for CV submissions. [neurips.cc/Conferences/2025](https://neurips.cc/Conferences/2025/CallForReproducibility)
