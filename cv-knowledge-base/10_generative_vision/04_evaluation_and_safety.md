# Evaluation and Safety of Generative Vision Models

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](./01_diffusion_models.md) | [GANs and VAEs](./00_gans_and_vaes.md) | [Evaluation Metrics](../11_datasets_and_benchmarks/04_evaluation_metrics.md) | [Video Generation](./02_video_generation.md)

> **🆕 October 2026 update:**
> - New section on **regulation and labelling mandates**: EU AI Act Art. 50 transparency obligations (applicable from 2 Aug 2026) and the accompanying Code of Practice on marking/labelling AI-generated content; China's AI-content labelling Measures (1 Sep 2025); the US TAKE IT DOWN Act (May 2025).
> - Watermarking/provenance updated: SynthID Detector and in-app SynthID verification, C2PA Content Credentials in Nano Banana 2, Sora 2 and Pixel 10 cameras.
> - New section on **human-preference arenas and next-generation benchmarks** (LMArena/Artificial Analysis arenas, GenEval saturation via RL, OneIG-Bench, VBench-2.0, Physics-IQ).
> - New material on likeness/consent controls and 2025–2026 copyright litigation; Key Papers, Open Problems and Further Reading extended.

---

## Overview

The rapid proliferation of high-quality generative vision systems — text-to-image, text-to-video, and 3D generation — has created an urgent need for evaluation methodologies that go beyond perceptual quality scores and address compositional correctness, alignment with human intent, safety, and authenticity. This file surveys the principal automatic and human evaluation benchmarks, the mathematical foundations of the most widely used metrics, and the safety and provenance tooling ecosystem that has emerged in response to the deepfake and synthetic media crisis.

Automatic metrics for generative image quality fall into two camps: **reference-based** metrics that compare the statistics of generated and real image distributions (FID, IS, FVD), and **alignment-based** metrics that measure how well a generated image matches its conditioning signal (CLIPScore, ImageReward, DINO score). Reference-based metrics require a held-out reference set (typically 30K–50K images from the training distribution) and are sensitive to biases in the reference set and the feature extractor (Inception V3, ViT-g). Alignment-based metrics are reference-free but depend on the quality of the underlying text-vision encoder.

The safety landscape for generative vision encompasses three overlapping concerns: (1) **deepfakes** — synthetic media that deliberately misrepresent real people or events, enabling fraud, misinformation, and non-consensual intimate imagery (NCII); (2) **memorization and data privacy** — generative models' tendency to reproduce near-verbatim copies of training data, enabling reconstruction of private or copyrighted material; and (3) **provenance and attribution** — the inability of downstream consumers to determine whether a piece of media was AI-generated. Responses to these concerns include invisible watermarking (SynthID [DeepMind2023], Stable Signature [Fernandez2023]), content provenance standards (C2PA), and dataset filtering for consent and licensing. Since 2025 these technical measures have become **legal obligations** in major jurisdictions: China's labelling Measures took effect on 1 September 2025, and the EU AI Act's Article 50 transparency obligations — machine-readable marking of synthetic outputs and disclosure of deepfakes — apply from 2 August 2026. A fourth concern grew sharply with photorealistic video-with-audio models (Veo 3, Sora 2, Seedance 2.0) and conversational image editors: **likeness and consent**, i.e. controlling who can be depicted.

---

## Fréchet Inception Distance (FID)

FID [Heusel2017] measures the distance between the distributions of real and generated images by comparing statistics of activations from the Inception V3 pool3 layer:

```
FID = ||mu_r - mu_g||^2
    + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^{1/2})

where:
  mu_r, Sigma_r = mean and covariance of real image features
  mu_g, Sigma_g = mean and covariance of generated image features
  Features extracted from Inception V3 pool3 layer (2048-d)
  Tr = matrix trace
  (Sigma_r * Sigma_g)^{1/2} = matrix square root (via eigendecomposition)

Properties:
  - Lower FID = better quality/diversity match to real data
  - FID(P, P) = 0; FID(P, Q) = FID(Q, P) (symmetric)
  - Sensitive to number of samples: N >= 10,000 strongly recommended
  - Sensitive to Inception V3 pretraining and resolution preprocessing

Common failure modes:
  - Biased by reference dataset (e.g., FFHQ vs. CelebA give different baselines)
  - Penalizes diversity: a model that memorizes training data can achieve low FID
  - Inception features are biased toward ImageNet object categories
```

**sFID** (spatial FID) uses intermediate Inception activations to capture spatial structure rather than global statistics, better penalizing spatial incoherence and object placement artifacts.

---

## Inception Score (IS)

The Inception Score [Salimans2016] measures both quality (the conditional label distribution $p(y|x)$ should be sharp) and diversity (the marginal $p(y)$ should be flat):

```
IS(G) = exp(E_{x~p_g} [KL(p(y|x) || p(y))])

where:
  p(y|x) = Inception V3 class probabilities for generated image x
  p(y) = marginal = (1/N) * sum_x p(y|x)
  KL = Kullback-Leibler divergence

Expanded form:
  IS = exp( H(p(y)) - E_x[H(p(y|x))] )
  = exp( high marginal entropy - low conditional entropy )

Interpretation:
  Good generator: high-quality images (low H(y|x)) AND diverse outputs (high H(y))

Limitations:
  - Only measures ImageNet class alignment; fails on non-ImageNet content
  - Insensitive to within-class diversity
  - Can be fooled by single sharp image per class
  - Not normalized: depends on Inception V3 version
```

---

## CLIPScore and DINO Score

**CLIPScore** [Hessel2021] measures text-image alignment using cosine similarity in CLIP's joint embedding space:

```
CLIPScore(image, text) = w * max(cos(CLIP_I(image), CLIP_T(text)), 0)

where:
  CLIP_I = CLIP image encoder output (normalized)
  CLIP_T = CLIP text encoder output (normalized)
  w = 2.5 (scaling constant in original paper)
  cos = cosine similarity

RefCLIPScore (with reference image):
  RefCLIPScore = harmonic_mean(CLIPScore(gen, text), CLIPScore(gen, ref))

Properties:
  - Reference-free; can evaluate any generated image against its prompt
  - Correlates with human judgment on text-image alignment
  - Weak on fine-grained attribute binding and counting
  - Saturates at high quality; not sensitive to small textual changes
```

**DINO Score** uses ViT-DINO features for intra-class consistency in personalization experiments (DreamBooth evaluation): high DINO score = generated images look like the subject in reference photos.

---

## Fréchet Video Distance (FVD)

FVD [Unterthiner2019] extends FID to video by replacing Inception V3 with an I3D network trained on Kinetics:

```
FVD = ||mu_r - mu_g||^2
    + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^{1/2})

where features are I3D activations on video clips.

Practical considerations:
  - N >= 2048 video clips recommended for stable estimates
  - Highly sensitive to temporal resolution and frame sampling
  - I3D biased toward Kinetics action classes
  - FVD alone does not capture physics plausibility or text-video alignment
```

VBench [Huang2024] provides a more comprehensive multi-dimensional video evaluation framework with 16 dimensions including subject consistency, background consistency, temporal flickering, motion smoothness, and text-video alignment, addressing the limitations of scalar FVD.

---

## GenEval

GenEval [Ghosh2023] is a compositional text-to-image benchmark evaluating specific generation capabilities:

```
Categories:
  1. Single object (baseline; should be easy)
  2. Two objects (co-occurrence)
  3. Counting (1-4 objects)
  4. Colors (attribute binding)
  5. Color attribution (which object has which color)
  6. Spatial relationships (left/right/above/below)
  7. Non-spatial relationships (holding, wearing)

Evaluation: Object detection + attribute verification on generated images
Score per category + overall score (0-1 range)

Representative scores (at time of publication):
  SDXL: 0.55 overall
  DALL-E 3: 0.67 overall
  FLUX.1-dev: 0.66 overall
  Ideogram 2.0: ~0.70 overall

GenEval 2 (2025): Expanded with more categories and adversarial prompts.
```

By 2025 GenEval is close to **saturation**: RL post-training with detector-based rewards (Flow-GRPO) lifts SD3.5-Medium from 0.63 to 0.95, so high GenEval scores no longer separate frontier systems and can reflect reward-hacking toward the detector rather than general compositional skill.

---

## T2I-CompBench and ImageReward

**T2I-CompBench** [Huang2023] evaluates compositional text-to-image generation across 8 dimensions (attribute binding, spatial relations, non-spatial relations, complex compositions) using VQA-based automatic evaluation:

```
Evaluation pipeline:
  1. Generate images from 8,000 compositional prompts
  2. Use VQA model (BLIP-VQA or UniDet) to verify attribute presence
  3. Score per dimension + overall weighted score

Dimensions:
  - Color binding ("a red cube")
  - Shape binding ("a round table")
  - Texture binding ("a striped shirt")
  - Spatial relationships (left/right/above)
  - Non-spatial relationships (riding, holding)
  - Complex compositions (combining all above)
```

**ImageReward** [Xu2023] trains a reward model on human preference data (collected from real text-to-image generation user sessions) to predict human rating of a generated image given its text prompt. Provides a scalar "human alignment" score that correlates better with human judgment than FID or CLIPScore on alignment-sensitive tasks. Used as a reward in RLHF fine-tuning of diffusion models (DPOK, DDPO).

---

## Human-Preference Arenas and Next-Generation Benchmarks (2025–2026)

With automatic compositional benchmarks saturating, the de facto leaderboard for frontier image and video models became **pairwise human-preference arenas**: LMArena's Text-to-Image and Image Edit arenas and the Artificial Analysis image/video arenas, which compute Elo / Bradley–Terry ratings from blind A/B votes. Vendors now announce launches by arena rank (e.g. OpenAI reported that gpt-image-2 reached #1 on Image Arena within hours of its April 2026 launch; Seedance 1.0 led the Artificial Analysis video arenas in mid-2025).

```
Bradley-Terry arena rating:
  P(model i beats model j) = sigma(s_i - s_j) = 1 / (1 + exp(-(s_i - s_j)))
  s = argmax_s  sum_{votes (i>j)} log sigma(s_i - s_j)  - lambda * ||s||^2
  Elo-style display: R_i = 400 * s_i / ln(10) + offset
Caveats: prompt distribution set by volunteer users; style/length confounders;
         ties and vendor-specific "arena tuning"; ratings drift as the pool changes.
```

Newer automatic benchmarks target what arenas and GenEval miss:

| Benchmark | Year | Focus |
|-----------|------|-------|
| OneIG-Bench | 2025 | Omni-dimensional T2I evaluation: alignment, text rendering, reasoning/knowledge, style, diversity |
| VBench-2.0 [Zheng2025VBench2] | 2025 | "Intrinsic faithfulness" of video: human fidelity, controllability, creativity, physics, commonsense |
| Physics-IQ [Motamed2025] | 2025 | Real filmed physical experiments; tests whether video models predict physically correct continuations |
| Video zero-shot reasoning suites [Wiedemer2025] | 2025 | Perception, modelling, manipulation and reasoning tasks solved by prompting a video model |

A consistent finding across Physics-IQ and VBench-2.0 is that **visual realism and physical understanding are only weakly correlated**: models rated most photorealistic by humans can still violate basic dynamics.

---

## Deepfakes and Detection

Deepfakes — AI-generated or AI-manipulated media that misrepresents real people — constitute a major societal risk. The detection landscape is an adversarial arms race:

```
Detection methods:
  1. Frequency-domain artifacts: GAN-generated images have systematic
     spectral artifacts (grid-like patterns) detectable by FFT analysis
  2. Physiological signals: Inconsistent blinking, pulse detection via
     rPPG; suppressed in modern models
  3. Semantic inconsistencies: Eye reflections, finger count, jewelry
     that violates physics; increasingly rare in frontier models
  4. Neural fingerprints: Reconstruction-based detectors trained on
     "is this from distribution X?" using model-specific artifacts
  5. Universal detectors: CNNDetect, UnivFD, trained across GAN/diffusion
     models; drop in accuracy as new models emerge (generalization gap)

Key challenge:
  Diffusion models (DDPM, LDM) do NOT exhibit the spectral grid artifacts
  of GANs, making frequency-based detectors fail.
  Modern detectors must train on diverse synthetic data.
```

**Likeness and consent controls (2025–2026).** With audio-visual generators able to produce convincing talking-head video of real people, providers added *upstream* controls in addition to downstream detection: Sora 2's opt-in "cameos" let a verified user decide who may generate their likeness and revoke access, public-figure and deceased-person depictions were restricted after complaints, and image editors refuse or watermark edits of real people. These controls failed visibly in several 2025–2026 incidents (e.g. widely reported misuse of chatbot-integrated image editing to create sexualized images of real people in early 2026, which prompted regulatory investigations, and studio objections to Seedance 2.0 clips of celebrities and copyrighted characters), underscoring that policy filters are not a substitute for robust provenance.

---

## Watermarking: SynthID and Stable Signature

**SynthID** [DeepMind2023, updated as SynthID-Image 2024] is Google DeepMind's production watermarking system for AI-generated images:
- Embeds an invisible watermark during image generation (not post-hoc) by fine-tuning a small detector alongside the generation model.
- Watermark is imperceptible to humans but detectable by a paired classifier.
- Robust to JPEG compression, resizing, cropping, and color transformations.
- Deployed in Imagen, Veo and all Gemini image models (Nano Banana, Nano Banana Pro, Nano Banana 2); SynthID-Text was open-sourced in 2024.
- The image/video watermarking model is not open-source. Google launched a **SynthID Detector** portal (May 2025) and, with Nano Banana Pro (Nov 2025), let users ask the Gemini app whether an image was generated or edited by Google AI. Nano Banana 2 (Feb 2026) outputs carry both SynthID and interoperable C2PA Content Credentials — illustrating the emerging "watermark + signed manifest" dual-layer approach that regulators expect.

**Stable Signature** [Fernandez2023] embeds a binary message (48 bits) into the latent diffusion decoder weights, so every image produced by the decoder carries the watermark automatically:

```
Training:
  1. Train watermark decoder W: image -> 48-bit message
  2. Fine-tune VAE decoder D to produce images x = D(z) such that:
     W(x) ~= target_message  (message recovery loss)
     ||x - D_orig(z)||_lpips < epsilon  (perceptual fidelity constraint)

Properties:
  - Watermark baked into decoder; no inference overhead
  - Robust to JPEG (Q=30), Gaussian noise, blur
  - Traceable: different models can have different embedded messages
  - Limitation: watermark is fixed to the decoder, not input-conditioned
```

---

## C2PA Content Provenance

The Coalition for Content Provenance and Authenticity (C2PA) defines an open standard for cryptographically signed content manifests that record creation and edit history:

```
C2PA manifest attached to media file:
  {
    "claim_generator": "StabilityAI/StableDiffusion3",
    "assertions": [
      {"label": "c2pa.training_and_data_mining", "data": {...}},
      {"label": "c2pa.ai_generative_info", "data": {...}}
    ],
    "signature": {
      "alg": "ES256",
      "cert": "<X.509 cert of signing authority>",
      "issuer": "StabilityAI"
    }
  }

Verification:
  - Certificate chain validation (PKI)
  - Hash of media content embedded in manifest
  - Tamper-evident: any pixel change invalidates hash

Adoption:
  - Adobe (Firefly), Microsoft, Sony, OpenAI (DALL-E 3), Leica, Qualcomm
  - "Content Credentials" badge visible in Adobe software
  - CAI (Content Authenticity Initiative) is industry implementation body
  - 2025-2026: OpenAI GPT Image and Sora 2 outputs; Google Nano Banana 2;
    Google Pixel 10 camera signs photos natively (Aug 2025)
```

C2PA provides provenance but not authenticity detection: a bad actor can generate media without C2PA credentials and simply omit them. It is most useful for establishing trust in signed media rather than detecting unsigned synthetic media. The industry therefore converged on **layered provenance**: (1) a signed C2PA manifest (rich, but strippable), (2) an invisible watermark that survives metadata stripping and moderate edits (SynthID, Stable Signature-style), (3) optionally a visible label (Sora 2's moving watermark), and (4) fingerprint/hash lookups for recovery when both are removed. Camera-side signing (Pixel 10, Leica, Sony) attacks the problem from the other end by making *authentic* capture verifiable.

---

## Regulation and Labelling Mandates (2025–2026)

| Jurisdiction | Instrument | In force | Requirements relevant to generative vision |
|--------------|-----------|----------|-------------------------------------------|
| EU | AI Act, Article 50 | 2 Aug 2026 | 50(2): providers of systems generating synthetic image/video/audio/text must mark outputs in a machine-readable format detectable as artificially generated, using effective, interoperable, robust and reliable techniques "as far as technically feasible"; 50(4): deployers must disclose deepfakes (reduced obligations for evidently artistic/satirical works) |
| EU | Code of Practice on marking and labelling AI-generated content | Drafting from late 2025; intended to support Art. 50 compliance | Voluntary code covering watermarking, metadata, fingerprinting/logging and common labels/icons for deepfakes |
| China | Measures for Labelling AI-Generated Synthetic Content + mandatory national standard GB 45438-2025 | 1 Sep 2025 | Explicit (visible/audible) labels plus implicit labels in file metadata; platforms must detect and label |
| USA (federal) | TAKE IT DOWN Act | Signed 19 May 2025; platform process due May 2026 | Criminalizes publishing non-consensual intimate imagery including AI-generated "digital forgeries"; 48-hour takedown on valid request |
| USA (California) | AI Transparency Act (SB 942, as amended) | Operative 2026 (reported) | Large GenAI providers must offer free detection tools and embed latent disclosures |

Implementation questions remain: what counts as "robust" marking for open-weight models where users can remove the watermark module; whether metadata-only C2PA satisfies Art. 50(2) given trivial stripping; and how to label partially AI-edited media. The Commission's November 2025 "Digital Omnibus" proposal also floated transitional relief for Art. 50(2) for systems already on the market; practitioners should check the final adopted text and the Commission's guidelines.

**Copyright and training-data litigation.** Rights-holder suits shape data practice as much as regulation: Getty Images v. Stability AI (UK High Court judgment, Nov 2025) largely rejected Getty's secondary-infringement theory that the model weights themselves are an "infringing copy" while making limited trademark findings on reproduced watermarks; Disney and Universal sued Midjourney (June 2025) over generated depictions of their characters; and studios moved toward licensing, e.g. Disney's December 2025 agreement with OpenAI to license characters for Sora alongside an equity investment. These cases make memorization research (next section) directly legally relevant.

---

## Memorization in Diffusion Models

Carlini et al. [Carlini2023] demonstrated that diffusion models memorize and reproduce training images verbatim at measurable rates:

```
Memorization types:
  1. Verbatim memorization: exact pixel-level reproduction of training image
  2. Template memorization: structural template reproduced with different content
  3. Concept memorization: specific concept (e.g., a person's face) reproduced

Measurement protocol [Carlini2023]:
  - Use captions from training set as prompts
  - Generate 500 images per caption with top-500 most-duplicated prompts
  - L2 nearest-neighbor search in pixel space against training set
  - Threshold L2/resolution < 0.1 -> "memorized"

Findings:
  - ~350 images memorized out of first 500K training images analyzed
  - Verbatim memorization rate ~0.03% of queried prompts
  - Heavily duplicated training images memorized more frequently
  - DDPM objective provides no formal privacy guarantee

Mitigations:
  - Deduplication of training data (reduces memorization ~5x)
  - Differential privacy training (large epsilon; quality penalty)
  - Membership inference tests during training; remove duplicates
  - Noise augmentation at inference (not fully effective)
```

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| GANs Trained by a Two Time-Scale Update Rule (FID) | Heusel et al. | 2017 | NeurIPS | Fréchet Inception Distance; became standard quality metric |
| Improved Techniques for Training GANs (IS) | Salimans et al. | 2016 | NeurIPS | Inception Score; quality + diversity metric |
| CLIPScore | Hessel et al. | 2021 | EMNLP | Reference-free image-caption alignment via CLIP |
| Towards Video Understanding (FVD) | Unterthiner et al. | 2019 | arXiv | Fréchet Video Distance; I3D features for video evaluation |
| GenEval | Ghosh et al. | 2023 | NeurIPS | Compositional T2I evaluation; object/attribute/spatial |
| T2I-CompBench | Huang et al. | 2023 | NeurIPS | Multi-dimensional compositional T2I benchmark; 8,000 prompts |
| ImageReward | Xu et al. | 2023 | NeurIPS | Human preference reward model for T2I; RLHF signal |
| Extracting Training Data from Diffusion Models | Carlini et al. | 2023 | USENIX Security | Quantified memorization in DDPM/LDM; verbatim reproduction |
| SynthID | Lim et al. (Google DeepMind) | 2024 | arXiv | Production-scale invisible watermarking; robustness analysis |
| Stable Signature | Fernandez et al. | 2023 | ICCV | Per-decoder binary watermark baked into LDM decoder |
| VBench | Huang et al. | 2024 | CVPR | 16-dimension video generation benchmark; comprehensive evaluation |
| WAVES: Benchmarking the Robustness of Image Watermarks | An et al. | 2024 | ICML | Standardized stress-testing of image watermarks under distortion, regeneration and adversarial removal attacks |
| Do generative video models understand physical principles? (Physics-IQ) | Motamed et al. | 2025 | arXiv | Real-world physics benchmark; shows realism ≠ physical understanding |
| VBench-2.0 | Zheng et al. | 2025 | arXiv | Intrinsic-faithfulness video benchmark (physics, commonsense, controllability, human fidelity) |
| Flow-GRPO | Liu et al. | 2025 | NeurIPS | RL post-training that saturates GenEval (0.63 → 0.95), exposing benchmark limits |

---

## Benchmark Performance

| Model | Benchmark | Metric | Score | Notes |
|-------|-----------|--------|-------|-------|
| DALL-E 3 | GenEval | Overall accuracy | 0.67 | Best at time of paper |
| FLUX.1-dev | GenEval | Overall accuracy | 0.66 | Open model competitor |
| SDXL 1.0 | GenEval | Overall accuracy | 0.55 | Dual-encoder; 1024px |
| SD 2.1 | MS-COCO 30K | FID | 11.7 | Standard reference |
| SDXL | MS-COCO 30K | FID | ~11 | Refiner included |
| HunyuanVideo | VBench (video) | Total score | 85.09 | Best open-source at release |
| SynthID | COCO 512px | AUROC (watermark detect) | 98.7% | After JPEG Q=80 compression |
| Stable Signature | COCO 512px | Bit accuracy (48-bit) | 96.2% | After JPEG Q=30 |
| UnivFD | New generators | AUROC | ~63% | Cross-generator deepfake detection |
| SD3.5-M + Flow-GRPO | GenEval | Overall accuracy | 0.95 | RL-tuned; GenEval near saturation |
| gpt-image-2 | Image Arena (LMArena) | Rank | #1 at launch (Apr 2026) | Vendor-reported; pairwise human preference |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| **FID/IS** | Widely used; enables cross-paper comparisons; captures quality+diversity together | Biased by Inception pretraining; sensitive to reference set; does not measure text alignment; easily gamed by low-diversity models |
| **CLIPScore** | Reference-free; directly measures prompt alignment; fast to compute | Saturates at high quality; blind to counting, spatial, fine-grained attributes |
| **GenEval/T2I-CompBench** | Compositional; measures specific failure modes; reproducible | Evaluation model (detector/VQA) itself can fail; distribution gap from real use |
| **SynthID/watermarking** | Invisible; robust to common transforms; traceable | Requires model cooperation; can be removed by strong adversarial attacks; model-specific |
| **C2PA** | Open standard; cryptographically secure; hardware camera support | Opt-in; doesn't detect unsigned synthetic media; metadata can be stripped |
| **Human-preference arenas** | Track real user preferences; hard to overfit with fixed prompts; cover editing and video | Volunteer prompt distribution; style confounders; opaque vendor pre-release testing; not reproducible offline |
| **Regulatory labelling (EU Art. 50, China Measures)** | Creates a uniform baseline and market incentive for provenance | Technical feasibility for open-weight models unclear; fragmentation across jurisdictions; enforcement just starting |

---

## Open Problems & Research Gaps

- **Unified quality-alignment-safety benchmark**: No single benchmark simultaneously measures perceptual quality (FID), text alignment (CLIPScore), compositional accuracy (GenEval), safety content (NSFW rate), and memorization. Researchers optimize metrics in isolation.
- **Generalization of deepfake detectors**: Detectors trained on one generation model family fail badly on another; the generalization gap is an arms-race problem without a fundamental solution.
- **Robust watermarking under adversarial removal**: Adversarial image transformations (JPEG, cropping, noise, generative adversarial removal with IP2P) can degrade watermark detection rates below threshold; embedding-level robustness guarantees are not achievable against unbounded attackers.
- **Legal and regulatory status**: EU AI Act Art. 50 (from Aug 2026) and China's labelling Measures (from Sep 2025) now *mandate* machine-readable marking, but neither prescribes a single technique; harmonized standards, interoperability between proprietary watermarks (SynthID vs. others) and treatment of open-weight models remain unresolved, and US requirements are a patchwork of state laws plus the TAKE IT DOWN Act.
- **Benchmark saturation and reward hacking**: GenEval-style detector-based metrics can be driven near 1.0 by RL post-training; arenas are harder to game but irreproducible. Evaluation of reasoning-heavy outputs (infographics, diagrams, multilingual text, factual correctness) lacks accepted automatic metrics.
- **Watermarks for open-weight generators**: Decoder-level watermarks (Stable Signature) can be fine-tuned away and post-hoc watermarks simply skipped; no scheme provides robust marking when the user controls the weights.
- **Membership inference beyond memorization**: Determining whether a specific image (not an exact duplicate) was in a model's training set — important for copyright enforcement — is unsolved beyond the verbatim memorization case.
- **Human evaluation scalability**: ImageReward and similar reward models are trained on limited human preference data; they fail to capture cultural diversity, contextual appropriateness, and evolving aesthetic standards at global scale.
- **Video authenticity**: FVD and VBench measure distributional quality; no established benchmark measures video-level deepfake authenticity, temporal consistency of synthetic persons, or audio-visual lip-sync fidelity.

---

## Further Reading

- [Heusel et al. (2017), "GANs Trained by a Two Time-Scale Update Rule," NeurIPS 2017](https://arxiv.org/abs/1706.08500)
- [Hessel et al. (2021), "CLIPScore: A Reference-free Evaluation Metric for Image Captioning," EMNLP 2021](https://arxiv.org/abs/2104.08718)
- [Carlini et al. (2023), "Extracting Training Data from Diffusion Models," USENIX Security 2023](https://arxiv.org/abs/2301.13188)
- [Fernandez et al. (2023), "The Stable Signature: Rooting Watermarks in Latent Diffusion Models," ICCV 2023](https://arxiv.org/abs/2303.15435)
- [Ghosh et al. (2023), "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment," NeurIPS 2023](https://arxiv.org/abs/2310.11513)
- [Huang et al. (2024), "VBench: Comprehensive Benchmark Suite for Video Generative Models," CVPR 2024](https://arxiv.org/abs/2311.17982)
- [Zheng et al. (2025), "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness"](https://arxiv.org/abs/2503.21755)
- [Motamed et al. (2025), "Do generative video models understand physical principles?"](https://arxiv.org/abs/2501.09038)
- [EU AI Act, Article 50: Transparency obligations for providers and deployers of certain AI systems](https://artificialintelligenceact.eu/article/50/)
- [C2PA technical specifications](https://c2pa.org/specifications/)
- [TAKE IT DOWN Act (S.146, 119th Congress)](https://www.congress.gov/bill/119th-congress/senate-bill/146)
- [OpenAI (2026), "System Card: ChatGPT Images 2.0 and Thinking mode"](https://deploymentsafety.openai.com/chatgpt-images-2-0/chatgpt-images-2-0.pdf)
- [Google (2026), "Nano Banana 2" announcement (SynthID + C2PA)](https://blog.google/innovation-and-ai/technology/ai/nano-banana-2/)
