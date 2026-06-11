# Evaluation and Safety of Generative Vision Models

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](./01_diffusion_models.md) | [GANs and VAEs](./00_gans_and_vaes.md) | [Evaluation Metrics](../11_datasets_and_benchmarks/04_evaluation_metrics.md) | [Video Generation](./02_video_generation.md)

---

## Overview

The rapid proliferation of high-quality generative vision systems — text-to-image, text-to-video, and 3D generation — has created an urgent need for evaluation methodologies that go beyond perceptual quality scores and address compositional correctness, alignment with human intent, safety, and authenticity. This file surveys the principal automatic and human evaluation benchmarks, the mathematical foundations of the most widely used metrics, and the safety and provenance tooling ecosystem that has emerged in response to the deepfake and synthetic media crisis.

Automatic metrics for generative image quality fall into two camps: **reference-based** metrics that compare the statistics of generated and real image distributions (FID, IS, FVD), and **alignment-based** metrics that measure how well a generated image matches its conditioning signal (CLIPScore, ImageReward, DINO score). Reference-based metrics require a held-out reference set (typically 30K–50K images from the training distribution) and are sensitive to biases in the reference set and the feature extractor (Inception V3, ViT-g). Alignment-based metrics are reference-free but depend on the quality of the underlying text-vision encoder.

The safety landscape for generative vision encompasses three overlapping concerns: (1) **deepfakes** — synthetic media that deliberately misrepresent real people or events, enabling fraud, misinformation, and non-consensual intimate imagery (NCII); (2) **memorization and data privacy** — generative models' tendency to reproduce near-verbatim copies of training data, enabling reconstruction of private or copyrighted material; and (3) **provenance and attribution** — the inability of downstream consumers to determine whether a piece of media was AI-generated. Responses to these concerns include invisible watermarking (SynthID [DeepMind2023], Stable Signature [Fernandez2023]), content provenance standards (C2PA), and dataset filtering for consent and licensing.

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

---

## Watermarking: SynthID and Stable Signature

**SynthID** [DeepMind2023, updated as SynthID-Image 2024] is Google DeepMind's production watermarking system for AI-generated images:
- Embeds an invisible watermark during image generation (not post-hoc) by fine-tuning a small detector alongside the generation model.
- Watermark is imperceptible to humans but detectable by a paired classifier.
- Robust to JPEG compression, resizing, cropping, and color transformations.
- Deployed in Imagen and Gemini image generation.
- The watermarking model is not fully open-source; detection API available.

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
```

C2PA provides provenance but not authenticity detection: a bad actor can generate media without C2PA credentials and simply omit them. It is most useful for establishing trust in signed media rather than detecting unsigned synthetic media.

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

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| **FID/IS** | Widely used; enables cross-paper comparisons; captures quality+diversity together | Biased by Inception pretraining; sensitive to reference set; does not measure text alignment; easily gamed by low-diversity models |
| **CLIPScore** | Reference-free; directly measures prompt alignment; fast to compute | Saturates at high quality; blind to counting, spatial, fine-grained attributes |
| **GenEval/T2I-CompBench** | Compositional; measures specific failure modes; reproducible | Evaluation model (detector/VQA) itself can fail; distribution gap from real use |
| **SynthID/watermarking** | Invisible; robust to common transforms; traceable | Requires model cooperation; can be removed by strong adversarial attacks; model-specific |
| **C2PA** | Open standard; cryptographically secure; hardware camera support | Opt-in; doesn't detect unsigned synthetic media; metadata can be stripped |

---

## Open Problems & Research Gaps

- **Unified quality-alignment-safety benchmark**: No single benchmark simultaneously measures perceptual quality (FID), text alignment (CLIPScore), compositional accuracy (GenEval), safety content (NSFW rate), and memorization. Researchers optimize metrics in isolation.
- **Generalization of deepfake detectors**: Detectors trained on one generation model family fail badly on another; the generalization gap is an arms-race problem without a fundamental solution.
- **Robust watermarking under adversarial removal**: Adversarial image transformations (JPEG, cropping, noise, generative adversarial removal with IP2P) can degrade watermark detection rates below threshold; embedding-level robustness guarantees are not achievable against unbounded attackers.
- **Legal and regulatory status**: OECD, EU AI Act, and US Executive Orders have conflicting requirements on AI watermarking and disclosure, with no harmonized technical standard; C2PA is voluntary.
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
