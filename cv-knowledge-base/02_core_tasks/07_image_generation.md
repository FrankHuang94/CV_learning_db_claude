# Image Generation: GANs, VAEs, Diffusion Models, and Controllable Synthesis

> **Last Updated:** June 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](../10_generative_vision/01_diffusion_models.md) | [GANs and VAEs](../10_generative_vision/00_gans_and_vaes.md) | [Video Understanding](./08_video_understanding.md) | [Research Frontier 2024–2026](../12_research_frontier_2024_2026/02_any_model_paradigm.md)

---

## Overview

Image generation — producing photorealistic or stylistically controlled images from noise, text, or partial observations — has undergone three major paradigm shifts since 2014. The first was the **Generative Adversarial Network (GAN)** era: adversarial training between a generator and discriminator produced the first sharp, diverse images, culminating in the StyleGAN family [Karras2019, Karras2020] which achieved unprecedented realism and disentanglement for high-resolution faces. In parallel, **Variational Autoencoders (VAEs)** [Kingma2014] offered tractable likelihoods and smooth latent spaces but historically produced blurrier outputs than GANs. A hybrid turning point was VQGAN+Transformer [Esser2021], which learned discrete visual codebooks with VQ-VAEs and then modeled their distribution autoregressively with transformers, enabling high-resolution synthesis from text or layout conditions.

The second paradigm shift came with **Denoising Diffusion Probabilistic Models (DDPM)** [Ho2020] and subsequent work, which reframed generation as iterative denoising of Gaussian noise. Diffusion models surpassed GANs on FID within two years. The key efficiency breakthrough was **Latent Diffusion Models (LDM)** [Rombach2022], which moved the diffusion process into the compressed latent space of a pretrained VAE encoder, reducing the computational cost by 2–4 orders of magnitude and enabling text-conditioned synthesis at scale. Stable Diffusion (based on LDM) democratized image generation. SDXL [Podell2023] further improved quality with a 3× larger UNet and multi-encoder text conditioning.

The third paradigm is **flow-matching foundation models** (FLUX.1 [Black Forest Labs 2024]) and integrated text-image architectures. FLUX uses rectified flow matching — a simpler, straighter ODE trajectory from noise to data — with a hybrid multimodal transformer (MM-DiT) processing text and image tokens jointly at 12B parameters. These models set a new quality bar for photorealism and prompt adherence. Controllable generation through adapters (ControlNet [Zhang2023], IP-Adapter [Ye2023]) has transformed generation from an open-ended creative tool to a precise design instrument.

---

## Generative Adversarial Networks

### GAN Fundamentals

A GAN [Goodfellow2014] consists of a generator G: ℤ → 𝒳 and discriminator D: 𝒳 → [0,1] trained via a minimax game:

```latex
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log D(x)\right] + \mathbb{E}_{z \sim p_z}\!\left[\log(1 - D(G(z)))\right]
```

In practice, non-saturating loss is used for G. Training is notoriously unstable: mode collapse (G produces limited diversity), vanishing gradients (D too strong), and checkerboard artifacts from transposed convolutions.

### StyleGAN (Karras 2019)

StyleGAN [Karras2019] (CVPR 2019) introduced a **style-based generator** that disentangles high-level attributes (pose, age) from stochastic details (hair strands, pores):

1. **Mapping network f: ℤ → 𝒲**: 8-layer MLP maps the input noise z to an intermediate latent space 𝒲, which is more disentangled than Z.
2. **Adaptive Instance Normalization (AdaIN):** At each synthesis network layer, style vectors w ∈ 𝒲 are transformed into scale/bias via learned affine transforms:

```latex
\text{AdaIN}(x_i, y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}
```

3. **Stochastic noise inputs:** Per-pixel independent noise is added to each synthesis layer, controlling fine-grained texture.

StyleGAN achieves FID of 4.40 on FFHQ 1024×1024. StyleGAN2 [Karras2020] (CVPR 2020) eliminates characteristic "water droplet" artifacts via weight demodulation and introduces path-length regularization for better perceptual path-length.

---

## Variational Autoencoders

VAEs [Kingma2014] parameterize a probabilistic encoder q_φ(z|x) and decoder p_θ(x|z) and maximize the ELBO:

```latex
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\!\left(q_\phi(z|x) \| p(z)\right)
```

The reparameterization trick z = μ + σ ⊙ ε, ε ~ N(0,I) enables gradient flow through the sampling operation. VAEs produce continuous, interpolatable latent spaces but blurry reconstructions due to the MSE reconstruction term. The KL divergence enforces a structured prior, enabling generation by sampling from p(z) = N(0,I).

**VQ-VAE** [van den Oord2017] replaces the continuous latent with a discrete codebook of embeddings, learning a vocabulary of visual tokens. VQGAN [Esser2021] adds an adversarial discriminator to the VQ-VAE objective, dramatically sharpening reconstructions and enabling codebook usage at higher compression rates. The Transformer then models the discrete token sequence autoregressively, supporting conditioned generation at megapixel resolution.

---

## Diffusion Models

### DDPM (Ho 2020)

DDPM [Ho2020] (NeurIPS 2020) defines a **forward process** that gradually corrupts data with Gaussian noise over T steps:

```latex
q(x_t | x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t \mathbf{I}\right)
```

with variance schedule β_1,...,β_T. By the reparameterization trick, sampling x_t directly from x_0:

```latex
q(x_t | x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) \mathbf{I}\right), \quad \bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)
```

A U-Net ε_θ(x_t, t) is trained to predict the added noise ε at each step, minimizing:

```latex
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, \epsilon, t}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon,\; t)\|^2\right]
```

Generation proceeds by iterative denoising from x_T ~ N(0,I) using the reverse process q(x_{t-1}|x_t).

### Latent Diffusion Models (Rombach 2022)

LDM [Rombach2022] (CVPR 2022) operates in the latent space z = E(x) of a pre-trained VQ-regularized autoencoder, reducing spatial dimensions by factor f ∈ {4, 8, 16}. The diffusion model ε_θ(z_t, t, c) is conditioned on context c (text, segmentation, etc.) via cross-attention:

```latex
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V, \quad Q = z_t W_Q,\; K = \tau_\theta(c) W_K,\; V = \tau_\theta(c) W_V
```

where τ_θ is a domain-specific encoder (CLIP text encoder for text-to-image). LDM achieves competitive FID with pixel-space DDPM while training 2–4× faster.

```mermaid
graph LR
    subgraph Training
        X[Real Image x] --> E[VAE Encoder E]
        E --> Z[Latent z]
        Z --> FW[Forward Diffusion: add noise]
        FW --> Zt[Noisy Latent z_t]
        C[Text Condition c] --> TE[CLIP Text Encoder τ_θ]
        TE --> UNET[LDM U-Net ε_θ]
        Zt --> UNET
        UNET --> eps[Predicted ε]
    end
    subgraph Inference
        ZT[z_T ~ N(0,I)] --> REV[Reverse Denoising T→0]
        C2[Text Prompt] --> TE2[CLIP Encoder]
        TE2 --> REV
        REV --> Z0[Latent z_0]
        Z0 --> D[VAE Decoder D]
        D --> Xhat[Generated Image]
    end
```

### SDXL (Podell 2023)

SDXL [Podell2023] improves LDM by: (1) a 3× larger U-Net backbone with more attention blocks; (2) two text encoders (OpenCLIP ViT-G + CLIP ViT-L) for richer conditioning; (3) multi-aspect-ratio training; (4) a secondary **refiner model** that applies a high-noise-level diffusion pass on generated latents, sharpening fine details without changing composition.

### FLUX.1 (Black Forest Labs 2024)

FLUX.1 [BlackForestLabs2024] is a 12B-parameter text-to-image model using **rectified flow matching** — training the model to predict the straightest possible path from noise to data by targeting the flow vector field:

```latex
\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, x_0, x_1}\!\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right], \quad x_t = (1-t)x_0 + t x_1
```

The architecture is a **hybrid multimodal transformer (MM-DiT)** that processes text and image tokens jointly in the same attention layers, replacing the cross-attention bottleneck. FLUX achieves state-of-the-art text-to-image fidelity on GenEval and other human preference benchmarks as of 2024. FLUX.1 Kontext (2025) extends this to unified generation and editing.

---

## Controllable Generation

### ControlNet (Zhang 2023)

ControlNet [Zhang2023] (ICCV 2023) adds spatial conditioning (edges, depth maps, segmentation, human pose, normals) to frozen pretrained diffusion models by cloning the encoder of the U-Net and connecting it to the original through **zero-initialized convolutions**:

```latex
\mathbf{y}_c = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c; \Theta_{z1}); \Theta_c); \Theta_{z2})
```

where Z(·; ·) is a zero-convolution layer (initialized to zero weight and bias). During early training, the zero convolutions produce zero output, ensuring no harmful gradients corrupt the pretrained backbone. ControlNet enables pixel-precise control (e.g., conditioning on a Canny edge map or human keypoint skeleton) while preserving generalization from the base model.

### IP-Adapter (Ye 2023)

IP-Adapter [Ye2023] (arXiv 2023) achieves image-prompt conditioning for frozen text-to-image models with only 22M additional parameters. A **decoupled cross-attention** mechanism adds image-feature cross-attention layers (separate from the text cross-attention) to each U-Net block:

```latex
\text{Attn}(Q, K_{\text{text}}, V_{\text{text}}) + \lambda \cdot \text{Attn}(Q, K_{\text{img}}, V_{\text{img}})
```

Image features are extracted via a frozen CLIP image encoder and projected to match the U-Net's cross-attention dimension. IP-Adapter generalizes across custom fine-tuned models sharing the same base and is composable with ControlNet.

---

## Evaluation Metrics

### FID (Fréchet Inception Distance)

FID [Heusel2017] measures the distance between feature distributions of real and generated images in the InceptionV3 feature space:

```latex
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
```

Lower FID → more realistic distribution. FID is sensitive to sample size and image resolution; comparisons require identical evaluation protocols. Common thresholds: FID < 5 is considered excellent for unconditional generation.

### Inception Score (IS)

```latex
\text{IS} = \exp\!\left(\mathbb{E}_{x}\left[D_{\text{KL}}\!\left(p(y|x) \| p(y)\right)\right]\right)
```

IS rewards high conditional confidence (sharp images) and high marginal entropy (diversity). IS is less reliable than FID and sensitive to Inception model choice.

### CLIPScore

For text-to-image, CLIPScore [Hessel2021] measures cosine similarity between CLIP embeddings of generated image and text prompt:

```latex
\text{CLIPScore}(I, T) = \max(100 \cdot \cos(\mathbf{e}_I, \mathbf{e}_T), 0)
```

Higher CLIPScore → better text-image alignment. Limitations: CLIP has known biases and cannot capture spatial or compositional accuracy.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|---|---|---|---|---|
| Generative Adversarial Networks | Goodfellow, Pouget-Abadie, Mirza et al. | 2014 | NeurIPS | Minimax adversarial training for implicit generative models |
| Auto-Encoding Variational Bayes (VAE) | Kingma, Welling | 2014 | ICLR | ELBO objective; reparameterization trick; continuous latent space generation |
| A Style-Based Generator Architecture for GANs (StyleGAN) | Karras, Laine, Aila | 2019 | CVPR | Mapping network; AdaIN; disentangled 𝒲 space; stochastic noise |
| Taming Transformers for High-Resolution Image Synthesis (VQGAN) | Esser, Rombach, Ommer | 2021 | CVPR (Oral) | VQ codebook + adversarial training + autoregressive transformer |
| Denoising Diffusion Probabilistic Models (DDPM) | Ho, Jain, Abbeel | 2020 | NeurIPS | Markovian diffusion forward process; noise-prediction U-Net; surpassed GANs |
| High-Resolution Image Synthesis with Latent Diffusion Models (LDM/SD) | Rombach, Blattmann, Lorenz, Esser, Ommer | 2022 | CVPR | Diffusion in VAE latent space; cross-attention conditioning; 100× compute reduction |
| SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis | Podell, English, Lacey et al. | 2023 | arXiv | 3× larger U-Net; dual text encoders; refiner stage |
| Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet) | Zhang, Rao, Agrawala | 2023 | ICCV | Zero-convolution spatial conditioning adapters for frozen diffusion models |
| IP-Adapter: Text Compatible Image Prompt Adapter | Ye, Zhang, Liu, Han, Yang | 2023 | arXiv | Decoupled cross-attention for image prompt conditioning; 22M parameters |

---

## Benchmark Performance

| Model | Dataset/Benchmark | Metric | Score | Notes |
|---|---|---|---|---|
| StyleGAN2 | FFHQ 256×256 | FID ↓ | 3.8 | Unconditional; as reported [Karras2020] |
| StyleGAN2 | FFHQ 1024×1024 | FID ↓ | 2.84 | Uncond high-res generation |
| DDPM | CIFAR-10 uncond | FID ↓ | 3.17 | Pixel-space; 1000 diffusion steps |
| LDM (f=4) | CelebA-HQ 256 | FID ↓ | 5.11 | Latent diffusion; 200 DDIM steps |
| Stable Diffusion 2.1 | COCO (30K val) | CLIPScore ↑ | 31.9 | Text-to-image; CLIP ViT-L/14 |
| SDXL | PartiPrompts | FID ↓ | ~14.0 | Not publicly reported at exact protocol; indicative |
| FLUX.1-dev | GenEval Overall | Accuracy ↑ | 0.66 | Prompt-following benchmark; 12B params |
| ControlNet (SD1.5) | COCO val2017 | FID ↓ | ~16.0 | Canny-conditioned; indicative, protocol-sensitive |

*Note: many FLUX and SDXL scores are not reported under standardized FID protocols; human preference evaluations (ELO ratings, user studies) are increasingly preferred.*

---

## Pros & Cons

| Aspect | Pros | Cons |
|---|---|---|
| GAN-based generation (StyleGAN) | Very fast inference (single forward pass); excellent high-resolution face quality; controllable latent space | Training instability; mode collapse risk; limited generalization to diverse prompts |
| VAE/VQGAN | Tractable likelihood; smooth interpolation; discrete tokens enable LLM-style modeling | Lower visual quality than GANs/diffusion without adversarial component; codebook collapse |
| Diffusion models (LDM/SD/SDXL) | High diversity; excellent text-image alignment; stable training; composable with adapters | Slow iterative inference (10–50 steps); memory intensive; harder to control spatial layout precisely |
| Flow matching (FLUX) | Straighter ODE paths → fewer inference steps; better scaling; multimodal joint attention | Extremely large models (12B+); high VRAM requirement; proprietary details not fully published |

---

## Open Problems & Research Gaps

- **Compositional generation failure:** Current text-to-image models struggle with attribute binding (e.g., "a red cube next to a blue sphere"), negation, and spatial relationships; structured scene-graph conditioning is an active direction.
- **Inference acceleration without quality loss:** Distillation (consistency models, SDXL-Lightning, TurboEdit) achieves 1–4 step generation but still lags quality of full diffusion at high guidance scales; flow-matching models partially address this.
- **3D consistency and multi-view generation:** Generating consistent images from multiple viewpoints (required for 3D reconstruction and NeRF initialization) remains challenging; score distillation sampling (SDS) approaches suffer from over-saturation and blurriness.
- **Fine-grained controllability beyond adapters:** ControlNet provides structural control but joint optimization of content, style, identity, and spatial layout simultaneously without artifacts remains unsolved.
- **Evaluation beyond FID:** FID correlates poorly with human preference for text-to-image; better metrics capturing attribute binding, counting accuracy, and compositional understanding are needed.
- **Memorization and copyright:** Large diffusion models memorize training samples; quantifying and mitigating unintended reproduction is an unresolved technical and legal challenge.
- **Video generation consistency:** Extending image generation to temporally consistent video (Sora, OpenAI; AnimateDiff; CogVideo) while maintaining per-frame quality and long-range coherence remains the frontier challenge.

---

## Further Reading

- [LDM / Stable Diffusion paper (arXiv:2112.10752)](https://arxiv.org/abs/2112.10752) — CVPR 2022 [Rombach2022]
- [DDPM project page (Ho et al.)](https://hojonathanho.github.io/diffusion/) — NeurIPS 2020 [Ho2020]
- [ControlNet paper (arXiv:2302.05543)](https://arxiv.org/abs/2302.05543) — ICCV 2023 [Zhang2023]
- [FLUX.1 official GitHub (black-forest-labs)](https://github.com/black-forest-labs/flux) — Official inference code and model cards
- [Taming Transformers / VQGAN project page](https://compvis.github.io/taming-transformers/) — CVPR 2021 [Esser2021]
- [IP-Adapter GitHub (tencent-ailab)](https://github.com/tencent-ailab/IP-Adapter) — Official implementation [Ye2023]
