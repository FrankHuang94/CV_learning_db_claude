# GANs and VAEs: Foundations of Deep Generative Vision

> **Last Updated:** October 2026
> **Level:** Advanced
> **Related Sections:** [Diffusion Models](./01_diffusion_models.md) | [3D Generation](./03_3d_generation.md) | [Evaluation and Safety](./04_evaluation_and_safety.md) | [Video Generation](./02_video_generation.md)

> **🆕 October 2026 update:**
> - New section on the 2024–2026 afterlife of GANs and VAEs: modern GAN baselines (R3GAN), adversarial distillation / post-training (ADD, LADD, Seaweed-APT) now used in production few-step generators such as Seedream 4.0.
> - New coverage of the "tokenizer renaissance": high-compression autoencoders (DC-AE, Wan 2.2 VAE, the open FLUX.2 VAE), representation-aligned latents (VA-VAE, REPA-E) and Representation Autoencoders (RAE), plus the pixel-space counter-trend (JiT).
> - Key Papers, Benchmark, Open Problems and Further Reading extended accordingly; the "hybrid GAN-diffusion is unexplored at scale" claim has been corrected.

---

## Overview

Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) represent the two foundational paradigms of learned generative modeling that preceded the diffusion era. Introduced within a year of each other — VAEs by [KingmaWelling2013] and GANs by [Goodfellow2014] — they encode fundamentally different inductive biases: VAEs ground generation in Bayesian inference and explicit latent structure, while GANs frame generation as an adversarial game between a generator and a discriminator. Together they drove a decade of progress in image synthesis, compression, video prediction, and representation learning.

The GAN framework produces photorealistic outputs by training a generator network $G$ to produce samples indistinguishable from real data under a discriminator $D$. The theoretical underpinning of the original minimax objective, combined with architectures such as DCGAN [Radford2015], progressive growing [KarrasLaine2018], and style-based generation [Karras2019], pushed GAN outputs from blurry 64×64 digits to near-photographic 1024×1024 faces by 2019. WGAN-GP [Gulrajani2017] resolved catastrophic training instabilities by replacing the Jensen–Shannon divergence with a Wasserstein distance and enforcing a Lipschitz constraint via gradient penalty, enabling stable training without mode collapse. The BigGAN [Brock2019] scaling experiment established that class-conditional ImageNet generation was tractable with careful truncation tricks. Conditional image-to-image translation (pix2pix [IsolaZhu2017]) and unpaired domain adaptation (CycleGAN [ZhuPark2017]) extended GANs to structured vision tasks with broad practical impact.

VAEs take a probabilistic view: the encoder learns to approximate the true posterior $q_\phi(\mathbf{z}|\mathbf{x})$, and the decoder learns $p_\theta(\mathbf{x}|\mathbf{z})$. Optimizing the Evidence Lower BOund (ELBO) trains the model end-to-end. Crucially, the reparameterization trick makes the stochastic sampling node differentiable. While early VAEs produced blurry reconstructions (attributed to the mean-seeking behavior of KL-diverged Gaussians), the VQ-VAE line [vanDenOord2017; Razavi2019] replaced the continuous bottleneck with a discrete codebook, enabling high-fidelity reconstruction and pairing naturally with autoregressive priors. VQGAN [Esser2021] fused VQ-VAE with a patch-discriminator, delivering the high-quality discrete tokenizer that underpins the Latent Diffusion / Stable Diffusion family.

---

## GAN: Minimax Objective

The original GAN game [Goodfellow2014] is defined by the following minimax value function:

```
V(D, G) = E_{x ~ p_data(x)} [log D(x)]
         + E_{z ~ p_z(z)} [log(1 - D(G(z)))]

Discriminator maximizes V(D, G)
Generator     minimizes V(D, G)

Optimal discriminator: D*(x) = p_data(x) / (p_data(x) + p_g(x))

At equilibrium (p_g = p_data):
  V(D*, G) = -log(4)
  C(G) = 2 * JSD(p_data || p_g) - log(4)
```

The minimax problem has a global minimum when $p_g = p_{data}$, corresponding to the generator perfectly capturing the data distribution. In practice, the non-saturating variant replaces the generator objective with maximizing $\log D(G(z))$ to avoid vanishing gradients at the start of training.

---

## WGAN-GP Gradient Penalty

Wasserstein-1 distance provides more stable gradients but requires the discriminator (called a "critic") to be 1-Lipschitz. WGAN-GP enforces this via:

```
L = E_{x~p_g}[D(x)] - E_{x~p_r}[D(x)]
  + lambda * E_{x~p_hat}[(||grad_x D(x)||_2 - 1)^2]

where x_hat = epsilon * x_real + (1 - epsilon) * x_fake,
      epsilon ~ U[0,1], lambda = 10 (recommended)

Critic trained for n_critic = 5 steps per generator step.
```

This gradient penalty replaces weight clipping and dramatically reduces sensitivity to hyperparameters [Gulrajani2017].

---

## StyleGAN Architecture Family

StyleGAN [Karras2019] introduced a **mapping network** that transforms the input noise $z \in \mathcal{Z}$ to an intermediate latent space $w \in \mathcal{W}$, which is then injected at each resolution level via adaptive instance normalization (AdaIN):

```
AdaIN(x_i, y) = y_{s,i} * (x_i - mu(x_i)) / sigma(x_i) + y_{b,i}

where y = (y_s, y_b) are affine transforms of w,
      x_i is the i-th feature map activation
```

StyleGAN2 [Karras2020] replaced AdaIN with weight demodulation to eliminate blob artifacts, and removed progressive growing in favor of a skip-generator / residual-discriminator architecture. StyleGAN3 [Karras2021] addressed texture sticking by enforcing continuous equivariance to translation and rotation, using sinc-filtered convolutions — important for video and animation applications.

---

## Pix2Pix and CycleGAN

Conditional image-to-image translation was formalized by pix2pix [IsolaZhu2017] as a cGAN with a paired dataset, PatchGAN discriminator, and L1 reconstruction loss. The PatchGAN discriminator classifies overlapping 70×70 image patches rather than the full image, encouraging high-frequency sharpness. CycleGAN [ZhuPark2017] extended this to unpaired data by introducing a cycle-consistency loss:

```
L_cyc(G, F) = E_{x~p(x)} [||F(G(x)) - x||_1]
             + E_{y~p(y)} [||G(F(y)) - y||_1]
```

where $G: X \to Y$ and $F: Y \to X$ are the two generators. This constraint prevents mode collapse and is sufficient to learn many structured mappings (horse→zebra, summer→winter, photo→painting).

---

## VAE: ELBO Derivation

The Variational Autoencoder [KingmaWelling2013] maximizes a lower bound on the log-likelihood:

```
log p(x) >= E_{q_phi(z|x)} [log p_theta(x|z)]
           - KL(q_phi(z|x) || p(z))
          = ELBO(x; phi, theta)

Reparameterization trick:
  z = mu_phi(x) + sigma_phi(x) * epsilon,  epsilon ~ N(0, I)
  => gradients flow through z back to phi

KL term (Gaussian prior, diagonal posterior):
  KL = -0.5 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)
```

The reconstruction term encourages the decoder to faithfully reproduce $x$, while the KL term regularizes the posterior toward the prior $\mathcal{N}(0, I)$, ensuring smooth latent interpolation.

---

## VQ-VAE, VQ-VAE-2, and VQGAN

Vector-Quantized VAE [vanDenOord2017] replaces the continuous Gaussian bottleneck with a discrete codebook $\{e_k\}_{k=1}^K$:

```
z_q = e_k,   k = argmin_j ||z_e - e_j||_2

Loss = ||x - decoder(z_q)||^2
     + ||sg[z_e] - e_k||^2       (codebook loss)
     + beta * ||z_e - sg[e_k]||^2  (commitment loss, beta=0.25)

Straight-through gradient: partial L / partial z_e
  approximated by copying gradient from z_q to z_e.
```

VQ-VAE-2 [Razavi2019] stacks two levels of quantization (bottom: 32×32, top: 8×8 for 256px images) capturing local and global structure separately, then fits PixelSnail autoregressive priors at both scales to achieve FID competitive with BigGAN at the time.

VQGAN [Esser2021] adds a patch-based discriminator and perceptual loss to VQ-VAE training, dramatically improving reconstruction sharpness. The resulting tokenizer encodes images into 16×16 or 32×32 discrete maps — the representation later used as the "perceptual compression" stage in Latent Diffusion Models [Rombach2022].

---

## Why Diffusion Models Overtook GANs

By 2021–2022, diffusion models surpassed GANs on most image generation benchmarks, driven by several structural advantages:

1. **Training stability**: No adversarial game; the denoising objective is a simple MSE/L2 regression that scales reliably with data and compute.
2. **Mode coverage**: GANs suffer from mode collapse — dropping entire semantic categories. Diffusion models optimize a likelihood-based objective and demonstrate better coverage.
3. **Conditioning flexibility**: Classifier-free guidance [Ho2022] enabled zero-shot conditioning on arbitrary text without retraining the discriminator. GANs require architecture changes per conditioning type.
4. **Diversity-fidelity tradeoff**: The guidance scale knob gives continuous control over this tradeoff; GANs require the truncation trick, which cuts the tails of $\mathcal{W}$.
5. **Scalability**: DDPM→LDM→SDXL demonstrated consistent quality gains by scaling model size and training compute, following patterns similar to LLM scaling. GANs showed training instability at very large scale.
6. **Editability**: DDIM inversion [Song2021] provides near-exact latent inversion for real images, enabling precise text-guided editing. GAN inversion is harder and less faithful at high resolution.

GANs retain advantages in inference speed (single forward pass vs. 20–1000 NFE for diffusion) and in structured generation tasks where paired data enables stable discriminator training (medical image synthesis, video super-resolution). GAN-derived components — especially VQGAN tokenizers — remain central to the latent diffusion pipeline.

---

## 2024–2026: The Afterlife of GANs and VAEs in the Diffusion Era

Although neither GANs nor VAEs are the headline generator in 2025–2026 frontier systems, both survive as **essential sub-components**: the adversarial loss as a distillation/post-training signal, and the (KL- or VQ-) autoencoder as the latent tokenizer on which almost every diffusion/flow model operates.

### A modern GAN baseline: R3GAN

Huang et al. [Huang2024R3GAN] ("The GAN is dead; long live the GAN!", NeurIPS 2024) revisited GAN training from first principles. They derive a well-behaved **regularized relativistic GAN loss** (RpGAN + zero-centred gradient penalties R1 and R2) with local convergence guarantees, which removes the need for most of the empirical tricks accumulated since DCGAN/StyleGAN. With this loss a deliberately simple, modernized ResNet-style backbone (grouped convolutions, inverted bottlenecks, no normalization) is competitive with StyleGAN2 and with some diffusion baselines on FFHQ, ImageNet, CIFAR-10 and Stacked-MNIST, while keeping single-step sampling.

```
Relativistic pairing loss with zero-centred penalties (R3GAN):
  L_D = E_{x~p_d, z~p_z}[ f( D(G(z)) - D(x) ) ]
        + (gamma/2) * E_{x~p_d}[ ||grad_x D(x)||^2 ]      (R1)
        + (gamma/2) * E_{x~p_g}[ ||grad_x D(x)||^2 ]      (R2)
  L_G = E_{x~p_d, z~p_z}[ f( D(x) - D(G(z)) ) ]
  f(t) = softplus(t) = log(1 + e^t)
```

### Adversarial losses as distillation and post-training signals

The most consequential 2024–2026 use of discriminators is to **collapse multi-step diffusion into 1–4 steps**:

- **ADD** (Adversarial Diffusion Distillation, SDXL-Turbo) [Sauer2023ADD] combines a score-distillation loss from a frozen teacher with an adversarial loss using a frozen DINOv2 feature discriminator, yielding 1–4-step SDXL sampling.
- **LADD** [Sauer2024LADD] moves the discriminator into latent space, using the teacher diffusion model's own features as the discriminator backbone — the basis of SD3-Turbo.
- **Seaweed-APT** [Lin2025APT] ("Diffusion Adversarial Post-Training") applies adversarial training *against real data* after diffusion pre-training to obtain **one-step** text-to-video and 1024px text-to-image generation.
- **Production use**: ByteDance's Seedream 4.0 technical report [Seedream2025] lists adversarial distillation together with distribution matching, quantization and speculative decoding to reach ~1.4 s per 2K image. DMD2 [Yin2024DMD2] likewise adds a GAN loss to distribution-matching distillation. See [Diffusion Models](./01_diffusion_models.md) for the full few-step landscape.

### The tokenizer renaissance

Since the generator's quality is upper-bounded by the autoencoder's reconstruction and by how *learnable* its latent space is, autoencoder design became an active research area again:

| Direction | Representative work | Idea |
|-----------|--------------------|------|
| High spatial compression | DC-AE [Chen2025DCAE]; Wan 2.2 VAE (16×16×4 spatio-temporal) | 32–64× spatial downsampling with residual "space-to-channel" shortcuts; fewer tokens for high-res image/video DiTs |
| Representation-aligned latents | VA-VAE / LightningDiT [Yao2025VAVAE]; REPA-E [Leng2025REPAE] | Align VAE latents with a frozen vision foundation model (DINOv2) or train VAE and diffusion end-to-end, resolving the reconstruction-vs-generation trade-off of high-dimensional latents |
| Representation Autoencoders | RAE [Zheng2025RAE] | Replace the VAE encoder by a *frozen* pretrained representation encoder (e.g. DINOv2, SigLIP) plus a trained decoder; DiTs trained in this semantic latent space converge faster and reach strong ImageNet FID |
| Open, reusable tokenizers | FLUX.2 VAE (Apache-2.0) [BFL2025FLUX2] | A standalone, permissively licensed autoencoder released separately from the (non-commercial) generator weights |
| Pixel space without a tokenizer | JiT [LiHe2025JiT] | Large-patch plain Transformers predicting the clean image (x-prediction) directly in pixel space, questioning whether a learned tokenizer is necessary at all |

The common theme is that the old VAE trade-off — reconstruction fidelity vs. a smooth, easily-modelled latent — is now explicitly engineered rather than left to the KL weight $\beta$.

---

## Mermaid: GAN vs. VAE Topology

```mermaid
graph LR
    subgraph GAN
        Z1[Noise z] --> G[Generator G]
        G --> Fake[Fake Image]
        Real[Real Image] --> D[Discriminator D]
        Fake --> D
        D -->|Gradient| G
    end

    subgraph VAE
        X[Input Image x] --> Enc[Encoder q_phi]
        Enc --> Mu[mu] & Sigma[sigma]
        Mu & Sigma -->|Reparam| Z2[Latent z]
        Z2 --> Dec[Decoder p_theta]
        Dec --> Xhat[Reconstruction]
        Xhat -->|ELBO| Loss
    end
```

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| Generative Adversarial Nets | Goodfellow et al. | 2014 | NeurIPS | Minimax adversarial framework; proved Nash equilibrium at $p_g=p_{data}$ |
| Unsupervised Representation Learning with DCGANs | Radford, Metz, Chintala | 2015 | arXiv/ICLR 2016 | All-convolutional architecture with BN; stable large-scale GAN training |
| Improved Training of WGANs | Gulrajani et al. | 2017 | NeurIPS | Gradient penalty Lipschitz constraint; replaced weight clipping |
| Large Scale GAN Training (BigGAN) | Brock et al. | 2019 | ICLR | Class-conditional 512px ImageNet; truncation trick; orthogonal regularization |
| A Style-Based Generator Architecture (StyleGAN) | Karras et al. | 2019 | CVPR | Mapping network, AdaIN injection, per-layer noise; FID 4.40 on FFHQ |
| Analyzing and Improving the Image Quality (StyleGAN2) | Karras et al. | 2020 | CVPR | Weight demodulation, path-length regularization; removed artifacts |
| Alias-Free GAN (StyleGAN3) | Karras et al. | 2021 | NeurIPS | Sinc-filtered convolutions; translation/rotation equivariance |
| Image-to-Image Translation with cGANs (pix2pix) | Isola et al. | 2017 | CVPR | PatchGAN discriminator; L1 + adversarial loss for paired translation |
| Unpaired Image-to-Image Translation (CycleGAN) | Zhu et al. | 2017 | ICCV | Cycle-consistency loss; unsupervised domain transfer |
| Auto-Encoding Variational Bayes | Kingma, Welling | 2013 | ICLR 2014 | VAE; ELBO; reparameterization trick |
| Neural Discrete Representation Learning (VQ-VAE) | van den Oord et al. | 2017 | NeurIPS | Discrete codebook bottleneck; straight-through estimator |
| Generating Diverse High-Fidelity Images (VQ-VAE-2) | Razavi et al. | 2019 | NeurIPS | Hierarchical two-level quantization; PixelSnail prior |
| Taming Transformers (VQGAN) | Esser et al. | 2021 | CVPR | Patch discriminator + perceptual loss on VQ-VAE; discrete tokens for transformers |
| Adversarial Diffusion Distillation (SDXL-Turbo) | Sauer et al. | 2023 | ECCV 2024 | Teacher score distillation + DINOv2-feature discriminator; 1–4-step text-to-image |
| The GAN is dead; long live the GAN! (R3GAN) | Huang et al. | 2024 | NeurIPS | Regularized relativistic loss (RpGAN + R1 + R2) with convergence guarantees; trick-free modern GAN baseline |
| Fast High-Resolution Image Synthesis with LADD | Sauer et al. | 2024 | SIGGRAPH Asia | Latent adversarial distillation using teacher-diffusion features as discriminator (SD3-Turbo) |
| Deep Compression Autoencoder (DC-AE) | Chen et al. | 2024 | ICLR 2025 | 32–128× spatial compression autoencoder for efficient high-res latent diffusion |
| Reconstruction vs. Generation (VA-VAE / LightningDiT) | Yao et al. | 2025 | CVPR | Vision-foundation-model-aligned VAE latents; resolves high-dim latent optimization dilemma |
| Diffusion Adversarial Post-Training (Seaweed-APT) | Lin et al. | 2025 | arXiv | Adversarial post-training against real data for one-step image and video generation |
| Diffusion Transformers with Representation Autoencoders (RAE) | Zheng et al. | 2025 | arXiv | Frozen representation encoder + trained decoder as the latent space for DiTs |

---

## Benchmark Performance

| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| StyleGAN2 | FFHQ 1024px | FID | 2.84 | Truncation ψ=0.5 |
| StyleGAN3-T | FFHQ 1024px | FID | 2.79 | Translation equivariant |
| BigGAN-deep | ImageNet 512px | FID | 3.24 | Truncation threshold 0.5 |
| BigGAN-deep | ImageNet 128px | IS | 232.5 | Best reported IS at time |
| VQ-VAE-2 | ImageNet 256px | FID | 31.11 | Autoregressive prior |
| VQGAN | ImageNet 256px | FID | 7.94 | Transformer prior (GPT-style) |
| CycleGAN | Horse→Zebra | AMT fool rate | 25.5% | Human perceptual evaluation |
| DCGAN | CelebA 64px | FID | ~37 | Baseline; modern implementations vary |
| LightningDiT + VA-VAE | ImageNet 256px | FID | 1.35 | Diffusion on foundation-model-aligned VAE latents [Yao2025VAVAE]; shows tokenizer design dominates |

---

## Pros & Cons

| Aspect | Pros | Cons |
|--------|------|------|
| **Training dynamics** | Fast single-step inference; mature tooling | Adversarial instability; mode collapse; sensitive to hyperparameters |
| **Sample quality (GANs)** | Sharp, high-frequency details; state-of-art portraits at 1024px+ | Coverage issues; hallucination of consistent 3D structure |
| **Latent structure (VAEs)** | Smooth, interpretable latent space; stable likelihood-based training | Blurry reconstructions in standard VAEs; posterior collapse in deep models |
| **Conditioning** | pix2pix/CycleGAN allow strong spatial conditioning | Arbitrary text conditioning requires architecture redesign vs. CFG in diffusion |
| **Computational cost** | Generator is a single forward pass; trivially fast at inference | GAN discriminator doubles training memory; VQ-VAE tokenizer training expensive |
| **Discrete tokenization (VQ)** | Natural for transformer priors; compresses $256^2$ images to $16^2$ tokens | Codebook collapse; commitment loss tuning required |

---

## Open Problems & Research Gaps

- **GAN scaling**: Despite BigGAN, no GAN architecture has matched the quality-diversity tradeoff of diffusion at equivalent parameter counts on diverse datasets beyond face/bedroom domains. Understanding whether this reflects inductive bias or data curriculum is open.
- **VAE sharpness**: The blurriness of VAE reconstructions under pixel-space likelihood is theoretically understood (mean-seeking vs. mode-seeking behavior) but practical solutions — perceptual losses, VQ bottlenecks, flow decoders — each introduce their own approximations.
- **Codebook utilization**: VQ-VAE codebooks frequently exhibit collapse (many codes unused). Entropy-constrained codebook learning, EMA updates, and residual quantization are active research directions.
- **Hybrid GAN-diffusion**: Adversarial objectives are now used at production scale for few-step distillation and post-training (ADD, LADD, DMD2, Seaweed-APT, Seedream 4.0). What remains open is *theory*: when adversarial post-training preserves the teacher's diversity versus when it silently re-introduces mode dropping, and how to choose discriminator feature spaces (DINOv2, teacher features, video encoders) in a principled way.
- **What should a tokenizer optimize?**: VA-VAE, REPA-E and RAE show that latent *learnability* matters as much as reconstruction PSNR/rFID, while JiT suggests the tokenizer can sometimes be dropped entirely. A unified criterion that predicts downstream generative FID from autoencoder properties does not yet exist.
- **Inversion quality**: GAN inversion (projecting real images into $\mathcal{W}$ or $\mathcal{W+}$ space) still struggles with rare faces, complex backgrounds, and accessories. No inversion method achieves both reconstruction fidelity and editability simultaneously.
- **Equivariance in generation**: StyleGAN3's equivariance result suggests rethinking convolution as the core operator in generators. Whether equivariant architectures generalize to video, 3D, and multimodal settings is open.
- **Theoretical analysis of ELBO tightness**: The gap between the ELBO and the true log-likelihood in VAEs is poorly characterized for high-dimensional image distributions; importance-weighted bounds improve estimates but the practical implications for generation quality are not fully understood.

---

## Further Reading

- [Goodfellow et al. (2014), "Generative Adversarial Nets," NeurIPS 2014](https://arxiv.org/abs/1406.2661)
- [Karras et al. (2020), "Analyzing and Improving the Image Quality of StyleGAN," CVPR 2020](https://arxiv.org/abs/1912.04958)
- [Kingma & Welling (2013), "Auto-Encoding Variational Bayes," ICLR 2014](https://arxiv.org/abs/1312.6114)
- [Esser et al. (2021), "Taming Transformers for High-Resolution Image Synthesis," CVPR 2021](https://arxiv.org/abs/2012.09841)
- [Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022](https://arxiv.org/abs/2112.10752)
- [Bond-Taylor et al. (2021), "Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models," TPAMI](https://arxiv.org/abs/2103.04922)
- [Huang et al. (2024), "The GAN is dead; long live the GAN! A Modern GAN Baseline," NeurIPS 2024](https://arxiv.org/abs/2501.05441)
- [Sauer et al. (2023), "Adversarial Diffusion Distillation"](https://arxiv.org/abs/2311.17042)
- [Lin et al. (2025), "Diffusion Adversarial Post-Training for One-Step Video Generation"](https://arxiv.org/abs/2501.08316)
- [Chen et al. (2024), "Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models," ICLR 2025](https://arxiv.org/abs/2410.10733)
- [Yao et al. (2025), "Reconstruction vs. Generation: Taming Optimization Dilemma with Visual Foundation Model Aligned Latent Diffusion Models," CVPR 2025](https://arxiv.org/abs/2501.01423)
- [Zheng et al. (2025), "Diffusion Transformers with Representation Autoencoders"](https://arxiv.org/abs/2510.11690)
- [Li & He (2025), "Back to Basics: Let Denoising Generative Models Denoise" (JiT)](https://arxiv.org/abs/2511.13720)
