# Image Formation: Cameras, Radiometry, and Color

> **Last Updated:** June 2026
> **Level:** Foundational
> **Related Sections:** [Image Processing](./02_image_processing.md) · [Classical Geometry](./04_classical_geometry.md) · [SfM and SLAM](../04_3d_vision_and_scene/03_sfm_and_slam.md) · [CV Overview](./00_overview.md)

---

## Overview

Image formation is the physical and mathematical process by which three-dimensional scenes are projected onto two-dimensional sensor planes, producing the discrete digital arrays that CV algorithms consume. Understanding this process is indispensable: every subsequent stage of a vision pipeline — from edge detection to 3D reconstruction — implicitly or explicitly inverts some aspect of image formation. Errors in the forward model (incorrect calibration, unmodelled distortion, wrong color space) propagate as systematic biases that no amount of downstream learning can fully correct.

The image formation pipeline encompasses three major phenomena. First, **geometric projection**: how 3D scene points are mapped to 2D image coordinates via the camera's optical system, modelled most usefully by the pinhole camera with intrinsic and extrinsic parameters. Second, **radiometry and photometry**: how light energy from a scene — described by BRDFs, radiance fields, and irradiance — is captured and converted to pixel values. Third, **sensor physics**: how photons are converted to electrons (photoelectric effect), sampled by a discrete pixel array, quantized to integer values, and corrupted by various noise sources. Each aspect produces a different class of distortion or uncertainty in the final image.

A rigorous treatment requires distinguishing quantities carefully: radiance (energy per solid angle per projected area, W·m⁻²·sr⁻¹) is a property of a ray in space; irradiance (W·m⁻²) is incident power per unit area on a surface; and pixel values (digital numbers) are a quantized, nonlinearly encoded transformation of sensor irradiance. Color science adds a further layer: what the sensor measures depends on spectral sensitivity, illuminant spectrum, and the color processing pipeline (demosaicing, white balance, gamma encoding).

---

## The Pinhole Camera Model

### Geometric Projection

The pinhole camera is a mathematical idealisation that captures the essential geometry of perspective projection. It consists of a centre of projection (the *optical centre* or *camera centre*) $\mathbf{C}$ and an image plane at distance $f$ (the *focal length*) from $\mathbf{C}$. A 3D point $\mathbf{X} = (X, Y, Z)^\top$ in camera coordinates is projected to the image point $(x, y)$ by similar triangles:

```latex
x = f \frac{X}{Z}, \quad y = f \frac{Y}{Z}
```

In homogeneous coordinates, this becomes the canonical projection matrix $\mathbf{P}_0 = [\mathbf{I} \mid \mathbf{0}]$ (diag$(f,f,1) \cdot [\mathbf{I}\mid\mathbf{0}]$ in pixel space):

```latex
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
= \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix}
\begin{pmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{pmatrix}
```

where $\lambda = Z$ (the depth) is the projective scale factor.

### Intrinsic Matrix $\mathbf{K}$

The **camera intrinsic matrix** $\mathbf{K}$ encodes the mapping from normalised camera coordinates to pixel coordinates:

```latex
\mathbf{K} = \begin{pmatrix}
f_x & s   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{pmatrix}
```

- $f_x, f_y$: focal lengths in pixels along $u$ and $v$ axes (often $f_x \approx f_y$ for square pixels)
- $(c_x, c_y)$: principal point — the image coordinates of the optical axis intersection
- $s$: skew coefficient, zero for most modern sensors

### Extrinsic Matrix $[\mathbf{R} \mid \mathbf{t}]$

The **extrinsic parameters** describe the rigid-body transformation from the world coordinate system to the camera coordinate system:

```latex
\mathbf{X}_{\text{cam}} = \mathbf{R} \mathbf{X}_w + \mathbf{t}
```

$\mathbf{R} \in SO(3)$ is a rotation matrix (3 degrees of freedom); $\mathbf{t} \in \mathbb{R}^3$ is a translation vector. Note that the camera centre in world coordinates is $\mathbf{C} = -\mathbf{R}^\top \mathbf{t}$.

### Full Projection Matrix

The complete **projection matrix** $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$ is a $3 \times 4$ rank-3 matrix:

```latex
\mathbf{P} = \begin{pmatrix}
f_x & s   & c_x & 0 \\
0   & f_y & c_y & 0 \\
0   & 0   & 1   & 0
\end{pmatrix}
\begin{pmatrix}
r_{11} & r_{12} & r_{13} & t_1 \\
r_{21} & r_{22} & r_{23} & t_2 \\
r_{31} & r_{32} & r_{33} & t_3 \\
0      & 0      & 0      & 1
\end{pmatrix}
```

This matrix has 11 degrees of freedom (5 intrinsic + 6 extrinsic), though in practice the skew is often fixed to zero leaving 10.

---

## Lens Distortion

Real lenses deviate from the ideal pinhole model due to manufacturing imperfections and the physics of refraction. The dominant terms are:

**Radial distortion** — the most significant in practice, caused by the lens bending light more at its edges than at its centre. Positive (barrel) distortion bows edges outward; negative (pincushion) bows inward:

```latex
x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6), \quad
y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
```

where $r^2 = x^2 + y^2$ and $(x, y)$ are normalised (undistorted) image coordinates.

**Tangential distortion** — caused by lens elements not being perfectly parallel to the image plane (decentering):

```latex
\Delta x = 2p_1 xy + p_2(r^2 + 2x^2), \quad
\Delta y = p_1(r^2 + 2y^2) + 2p_2 xy
```

The full distortion model thus requires at minimum five parameters $(k_1, k_2, p_1, p_2, k_3)$ — the standard in OpenCV and Zhang's calibration method (Zhang, IEEE PAMI, 2000).

---

## Radiometry: Light, Surfaces, and Sensors

### Key Radiometric Quantities

| Quantity | Symbol | Units | Definition |
|----------|--------|-------|------------|
| Radiant energy | $Q$ | J | Total energy |
| Radiant flux (power) | $\Phi$ | W | Energy per unit time |
| Irradiance | $E$ | W·m⁻² | Incident flux per unit area |
| Radiance | $L$ | W·m⁻²·sr⁻¹ | Flux per projected area per solid angle |

**Radiance** $L$ is the fundamental quantity invariant under lossless propagation along a ray and is what the camera sensor ultimately measures (modulo lens losses).

### BRDF

The **Bidirectional Reflectance Distribution Function** (BRDF), originally defined by Nicodemus et al. (National Bureau of Standards, 1977), characterises how a surface reflects incident light:

```latex
f_r(\omega_i \to \omega_r) = \frac{dL_r(\omega_r)}{L_i(\omega_i) \cos\theta_i \, d\omega_i}
```

where $\omega_i$ is the incoming direction, $\omega_r$ the outgoing (reflected) direction, $\theta_i$ the angle of incidence, and units are sr⁻¹. The BRDF must satisfy:
- **Positivity**: $f_r \geq 0$
- **Helmholtz reciprocity**: $f_r(\omega_i \to \omega_r) = f_r(\omega_r \to \omega_i)$
- **Energy conservation**: $\int_\Omega f_r(\omega_i \to \omega_r) \cos\theta_r \, d\omega_r \leq 1$

The **rendering equation** (Kajiya, 1986) integrates the BRDF over the hemisphere of incoming directions:

```latex
L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_\Omega f_r(\omega_i \to \omega_o) L_i(\mathbf{x}, \omega_i) \cos\theta_i \, d\omega_i
```

### Irradiance at the Sensor

For a thin lens of focal length $f$ and aperture diameter $d$, the irradiance at the image plane from a Lambertian source of radiance $L$ is:

```latex
E = \frac{\pi}{4} \left(\frac{d}{f}\right)^2 L \cos^4\alpha
```

where $\alpha$ is the angle from the optical axis ($\cos^4\alpha$ vignetting). The $\cos^4$ falloff explains the characteristic darkening at image corners in wide-angle lenses.

---

## Color: Spectral Sensitivity, Demosaicing, and Color Spaces

### Spectral Integration

A sensor channel $c$ produces output:

```latex
I_c = \int_\lambda S_c(\lambda) \cdot L(\lambda) \, d\lambda
```

where $S_c(\lambda)$ is the spectral sensitivity of channel $c$ and $L(\lambda)$ is the spectral radiance. The RGB sensitivity curves of a camera differ from the CIE cone fundamentals, so raw camera RGB values are not perceptually uniform.

### Bayer CFA and Demosaicing

Most digital cameras use a single-chip sensor overlaid with a **Color Filter Array (CFA)** — most commonly the Bayer pattern (patented by Bryce Bayer at Kodak, 1976), arranged as a 2×2 tile:

```
G R
B G
```

The pattern is 50% green, 25% red, 25% blue, reflecting human luminance sensitivity peaking in the green. Each pixel records only one color channel; the missing channels are **interpolated** in the demosaicing step. Common demosaicing algorithms range from bilinear interpolation (fast, low quality) to gradient-corrected (AHD, VNG) and deep learning-based methods (RCAN, etc.). Demosaicing errors manifest as color fringing (zipper artifacts) at high-contrast edges.

### Color Spaces

| Space | Description | Use Case |
|-------|-------------|----------|
| Linear RGB | Camera-native linear response | Radiometric computation |
| sRGB | Gamma-encoded ($\gamma \approx 2.2$) standard | Display, web |
| HSV / HSL | Hue-Saturation-Value/Lightness | Color-based segmentation |
| LAB (CIE L\*a\*b\*) | Perceptually uniform (D65 illuminant) | Color distance metrics |
| YCbCr | Luma + chroma (JPEG, video) | Compression |
| XYZ (CIE 1931) | Device-independent human color space | Color management |

The **gamma correction** applied in standard pipelines encodes linear irradiance values $E$ as:

```latex
V_{\text{out}} = \begin{cases}
12.92 \, E & E \leq 0.0031308 \\
1.055 \, E^{1/2.4} - 0.055 & E > 0.0031308
\end{cases}
```

(sRGB IEC 61966-2-1 standard). Failing to account for gamma before convolution or gradient computation is a common source of subtle errors in CV pipelines.

---

## Sampling, Quantization, and Noise Models

### Nyquist Sampling

The Shannon–Nyquist theorem requires that to avoid aliasing, the sampling rate $f_s$ must exceed twice the maximum spatial frequency $B$ in the signal:

```latex
f_s > 2B \quad \text{(Nyquist criterion)}
```

Camera lenses act as low-pass filters; if the optical cutoff frequency exceeds the pixel sampling rate, aliasing appears as moiré patterns. Camera manufacturers place optical low-pass filters (OLPF) before the sensor to band-limit the scene.

### Quantization

An 8-bit sensor maps the analog voltage range $[0, V_{\max}]$ to 256 integer levels, yielding **quantization noise** with RMS value:

```latex
\sigma_q = \frac{\Delta}{2\sqrt{3}}, \quad \Delta = \frac{V_{\max}}{2^N}
```

where $N$ is the bit depth. Professional cameras use 12–16 bit RAW formats to defer quantization to post-processing.

### Noise Sources

The dominant noise sources in digital image sensors, from physical first principles:

| Source | Distribution | Description |
|--------|-------------|-------------|
| **Photon shot noise** | Poisson($\lambda$), $\sigma = \sqrt{\lambda}$ | Quantum fluctuations in photon arrival |
| **Read noise** | Gaussian | Electronic noise in readout amplifiers |
| **Dark current** | Poisson (temp. dependent) | Thermal generation of electron-hole pairs |
| **Fixed-pattern noise** | Deterministic bias | Per-pixel gain/offset variations |
| **Quantization noise** | Uniform $[-\Delta/2, \Delta/2]$ | ADC rounding |

At high light levels (ISO 100), shot noise dominates and $\sigma \propto \sqrt{\text{signal}}$. At high ISO or long exposures, read noise and dark current become significant. The combined model commonly used is:

```latex
\sigma^2_{\text{total}}(I) = \alpha I + \sigma_r^2
```

where $\alpha$ is the shot noise gain and $\sigma_r^2$ is the read noise variance. This is the basis of **noise modeling** used in denoising algorithms (BM3D, DnCNN, etc.) and in calibration pipelines.

---

## Key Papers

| Paper | Authors | Year | Venue | Key Contribution |
|-------|---------|------|-------|-----------------|
| "A Flexible New Technique for Camera Calibration" | Z. Zhang | 2000 | IEEE PAMI | Practical planar-pattern calibration method; closed-form + MLE refinement |
| "Geometrical Considerations and Nomenclature for Reflectance" | F. Nicodemus et al. | 1977 | NBS Monograph 160 | Formal definition of the BRDF |
| "The Rendering Equation" | J. T. Kajiya | 1986 | SIGGRAPH | Integral formulation of light transport |
| "Color Filter Array" (Bayer patent) | B. Bayer (Kodak) | 1976 | US Patent 3,971,065 | RGGB mosaic pattern still used in most cameras today |
| "Camera Models and Fundamental Concepts Used in Geometric Computer Vision" | P. Sturm et al. | 2011 | Foundations and Trends in Computer Graphics and Vision | Comprehensive unified treatment of camera models |
| *Multiple View Geometry in Computer Vision* (Ch. 6–7) | R. Hartley, A. Zisserman | 2004 | Cambridge University Press | Canonical reference for projection geometry |
| "Deep Learning for Camera Calibration and Beyond: A Survey" | Y. Cao et al. | 2023 | arXiv 2303.10559 | Survey of learned calibration methods |

---

## Benchmark Performance

Camera calibration accuracy is typically reported as **reprojection error** — the RMS pixel distance between projected 3D calibration points and their detected 2D image locations. Zhang's method achieves sub-pixel accuracy (< 0.5 px RMS) with as few as 10–15 images of a planar checkerboard using the standard 5-parameter distortion model.

| Method | Reprojection Error | Notes |
|--------|-------------------|-------|
| Zhang (2000) — classical | < 0.3–0.5 px RMS | Standard baseline; used in OpenCV `calibrateCamera` |
| DNN-based (single image) | ~1–2 px RMS | No calibration target needed |
| Self-calibration (SfM) | ~0.5–1.0 px | Requires scene texture |

---

## Pros & Cons: Camera Model Choices

| Model | Pros | Cons |
|-------|------|------|
| Pinhole (no distortion) | Simple, analytically tractable | Inaccurate for wide-angle or cheap lenses |
| Pinhole + radial/tangential | Standard; covers 95% of use cases | Inadequate for fisheye (>90° FOV) |
| Division model (Fitzgibbon 2001) | Single parameter, closed-form inversion | Less accurate than polynomial for large distortion |
| Kannala-Brandt fisheye | Handles > 180° FOV | Requires specialised tooling; more parameters |
| Unified sphere model | Handles catadioptric systems | More complex; less standard tooling |

---

## Open Problems & Research Gaps

- **Self-supervised / in-the-wild calibration.** Current methods require controlled calibration targets or textured scenes; robust single-image calibration from arbitrary content remains challenging.
- **Dynamic calibration.** Thermal expansion and mechanical vibration cause intrinsics to drift over time; continuous online recalibration is an open engineering and algorithmic problem.
- **Non-Lambertian and global illumination effects.** BRDFs in real scenes (specularities, inter-reflections, subsurface scattering) violate the assumptions of most calibration and reconstruction methods.
- **Event cameras.** Neuromorphic (event-based) sensors produce asynchronous pixel-level brightness changes rather than frames; their formation model, calibration, and integration with standard CV pipelines is actively being developed.
- **Learned image signal processors (ISPs).** The gap between RAW sensor data and processed images involves dozens of nonlinear steps (noise reduction, sharpening, tonemapping); learning the full ISP end-to-end for downstream CV tasks is an active area.
- **Polarimetric and multispectral imaging.** Sensors that capture polarization or wavelengths beyond visible RGB encode additional scene information; models and algorithms for these modalities are less mature.
- **Unified noise models for modern stacked sensors.** Stacked CMOS sensors with in-pixel processing have noise characteristics that depart from the classical Poisson-Gaussian model.

---

## Further Reading

- [Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. *IEEE PAMI*, 22(11), 1330–1334.](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/) — Original calibration paper.
- [Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed.](https://www.robots.ox.ac.uk/~vgg/hzbook/) — Chapters 6–7 cover camera models and calibration exhaustively.
- [Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed., Ch. 2.](https://szeliski.org/Book/) — Accessible treatment of image formation and sensors.
- [Cao, Y. et al. (2023). Deep Learning for Camera Calibration and Beyond: A Survey. arXiv:2303.10559.](https://arxiv.org/abs/2303.10559) — Modern learned calibration overview.
- [Nicodemus, F. et al. (1977). *Geometrical Considerations and Nomenclature for Reflectance*. NBS Monograph 160.](https://nvlpubs.nist.gov/nistpubs/Legacy/MONO/nbsmonograph160.pdf) — BRDF definition source.
- [Wikipedia: Color filter array.](https://en.wikipedia.org/wiki/Color_filter_array) — Overview of Bayer and alternative CFA patterns.
