# 图像形成：相机、辐射度学与颜色

> **英文原文：** [cv-knowledge-base/00_foundations/01_image_formation.md](../../cv-knowledge-base/00_foundations/01_image_formation.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 基础
> **相关章节：** [图像处理](./02_image_processing.md) · [经典几何](./04_classical_geometry.md) · [SfM 与 SLAM](../04_3d_vision_and_scene/03_sfm_and_slam.md) · [CV 概述](./00_overview.md)

> **🆕 2026年10月更新：**
> - 图像形成的物理原理没有变化；主要进展在*求逆*一侧：与模型无关的单图像标定（AnyCalib，ICCV 2025），以及顺带预测内参/视场角的三维基础模型（VGGT、MoGe-2、Depth Anything 3）。新增一节对此进行总结。
> - 相机模型表新增通用逐像素射线模型、EUCM 和等距柱状（360°）模型，反映了 COLMAP 4.x（2026）目前支持的模型。
> - 更新了关于野外（in-the-wild）标定的开放问题，以反映上述进展。

---

## 概述

图像形成是三维场景投影到二维传感器平面、进而产生供 CV 算法使用的离散数字阵列的物理与数学过程。理解这一过程不可或缺：视觉流水线的每个后续阶段，从边缘检测到三维重建，都隐式或显式地对图像形成的某个方面进行求逆。前向模型中的误差（标定错误、未建模的畸变、错误的色彩空间）会以系统性偏差的形式传播，无论下游学习多么充分都无法完全纠正。

图像形成流水线包含三大类现象。第一，**几何投影**：三维场景点如何经由相机光学系统映射到二维图像坐标，最实用的建模方式是带内参和外参的针孔相机模型。第二，**辐射度学与光度学**：来自场景的光能（由 BRDF、辐射场和辐照度描述）如何被捕获并转换为像素值。第三，**传感器物理**：光子如何被转换为电子（光电效应），由离散像素阵列采样，量化为整数值，并受到各种噪声源的污染。每个方面都会在最终图像中引入不同类别的失真或不确定性。

严谨的处理需要仔细区分各个物理量：辐亮度（radiance，单位投影面积、单位立体角的能量，W·m⁻²·sr⁻¹）是空间中一条光线的属性；辐照度（irradiance，W·m⁻²）是入射到表面上单位面积的功率；而像素值（数字量）是传感器辐照度经过量化和非线性编码后的变换。色彩科学又增加了一层：传感器测量到的内容取决于光谱灵敏度、光源光谱以及色彩处理流水线（去马赛克、白平衡、伽马编码）。

---

## 针孔相机模型

### 几何投影

针孔相机是一种数学理想化模型，刻画了透视投影的基本几何。它由一个投影中心（*光心*或*相机中心*）$\mathbf{C}$ 和一个距 $\mathbf{C}$ 为 $f$（*焦距*）的像平面组成。相机坐标系中的三维点 $\mathbf{X} = (X, Y, Z)^\top$ 由相似三角形投影到图像点 $(x, y)$：

```latex
x = f \frac{X}{Z}, \quad y = f \frac{Y}{Z}
```

在齐次坐标下，这成为标准投影矩阵 $\mathbf{P}_0 = [\mathbf{I} \mid \mathbf{0}]$（在像素空间中为 diag$(f,f,1) \cdot [\mathbf{I}\mid\mathbf{0}]$）：

```latex
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
= \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix}
\begin{pmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{pmatrix}
```

其中 $\lambda = Z$（深度）是射影尺度因子。

### 内参矩阵 $\mathbf{K}$

**相机内参矩阵** $\mathbf{K}$ 编码了从归一化相机坐标到像素坐标的映射：

```latex
\mathbf{K} = \begin{pmatrix}
f_x & s   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{pmatrix}
```

- $f_x, f_y$：沿 $u$ 轴和 $v$ 轴、以像素为单位的焦距（对于方形像素通常 $f_x \approx f_y$）
- $(c_x, c_y)$：主点，即光轴与像平面交点的图像坐标
- $s$：倾斜系数，对于大多数现代传感器为零

### 外参矩阵 $[\mathbf{R} \mid \mathbf{t}]$

**外参**描述了从世界坐标系到相机坐标系的刚体变换：

```latex
\mathbf{X}_{\text{cam}} = \mathbf{R} \mathbf{X}_w + \mathbf{t}
```

$\mathbf{R} \in SO(3)$ 是旋转矩阵（3 个自由度）；$\mathbf{t} \in \mathbb{R}^3$ 是平移向量。注意，相机中心在世界坐标系中为 $\mathbf{C} = -\mathbf{R}^\top \mathbf{t}$。

### 完整投影矩阵

完整的**投影矩阵** $\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$ 是一个秩为 3 的 $3 \times 4$ 矩阵：

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

该矩阵有 11 个自由度（5 个内参 + 6 个外参），但在实践中倾斜系数通常固定为零，剩下 10 个。

---

## 镜头畸变

由于制造缺陷和折射物理，真实镜头会偏离理想针孔模型。主要的畸变项如下：

**径向畸变**：实践中最显著的畸变，由镜头边缘对光线的弯折强于中心所致。正向（桶形）畸变使边缘向外凸出；负向（枕形）畸变使边缘向内凹陷：

```latex
x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6), \quad
y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
```

其中 $r^2 = x^2 + y^2$，$(x, y)$ 为归一化（无畸变）图像坐标。

**切向畸变**：由镜片与像平面不完全平行（偏心）所致：

```latex
\Delta x = 2p_1 xy + p_2(r^2 + 2x^2), \quad
\Delta y = p_1(r^2 + 2y^2) + 2p_2 xy
```

因此，完整的畸变模型至少需要五个参数 $(k_1, k_2, p_1, p_2, k_3)$，这是 OpenCV 和 Zhang 标定方法（Zhang，IEEE PAMI，2000）中的标准做法。

---

## 辐射度学：光、表面与传感器

### 关键辐射度量

| 物理量 | 符号 | 单位 | 定义 |
|----------|--------|-------|------------|
| 辐射能（Radiant energy） | $Q$ | J | 总能量 |
| 辐射通量（功率） | $\Phi$ | W | 单位时间的能量 |
| 辐照度（Irradiance） | $E$ | W·m⁻² | 单位面积上的入射通量 |
| 辐亮度（Radiance） | $L$ | W·m⁻²·sr⁻¹ | 单位投影面积、单位立体角的通量 |

**辐亮度** $L$ 是在沿光线无损传播时保持不变的基本物理量，也是相机传感器最终测量的量（不计镜头损耗）。

### BRDF

**双向反射分布函数**（Bidirectional Reflectance Distribution Function, BRDF）最初由 Nicodemus 等人（美国国家标准局，1977）定义，刻画了表面如何反射入射光：

```latex
f_r(\omega_i \to \omega_r) = \frac{dL_r(\omega_r)}{L_i(\omega_i) \cos\theta_i \, d\omega_i}
```

其中 $\omega_i$ 为入射方向，$\omega_r$ 为出射（反射）方向，$\theta_i$ 为入射角，单位为 sr⁻¹。BRDF 必须满足：
- **非负性**：$f_r \geq 0$
- **Helmholtz 互易性**：$f_r(\omega_i \to \omega_r) = f_r(\omega_r \to \omega_i)$
- **能量守恒**：$\int_\Omega f_r(\omega_i \to \omega_r) \cos\theta_r \, d\omega_r \leq 1$

**渲染方程**（Kajiya，1986）在入射方向半球上对 BRDF 进行积分：

```latex
L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_\Omega f_r(\omega_i \to \omega_o) L_i(\mathbf{x}, \omega_i) \cos\theta_i \, d\omega_i
```

### 传感器处的辐照度

对于焦距为 $f$、孔径直径为 $d$ 的薄透镜，来自辐亮度为 $L$ 的朗伯光源在像平面处的辐照度为：

```latex
E = \frac{\pi}{4} \left(\frac{d}{f}\right)^2 L \cos^4\alpha
```

其中 $\alpha$ 是相对于光轴的夹角（$\cos^4\alpha$ 渐晕）。$\cos^4$ 衰减解释了广角镜头图像四角特有的变暗现象。

---

## 学习式标定与前馈式相机估计（2025–2026）

上述前向模型是经典的，但对其*求逆*，即在没有标定靶的情况下恢复 $\mathbf{K}$ 和畸变，自 2025 年以来已发生了实质性变化。

**基于射线、与模型无关的标定。** 近期方法不再回归某一特定模型的参数（例如 $f, c_x, c_y, k_1$），而是预测*逐像素视线射线* $\mathbf{r}(u,v) = \mathbf{K}^{-1}(u, v, 1)^\top$（归一化形式，或更一般地，任意中心相机模型的反投影），然后以闭式解拟合所选的参数化模型。AnyCalib [TiradoGarin2025]（ICCV 2025）在流形上从单幅野外图像完成这一过程，并从同一稠密射线预测中恢复针孔、Brown–Conrady 和 Kannala–Brandt 模型参数；它还能处理经过裁剪和拉伸的图像。由于参数化模型是在网络*之后*拟合的，相机模型可以在推理时再选择。AnyCalib 的 ONNX 导出版本已随 COLMAP 的发布资源一同分发（2026年9月）。

```latex
\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \sum_{(u,v)} \left\| \pi^{-1}_{\boldsymbol{\theta}}(u,v) - \hat{\mathbf{r}}(u,v) \right\|^2
% 将相机模型参数 θ 拟合到网络预测的稠密射线场 r̂
```

**标定作为三维基础模型的副产品。** 前馈式几何网络如今会与深度一起联合预测相机：VGGT [Wang2025] 为每个输入视图回归内参和外参；MoGe-2（arXiv 2507.02546）预测度量尺度的点图，并在未提供视场角时对其进行估计；Depth Anything 3（2025年11月）使用“深度 + 射线”表示，可由此推导出位姿和内参。这些估计通常足以用于初始化 SfM，但它们继承了训练数据分布中的相机先验，在非常规光学系统（强鱼眼、折反射、经过大幅裁剪的图像）上的精度，远不如基于标定靶的标定那样得到充分刻画。

**工具链。** COLMAP 4.0–4.2（2026年3月至9月）新增了除法模型、鱼眼（无畸变）、等距柱状和 EUCM 相机模型，其全局建图器通过视图图标定（view-graph calibration）来估计内参。对于计量级精度的工作，Zhang 的基于标定靶的方法仍是参考标准。

---

## 颜色：光谱灵敏度、去马赛克与色彩空间

### 光谱积分

传感器通道 $c$ 的输出为：

```latex
I_c = \int_\lambda S_c(\lambda) \cdot L(\lambda) \, d\lambda
```

其中 $S_c(\lambda)$ 是通道 $c$ 的光谱灵敏度，$L(\lambda)$ 是光谱辐亮度。相机的 RGB 灵敏度曲线与 CIE 视锥基函数不同，因此原始相机 RGB 值在感知上并不均匀。

### Bayer CFA 与去马赛克

大多数数码相机使用覆盖有**彩色滤光阵列**（Color Filter Array, CFA）的单芯片传感器，最常见的是 Bayer 图案（由柯达的 Bryce Bayer 于 1976 年申请专利），以 2×2 单元排列：

```
G R
B G
```

该图案中绿色占 50%，红色和蓝色各占 25%，这反映了人眼亮度灵敏度在绿色波段达到峰值。每个像素只记录一个颜色通道；缺失的通道在去马赛克步骤中通过**插值**得到。常见的去马赛克算法从双线性插值（快速、质量低）到梯度校正方法（AHD、VNG），再到基于深度学习的方法（RCAN 等）。去马赛克误差表现为高对比度边缘处的彩色镶边（拉链伪影）。

### 色彩空间

| 色彩空间 | 描述 | 用途 |
|-------|-------------|----------|
| 线性 RGB | 相机原生的线性响应 | 辐射度计算 |
| sRGB | 伽马编码（$\gamma \approx 2.2$）的标准 | 显示、网页 |
| HSV / HSL | 色相-饱和度-明度/亮度 | 基于颜色的分割 |
| LAB（CIE L\*a\*b\*） | 感知均匀（D65 光源） | 颜色距离度量 |
| YCbCr | 亮度 + 色度（JPEG、视频） | 压缩 |
| XYZ（CIE 1931） | 与设备无关的人类色彩空间 | 色彩管理 |

标准流水线中应用的**伽马校正**将线性辐照度值 $E$ 编码为：

```latex
V_{\text{out}} = \begin{cases}
12.92 \, E & E \leq 0.0031308 \\
1.055 \, E^{1/2.4} - 0.055 & E > 0.0031308
\end{cases}
```

（sRGB IEC 61966-2-1 标准。）在卷积或梯度计算之前未考虑伽马，是 CV 流水线中常见的隐蔽误差来源。

---

## 采样、量化与噪声模型

### Nyquist 采样

Shannon–Nyquist 定理要求，为避免混叠，采样率 $f_s$ 必须大于信号中最高空间频率 $B$ 的两倍：

```latex
f_s > 2B \quad \text{(Nyquist criterion)}
```

相机镜头起到低通滤波器的作用；若光学截止频率超过像素采样率，混叠就会表现为摩尔纹。相机制造商在传感器前放置光学低通滤波器（OLPF），以对场景进行带限。

### 量化

8 位传感器将模拟电压范围 $[0, V_{\max}]$ 映射到 256 个整数级，产生均方根值如下的**量化噪声**：

```latex
\sigma_q = \frac{\Delta}{2\sqrt{3}}, \quad \Delta = \frac{V_{\max}}{2^N}
```

其中 $N$ 为位深。专业相机使用 12–16 位的 RAW 格式，将量化推迟到后期处理。

### 噪声源

从物理第一性原理出发，数字图像传感器中的主要噪声源如下：

| 噪声源 | 分布 | 描述 |
|--------|-------------|-------------|
| **光子散粒噪声** | Poisson($\lambda$)，$\sigma = \sqrt{\lambda}$ | 光子到达的量子涨落 |
| **读出噪声** | 高斯分布 | 读出放大器中的电子噪声 |
| **暗电流** | 泊松分布（与温度有关） | 电子-空穴对的热生成 |
| **固定模式噪声** | 确定性偏差 | 逐像素的增益/偏移差异 |
| **量化噪声** | 均匀分布 $[-\Delta/2, \Delta/2]$ | ADC 舍入 |

在高光照水平（ISO 100）下，散粒噪声占主导，$\sigma \propto \sqrt{\text{signal}}$。在高 ISO 或长曝光时，读出噪声和暗电流变得显著。常用的组合模型为：

```latex
\sigma^2_{\text{total}}(I) = \alpha I + \sigma_r^2
```

其中 $\alpha$ 为散粒噪声增益，$\sigma_r^2$ 为读出噪声方差。这是去噪算法（BM3D、DnCNN 等）和标定流水线中所用**噪声建模**的基础。

---

## 关键论文

| 论文 | 作者 | 年份 | 发表场所 | 关键贡献 |
|-------|---------|------|-------|-----------------|
| "A Flexible New Technique for Camera Calibration" | Z. Zhang | 2000 | IEEE PAMI | 实用的平面标定板标定方法；闭式解 + 最大似然估计优化 |
| "Geometrical Considerations and Nomenclature for Reflectance" | F. Nicodemus et al. | 1977 | NBS Monograph 160 | BRDF 的正式定义 |
| "The Rendering Equation" | J. T. Kajiya | 1986 | SIGGRAPH | 光传输的积分形式表述 |
| "Color Filter Array" (Bayer patent) | B. Bayer (Kodak) | 1976 | US Patent 3,971,065 | 至今仍用于大多数相机的 RGGB 马赛克图案 |
| "Camera Models and Fundamental Concepts Used in Geometric Computer Vision" | P. Sturm et al. | 2011 | Foundations and Trends in Computer Graphics and Vision | 对相机模型全面而统一的论述 |
| *Multiple View Geometry in Computer Vision* (Ch. 6–7) | R. Hartley, A. Zisserman | 2004 | Cambridge University Press | 投影几何的经典参考 |
| "Deep Learning for Camera Calibration and Beyond: A Survey" | Y. Cao et al. | 2023 | arXiv 2303.10559 | 学习式标定方法综述 |
| "AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration" | J. Tirado-Garín, J. Civera | 2025 | ICCV | 稠密射线回归 + 闭式拟合，从单幅图像适配任意中心相机模型 |
| "VGGT: Visual Geometry Grounded Transformer" | J. Wang et al. | 2025 | CVPR（最佳论文） | 对多视图前馈预测内参、外参和深度 |

---

## 基准性能

相机标定精度通常以**重投影误差**报告，即投影的三维标定点与其检测到的二维图像位置之间的均方根像素距离。使用标准的 5 参数畸变模型，Zhang 的方法仅需 10–15 幅平面棋盘格图像即可达到亚像素精度（< 0.5 px RMS）。

| 方法 | 重投影误差 | 说明 |
|--------|-------------------|-------|
| Zhang (2000)：经典方法 | < 0.3–0.5 px RMS | 标准基线；用于 OpenCV `calibrateCamera` |
| 基于 DNN（单图像） | ~1–2 px RMS | 无需标定靶 |
| 自标定（SfM） | ~0.5–1.0 px | 需要场景纹理 |

单图像学习式标定器（GeoCalib、AnyCalib）和三维基础模型通常以*内参*误差（例如相对焦距误差、视场角误差）和位姿 AUC 来评估，而不是以相对标定靶的重投影误差来评估，因此它们的数值不能与上表各行直接比较。

---

## 优缺点：相机模型的选择

| 模型 | 优点 | 缺点 |
|-------|------|------|
| 针孔（无畸变） | 简单，便于解析处理 | 对广角或廉价镜头不准确 |
| 针孔 + 径向/切向畸变 | 标准做法；覆盖 95% 的使用场景 | 不适用于鱼眼（>90° 视场） |
| 除法模型（Fitzgibbon 2001） | 单参数，可闭式求逆 | 对大畸变而言精度不如多项式模型 |
| Kannala-Brandt 鱼眼 | 可处理 > 180° 视场 | 需要专门工具；参数更多 |
| 统一球面模型 | 可处理折反射系统 | 更复杂；标准工具支持较少 |
| 增强统一模型（EUCM） | 参数少；适用于鱼眼和广角；COLMAP 4.1+ 支持 | 在较旧的工具链中不常见 |
| 等距柱状（360°） | 全景/球面相机的原生模型 | 两极附近畸变强烈；需要专门的匹配方法 |
| 通用逐像素射线（学习式，例如 AnyCalib） | 与模型无关；可从单幅野外图像工作 | 精度受学习先验限制；达不到计量级 |

---

## 开放问题与研究空白

- **自监督 / 野外标定。** 经典方法需要受控的标定靶或有纹理的场景。学习式单图像标定器（AnyCalib，2025）和三维基础模型（VGGT、MoGe-2）已使野外估计变得实用，但它们在非常规光学系统上的误差及其不确定性校准仍缺乏充分刻画，精度也尚未达到基于标定靶的方法。
- **动态标定。** 热膨胀和机械振动会导致内参随时间漂移；持续的在线重标定是一个开放的工程与算法问题。
- **非朗伯与全局光照效应。** 真实场景中的 BRDF（高光、互反射、次表面散射）违背了大多数标定与重建方法的假设。
- **事件相机。** 神经形态（基于事件的）传感器产生的是异步的像素级亮度变化而非帧；其成像模型、标定以及与标准 CV 流水线的集成仍在积极研究中。
- **学习式图像信号处理器（ISP）。** 从 RAW 传感器数据到处理后图像之间涉及数十个非线性步骤（降噪、锐化、色调映射）；为下游 CV 任务端到端学习完整的 ISP 是一个活跃的研究领域。
- **偏振与多光谱成像。** 捕获偏振信息或可见 RGB 以外波长的传感器编码了额外的场景信息；面向这些模态的模型和算法尚不成熟。
- **面向现代堆栈式传感器的统一噪声模型。** 具有像素内处理能力的堆栈式 CMOS 传感器，其噪声特性偏离了经典的泊松-高斯模型。

---

## 延伸阅读

- [Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. *IEEE PAMI*, 22(11), 1330–1334.](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)：原始标定论文。
- [Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed.](https://www.robots.ox.ac.uk/~vgg/hzbook/)：第 6–7 章详尽介绍了相机模型与标定。
- [Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed., Ch. 2.](https://szeliski.org/Book/)：对图像形成与传感器通俗易懂的介绍。
- [Cao, Y. et al. (2023). Deep Learning for Camera Calibration and Beyond: A Survey. arXiv:2303.10559.](https://arxiv.org/abs/2303.10559)：现代学习式标定概览。
- [Nicodemus, F. et al. (1977). *Geometrical Considerations and Nomenclature for Reflectance*. NBS Monograph 160.](https://nvlpubs.nist.gov/nistpubs/Legacy/MONO/nbsmonograph160.pdf)：BRDF 定义的出处。
- [Wikipedia: Color filter array.](https://en.wikipedia.org/wiki/Color_filter_array)：Bayer 及其他 CFA 图案概览。
- [Tirado-Garín, J. & Civera, J. (2025). AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration. *ICCV 2025*.](https://arxiv.org/abs/2503.12701)：与模型无关的单图像标定；代码见 [javrtg/AnyCalib](https://github.com/javrtg/AnyCalib)。
- [MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details. arXiv:2507.02546.](https://github.com/microsoft/MoGe)：带视场角估计的单目度量点图。
- [COLMAP 发布说明。](https://github.com/colmap/colmap/releases)：4.x（2026）中新增的相机模型与标定选项。
