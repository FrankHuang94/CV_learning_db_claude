# 3D 表示：点云、体素、网格、隐式曲面与神经场

> **英文原文：** [cv-knowledge-base/04_3d_vision_and_scene/00_3d_representations.md](../../cv-knowledge-base/04_3d_vision_and_scene/00_3d_representations.md)（本中文版供参考，内容以英文原文为准）

> **最后更新：** 2026年10月
> **难度级别：** 高级
> **相关章节：** [NeRF 及其变体](./01_nerf_and_variants.md) · [高斯泼溅](./02_gaussian_splatting.md) · [SfM 与 SLAM](./03_sfm_and_slam.md) · [3D 生成](./04_3d_generation.md)

> **🆕 2026年10月更新：**
> - 新增一节，介绍**作为学习型 3D 表示的点图（pointmap）与射线图（ray map）**。它们是 VGGT（CVPR 2025 最佳论文）、π³、MapAnything 和 Depth Anything 3 等前馈重建模型的输出格式。
> - 新增一节，介绍**内蕴曲面纹理**：ECCV 2026 最佳论文 Heat Kernel Textures（HKTex）用直接定义在网格上的各向异性热核（“测地高斯”）取代 UV 图集。
> - 新增一节，介绍**结构化稀疏体素潜变量**（TRELLIS 的 SLAT 与 TRELLIS.2 的 O-Voxel，后者为 CVPR 2026 最佳学生论文）。它们是当前最先进 3D 生成背后的表示。
> - 随着 Khronos `KHR_gaussian_splatting` glTF 扩展获得正式批准，3DGS 成为标准交换格式。光线追踪的基元变体（3DGRT/3DGUT、EVER、Radiant Foam）模糊了泼溅与体表示之间的界限。
> - 相应扩充了权衡表、关键论文、基准性能、开放问题与延伸阅读。

---

## 概述

3D 表示的选择是任何 3D 视觉系统中最基础的架构决策。不同表示在内存占用、几何保真度、渲染速度、可编辑性和可学习性之间做出了根本不同的权衡。过去十年间，该领域从经典的离散表示（点云、体素、网格）走向连续隐式函数（SDF、占用场），近年又走向神经场（NeRF）和显式概率基元（3D 高斯泼溅，3D Gaussian Splatting）。要为从机器人操作到照片级内容创作等各类任务选择合适的骨干表示，就必须严谨地理解这些权衡。

3D 表示的分类大体分为**显式**和**隐式**两种范式。显式表示（点云、体素、网格、高斯）直接在内存中存储几何基元，渲染快、易于操作，但在分辨率与效率之间的权衡空间有限。隐式表示把几何编码为某个学习得到的或解析的标量场的水平集，例如有符号距离函数、占用概率或辐射场。它在理论上可达到无限分辨率，代价是查询操作开销大。随着 Instant-NGP [MüllerEtAl2022] 等混合方法的出现，这两类之间的界限已大为模糊；Instant-NGP 把隐式 MLP 查询嵌入到显式哈希网格查找之中。

一个关键却常被低估的维度是**监督信号兼容性**：点云天然来自 LiDAR 和立体视觉，体素来自 CT/MRI，网格来自美术制作管线和 MVS 的 Marching Cubes，而隐式神经场则直接由带位姿的 RGB 图像训练得到。获取途径的多样性意味着，真实世界的 3D 视觉管线常常必须在不同表示之间转换。例如，先从图像拟合 NeRF，再提取网格用于物理仿真，然后转换为点云供下游学习使用。

---

## 点云

点云是最原始的 3D 表示：一个无序集合 $\mathcal{P} = \{p_i \in \mathbb{R}^d\}_{i=1}^N$，其中每个点至少携带 $(x, y, z)$ 坐标，并可选地带有颜色、法向和语义属性。点云是 LiDAR 传感器以及立体/深度重建算法的直接输出，不需要任何连接关系信息。

### PointNet 与 PointNet++ [Qi2017a, Qi2017b]

点云深度学习的开创性突破是 PointNet [Qi2017a]，由 Charles R. Qi、Hao Su、Kaichun Mo 和 Leonidas J. Guibas（斯坦福大学）发表于 CVPR 2017。PointNet 解决的核心难题是**置换不变性**：点云 $\{p_1, \ldots, p_N\}$ 无论点的顺序如何都表示同一形状，而标准卷积无法处理这一点。PointNet 的方案在架构上十分优雅：对每个点逐点应用共享 MLP，把它提升到高维特征，再用对称函数（全局最大池化）聚合，得到与顺序无关的全局描述子。

```
# PointNet 全局特征提取（伪代码）
for p_i in point_cloud:
    f_i = MLP(p_i)           # 共享权重，与点无关
global_feat = maxpool({f_i}) # 对称聚合
```

一个关键的正则化器是 **T-Net**（一个小型 PointNet，先后输出 3×3 和 64×64 的对齐矩阵），它学习在特征提取之前把输入规范化。PointNet 分别在 ModelNet40 上的分类和 ShapeNet Part 上的部件分割中取得了强劲表现，但缺少捕获局部几何结构的机制。

同一团队在 NeurIPS 2017 发表的 PointNet++ [Qi2017b] 通过在嵌套的局部邻域上递归应用 PointNet 解决了这一问题。该架构引入了**最远点采样**（FPS）来选取具有代表性的中心点，用**球查询**（ball query）对半径 $r$ 内的邻居进行分组，并针对非均匀密度采用**多尺度分组**（MSG）策略。形式上，一个集合抽象层把 $\{(x_i, f_i)\}$ 映射为 $\{(y_j, g_j)\}$，其中 $y_j$ 是下采样后的中心点，$g_j$ 是从局部邻域聚合得到的特征。

**点云的局限：** 没有内在拓扑，使得曲面渲染代价高昂（例如需要泼溅或网格化）；存储量随场景复杂度线性增长；固定密度会在边界附近产生伪影；在大规模场景下，处理点的深度网络面临 $O(N^2)$ 的注意力开销或 $O(N \log N)$ 的最近邻开销。

---

## 体素网格与稀疏体素

体素表示把 3D 空间离散化为规则的立方体单元网格，从而可以直接应用 3D 卷积。分辨率为 $V^3$ 的稠密体素网格需要 $O(V^3)$ 的内存：$64^3 = 262,144$ 个单元，$256^3 = 16.7M$ 个单元。对于大场景或高分辨率几何，这一开销令人望而却步。

### MinkowskiNet [Choy2019]

Choy、Gwak 和 Savarese 在 CVPR 2019 提出了 **4D 时空卷积网络（MinkowskiNet）**，通过 Minkowski Engine 实现了**稀疏卷积**。它不为全部 $V^3$ 个体素分配内存，而只把被占据的体素存储为以整数坐标索引的稀疏张量。广义稀疏卷积只在被占据的位置上应用标准卷积核，对真实世界的 3D 数据（通常占据率不到 1%）带来了显著的内存与计算节省。该库为稀疏操作的反向传播提供自动微分，对实现大规模室外 3D 语义分割（例如在 SemanticKITTI 上）至关重要。

**VoxNet、占用网格与 TSDF：** 更早的工作（Maturana & Scherer 2015，ICRA）在稠密的占用体素网格上做目标检测。**截断有符号距离函数（Truncated Signed Distance Function, TSDF）** 是一种体素网格，每个单元存储到最近曲面的截断距离。它是 RGB-D SLAM 中标准的稠密 3D 地图格式（例如 KinectFusion、InfiniTAM）。

---

## 多边形网格

三角网格 $\mathcal{M} = (\mathcal{V}, \mathcal{F})$ 由顶点集 $\mathcal{V} \subset \mathbb{R}^3$ 和面集 $\mathcal{F} \subset \mathbb{N}^3$ 组成，是计算机图形学、动画和物理仿真中的主导表示。网格紧凑，可在 GPU 上光栅化，并支持 UV 纹理。然而，由于拓扑约束，从原始传感器数据或神经优化中生成网格是出了名的困难：可微的网格提取通常需要预先知道拓扑，或事后使用 Marching Cubes。

**基于网格的可微渲染：** PyTorch3D（Ravi et al. 2020）和 SoftRas（Liu et al. 2019）引入了可微光栅化器，使梯度能够从渲染图像回传到顶点位置，从而实现基于网格的逆渲染。**NMR（Neural Mesh Renderer）** [Kato 2018] 是早期的近似方法。MeshGPT [Siddiqui2024, CVPR 2024] 从生成的角度重新审视网格：它训练一个仅解码器的 GPT 式 Transformer 自回归地生成三角形序列，直接产出紧凑的、美术品质的网格，而非密度场。

---

## 隐式表示：SDF 与占用网络

### DeepSDF [Park2019]

Park、Florence、Straub、Newcombe 和 Lovegrove（Facebook Reality Labs）在 CVPR 2019 提出了 **DeepSDF**。有符号距离函数 $f: \mathbb{R}^3 \to \mathbb{R}$ 把每个查询点 $x$ 映射到它到最近曲面的有符号距离：内部为负，外部为正，曲面上为零。曲面即零水平集 $\mathcal{S} = \{x : f(x) = 0\}$。

DeepSDF 通过**概率自解码器**（probabilistic auto-decoder）把 $f_\theta(x, z)$ 参数化为以潜码 $z \in \mathbb{R}^{256}$ 为条件的前馈 MLP：训练时联合优化潜码与解码器权重；测试时通过梯度下降推断 $z$，以最小化在观测曲面上的重建损失。这种自解码器设计（训练时无需编码器）比编码器-解码器架构具有更好的泛化能力。

```
# SDF 曲面条件
f(x) = 0   ← 曲面
f(x) < 0   ← 内部
f(x) > 0   ← 外部

# 有效 SDF 的 Eikonal 约束：
||∇_x f(x)|| = 1   （几乎处处成立）
```

### 占用网络 [Mescheder2019]

Mescheder、Oechsle、Niemeyer、Nowozin 和 Geiger（马克斯·普朗克研究所 / 图宾根大学）与 DeepSDF 同期在 CVPR 2019 发表了**占用网络（Occupancy Networks）**。它们不预测有符号距离，而是预测二值占用概率 $o_\theta(x, z) \in [0, 1]$，即在给定潜变量 $z$ 的情况下点 $x$ 是否位于形状内部。曲面是 $0.5$ 等值面。一个 5 块的 ResNet 联合处理 $(x, z)$。推理时通过多分辨率 MISE（带自适应细分的 Marching Cubes）提取曲面。

**与 DeepSDF 的关键区别：** 占用更容易监督（二值标签），并能自然地处理非水密曲面；但 DeepSDF 的有符号距离提供了更丰富的几何信息（梯度直接给出曲面法向）。

---

## 神经辐射场（NeRF）

NeRF [Mildenhall2020, ECCV 2020] 把场景表示重新定义为一个连续的体函数，将 5D 输入 $(x, y, z, \theta, \phi)$（3D 位置与 2D 视角方向）映射为体密度 $\sigma$ 和视角相关的颜色 $c = (r, g, b)$。新视角通过可微体渲染合成（完整的渲染方程见 [01_nerf_and_variants.md](./01_nerf_and_variants.md)）。NeRF 的关键性质是：(1) 隐式表示，无需显式几何；(2) 通过方向输入实现视角相关外观；(3) 仅凭带位姿的 RGB 图像、通过光度损失进行训练；(4) 连续性使其可以任意分辨率渲染。

其主要弱点是训练慢（每个场景数小时到数天）和渲染慢（每帧数秒），由此催生了 Instant-NGP → Zip-NeRF 的加速路线，并最终催生了用显式高斯基元完全取代隐式场的 3DGS。

---

## 3D 高斯泼溅（3DGS）

Kerbl、Kopanas、Leimkühler 和 Drettakis（INRIA）在 SIGGRAPH 2023 提出了 **3D 高斯泼溅**，把场景表示为一组各向异性的 3D 高斯。每个高斯由均值 $\mu \in \mathbb{R}^3$、协方差矩阵 $\Sigma \in \mathbb{R}^{3 \times 3}$（通过旋转 $R$ 和缩放 $S$ 参数化以保证半正定）、不透明度 $\alpha$ 以及用于视角相关颜色的球谐系数定义。渲染采用基于图块（tile）的可微光栅化（泼溅），而非体步进，在 1080p 下可达 100+ FPS。完整细节见 [02_gaussian_splatting.md](./02_gaussian_splatting.md)。

**作为交换格式的 3DGS（2026）。** 到 2026 年，高斯泼溅已从研究产物变为标准资产类型。Khronos Group 正式批准了 `KHR_gaussian_splatting` glTF 2.0 扩展，贡献者来自 Cesium、Niantic Spatial、Esri、NVIDIA、华为和 Autodesk。该扩展把每个泼溅点存储为一个 `POINTS` 图元，包含 `POSITION`、单位四元数 `ROTATION`、线性 `SCALE`、线性 `OPACITY`，以及最高 3 阶的 SH 系数。另一方面，光线追踪的高斯/椭球变体（3DGRT [MoenneLoccoz2024]、3DGUT [Wu2025, CVPR 2025 Oral]、EVER [Mai2024]）以及 Radiant Foam 的光线追踪 Delaunay/Voronoi 网格 [Govindarajan2025] 表明，同样的基元可以被*精确地*（以体渲染方式）渲染，而不是通过近似的泼溅。这缩小了 3DGS 与 NeRF 式体表示之间的概念差距。

---

## 点图与射线图：学习得到的逐像素几何

自 2024 年以来变得至关重要的一种表示是**点图（pointmap）**：一个与图像对齐的稠密张量 $X \in \mathbb{R}^{H \times W \times 3}$，为每个像素分配一个在选定参考系下表达的 3D 点。DUSt3R [Wang2024] 把点图作为回归目标引入。由于相机、深度和对应关系都可以从点图中读出，这一表示统一了经典管线分别处理的多个任务（见 [03_sfm_and_slam.md](./03_sfm_and_slam.md)）。

自 2025 年起，一系列大型前馈模型使点图（及其分解变体）成为多视图 3D 感知的默认输出：

- **VGGT** [Wang2025, CVPR 2025 最佳论文]（牛津 VGG + Meta AI）是一个约 1B 参数的 Transformer。对于一到数百个视图，它在单次前向传播中预测相机内参/外参、深度图、点图和 3D 点轨迹，通常耗时不到一秒。
- **π³** [Wang2025b, ICLR 2026] 去掉了固定的参考视图。其完全置换等变的架构预测仿射不变的相机位姿和尺度不变的*局部*点图。
- **MapAnything** [Keetha2026, 3DV 2026]（Meta Reality Labs + CMU）把几何分解为逐视图的**深度图、局部射线图（逐像素射线方向）、相机位姿以及一个统一的度量尺度因子**。它可以接受可选的内参、位姿或深度作为额外输入，用一个模型覆盖 12 种以上的重建任务。
- **Depth Anything 3** [Lin2026, ICLR 2026]（字节跳动 Seed）主张极简设计：使用普通的 DINO 式 ViT 骨干，搭配单一的**深度-射线**预测目标。

```
# 分解式“深度 + 射线”表示（MapAnything / DA3 风格）
For pixel (u,v) in view i:
  r_i(u,v) ∈ S^2       ← 相机坐标系下的单位射线方向（编码了内参）
  d_i(u,v) ∈ R_+       ← 沿射线的深度
  X_i^cam(u,v) = d_i(u,v) · r_i(u,v)           ← 局部点图
  X_i^world(u,v) = s · (R_i X_i^cam(u,v) + t_i) ← 全局度量点图
  (R_i, t_i) ∈ SE(3) 每个视图一个，s ∈ R_+ 为共享的度量尺度
```

与经典的稀疏 SfM 点相比，点图是稠密、可微且可由网络直接预测的。然而，点图依赖于视图：每个视图都携带一份冗余的曲面副本。在成为单一一致的场景表示之前，还需要进行融合（TSDF、高斯、网格化）。

---

## 内蕴曲面纹理：Heat Kernel Textures（ECCV 2026 最佳论文）

经典网格通过 **UV 图集**承载外观，即把曲面参数化到平面上。UV 展开会引入接缝、畸变、图集空间浪费、顶点重复以及不均匀的纹素密度。Simone Foti、Caner Korkmaz、Stefanos Zafeiriou 和 Tolga Birdal（帝国理工学院）提出的 **Heat Kernel Textures（HKTex）** [Foti2026] 以一种内蕴的替代方案获得了 **ECCV 2026 最佳论文奖**。外观被建模为**网格曲面上各向异性热核**之和，热核是高斯的测地类比。核的放置、优化、剪枝和致密化都在曲面本身上进行，思路与 3DGS 的自适应密度控制相同，但不向图像平面投影（“泼溅”）。副标题“the Geodesic Gaussians That Do Not Splat”（不泼溅的测地高斯）由此而来。

```
# 黎曼曲面 M 上的热核（各向同性情形）
∂u/∂t = Δ_M u ,   u(x, 0) = δ_y(x)
k_t(x, y) = Σ_i exp(-λ_i t) φ_i(x) φ_i(y)      ← （Laplace–Beltrami 特征对 λ_i, φ_i）

# 在平坦的 R^2 上：  k_t(x,y) = (4πt)^{-1} exp(-||x - y||^2 / 4t)   ← 即一个高斯
# HKTex 风格纹理的示意（仅作说明；确切参数化见论文）：
#   c(x) = Σ_j w_j · k^{aniso}_{t_j}(x, y_j) · a_j
#   y_j ∈ M 为可学习的核中心；各向异性/尺度 t_j 和外观 a_j 均可学习
```

据作者介绍，该方法无需 UV 展开，既可以从已有带纹理网格拟合纹理，也可以从多视图图像拟合；可接入可微渲染器和基于物理的渲染器（例如 Mitsuba），并可在单张消费级 GPU 上运行。代码以 MIT 许可证发布。已有后续预印本提出了无需计算网格特征系统即可加速 HKTex 的方法。HKTex 标志着一个更广泛的趋势：在 3DGS 中得到验证的思想（显式、自适应致密化、可微的基元）正被迁回到经典曲面表示上。

---

## 用于生成的结构化稀疏体素潜变量

3D 生成模型需要一种既足够规则以适配 Transformer、又足够稀疏以便扩展的表示。微软研究院 / 清华大学 / 中国科学技术大学的两项工作确立了当前的标准：

- TRELLIS [Xiang2025, CVPR 2025 Spotlight] 中的 **SLAT（Structured LATent，结构化潜变量）** 把从稠密多视图 DINOv2 特征聚合得到的潜在特征，附着到稀疏 $64^3$ 网格中的活跃体素上。同一个潜变量可以解码为辐射场、3D 高斯或网格。
- TRELLIS.2 [Xiang2026, CVPR 2026 最佳学生论文，“Native and Compact Structured Latents for 3D Generation”] 中的 **O-Voxel** 是一种“无场”（field-free）稀疏体素结构，联合编码几何与 **PBR 材质**属性。它能处理开放曲面、非流形几何以及封闭的内部结构，而这些是 SDF/占用场无法表示的。一个稀疏压缩 VAE 把它转换为紧凑的潜空间，供约 4B 参数的流匹配（flow matching）Transformer 在 $512^3$ 到 $1536^3$ 的分辨率下生成资产。

在上述分类中，这些潜变量属于混合表示。它们是显式的（稀疏体素），但每个体素携带一个学习得到的编码，解码器可以把它转换为下游所需的任何表示。见 [04_3d_generation.md](./04_3d_generation.md)。

---

## 表示权衡汇总

```
表示           | 内存       | 渲染速度        | 几何质量         | 可学习性     | 可编辑性
---------------------------------------------------------------------------
点云           | O(N)       | 中（泼溅）      | 中               | 高           | 高
体素（稠密）   | O(V^3)     | 快（光栅化）    | 中               | 高           | 中
体素（稀疏）   | O(N_occ)   | 快              | 中-高            | 高           | 中
网格           | O(V+F)     | 很快（GPU）     | 高               | 中           | 很高
DeepSDF/OccNet | O(θ)       | 慢（查询）      | 很高             | 高           | 低
NeRF           | O(θ)       | 很慢            | 高               | 高           | 低
3DGS           | O(N_gauss) | 很快            | 高               | 中           | 中
点图/射线图    | O(K·H·W)   | 快（点）        | 中-高            | 很高         | 低
HKTex（网格上）| O(V+F+N_k) | 快（光栅化）    | 高（网格）       | 中           | 高
稀疏体素       | O(N_occ·C) | 取决于解码器    | 高               | 很高         | 中
  潜变量（SLAT/O-Voxel）
```

*K = 视图数；N_k = 热核数；C = 每个活跃体素的潜在通道数。*

---

## 关键论文

| 论文 | 作者 | 年份 | 会议/期刊 | 主要贡献 |
|-------|---------|------|-------|-----------------|
| PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | Qi, Su, Mo, Guibas | 2017 | CVPR | 通过共享 MLP + 全局最大池化，首个直接作用于原始点云的置换不变深度网络 |
| PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space | Qi, Yi, Su, Guibas | 2017 | NeurIPS | 基于最远点采样和球查询分组的层次化特征学习 |
| 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks | Choy, Gwak, Savarese | 2019 | CVPR | 借助 Minkowski Engine 实现稀疏 3D/4D 卷积，支持大规模体素学习 |
| DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation | Park, Florence, Straub, Newcombe, Lovegrove | 2019 | CVPR | 用于类别条件连续 SDF 的自解码器；潜在形状插值 |
| Occupancy Networks: Learning 3D Reconstruction in Function Space | Mescheder, Oechsle, Niemeyer, Nowozin, Geiger | 2019 | CVPR | 隐式占用场；基于 MISE 的任意分辨率网格提取 |
| NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng | 2020 | ECCV | 5D 辐射场 + 可微体渲染，实现照片级新视角合成 |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl, Kopanas, Leimkühler, Drettakis | 2023 | SIGGRAPH | 各向异性 3D 高斯 + 图块光栅化；100+ FPS 的实时新视角合成 |
| MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers | Siddiqui, Alliegro, Artemov et al. | 2024 | CVPR | 在学习得到的网格词表上用 GPT 式 Transformer 自回归生成三角网格 |
| DUSt3R: Geometric 3D Vision Made Easy | Wang, Leroy, Cabon, Chidlovskii, Revaud | 2024 | CVPR | 以点图回归作为统一的 3D 表示，覆盖位姿、深度与匹配 |
| VGGT: Visual Geometry Grounded Transformer | Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny | 2025 | CVPR（最佳论文） | 约 1B 参数的前馈 Transformer，从 1 到数百个视图预测相机、深度、点图和轨迹 |
| Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS) | Xiang, Lv, Xu, Deng, Wang, Zhang, Chen, Tong, Yang | 2025 | CVPR（Spotlight） | SLAT 稀疏体素潜变量，可解码为高斯、辐射场或网格 |
| MapAnything: Universal Feed-Forward Metric 3D Reconstruction | Keetha et al. (Meta, CMU) | 2026 | 3DV | 分解为深度 + 射线图 + 位姿 + 度量尺度；一个模型覆盖 12 种以上任务 |
| Native and Compact Structured Latents for 3D Generation (TRELLIS.2) | Xiang et al. (Tsinghua, MSR, USTC, Microsoft AI) | 2026 | CVPR（最佳学生论文） | O-Voxel：带 PBR 属性的无场稀疏体素；支持任意拓扑 |
| Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat | Foti, Korkmaz, Zafeiriou, Birdal | 2026 | ECCV（最佳论文） | 基于曲面各向异性热核的无 UV 内蕴网格纹理 |

---

## 基准性能

| 模型 | 数据集 | 指标 | 得分 | 备注 |
|-------|---------|--------|-------|-------|
| PointNet++ (MSG) | ModelNet40 | 分类准确率 | 91.9% | 优于 PointNet（89.2%） |
| MinkowskiNet | SemanticKITTI | mIoU | 65.4% | 稀疏 3D 卷积，21 类 |
| DeepSDF | ShapeNet Cars | Chamfer-L1 | — | 插值效果优于 AtlasNet |
| OccNet | ShapeNet | IoU (3D) | 0.761 | 体素基线约为 0.66 |
| NeRF | NeRF-Synthetic | PSNR | 31.01 dB | 8 个场景，每个场景训练数小时 |
| Instant-NGP | NeRF-Synthetic | PSNR | 33.18 dB | 约 5 分钟训练；SIGGRAPH 2022 最佳论文 |
| 3DGS | Tanks & Temples | PSNR | 23.14 dB | 1080p 实时渲染 |
| Mip-Splatting | Mip-NeRF 360 | PSNR | 27.92 dB | CVPR 2024 最佳学生论文 |
| VGGT (1B) | Co3D | 相机位姿 AUC@30 | ≈90 | 前馈的点图/相机预测；据官方仓库 |
| TRELLIS.2 (4B) | — | 生成时间（H100） | ~3 s @512³, ~17 s @1024³, ~60 s @1536³ | O-Voxel 潜变量；据官方仓库 |

---

## 优缺点

| 方面 | 优点 | 缺点 |
|--------|------|------|
| 点云 | LiDAR/立体视觉的直接输出；置换不变学习（PointNet++）；易于流式处理 | 无连接关系；渲染需要泼溅或重建；密度不均 |
| 隐式神经表示（SDF/OccNet） | 无限分辨率；几何平滑；紧凑（MLP 权重）；解析可微 | 逐点查询慢；无法直接渲染；网格化步骤会丢失连续性 |
| 3D 高斯泼溅 | 实时渲染；优化快；PSNR 高；易于致密化/剪枝 | 默认无曲面法向；每个场景内存大；漂浮伪影；拓扑表达有限 |
| 点图 / 射线图 | 可由前馈网络直接预测；统一位姿、深度与匹配；稠密 | 逐视图冗余；需融合成一致场景；除非使用度量预测头，否则存在尺度/坐标系歧义 |
| 内蕴纹理（HKTex） | 无 UV 接缝或畸变；曲面上的自适应分辨率；可微 | 需要先有良好网格；核计算比纹素查找开销大；生态尚不成熟 |
| 稀疏体素潜变量（SLAT/O-Voxel） | 适合 Transformer 扩展；一个潜变量可解码为多种输出；能处理任意拓扑（O-Voxel） | 需要大规模 3D 数据集和学习得到的解码器；分辨率受体素网格限制 |

---

## 开放问题与研究空白

- **表示的统一：** 没有任何单一表示能同时在所有指标上占优。当前研究积极探索混合表示（例如锚定在网格曲面上的高斯基元），以兼顾渲染速度与几何精度。
- **扩展到城市尺度：** 现有方法（NeRF、3DGS）超出有界场景后扩展性很差。城市尺度的神经表示需要层次化或流式设计，这是一个尚未解决的工程与算法难题。
- **动态场景表示：** 把静态表示扩展到 4D（时空）而不引起内存指数增长仍是开放问题。当前的 4D-GS [Wu2023] 和 SC-GS [Huang2024] 适用于短的单目片段，但难以处理长时程、多物体的动态。
- **物理感知表示：** 能同时编码几何与物理属性（材质、可变形性）以用于仿真的表示尚处于起步阶段。神经 SDF 可以与弹性求解器耦合，但训练管线复杂。
- **跨场景泛化：** 逐场景拟合的方法（NeRF、3DGS）无法泛化，每个场景都需要重新优化。前馈模型正在缩小差距：物体层面有 LRM [Hong2023] 和 TripoSR [Tochilkin2024]，场景几何层面有 VGGT、π³、MapAnything 和 Depth Anything 3（2025–2026）。逐场景优化在最终的光度保真度上仍占优势。开放的问题是：单个网络应该输出哪种表示——点图、高斯还是潜变量？
- **连接曲面与基元：** HKTex（ECCV 2026 最佳论文）表明，3DGS 式的自适应基元可以存在于网格*之上*。几何（而不仅仅是外观）能否以同样方式表示，以及这类内蕴表示如何与变形和重新网格化相互作用，仍是开放问题。
- **标准化与流式传输：** 随着 `KHR_gaussian_splatting` 获批，泼溅数据的压缩（不止于截断 SH）、细节层次（LOD）和流式传输规范是下一步的标准化空白。
- **语义化与可编辑表示：** 在不牺牲渲染质量的前提下，为神经表示加入语义场（LangSplat [Qin2023]）或便于编辑的结构（Scaffold-GS [Lu2023]）是一个活跃领域。
- **拓扑变化与非刚性：** 隐式表示和高斯都难以处理大的拓扑变化（例如物体被撕裂）。能处理拓扑变化的神经可变形表示是一个开放问题。

---

## 延伸阅读

- [NeRF 项目主页（Mildenhall et al.）](https://www.matthewtancik.com/nerf) — 原始 NeRF，含视频和预训练模型
- [3D Gaussian Splatting 官方仓库（Kerbl et al.）](https://github.com/graphdeco-inria/gaussian-splatting) — 参考实现
- [Neural Fields in Visual Computing（Xie et al. 2022）](https://arxiv.org/abs/2111.11426) — 隐式神经表示的全面综述
- [PointNet 项目主页（Qi et al.）](https://stanford.edu/~rqi/pointnet/) — 斯坦福项目主页，含代码和数据
- [Occupancy Networks 博客（Autonomous Vision Group）](https://autonomousvision.github.io/occupancy-networks/) — 通俗易懂的概述
- [Open3D 文档](http://www.open3d.org/docs/release/) — 点云与网格处理的标准库
- [VGGT GitHub（Meta / 牛津 VGG）](https://github.com/facebookresearch/vggt) — CVPR 2025 最佳论文；代码、检查点、COLMAP 导出
- [MapAnything GitHub（Meta）](https://github.com/facebookresearch/map-anything) — 分解式度量重建；封装了 DUSt3R、MASt3R、VGGT、π³、DA3
- [HKTex GitHub（帝国理工学院）](https://github.com/circle-group/hktex) — Heat Kernel Textures，ECCV 2026 最佳论文（MIT 许可）
- [TRELLIS.2 GitHub（微软）](https://github.com/microsoft/TRELLIS.2) — O-Voxel 结构化潜变量，CVPR 2026 最佳学生论文
- [KHR_gaussian_splatting 规范（Khronos）](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) — 已获批的高斯泼溅 glTF 扩展
