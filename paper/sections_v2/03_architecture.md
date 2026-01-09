# III. PDPL-Net架构设计

本节提出PDPL-Net（Primal-Dual Proximal Learning Network），一种用于快速求解对偶距离优化问题的深度展开网络。在系统层面，我们沿用NeuPAN的整体MPC/NRMP规划-控制框架，仅将其深度展开神经编码器（DUNE，用于预测每个障碍点对应的对偶变量/LDF）替换为PDPL-Net，从而保持后端求解流程与接口不变。给定稠密障碍物点云$\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^N$，PDPL-Net通过并行求解$N$个对偶优化问题，将每个点$\mathbf{p}_i$映射到其对偶变量$\mu_i \in \mathbb{R}^E$（并由$\lambda_i = -\mathbf{G}^\top \mu_i$得到辅助变量），其中$E$为机器人多边形的边数。这些对偶变量可直接用于MPC控制器的碰撞回避约束，为安全导航提供实时保障。与NeuPAN中基于PIBCD的展开不同，本文选择将原始-对偶混合梯度（PDHG）算法展开为可解释神经网络，并通过末端硬投影层在结构上保证对偶可行性。

---

## A. 对偶问题与计算复杂度

### 1) 对偶优化问题的数学形式

回顾第II节的对偶重构，对于给定的障碍物点$\mathbf{p} \in \mathbb{R}^2$和机器人几何$(\mathbf{G}, \mathbf{g})$，单点对偶问题可写为：

$$
\begin{aligned}
\mathcal{D}(\mathbf{p}): \quad \max_{\mu, \lambda} \quad & -\mathbf{g}^\top \mu - \mathbf{p}^\top \lambda \\
\text{s.t.} \quad & \mathbf{G}^\top \mu + \lambda = \mathbf{0} \\
& \mu \geq \mathbf{0}, \quad \|\lambda\|_2 \leq 1
\end{aligned}
$$

其中$\mathbf{G} \in \mathbb{R}^{E \times 2}$为机器人几何矩阵，$\mathbf{g} \in \mathbb{R}^E$为几何偏移向量。通过消去辅助变量$\lambda = -\mathbf{G}^\top \mu$，上述问题可等价地写成仅含$\mu$的形式：

$$
\begin{aligned}
\mathcal{D}(\mathbf{p}): \quad \max_{\mu} \quad & (\mathbf{p}^\top \mathbf{G}^\top - \mathbf{g}^\top)\mu \\
\text{s.t.} \quad & \mu \geq \mathbf{0}, \quad \|\mathbf{G}^\top \mu\|_2 \leq 1
\end{aligned}
$$

定义对偶可行域$\mathcal{C}_{dual} = \{\mu \in \mathbb{R}^E : \mu \geq \mathbf{0}, \|\mathbf{G}^\top \mu\|_2 \leq 1\}$，则上述问题是一个定义在闭凸集上的线性最大化问题。

### 2) 传统求解方法的计算瓶颈

对于包含$N$个障碍物点的点云，需要在每个控制周期并行求解$N$个上述对偶问题。传统优化方法的计算复杂度为：

- **内点法**：每个问题需要$O(E^3)$的线性系统求解，总复杂度为$O(NE^3)$
- **标准PDHG**：每个问题需要$K$次迭代（$K \in [500, 1000]$），总复杂度为$O(NEK)$

在典型场景中，$N \approx 1000$（稠密点云），$E = 4$（四边形机器人），即使采用最快的PDHG算法，单次求解也需要数十至数百毫秒，无法满足实时控制（通常要求$<10$ms）的需求。

为解决这一计算瓶颈，我们提出将PDHG算法展开为深度神经网络，通过数据驱动的方式学习问题结构的先验知识，将迭代次数从数百降至个位数，从而实现数量级的加速。

---

## B. 原始-对偶混合梯度算法回顾

### 1) 鞍点形式与PDHG迭代

为将优化算法展开为神经网络，我们首先将问题(2)重写为鞍点形式。引入辅助变量$y = \lambda = -\mathbf{G}^\top \mu$，则问题可表示为：

$$
\min_{\mu \in \mathcal{C}_\mu} \max_{y \in \mathcal{C}_y} \mathcal{L}(\mu, y) = \langle \mathbf{K}\mu, y \rangle - f(\mu)
$$

其中线性算子$\mathbf{K} = \mathbf{G}^\top \in \mathbb{R}^{2 \times E}$，原始可行域$\mathcal{C}_\mu = \mathbb{R}_+^E$（非负锥），对偶可行域$\mathcal{C}_y = \mathcal{B}_2$（单位球），$f(\mu) = \mathbf{g}^\top \mu - \mathbf{p}^\top \mathbf{G}^\top \mu$为线性目标函数。

原始-对偶混合梯度（PDHG）算法通过交替更新原始变量和对偶变量求解上述鞍点问题：

$$
\begin{cases}
y^{(k+1)} = \Pi_{\mathcal{B}_2}(y^{(k)} + \sigma \mathbf{G}^\top \mu^{(k)}) \\
\mu^{(k+1)} = \Pi_{\mathbb{R}_+^E}(\mu^{(k)} + \tau (\mathbf{G}\mathbf{p} - \mathbf{g} - \mathbf{G} y^{(k+1)}))
\end{cases}
$$

其中$\tau, \sigma > 0$为步长参数，$\Pi_{\mathcal{C}}(\cdot)$表示到集合$\mathcal{C}$的欧氏投影。

**收敛性保证**（Chambolle & Pock, 2011）。设$\|\mathbf{G}\|_{op}$为几何矩阵的算子范数。若步长参数满足$\tau \sigma \|\mathbf{G}\|_{op}^2 < 1$，则对于任意初始点$(\mu^{(0)}, y^{(0)})$，PDHG迭代产生的序列收敛到鞍点问题的解。

### 2) 从迭代到层的映射

PDHG算法产生的迭代序列可视为一系列映射的复合：

$$
(\mu^{(0)}, y^{(0)}) \xrightarrow{g_1} (\mu^{(1)}, y^{(1)}) \xrightarrow{g_2} \cdots \xrightarrow{g_J} (\mu^{(J)}, y^{(J)})
$$

其中每个映射$g_j$由PDHG的一次迭代组成。关键观察是：这些映射仅涉及矩阵乘法和投影操作，两者均可表示为神经网络的基本运算。因此，我们可以将$J$次迭代展开为$J$层神经网络，每层对应一次PDHG迭代。

深度展开的核心优势在于：
1. **结构先验继承**：网络架构继承优化算法的收敛性保证
2. **参数可学习**：步长、权重等参数可通过训练自适应调整
3. **早停加速**：通过学习问题结构先验，仅需少量层数即可收敛

---

## C. PDPL-Net网络结构

基于上述思想，我们提出PDPL-Net架构，如图3所示。网络由三个级联模块组成：特征编码器、PDHG展开层和硬投影层。

### 1) 特征编码器（Feature Encoder）

传统PDHG从零向量或随机向量开始迭代，忽略了不同问题实例之间的结构相似性。我们设计一个轻量级MLP编码器实现"热启动"：

$$
\mathbf{h} = \text{ReLU}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{p} + \mathbf{b}_1) + \mathbf{b}_2)
$$

其中$\mathbf{p} \in \mathbb{R}^2$为输入点坐标，$\mathbf{h} \in \mathbb{R}^{32}$为隐藏特征。编码器通过两个独立的线性头初始化$(\mu^{(0)}, y^{(0)})$：

$$
\begin{aligned}
\mu^{(0)} &= \text{ReLU}(\mathbf{W}_\mu \mathbf{h} + \mathbf{b}_\mu) \in \mathbb{R}^E \\
y^{(0)} &= \text{Tanh}(\mathbf{W}_y \mathbf{h} + \mathbf{b}_y) \in \mathbb{R}^2
\end{aligned}
$$

$\mu$头使用ReLU激活保证初始值非负，$y$头使用Tanh激活保证初始值有界（在$[-1, 1]$内）。

### 2) PDHG展开层（Unrolling Layers）

第$j$层（$j = 1, \ldots, J$）对应PDHG的第$j$次迭代，但引入了两项关键改进：

$$
\begin{cases}
y^{(j+1)} = \Pi_{\mathcal{B}_2}\left( y^{(j)} + \sigma_j \mathbf{G}^\top \mu^{(j)} \right) \\
\tilde{\mu}^{(j+1)} = \mu^{(j)} + \tau_j \left( \mathbf{G}\mathbf{p} - \mathbf{g} - \mathbf{G} y^{(j+1)} \right) \\
\mu^{(j+1)} = \mathcal{P}_{dual}\left( \tilde{\mu}^{(j+1)} + \alpha \cdot \mathcal{R}_\theta(\mathbf{G}^\top \tilde{\mu}^{(j+1)}) \right)
\end{cases}
$$

**改进1：可学习步长**。每层使用独立的步长参数$\tau_j$和$\sigma_j$，在训练中学习自适应调整更新幅度。

**改进2：可学习近端算子**。残差模块$\mathcal{R}_\theta: \mathbb{R}^2 \to \mathbb{R}^E$是一个轻量级MLP：

$$
\mathcal{R}_\theta(z) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 z + \mathbf{b}_1) + \mathbf{b}_2
$$

其中$z = \mathbf{G}^\top \mu \in \mathbb{R}^2$，隐藏层维度为32，输出维度为$E$。$\alpha \in (0, 1)$为残差缩放因子（默认0.5）。

残差模块的作用是学习从当前迭代点到最优解的"修正方向"，将保守的一阶梯度转变为准二阶更新方向。若$\mathcal{R}_\theta$能够预测$\mathcal{R}_\theta(\mathbf{G}^\top \mu) \approx (\mu^* - \mu)/\alpha$，则单次迭代即可收敛到最优解，将复杂度从$O(1/\epsilon)$压缩至$O(1)$。

### 3) 硬投影层（Hard Projection Layer）

为确保100%的约束满足率，我们在网络末端引入硬投影层：

$$
\mathcal{P}_{dual}(\mu) = \frac{\max(\mathbf{0}, \mu)}{\max(1, \|\mathbf{G}^\top \max(\mathbf{0}, \mu)\|_2)}
$$

该操作分为两步：
1. **非负锥投影**：$\hat{\mu} = \max(\mathbf{0}, \mu) = \text{ReLU}(\mu)$
2. **二阶锥缩放**：$\mu^* = \hat{\mu} / \max(1, \|\mathbf{G}^\top \hat{\mu}\|_2)$

经过硬投影后，输出$\mu^*$满足$\mu^* \geq \mathbf{0}$且$\|\mathbf{G}^\top \mu^*\|_2 \leq 1$，即$\mu^* \in \mathcal{C}_{dual}$。这一设计从根本上消除了约束违背的可能性，为安全攸关的机器人控制提供了确定性保障。

### 4) 计算复杂度分析

PDPL-Net的计算复杂度分析如下：

- **特征编码器**：$O(2 \times 32 + 32 \times 32 + 32 \times E) = O(1)$（$E$为常数）
- **每个展开层**：$O(E \times 2 + 2 \times E + 2 \times 32 + 32 \times E) = O(E)$
- **总复杂度**：$O(J \cdot E)$

由于$J \in [1, 3]$且$E = 4$（四边形机器人），PDPL-Net的总复杂度为$O(1)$。与标准PDHG的$O(EK)$相比（$K \approx 1000$），实现了约$K/J \approx 500 \sim 1000$倍的加速。

表I汇总了PDPL-Net的网络结构和参数量。整个网络极其轻量级，总参数量约为1600（$J=2$时），远小于PointNet++（数百万参数）和Point Transformer（数百万至数千万参数）。

**表I：PDPL-Net网络架构参数**

| 模块 | 组件 | 输入维度 | 隐藏/输出维度 | 激活函数 | 参数量 |
|:-----|:-----|:--------|:-------------|:---------|-------:|
| **Feature Encoder** | FC Layer 1 | 2 | 32 | ReLU | ~100 |
| | FC Layer 2 | 32 | 32 | ReLU | ~1,100 |
| | Init Head ($\mu$) | 32 | $E$ | ReLU | ~150 |
| | Init Head ($y$) | 32 | 2 | Tanh | ~70 |
| **PDHG Unrolling** | Residual Module | 2 | 32 → $E$ | ReLU | ~200$\times J$ |
| ($J$ layers) | Step Sizes | - | - | - | 2$\times J$ |
| **Hard Projection** | ReLU + Norm | - | - | - | 0 |
| **Total ($J=2, E=4$)** | | | | | **~1,600** |

---

## D. 损失函数与训练策略

### 1) 监督损失函数

网络采用有监督学习策略，使用精确求解器（如CVXPY ECOS）生成的标签$\mu^*$训练。基础的监督损失为均方误差：

$$
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N \|\mu_i - \mu_i^*\|_2^2
$$

然而，仅优化MSE未考虑对偶变量的物理意义。因此我们设计多目标损失函数：

$$
\begin{aligned}
\mathcal{L}_{sup} &= \mathcal{L}_{MSE} + \mathcal{L}_{obj} \\
\mathcal{L}_{obj} &= \frac{1}{N} \sum_{i=1}^N \left( f(\mu_i) - f(\mu_i^*) \right)^2
\end{aligned}
$$

其中$f(\mu) = (\mathbf{p}_i^\top \mathbf{G}^\top - \mathbf{g}^\top)\mu_i$为对偶目标函数。通过同时优化$\mu$的数值和目标函数值，确保网络输出在物理意义上正确。

### 2) KKT正则化损失

为了进一步提升网络输出的最优性和约束满足，我们引入基于KKT条件的正则化损失。对偶问题(2)的KKT条件为：

$$
\begin{cases}
\text{原始可行性}： & \mathbf{G}^\top \mu + \lambda = \mathbf{0} \\
\text{对偶可行性}： & \mu \geq \mathbf{0}, \|\lambda\|_2 \leq 1 \\
\text{互补松弛性}： & \mu_i \cdot (-\|\mathbf{G}^\top \mu\|_2 + 1) = 0, \forall i
\end{cases}
$$

基于上述条件，定义KKT残差：

$$
\begin{aligned}
\mathbf{r}_{prim} &= \mathbf{G}^\top \mu + \lambda \\
r_{dual\_1} &= \|\max(-\mu, 0)\|_2^2 \\
r_{dual\_2} &= \max(0, \|\lambda\|_2 - 1)^2 \\
\mathbf{r}_{comp} &= \mu \odot \max(0, \|\lambda\|_2 - 1)
\end{aligned}
$$

KKT正则化损失为：

$$
\mathcal{L}_{KKT} = \|\mathbf{r}_{prim}\|_2^2 + r_{dual\_1} + r_{dual\_2} + \|\mathbf{r}_{comp}\|_2^2
$$

该损失驱动网络输出满足KKT最优性条件，使其不仅在数值上接近最优解，更在数学结构上满足最优性的必要条件。

综合监督损失和KKT正则化，总损失函数为：

$$
\mathcal{L}_{total} = \mathcal{L}_{sup} + \beta \cdot \mathcal{L}_{KKT}
$$

其中$\beta > 0$为正则化权重。实验中我们发现$\beta = 0.1 \sim 0.5$能够在监督学习和物理约束之间取得良好平衡。

### 3) 数据生成与训练策略

对于给定的机器人几何$(\mathbf{G}, \mathbf{g})$，训练数据通过以下方式生成：

1. 在机器人局部坐标系的采样范围$[r_l, r_h] \times [-\pi, \pi]$内随机生成$N_g$个障碍物点$\{\mathbf{p}_i\}_{i=1}^{N_g}$
2. 使用精确求解器（CVXPY ECOS）求解每个点对应的最优对偶变量$\{\mu_i^*\}_{i=1}^{N_g}$
3. 构建训练集$\mathcal{T} = \{(\mathbf{p}_i, \mu_i^*)\}_{i=1}^{N_g}$

表II列出了训练的超参数设置。

**表II：PDPL-Net训练超参数**

| 超参数 | 符号 | 默认值 | 说明 |
|:-----|:-----|:------|:-----|
| 采样点数 | $N_g$ | 50,000 | 训练集大小 |
| 采样范围 | $[r_l, r_h]$ | $[0.1, 5.0]$ | 米 |
| 训练轮数 | epochs | 100 | Adam优化器 |
| 批大小 | batch size | 512 | |
| 初始学习率 | $lr_0$ | 1e-3 | 余弦退火 |
| 学习率衰减 | $dr$ | 0.1 | 每30轮 |
| 残差缩放 | $\alpha$ | 0.5 | |
| 正则化权重 | $\beta$ | 0.2 | KKT损失 |
| 展开层数 | $J$ | 2 | 推理时 |

训练采用两阶段课程学习策略：

**阶段1（前70%轮数）**：仅使用监督损失$\mathcal{L}_{sup}$，使网络快速学习从输入到输出的映射。

**阶段2（后30%轮数）**：加入KKT正则化$\mathcal{L}_{KKT}$，微调网络使其满足最优性条件。

这种渐进式训练策略有效平衡了监督学习与物理约束的训练目标，避免了早期训练时KKT损失主导导致的收敛困难。
