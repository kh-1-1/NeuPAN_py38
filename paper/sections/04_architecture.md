# 4. PDPL-Net 网络架构

本章详细阐述PDPL-Net的架构设计。PDPL-Net的设计不仅仅是神经网络层级的堆叠，而是对凸优化理论中**Karush-Kuhn-Tucker (KKT) 最优性条件**的深度参数化建模。我们提出了一种“三角支撑”的架构设计理念：**深度展开PDHG**负责逼近最优性（Optimality），**残差学习**负责提升收敛效率（Efficiency），而**硬投影层**则负责严格保障对偶可行性（Feasibility）。

## 4.1 总体架构：寻找KKT鞍点

PDPL-Net的数学本质是一个求解对偶优化问题的参数化求解器。给定障碍物点 $\mathbf{p}_i$ 和机器人几何 $\mathbf{G}, \mathbf{g}$，我们的目标是找到一组对偶变量 $(\mu^*, \lambda^*)$，使其满足以下KKT条件：
1.  **平稳性 (Stationarity)**：梯度为零。
2.  **原始/对偶可行性 (Primal/Dual Feasibility)**：满足 $\mu \ge 0, \|\mathbf{G}^\top \mu\|_2 \le 1$ 等约束。
3.  **互补松弛性 (Complementary Slackness)**：$\mu$ 与约束边界的对齐关系。

如图1所示，PDPL-Net通过三个耦合模块来分别攻克这些条件：
1.  **几何特征初始化**：提供高质量初值，减少迭代需求。
2.  **带残差的PDHG展开**：通过交替更新逼近平稳性和互补松弛性。
3.  **硬投影层**：强制满足对偶可行性约束。

[图1: PDPL-Net系统总体架构图。展示了从几何特征编码、PDHG展开层迭代到硬投影输出的完整数据流。]

## 4.2 几何特征编码与初始化

传统优化算法通常从零向量开始迭代，这忽略了问题实例间的几何相似性。为了实现“热启动”，我们设计了一个几何感知编码器：

$$
\mathbf{z}_0 = \text{MLP}_{enc}([\mathbf{p}_i, \text{vec}(\mathbf{G}), \mathbf{g}])
$$

该模块将障碍物点 $\mathbf{p}_i$ 与机器人多边形参数 $\mathbf{G}, \mathbf{g}$ 拼接作为输入，通过两层全连接层提取高维几何特征，并直接回归出初始对偶变量 $\mu^{(0)}$ 和 $\mathbf{y}^{(0)}$。这相当于赋予了优化器一种“经验直觉”，使其起始点能够跳过初始的探索阶段，直接落在最优解的吸引域附近。

## 4.3 深度展开PDHG与残差学习：逼近最优性

PDHG算法的核心在于通过交替更新原始变量 $\mu$ 和对偶变量 $\mathbf{y}$ 来寻找拉格朗日函数的鞍点。在PDPL-Net中，我们将这一过程展开，并引入**残差学习机制**来加速对KKT平稳性条件的逼近。

第 $k$ 次迭代的更新公式为：

$$
\begin{cases}
\mu^{(k)} = \mathcal{P}_{\mu}(\mu^{(k-1)} - \tau_k \mathbf{K}^\top \bar{\mathbf{y}}^{(k-1)} + \mathcal{R}_{\mu}(\mu^{(k-1)})) \\
\mathbf{y}^{(k)} = \mathcal{P}_{\mathbf{y}}(\mathbf{y}^{(k-1)} + \sigma_k \mathbf{K} \mu^{(k)} + \mathcal{R}_{\mathbf{y}}(\mathbf{y}^{(k-1)})) \\
\bar{\mathbf{y}}^{(k)} = \mathbf{y}^{(k)} + \theta_k (\mathbf{y}^{(k)} - \mathbf{y}^{(k-1)})
\end{cases}
$$

[图2: PDHG展开层内部结构示意图。详细展示了残差模块 $\mathcal{R}$ 如何作为非线性修正项加入到传统的线性更新步骤中。]

### 4.3.1 可学习残差算子 ($\mathcal{R}_{\mu}, \mathcal{R}_{\mathbf{y}}$)

残差模块 $\mathcal{R}(\cdot)$ 由ResNet块构成，其物理意义在于**修正梯度流**。在优化曲面复杂的区域，标准梯度方向往往不是指向KKT点的最短路径。残差模块通过学习流形的局部曲率，预测出一个“捷径（Shortcut）”，从而在极少的步数（$J=1\sim3$）内将变量推向满足**互补松弛性**的状态。

具体参数配置如表1所示。

**表1：PDPL-Net网络架构参数详情**
| 模块 (Module) | 组件 (Component) | 维度/配置 (Configuration) | 参数量 (Parameters) | 说明 (Description) |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Encoder** | MLP Layer 1 | Input: $2+N_{geo}$, Hidden: 32 | $\approx 200$ | 提取相对几何特征 |
| | MLP Layer 2 | Hidden: 32, Output: 32 | $\approx 1000$ | ReLU激活 |
| | Init Head ($\mu$) | Input: 32, Output: $N_{\mu}$ | $\approx 150$ | 预测初始对偶变量 |
| | Init Head ($y$) | Input: 32, Output: $N_{y}$ | $\approx 150$ | 预测初始辅助变量 |
| **PDHG Unrolling** | Primal Update ($\mu$) | ResNet Block (Hidden: 16) | $\approx 100 \times J$ | 可学习近端算子 $\mathcal{N}_{\mu}$ |
| ($J$ Layers) | Dual Update ($y$) | ResNet Block (Hidden: 16) | $\approx 100 \times J$ | 可学习近端算子 $\mathcal{N}_{y}$ |
| | Step Sizes | $\tau_j, \sigma_j, \theta_j$ | $3 \times J$ | 每层独立的可学习步长 |
| **Projection** | Hard Projection | Non-parametric | 0 | 强制对偶可行性 |
| **Total** | | **$J=2$** | **$\approx 1600$** | **极其轻量级** |

## 4.4 硬投影层：严格保障对偶可行性

这是PDPL-Net区别于其他Deep Unrolling方法的关键组件。在KKT条件中，**对偶可行性**（Dual Feasibility）是保证距离计算有效性的红线。任何违反 $\mu \geq 0$ 或 $\| \mathbf{G}^\top \mu \|_2 \leq 1$ 的输出都会导致安全距离被高估，从而引发碰撞风险。

为此，我们设计了一个无参的可微层，显式地执行几何投影：

1.  **非负锥投影 (Project to Non-negative Cone)**：
    $$ \hat{\mu} = \text{ReLU}(\mu^{(J)}) = \max(0, \mu^{(J)}) $$
    这一步确保了 $\mu$ 的每个分量非负，对应于不等式约束的拉格朗日乘子性质。

2.  **二阶锥投影 (Project to Second-order Cone)**：
    $$ \mu^* = \frac{\hat{\mu}}{\max(1, \| \mathbf{G}^\top \hat{\mu} \|_2)} $$
    这一步将向量拉回到单位球 $\mathcal{B}_2$ 内。

[图3: 硬投影层几何意义示意图。展示了如何将网络输出的不可行解（灰色点）通过两步投影操作拉回到单位对偶球（Unit Dual Ball）的可行域内（绿色区域）。]

**理论分析**：
该硬投影层具有两个重要性质：
1.  **严格可行性**：无论前序网络的输出如何发散，经过该层后的 $\mu^*$ 恒定满足 $\mu^* \in \mathcal{C}_{dual}$。这意味着网络输出的距离估计 $d(\mathbf{x}) = -\mathbf{g}^\top \mu^* - \mathbf{p}^\top \lambda^*$ **始终是一个合法的安全下界**。
2.  **梯度保真性**：投影操作是分段线性和平滑的（除边界外），允许梯度在反向传播时无损地流回前面的展开层，指导网络学习如何生成“由于已经在可行域内而不需要被投影截断”的优质解。

## 4.5 高效推理：行优先并行计算

（本节保持不变，强调并行计算带来的速度优势。）
为了应对大规模点云（每帧数千点）的实时处理需求，PDPL-Net在实现上采用了**行优先（Row-Major）**的并行计算策略。所有线性变换层（MLP和ResNet）均使用 $1 \times 1$ 卷积或批处理矩阵乘法实现，充分利用GPU并行加速，实现微秒级推理。
