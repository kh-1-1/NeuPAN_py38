# 5. 训练策略

为了训练出既具备高性能又严格遵守物理约束的PDPL-Net，本文提出了一套融合了全合成数据生成与物理信息引导的训练框架。本章将详细介绍数据集的构建流程，并深入阐述核心的KKT残差正则化损失函数，这是连接深度学习与凸优化理论的关键桥梁。

## 5.1 全合成数据集生成

由于PDPL-Net是在点级别（Point-level）进行对偶变量预测，我们需要构建一个覆盖机器人周围工作空间的高质量数据集。与依赖人工标注的传统数据集不同，我们采用“求解器在环”的全合成方式生成数据，这不仅保证了标签的数学精确性（Ground Truth），还实现了对数据分布的完全控制。

数据集 $\mathcal{D} = \{(\mathbf{p}_i, \mathbf{x}_i, \mu^*_i, \mathbf{y}^*_i)\}_{i=1}^M$ 包含 $M=150,000$ 个样本。生成过程遵循以下原则：

1.  **重要性采样 (Importance Sampling)**：在机器人周围 $10m \times 10m$ 的局部区域内进行采样。考虑到避障约束在接近边界时最为活跃（Active Constraints），我们在机器人轮廓附近的 $0.5m$ 范围内采用了更高密度的采样策略。
2.  **精确标签生成**：对于每一个采样点 $\mathbf{p}_i$，我们调用工业级二阶锥规划（SOCP）求解器（ECOS）求解第3章定义的对偶优化问题。求解器返回的最优对偶变量 $\mu^*_i$ 和 $\mathbf{y}^*_i$ 被记录为训练标签。
3.  **标准化预处理**：为了加速训练收敛，我们对输入坐标 $\mathbf{p}_i$ 和机器人几何参数进行了标准化处理，使其分布在单位超球附近。

## 5.2 物理信息驱动的损失函数

传统的深度学习任务通常仅使用均方误差（MSE）来拟合标签。然而，对于受约束的优化问题，仅拟合输出值是不够的。我们需要训练网络“理解”优化问题的内在结构，即Karush-Kuhn-Tucker（KKT）条件。

为此，我们构建了一个复合损失函数，将数据驱动的监督信号与物理驱动的无监督信号相结合：

$$
\mathcal{L} = \mathcal{L}_{MSE} + w_{KKT} \mathcal{L}_{KKT}
$$

### 5.2.1 数据拟合项 ($\mathcal{L}_{MSE}$)

该项负责引导网络快速逼近求解器的解，为优化提供良好的初值：

$$
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N \left( \|\mu_i - \mu^*_i\|_2^2 + \|\mathbf{y}_i - \mathbf{y}^*_i\|_2^2 \right)
$$

### 5.2.2 KKT物理正则化项 ($\mathcal{L}_{KKT}$)

这是本文训练策略的核心。KKT条件是凸优化问题最优解的充分必要条件。通过最小化KKT残差，我们实际上是在训练一个“物理信息神经网络 (PINN)”。具体包括三部分：

1.  **原始可行性残差 (Primal Feasibility)**：
    对应于对偶问题的线性等式约束 $\mathbf{G}^\top \mu + \lambda = 0$：
    $$ r_{pri} = \|\mathbf{G}^\top \mu + \lambda\|_2 $$

2.  **对偶可行性残差 (Dual Feasibility)**：
    对应于不等式约束 $\mu \geq 0$ 和 $\|\lambda\|_2 \leq 1$。尽管硬投影层在推理阶段保证了这一点，但在训练早期引入该损失有助于网络权重更快收敛到可行域流形附近：
    $$ r_{dual} = \|\min(0, \mu)\|_2 + \max(0, \|\lambda\|_2 - 1) $$

3.  **互补松弛性残差 (Complementary Slackness)**：
    这是衡量解的“最优性”的关键指标。它要求对偶变量 $\mu$ 仅在原始约束被激活（即障碍物接触边界）时才非零：
    $$ r_{comp} = |\mu^\top (\mathbf{G}\mathbf{x} - \mathbf{g})| $$

最终的KKT损失定义为：
$$ \mathcal{L}_{KKT} = r_{pri} + r_{dual} + r_{comp} $$

## 5.3 两阶段课程学习 (Curriculum Learning)

由于引入物理约束后的损失曲面变得复杂且非凸，直接进行联合训练容易陷入局部极小。为此，我们设计了两阶段的训练课程：

[图4: 两阶段训练课程示意图。Phase 1仅使用监督损失进行快速预训练；Phase 2引入KKT物理损失进行微调，提升解的质量。]

*   **阶段一：模仿学习 (Imitation Phase)**
    *   **目标**：快速建立从输入到输出的粗略映射。
    *   **策略**：仅使用 $\mathcal{L}_{MSE}$，学习率设为 $5 \times 10^{-5}$。
    *   **结果**：网络能够学会基本的避障直觉，但在边界处不够精确。

*   **阶段二：物理微调 (Physics-Informed Fine-tuning)**
    *   **目标**：利用KKT条件精细修正网络参数，提升解的质量。
    *   **策略**：引入 $\mathcal{L}_{KKT}$ ($w_{KKT} = 10^{-3}$)，降低学习率为 $5 \times 10^{-6}$。同时，冻结特征编码器层，仅微调PDHG展开层和近端算子。
    *   **结果**：实验表明，经过此阶段后，网络输出的KKT残差显著降低，且在未见过的测试场景（Out-of-Distribution）中表现出更强的泛化能力。

训练算法的伪代码如下：

```algorithm
Algorithm 1: PDPL-Net Two-Stage Training Strategy
Input: Dataset D, MaxEpochs E, Weights w_KKT
Output: Trained Parameters \theta

# Phase 1: Imitation Learning
Initialize \theta randomly
for epoch = 1 to E/2 do
    Sample batch (p, G, g, \mu*, y*) from D
    (\hat{\mu}, \hat{y}) = PDPL_Net(p, G, g; \theta)
    Loss = MSE(\hat{\mu}, \mu*) + MSE(\hat{y}, y*)
    Update \theta using Adam
end for

# Phase 2: Physics-Informed Fine-tuning
Freeze encoder parameters \theta_enc
Reduce learning rate (lr = lr * 0.1)
for epoch = E/2 + 1 to E do
    Sample batch (p, G, g, \mu*, y*) from D
    (\hat{\mu}, \hat{y}) = PDPL_Net(p, G, g; \theta)
    L_mse = MSE(\hat{\mu}, \mu*) + MSE(\hat{y}, y*)
    L_kkt = calc_kkt_residual(\hat{\mu}, \hat{y}, G, g)
    Loss = L_mse + w_KKT * L_kkt
    Update \theta_pdhg using Adam
end for
```

详细的训练超参数列于表2中。

**表2：训练超参数设置**
| 阶段 (Stage) | 参数 (Parameter) | 值 (Value) | 说明 (Description) |
| :--- | :--- | :--- | :--- |
| **Global** | Batch Size | 256 | 批次大小 |
| | Optimizer | Adam | 优化器 |
| | Data Range | $[-10, 10] \times [-10, 10]$ | 采样范围 (米) |
| | Dataset Size | 150,000 | 训练样本总数 |
| **Stage 1** | Epochs | 2500 | 预训练轮数 |
| (Pre-training) | Learning Rate | $5 \times 10^{-5}$ | 初始学习率 |
| | LR Scheduler | Step (0.5 / 300 epochs) | 学习率衰减 |
| | Loss Weights | $w_{MSE}=1.0, w_{KKT}=0$ | 仅使用监督损失 |
| **Stage 2** | Epochs | 2500 | 微调轮数 |
| (KKT Fine-tuning)| Learning Rate | $5 \times 10^{-6}$ | 降低学习率 |
| | Loss Weights | $w_{MSE}=1.0, w_{KKT}=10^{-3}$ | 引入KKT物理正则化 |
| | KKT Components | Primal, Dual, Comp. Slack | 包含三项残差 |
| | Frozen Layers | Feature Encoder | 冻结编码器防止遗忘 |
