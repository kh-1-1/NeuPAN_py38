# 摘要

移动机器人在复杂环境中的实时避障导航是机器人学领域的核心挑战之一。传统的基于欧氏符号距离场（Euclidean Signed Distance Field, ESDF）的方法虽然在理论上具有完备性，但其高昂的计算代价严重制约了系统的实时性能。近年来，基于学习的方法虽然在计算效率上取得了显著突破，但黑盒神经网络难以保证输出满足优化问题的约束条件，这在安全攸关的机器人控制任务中构成了潜在的风险。针对上述问题，本文提出了PDPL-Net（Primal-Dual Proximal Learning Network），一种面向模型预测控制（Model Predictive Control, MPC）的快速约束保证点云感知网络。

PDPL-Net的核心创新在于将经典的原始-对偶混合梯度（Primal-Dual Hybrid Gradient, PDHG）优化算法展开为可训练的神经网络层，并引入可学习的近端算子以加速收敛。与传统的黑盒神经网络不同，本文设计了硬投影层（Hard Projection Layer），通过显式的几何投影操作严格保证网络输出满足对偶可行性约束，即 $\mu \geq 0$ 和 $\|G^\top \mu\|_2 \leq 1$。此外，本文提出了基于Karush-Kuhn-Tucker（KKT）条件的残差正则化训练策略，通过在损失函数中引入原始可行性、对偶可行性和互补松弛性的残差项，使网络输出在统计意义上更加接近最优解。

在实验验证方面，本文构建了包含12种基线方法的综合评测框架，涵盖传统优化方法（CVXPY、ESDF-MPC）、黑盒神经网络（PointNet++、MLP）、算法展开方法（ISTA、ADMM）以及最新的学习优化方法（DeepInverse、Point Transformer V3）。实验结果表明，PDPL-Net在保持与精确求解器相当精度的同时（MSE $1.07 \times 10^{-5}$），实现了**954倍**的计算加速（2.2ms vs 2099ms）；与黑盒神经网络相比，约束满足率从0%提升至**100%**；在闭环MPC导航任务中，所提方法在多种典型场景下均取得了最优或次优的导航性能。消融实验进一步验证了硬投影层、可学习近端算子和KKT正则化训练策略各自的贡献。

**关键词**：点云感知；模型预测控制；算法展开；约束保证；移动机器人导航

---

# Abstract

Real-time obstacle avoidance navigation of mobile robots in complex environments remains a core challenge in robotics. Traditional methods based on Euclidean Signed Distance Field (ESDF), while theoretically complete, suffer from prohibitive computational costs that severely limit real-time performance. Recent learning-based approaches have achieved significant breakthroughs in computational efficiency; however, black-box neural networks cannot guarantee that their outputs satisfy the constraints of optimization problems, posing potential risks in safety-critical robot control tasks. To address these challenges, this paper proposes PDPL-Net (Primal-Dual Proximal Learning Network), a fast and constraint-guaranteed point cloud perception network for Model Predictive Control (MPC).

The core innovation of PDPL-Net lies in unrolling the classical Primal-Dual Hybrid Gradient (PDHG) optimization algorithm into trainable neural network layers, augmented with learnable proximal operators to accelerate convergence. Unlike conventional black-box neural networks, we design a Hard Projection Layer that strictly guarantees dual feasibility constraints through explicit geometric projection, ensuring $\mu \geq 0$ and $\|G^\top \mu\|_2 \leq 1$. Furthermore, we propose a KKT (Karush-Kuhn-Tucker) residual regularization training strategy that incorporates primal feasibility, dual feasibility, and complementary slackness residuals into the loss function, enabling network outputs to statistically approximate optimal solutions.

Experimental validation is conducted on a comprehensive benchmark comprising 12 baseline methods, including traditional optimization (CVXPY, ESDF-MPC), black-box neural networks (PointNet++, MLP), algorithm unrolling methods (ISTA, ADMM), and state-of-the-art learning-based optimization (DeepInverse, Point Transformer V3). Results demonstrate that PDPL-Net achieves **954× speedup** (2.2ms vs 2099ms) while maintaining accuracy comparable to exact solvers (MSE $1.07 \times 10^{-5}$); constraint satisfaction rate improves from 0% to **100%** compared to black-box networks; and the proposed method achieves optimal or near-optimal navigation performance across various typical scenarios in closed-loop MPC navigation tasks. Ablation studies further validate the individual contributions of the hard projection layer, learnable proximal operators, and KKT regularization training strategy.

**Keywords**: Point cloud perception; Model predictive control; Algorithm unrolling; Constraint guarantee; Mobile robot navigation
