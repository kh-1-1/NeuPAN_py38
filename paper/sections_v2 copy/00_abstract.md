# PDPL-Net: 面向模型预测控制的快速约束保证点云感知网络

**PDPL-Net: Fast and Constraint-Guaranteed Point Cloud Perception for Model Predictive Control**

---

# 摘要

移动机器人在复杂环境中的实时安全导航是机器人学领域的核心挑战。模型预测控制（MPC）因其能够显式处理系统约束而成为主流方法，但传统MPC在处理稠密点云障碍物时面临严重的计算瓶颈——精确求解带有点级避障约束的优化问题通常需要数百毫秒，无法满足10–100Hz的实时控制需求。为解决这一问题，本文提出PDPL-Net（Primal-Dual Proximal Learning Network，原始-对偶近端学习网络），一种将优化理论的严谨性与深度学习高效性相结合的点云感知网络架构。

PDPL-Net的核心创新包括三个方面。首先，我们将原始-对偶混合梯度（PDHG）优化算法展开为可训练的神经网络层，引入可学习近端算子实现快速收敛——仅需1–2层展开即可达到与精确求解器相当的精度。其次，我们在网络架构中嵌入硬投影层，通过显式的非负锥投影和二阶锥投影操作，从数学上严格保证对偶可行性约束$\mu \geq 0$和$\|\mathbf{G}^\top \mu\|_2 \leq 1$的满足，实现100%的约束满足率。最后，我们提出基于KKT条件的残差正则化训练策略，将优化问题的最优性条件编码为损失函数，提升模型的物理一致性和分布外泛化能力。

在包含12种代表性基线方法的综合评测中，PDPL-Net展现了显著优势：相比工业级CVXPY求解器实现954倍加速（2.22ms vs 2099.8ms），MSE误差保持在$10^{-5}$量级；约束满足率从黑盒神经网络的0–88%提升至100%。闭环MPC导航实验验证了改进的前端模块转化为可观测的后端收益：对偶违背率从43–50%降至0%，计算效率提升7–8%，路径质量随展开层数渐进优化。理论分析严格证明了PDPL-Net的指数收敛性、约束满足保证和距离估计的保守性。本文证明了"结构化展开+硬投影保障"的设计范式是实现机器人安全高效控制的有效途径。

**关键词**：模型预测控制，点云感知，算法展开，约束满足，移动机器人导航

---

## Abstract

Real-time safe navigation of mobile robots in complex environments represents a core challenge in robotics. Model Predictive Control (MPC) has become the dominant approach due to its ability to explicitly handle system constraints. However, traditional MPC faces severe computational bottlenecks when processing dense point cloud obstacles—precisely solving optimization problems with point-level collision avoidance constraints typically requires hundreds of milliseconds, failing to meet the 10–100 Hz real-time control requirements. To address this challenge, this paper proposes PDPL-Net (Primal-Dual Proximal Learning Network), a point cloud perception architecture that combines the rigor of optimization theory with the efficiency of deep learning.

The core innovations of PDPL-Net encompass three aspects. First, we unroll the Primal-Dual Hybrid Gradient (PDHG) optimization algorithm into trainable neural network layers, introducing learnable proximal operators that achieve rapid convergence—requiring only 1–2 unrolling layers to attain accuracy comparable to exact solvers. Second, we embed a hard projection layer within the network architecture that, through explicit non-negative cone projection and second-order cone projection operations, mathematically guarantees the satisfaction of dual feasibility constraints $\mu \geq 0$ and $\|\mathbf{G}^\top \mu\|_2 \leq 1$, achieving 100% constraint satisfaction rate. Third, we propose a KKT condition-based residual regularization training strategy that encodes the optimality conditions of the optimization problem into the loss function, enhancing the model's physical consistency and out-of-distribution generalization capability.

In comprehensive evaluation against 12 representative baseline methods, PDPL-Net demonstrates significant advantages: achieving 954× speedup compared to industrial-grade CVXPY solver (2.22ms vs 2099.8ms) while maintaining MSE error at the $10^{-5}$ level; improving constraint satisfaction rate from 0–88% (black-box neural networks) to 100%. Closed-loop MPC navigation experiments validate that the improved front-end module translates to observable back-end benefits: dual violation rate reduced from 43–50% to 0%, computational efficiency improved by 7–8%, and path quality progressively optimized with increasing unrolling layers. Theoretical analysis rigorously proves PDPL-Net's exponential convergence, constraint satisfaction guarantee, and conservativeness of distance estimation. This paper demonstrates that the "structured unrolling + hard projection guarantee" design paradigm is an effective approach for achieving safe and efficient robot control.

**Keywords**: Model Predictive Control, Point Cloud Perception, Algorithm Unrolling, Constraint Satisfaction, Mobile Robot Navigation

