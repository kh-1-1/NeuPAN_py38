# 2. 相关工作 (Related Work)

## 2.1 基于优化的运动规划 (Optimization-based Motion Planning)
基于优化的运动规划方法通过将导航任务建模为带有约束的优化问题，能够显式地处理机器人的动力学约束和环境避障约束。模型预测控制（MPC）是其中的代表性方法，它通过在有限时域内求解最优控制序列来实现轨迹跟踪和避障。

为了处理避障约束，传统方法通常依赖于对环境的精确几何建模。例如，基于欧几里得符号距离场（ESDF）的方法 [1] 利用梯度的距离信息构建避障势场，但 ESDF 的构建和更新在复杂三维环境中计算成本高昂，难以支持高频实时规划。另一种常见策略是凸分解（Convex Decomposition）[2]，将自由空间分解为一系列凸多面体（如安全飞行走廊），从而将非凸避障约束转化为凸约束。然而，凸分解过程本身耗时较长，且生成的安全走廊往往过于保守，限制了机器人在狭窄空间中的机动性。

在求解器方面，虽然 OSQP、ECOS 和 IPOPT 等通用求解器已被广泛应用，但在处理非凸、非线性的避障约束时，它们通常需要大量的迭代次数才能收敛，且容易陷入局部最优。这限制了它们在计算资源受限的嵌入式移动机器人上的应用。

## 2.2 基于学习的运动规划 (Learning-based Motion Planning)
随着深度学习的发展，基于学习的运动规划方法逐渐受到关注。这类方法主要包括模仿学习（Imitation Learning, IL）和深度强化学习（Deep Reinforcement Learning, DRL）。

端到端的方法 [3] 直接将传感器数据（如激光雷达点云或RGB图像）映射到控制指令，省略了显式的地图构建和定位过程，具有极高的推理速度。DRL 方法 [4] 通过与环境的交互试错来学习策略，能够在未知和动态环境中表现出良好的适应性。然而，基于学习的方法通常被视为“黑盒”系统，缺乏可解释性和理论上的安全保证。虽然可以通过设计包含碰撞惩罚的奖励函数来引导智能体避障，但在训练数据分布之外的场景（Out-of-Distribution, OOD）中，网络输出的行为往往不可预测，存在所谓的“长尾风险”。

为了提高安全性，一些工作引入了安全过滤器（Safety Filter），如基于控制障碍函数（Control Barrier Functions, CBF）的方法，对网络输出的动作进行后处理修正。但这种方法依赖于精确的系统模型和障碍物状态估计，在非结构化环境中仍然面临挑战。

## 2.3 神经优化与算法展开 (Neural Optimization and Algorithm Unrolling)
神经优化，特别是算法展开（Algorithm Unrolling/Unfolding），旨在结合优化方法的理论严谨性和深度学习的高效拟合能力。该技术将传统的迭代优化算法（如梯度下降、ADMM、PDHG）展开为深度神经网络的层级结构，并将算法中的超参数（如步长、惩罚系数）设为可学习的权重。

Gregor 等人 [5] 提出的 LISTA 是该领域的开创性工作，证明了展开网络在稀疏编码任务中能够比传统算法更快收敛。在控制领域，Amos 等人提出的 OptNet [6] 将二次规划（QP）求解器嵌入到神经网络中，实现了端到端的微分优化。然而，现有的展开方法大多局限于处理凸优化问题，或者采用软惩罚（Soft Penalty）的方式处理约束，无法严格保证硬约束的满足。

本文提出的 NeuPAN 框架基于 PDHG 算法的展开，专门针对非凸的避障约束设计。与现有工作不同，NeuPAN 在网络层中显式集成了硬投影算子，并利用神经网络预测对偶变量，从而在保证硬约束满足的同时，实现了毫秒级的实时推理速度。

---
**参考文献 (References)**:
[1] Oleynikova, H., et al. "Voxblox: Incremental 3D Euclidean signed distance fields for on-board MAV planning." IROS 2017.
[2] Liu, S., et al. "Planning dynamically feasible trajectories for quadrotors using safe flight corridors in 3-d complex environments." ICRA 2017.
[3] Bojarski, M., et al. "End to end learning for self-driving cars." arXiv preprint 2016.
[4] Pfeiffer, M., et al. "From perception to decision: A data-driven approach to end-to-end motion planning for autonomous ground robots." ICRA 2017.
[5] Gregor, K., & LeCun, Y. "Learning fast approximations of sparse coding." ICML 2010.
[6] Amos, B., & Kolter, J. Z. "OptNet: Differentiable optimization as a layer in neural networks." ICML 2017.
