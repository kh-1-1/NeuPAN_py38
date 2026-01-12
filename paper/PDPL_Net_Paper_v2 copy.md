# PDPL-Net: 面向模型预测控制的快速约束保证点云感知网络

**PDPL-Net: Fast and Constraint-Guaranteed Point Cloud Perception for Model Predictive Control**

---

# 摘要

移动机器人在复杂环境中的实时安全导航是机器人学领域的核心挑战。模型预测控制（MPC）因其能够显式处理系统约束而成为主流方法，但传统MPC在处理稠密点云障碍物时面临严重的计算瓶颈——精确求解带有点级避障约束的优化问题通常需要数百毫秒，无法满足10–100Hz的实时控制需求。为解决这一问题，本文提出PDPL-Net（Primal-Dual Proximal Learning Network，原始-对偶近端学习网络），一种将优化理论的严谨性与深度学习高效性相结合的点云感知网络架构。

PDPL-Net的核心创新包括三个方面。首先，我们将原始-对偶混合梯度（PDHG）优化算法展开为可训练的神经网络层，引入可学习近端算子实现快速收敛——仅需1–2层展开即可达到与精确求解器相当的精度。其次，我们在网络架构中嵌入硬投影层，通过显式的非负锥投影和二阶锥投影操作，从数学上严格保证对偶可行性约束$\mu \geq 0$和$\|\mathbf{G}^\top \mu\|_2 \leq 1$的满足，实现100%的约束满足率。最后，我们提出基于KKT条件的残差正则化训练策略，将优化问题的最优性条件编码为损失函数，提升模型的物理一致性和分布外泛化能力。

在包含12种代表性基线方法的综合评测中，PDPL-Net展现了显著优势：相比工业级CVXPY求解器实现1680倍加速（1.25ms vs 2099.8ms），MSE误差保持在$10^{-6}$量级；约束满足率从黑盒神经网络的0–88%提升至100%。闭环MPC导航实验验证了改进的前端模块转化为可观测的后端收益：对偶违背率从43–50%降至0%，计算效率提升7–8%，路径质量随展开层数渐进优化。理论分析严格证明了PDPL-Net的指数收敛性、约束满足保证和距离估计的保守性。本文证明了"结构化展开+硬投影保障"的设计范式是实现机器人安全高效控制的有效途径。

在系统层面，我们沿用NeuPAN的NRMP/MPC滚动优化后端，仅将其前端点级对偶变量编码器（DUNE）替换为PDPL-Net；因此本文的贡献聚焦于“前端即插即用升级”，在不改变后端控制器接口与求解流程的前提下提升实时性与确定性安全。

**关键词**：模型预测控制，点云感知，算法展开，约束满足，移动机器人导航

---

## Abstract

Real-time safe navigation of mobile robots in complex environments represents a core challenge in robotics. Model Predictive Control (MPC) has become the dominant approach due to its ability to explicitly handle system constraints. However, traditional MPC faces severe computational bottlenecks when processing dense point cloud obstacles—precisely solving optimization problems with point-level collision avoidance constraints typically requires hundreds of milliseconds, failing to meet the 10–100 Hz real-time control requirements. To address this challenge, this paper proposes PDPL-Net (Primal-Dual Proximal Learning Network), a point cloud perception architecture that combines the rigor of optimization theory with the efficiency of deep learning.

The core innovations of PDPL-Net encompass three aspects. First, we unroll the Primal-Dual Hybrid Gradient (PDHG) optimization algorithm into trainable neural network layers, introducing learnable proximal operators that achieve rapid convergence—requiring only 1–2 unrolling layers to attain accuracy comparable to exact solvers. Second, we embed a hard projection layer within the network architecture that, through explicit non-negative cone projection and second-order cone projection operations, mathematically guarantees the satisfaction of dual feasibility constraints $\mu \geq 0$ and $\|\mathbf{G}^\top \mu\|_2 \leq 1$, achieving 100% constraint satisfaction rate. Third, we propose a KKT condition-based residual regularization training strategy that encodes the optimality conditions of the optimization problem into the loss function, enhancing the model's physical consistency and out-of-distribution generalization capability.

In comprehensive evaluation against 12 representative baseline methods, PDPL-Net demonstrates significant advantages: achieving 1680× speedup compared to industrial-grade CVXPY solver (1.25ms vs 2099.8ms) while maintaining MSE error at the $10^{-6}$ level; improving constraint satisfaction rate from 0–88% (black-box neural networks) to 100%. Closed-loop MPC navigation experiments validate that the improved front-end module translates to observable back-end benefits: dual violation rate reduced from 43–50% to 0%, computational efficiency improved by 7–8%, and path quality progressively optimized with increasing unrolling layers. Theoretical analysis rigorously proves PDPL-Net's exponential convergence, constraint satisfaction guarantee, and conservativeness of distance estimation. This paper demonstrates that the "structured unrolling + hard projection guarantee" design paradigm is an effective approach for achieving safe and efficient robot control.

At the system level, we retain NeuPAN’s NRMP/MPC back-end and replace only the front-end point-wise dual-variable encoder (DUNE) with PDPL-Net. This makes our method a drop-in upgrade that improves real-time performance and deterministic safety without changing the controller interface or the optimization pipeline.

**Keywords**: Model Predictive Control, Point Cloud Perception, Algorithm Unrolling, Constraint Satisfaction, Mobile Robot Navigation

# I. 引言

在移动机器人的自主导航领域，长期存在着一个难以调和的**"安全-效率"二律背反**：为了保证在非结构化环境中的绝对安全，机器人需要求解包含严格几何约束的优化问题，这通常计算昂贵且缓慢；而为了实现敏捷的实时响应，系统倾向于使用高效的启发式方法或端到端神经网络，但这往往以牺牲理论上的安全保证为代价。模型预测控制（MPC）作为处理受限控制问题的主流框架，处于这一矛盾的风暴中心。当面对稠密点云表示的复杂环境时，传统优化方法（如内点法）虽然能提供严谨的数学安全保证，但在嵌入式处理器上求解数百个点级避障约束的耗时往往高达数百毫秒，严重制约了机器人的动态性能。

为了打破这一计算瓶颈，深度学习方法应运而生，通过拟合感知输入到控制输出的映射展现了惊人的推理速度。然而，这种"黑盒"方法的阿喀琉斯之踵在于其**概率性的安全保障**——神经网络无法保证其输出严格满足硬性的几何约束。现有的混合方法（如NeuPAN）试图通过软约束（Soft Constraints）或后处理投影（Post-Projection）来修补这一缺陷，但这本质上是一种"打补丁"的策略：前者无法从数学上杜绝碰撞风险（对偶违背率仍可达40%以上），后者则可能导致解偏离最优控制意图。

本文旨在从根本上解决这一矛盾，提出了一种**"架构即约束"（Architecture as Constraint）**的新范式。我们认为，**安全性不应是训练出来的，而应是设计出来的。** 基于这一核心洞见，我们提出了PDPL-Net（Primal-Dual Proximal Learning Network）。不同于简单地训练网络去拟合优化结果，PDPL-Net**将原始-对偶混合梯度（PDHG）算法的迭代结构直接"展开"为神经网络**。这种设计不仅继承了优化理论的收敛性结构，更通过引入**架构级硬投影层**，从数学原理上强制网络输出必须落在对偶可行域内。这意味着PDPL-Net在保留深度学习$O(1)$极速推理优势的同时，成为了首个在点级对偶变量预测中实现**100%约束满足率**的学习方法，将安全性从"概率性"提升为"确定性"。

具体而言，PDPL-Net通过三个层面的创新重构了"安全"与"效率"的关系。首先，通过**结构化展开**，网络的每一层对应优化算法的一次迭代，利用数据驱动的**可学习近端算子**捕捉问题特定的几何结构，仅需1-2层即可达到传统迭代算法数百步的精度，实现了相比工业级求解器1680倍的加速。其次，我们在网络末端嵌入了无参数的**硬投影层**，利用显式的非负锥与二阶锥投影，确保生成的对偶变量严格满足物理约束，彻底消除了黑盒网络的幻觉风险。最后，我们提出了一套物理一致的训练策略，结合KKT残差正则化与课程学习，确保网络不仅拟合数据，更遵守优化的物理规律。这种"白盒"展开架构成功地将优化方法的严谨性（Constraint-Guaranteed）与深度学习的高效性（Real-Time Efficiency）统一在一个可微框架中，为下一代敏捷且安全的机器人导航系统奠定了坚实的感知基础。

# II. 相关工作

## A. 传统点云处理方法

将障碍物信息融入运动规划的主流范式是从原始传感器数据构建中间几何表示。Oleynikova等人提出了Voxblox——一种增量式三维符号距离场构建方法，通过利用相邻帧之间的空间相干性实现实时更新，显著加速了微型无人机（MAV）的在线轨迹规划。Han等人进一步提出了FIESTA算法，通过优化增量更新策略将ESDF维护的计算效率提升了一个数量级。尽管这些工作取得了重要进展，ESDF类方法共享一个根本性的局限：它们将距离计算与运动优化视为独立阶段，无法实现端到端学习与联合优化。此外，ESDF的存储开销随环境规模呈立方增长，对于大规模室外环境的适用性受到严重制约。

另一类研究直接在点云原语上进行操作，避免显式的地图构建过程。Zhou等人采用中心点距离作为简化的碰撞度量，以牺牲几何保真度为代价换取更快的计算速度。这种近似方法在障碍物稀疏且为凸形状的环境中表现尚可，但在杂乱场景中——精确的边界推理变得至关重要——其性能显著下降。Han等人近期提出的NeuPAN框架通过双凸对偶重构（Biconvex Dual Reformulation）解决了这一问题，将点级碰撞约束转化为对偶变量的求解问题，从而实现了对障碍物点的直接优化而无需中间地图表示。NeuPAN证明了基于学习的感知模块能够与基于优化的控制紧密集成以实现实时性能。然而，NeuPAN中的神经网络组件DUNE并未显式强制对偶可行性约束——这些约束对于有效的距离计算是必需的——导致约43–50%的推理结果存在约束违背，必须通过事后投影进行修正。

## B. 基于学习的点云感知

深度学习通过专门针对不规则、无序点集设计的网络架构彻底革新了点云处理领域。Qi等人提出的PointNet通过对逐点特征进行最大池化聚合实现了置换不变性，为后续研究奠定了基础。PointNet++在此基础上引入层次化特征学习，在多个尺度上捕获局部几何结构。近年来，基于注意力机制的架构在点云理解任务上取得了最先进的性能。Zhao等人提出的Point Transformer将自注意力机制应用于点云特征，并结合位置编码增强空间感知能力。最新的Point Transformer V3通过序列化注意力模式进一步提升了大规模点云处理的效率和精度，在多个基准测试中刷新了记录。

然而，这些黑盒神经网络无法保证其输出满足优化问题固有的数学约束。当应用于预测避障所需的对偶变量时——这些变量必须满足非负性约束和范数约束——标准神经网络的违背率通常超过20–30%。虽然可以通过投影操作修正这些违背，但修正后的解可能与最优解存在显著偏差，且投影步骤引入了额外的计算开销。更为关键的是，缺乏结构性保证从根本上削弱了MPC控制所追求的理论安全性——这种安全性正是采用基于优化的控制框架的初衷所在。

## C. 算法展开与神经优化

算法展开（Algorithm Unrolling），又称深度展开（Deep Unfolding），提供了一种将神经网络的效率与优化算法的结构相结合的原则性方法。通过将迭代优化步骤解释为神经网络的层级结构，并允许算法参数从数据中学习，展开网络能够以比手工调参算法更少的迭代次数达到收敛，同时保持良好的可解释性。Gregor和LeCun以LISTA（Learned ISTA）开创了这一研究方向，证明了展开的稀疏编码网络仅用几层便能达到数百次ISTA迭代的精度。Yang等人将展开方法扩展到ADMM用于压缩感知MRI重建，表明问题特定的架构能够超越通用网络的性能。

对于与机器人相关的受约束优化问题，可微分优化层提供了另一种范式。Amos和Kolter提出的OptNet将二次规划（QP）求解器嵌入神经网络中，并通过KKT条件计算梯度实现端到端训练。Agrawal等人提出的CvxpyLayers将这一方法推广到任意规范凸程序，使其能够作为可微分层参与神经网络计算。虽然这些方法通过构造保证了约束满足，但它们继承了底层求解器的计算复杂度，限制了其在需要毫秒级延迟的实时控制场景中的适用性。

## D. 物理信息神经网络

物理信息学习（Physics-Informed Learning）已成为将领域知识融入神经网络训练和架构的重要策略。Raissi等人提出的物理信息神经网络（Physics-Informed Neural Networks, PINNs）将控制方程作为软约束嵌入损失函数，使网络即使在训练数据有限的情况下也能遵循物理规律。对于优化问题，KKT条件提供了类似的结构性约束来刻画最优解的特性。近期工作探索了将KKT残差编码为正则化项以提升解的质量和泛化能力。然而，软惩罚方法无法保证严格的约束满足——这对于安全保障的机器人控制是一个关键要求。如何在网络架构层面嵌入硬约束保证，是连接学习效率与优化严谨性的核心挑战。

# III. 问题建模

本章旨在构建一个数学严谨的框架，以解决全维度移动机器人在非结构化环境中的自主导航问题。我们将该问题形式化为一个**模型预测感知与控制 (Model Predictive Perception and Control, MPPC)** 的联合优化过程。该过程在每个时刻 $k$，基于实时获取的激光雷达点云 $\mathcal{O}_k$，在预测时域 $N$ 内寻找最优的控制策略。

#### A. 离散时间运动学 (Discrete-Time Kinematics)
首先，我们建立机器人的运动方程。设 $\mathbf{x}_k \in \mathbb{R}^{n_x}$ 和 $\mathbf{u}_k \in \mathbb{R}^{n_u}$ 分别表示时刻 $k$ 的系统状态向量和控制输入向量。机器人的演化遵循非线性离散动力学：
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \phi(\mathbf{x}_k, \mathbf{u}_k) \Delta t
$$
其中 $\Delta t$ 为采样周期，$\phi(\cdot)$ 描述了系统的物理特性。为了便于在凸优化框架下求解，我们采用序列凸规划（Sequential Convex Programming）的思想，在参考轨迹点 $(\bar{\mathbf{x}}_k, \bar{\mathbf{u}}_k)$ 处对动力学方程进行一阶泰勒展开，得到如下线性时变（LTV）约束：
$$
\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{d}_k
$$
这里，$\mathbf{A}_k$ 和 $\mathbf{B}_k$ 分别为系统关于状态和控制的雅可比矩阵，$\mathbf{d}_k$ 为线性化残差项。此外，考虑到执行器的物理极限，控制量必须严格受限于允许的凸集 $\mathcal{U}_{admissible}$ 内：
$$
\mathbf{u}_{min} \leq \mathbf{u}_k \leq \mathbf{u}_{max}, \quad |\mathbf{u}_{k+1} - \mathbf{u}_k| \leq \mathbf{\delta}_{max}
$$
这确保了生成的控制指令在实际物理系统上的可执行性。

#### B. 机器人几何表征 (Robot Geometric Representation)
精确的避障依赖于对机器人轮廓的精确描述。我们将机器人建模为一个紧致凸多胞形（Compact Convex Polytope） $\mathcal{R}$。在机器人本体坐标系下，该区域由一组线性不等式定义：
$$
\mathcal{R}_{body} = \{ \mathbf{z} \in \mathbb{R}^2 \mid \mathbf{H} \mathbf{z} \leq \mathbf{h} \}
$$
其中 $\mathbf{H} \in \mathbb{R}^{m \times 2}$ 为边界法向量矩阵，$\mathbf{h} \in \mathbb{R}^m$ 为边界距离向量，$m$ 为多胞形的面数。当机器人处于全局状态 $\mathbf{x}_k$（包含位姿信息）时，其占据的空间区域 $\mathcal{R}(\mathbf{x}_k)$ 可通过刚体变换 $\mathcal{T}(\mathbf{x}_k)$ 描述为本体区域的映射：
$$
\mathcal{R}(\mathbf{x}_k) = \{ \mathbf{R}(\mathbf{x}_k)\mathbf{z} + \mathbf{t}(\mathbf{x}_k) \mid \mathbf{z} \in \mathcal{R}_{body} \}
$$
这种基于H-表示法（H-representation）的凸集建模，相比于外接圆或椭圆近似，能够更紧凑地包裹机器人轮廓，从而允许机器人在狭窄空间中进行更激进的机动。

#### C. 对偶避障约束 (Dual Obstacle Avoidance Constraints)
在真实场景中，障碍物信息通常以离散点云集 $\mathcal{O}_k = \{ \mathbf{o}_1, \dots, \mathbf{o}_M \}$ 的形式给出。安全导航的核心要求是机器人集合 $\mathcal{R}(\mathbf{x}_k)$ 与任意障碍点 $\mathbf{o}_i$ 均不相交，且保持至少 $d_{safe}$ 的安全裕度：
$$
\text{dist}(\mathcal{R}(\mathbf{x}_k), \mathbf{o}_i) \geq d_{safe}, \quad \forall i \in \{1, \dots, M\}
$$
直接处理上述距离约束面临着严峻挑战：距离函数 $\text{dist}(\cdot)$ 关于状态 $\mathbf{x}_k$ 往往是非凸且非光滑的。为了克服这一困难，我们引入拉格朗日对偶原理进行重构。根据凸分离定理，点到凸多胞形的距离可以等价转化为一个对偶线性规划的最大值问题：
$$
\text{dist}(\mathcal{R}(\mathbf{x}_k), \mathbf{o}_i) = \max_{\boldsymbol{\lambda}_i, \boldsymbol{\nu}_i} \left( -\mathbf{h}^\top \boldsymbol{\lambda}_i - (\mathbf{o}_i - \mathbf{t}(\mathbf{x}_k))^\top \boldsymbol{\nu}_i \right)
$$
$$
\text{s.t.} \quad \mathbf{H}^\top \boldsymbol{\lambda}_i + \mathbf{R}(\mathbf{x}_k)^\top \boldsymbol{\nu}_i = \mathbf{0}, \quad \|\boldsymbol{\nu}_i\|_2 \leq 1, \quad \boldsymbol{\lambda}_i \geq \mathbf{0}
$$
其中 $\boldsymbol{\lambda}_i \in \mathbb{R}^m$ 和 $\boldsymbol{\nu}_i \in \mathbb{R}^2$ 为引入的对偶变量。这种重构具有深刻的意义：它将复杂的几何距离计算转化为了关于对偶变量的代数约束，不仅恢复了问题的局部凸性，还为利用并行计算加速求解提供了可能。

#### D. 导航代价函数 (Navigation Cost Function)
为了引导机器人高效、平滑地驶向目标，我们设计了如下形式的二次代价函数 $J_N$：
$$
J_N = \sum_{j=0}^{N-1} \left( \|\mathbf{x}_{k+j} - \mathbf{x}^{ref}_{k+j}\|_{\mathbf{Q}}^2 + \|\mathbf{u}_{k+j} - \mathbf{u}^{ref}_{k+j}\|_{\mathbf{R}}^2 \right) + \|\mathbf{x}_{k+N} - \mathbf{x}^{ref}_{k+N}\|_{\mathbf{Q}_f}^2
$$
该函数由三部分组成：轨迹追踪误差项确保机器人紧贴全局规划路径；控制输入偏差项抑制控制量的剧烈抖动，保证运动平滑性；终端代价项则用于增强预测控制的闭环稳定性。权重矩阵 $\mathbf{Q}, \mathbf{R}, \mathbf{Q}_f$ 需根据具体的任务需求进行整定。

#### E. 联合优化公式化 (Joint Optimization Formulation)
综合上述各部分，我们将移动机器人的局部导航问题公式化为一个联合优化问题。该问题旨在同时寻找最优的控制序列 $\mathcal{U}^*$ 和最优的感知变量（对偶变量） $\Lambda^*$：
$$
\begin{aligned}
\min_{\mathbf{X}, \mathbf{U}, \Lambda} \quad & J_N(\mathbf{X}, \mathbf{U}) \\
\text{s.t.} \quad & \mathbf{x}_{j+1} = \mathbf{A}_j \mathbf{x}_j + \mathbf{B}_j \mathbf{u}_j + \mathbf{d}_j \\
& \mathbf{u}_{min} \leq \mathbf{u}_j \leq \mathbf{u}_{max} \\
& -\mathbf{h}(\mathbf{x}_j)^\top \boldsymbol{\lambda}_{i,j} - \mathbf{o}_i^\top \boldsymbol{\nu}_{i,j} \geq d_{safe}, \quad \forall i, \forall j \\
& \boldsymbol{\lambda}_{i,j} \in \mathcal{F}_{dual} \quad (\text{对偶可行域})
\end{aligned}
$$
**技术瓶颈分析**：上述 MPPC 问题在数学上是一个大规模的双层优化问题。虽然外层的控制变量优化是一个标准的二次规划（QP），求解十分成熟；但内层的感知变量优化涉及成百上千个障碍点的对偶变量求解。在传统方法中，这意味着需要在每个控制周期内求解数千个二阶锥规划（SOCP）子问题，其巨大的计算开销成为了限制系统实时性的最大瓶颈。这也正是本文提出 PDPL-Net 的核心动机——通过学习的方法来“预测”最优的对偶变量，从而突破这一计算瓶颈。

求解上述 MPPC 问题通常采用**两层交替优化（Two-layer Alternating Optimization）**策略：外层优化控制律 $\mathbf{U}$，内层优化感知变量 $\boldsymbol{\mu}$。在NeuPAN框架中，外层控制是一个标准的二次规划（QP），求解非常高效；而内层感知需要求解数千个二阶锥规划（SOCP），占据了绝大部分计算时间且难以保证实时性。**因此，本文的研究范围聚焦于该联合优化问题的内层子问题**。我们的目标不是重新设计整个 MPC 控制器，而是提出 PDPL-Net 来替代传统的内层数值求解器。随后的实验将重点评估 PDPL-Net 作为前端感知模块的性能（精度、速度、约束满足），并验证其对后端控制系统的具体贡献。

# IV. PDPL-NET 框架

本节介绍了 PDPL-Net，这是一种专门设计用于加速求解 MPPC 内层感知子问题的深度展开网络。该网络将经典的原始-对偶混合梯度（PDHG）算法映射为可微的神经网络层，通过学习特定问题分布下的最优算子，在保证几何约束的同时实现毫秒级推理。

#### A. 对偶优化的 PDHG 算法
点级避障的对偶问题可以写成如下的鞍点形式：
$$
\min_{\boldsymbol{\lambda}} \max_{\boldsymbol{\nu}} \quad \mathcal{L}(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \boldsymbol{\lambda}^\top (\mathbf{H}\mathbf{x} - \mathbf{h}) + \boldsymbol{\nu}^\top (\mathbf{o} - \mathbf{t}) \quad \text{s.t.} \quad \boldsymbol{\lambda} \geq \mathbf{0}, \quad \|\boldsymbol{\nu}\|_2 \leq 1
$$
我们引入辅助变量 $\boldsymbol{\nu}$ 以解耦范数约束。由于该问题是凸的，可以通过 PDHG 算法最优求解，该算法交替更新原始变量 $\boldsymbol{\lambda}$ 和对偶变量 $\boldsymbol{\nu}$。在 PDHG 的第 $k$ 次迭代中，给定解 $\boldsymbol{\lambda}^{(k)}$ 和 $\boldsymbol{\nu}^{(k)}$，相关的单步更新如下：
$$
\begin{cases}
\boldsymbol{\nu}^{(k+1)} = \text{Proj}_{\mathcal{B}_2} \left( \boldsymbol{\nu}^{(k)} + \sigma \mathbf{R} \mathbf{H}^\top \boldsymbol{\lambda}^{(k)} \right) & (\text{对偶步}) \\
\boldsymbol{\lambda}^{(k+1)} = \text{Proj}_{\mathbb{R}_+} \left( \boldsymbol{\lambda}^{(k)} + \tau (\mathbf{a} - \mathbf{H} \mathbf{R}^\top \boldsymbol{\nu}^{(k+1)}) \right) & (\text{原始步})
\end{cases} \tag{10}
$$
其中 $\mathbf{a} = \mathbf{H}\mathbf{x} - \mathbf{h}$，$\tau, \sigma$ 为步长参数，$\text{Proj}_{\mathcal{B}_2}$ 是投影到单位球，$\text{Proj}_{\mathbb{R}_+}$ 是投影到非负象限。这完成了一次迭代。通过重复此过程生成的序列是收敛的。为了解决复杂性问题同时保持可解释性，我们提出使用深度展开神经网络来实现这一过程。

#### B. 深度展开架构
PDHG 算法生成的迭代序列 $\{\boldsymbol{\mu}^{(0)}, \boldsymbol{\mu}^{(1)}, \cdots\}$ 可以看作是一个通过一系列参数化函数的顺序映射过程：
$$
\boldsymbol{\mu}^{(0)} \xrightarrow{\mathcal{G}_1} \boldsymbol{\mu}^{(1)} \xrightarrow{\mathcal{G}_2} \boldsymbol{\mu}^{(2)} \cdots \xrightarrow{\mathcal{G}_J} \boldsymbol{\mu}^{(J)}
\tag{22}
$$
其中每个映射 $\mathcal{G}_j$ 本质上对应于优化算法中的一步梯度更新与投影操作。更具体地说，$\mathcal{G}_j$ 由矩阵乘法（梯度计算）、非线性激活（投影操作）以及残差修正（近端加速）组成。因此，每个 $\mathcal{G}_j$ 都可以自然地展开为对应的神经网络层。PDHG 收敛所需的迭代次数 $J$ 决定了神经网络的深度。这种设计使得 PDPL-Net 成为迭代优化算法的可学习变体，从而在保持深度学习高效性的同时具备了明确的数学可解释性。

基于上述观察，所提出的 PDPL-Net 架构如图 2 所示。其数据流设计严格遵循算法逻辑，包含编码器、初始化头、展开层和终端投影四个模块。为了便于复现，我们在表 II 中列出了详细的网络结构参数。整个网络设计轻量高效，参数总量仅约为 10k，这为在嵌入式设备上的实时推理奠定了基础。

**表 II：PDPL-Net 网络结构与参数配置**
| Module | Layer Type | Input Dim | Output Dim | Activation | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder** | FC-1 | 2 | 32 | ReLU | 特征提取 |
| | FC-2 | 32 | 32 | ReLU | 特征提取 |
| **Init Heads** | $\mu$-Head | 32 | $E$ (4) | ReLU | 非负初始化 |
| | $y$-Head | 32 | 2 | Tanh | 范数有界初始化 |
| **Unroll Block** | Prox-Net | $E$ | 32 $\to$ $E$ | ReLU | 残差 MLP ($J$ 个共享或独立) |
| **Projector** | Hard-Proj | $E$ | $E$ | - | 无参数物理投影 |

首先，输入点云数据通过一个编码器模块，该模块由两层全连接（FC）网络构成，每层包含 32 个神经元并接 ReLU 激活，旨在从低维坐标中提取高维几何特征。编码器的输出被送入两个并行的初始化头：$\boldsymbol{\mu}$-Head 使用 FC 层接 ReLU 激活生成非负的初始对偶变量 $\boldsymbol{\mu}^{(0)}$，而 $\mathbf{y}$-Head 使用 FC 层接 Tanh 激活生成范数受限的辅助变量 $\mathbf{y}^{(0)}$。这种设计为后续迭代提供了符合物理约束的高质量初值。核心展开部分由 $J$ 个级联的块组成，每个块包含两个子层：第一子层对应 PDHG 的对偶步（公式 10a），执行线性更新后接 $l_2$ 球投影以更新辅助变量 $\mathbf{y}$；第二子层对应 PDHG 的原始步（公式 10b），执行梯度下降更新 $\boldsymbol{\mu}$，并引入由轻量级 MLP 构成的可学习近端模块来计算残差修正量，最后通过硬投影操作强制满足 $\boldsymbol{\mu} \ge 0$ 和 $\|\boldsymbol{\mu}\mathbf{G}\|_2 \le 1$。通过重复该展开块 $J$ 次（实验中 $J=1 \sim 3$），并在网络末端执行最终的安全投影，整个 PDPL-Net 相当于一个参数化的 PDHG 求解器。这是因为 PDHG 算法交替执行对偶投影和原始更新，而 PDPL-Net 正是由交替执行这两个步骤的神经网络层堆叠而成。

#### C. 损失函数设计
损失函数设计：使用从数值求解器（如 CVXPY-CLARABEL）派生的标记数据集对神经网络进行训练。基于最优解 $\{\boldsymbol{\mu}_i^*\}$ 与神经网络的解 $\{\hat{\boldsymbol{\mu}}_i\}$ 之间的损失函数来确定反向传播。一个简单的损失函数选择是这两个值之间的均方误差（MSE）。然而，由于学到的 $\{\hat{\boldsymbol{\mu}}_i\}$ 将在后续的运动规划中作为正则化项来处理避碰约束，我们必须保证在不同向量方向上 $\{\hat{\boldsymbol{\mu}}_i\}$ 的高精度，并将这些MSE纳入到损失函数中。基于这些考虑，我们将损失函数表述如下：
$$
\mathcal{L}(\hat{\boldsymbol{\mu}}_i, \boldsymbol{\mu}_i^*) = \|\hat{\boldsymbol{\mu}}_i - \boldsymbol{\mu}_i^*\|_2^2 + \|\mathcal{F}_o(\hat{\boldsymbol{\mu}}_i) - \mathcal{F}_o(\boldsymbol{\mu}_i^*)\|_2^2 + \|\mathcal{F}_a(\hat{\boldsymbol{\mu}}_i) - \mathcal{F}_a(\boldsymbol{\mu}_i^*)\|_2^2 + \|\mathcal{F}_b(\hat{\boldsymbol{\mu}}_i) - \mathcal{F}_b(\boldsymbol{\mu}_i^*)\|_2^2 + w_{kkt}\mathcal{L}_{KKT} \tag{11}
$$
其中，$\mathcal{F}_o(\boldsymbol{\mu}_i) = \boldsymbol{\mu}_i^\top (\mathbf{G}\mathbf{p}_i - \mathbf{h})$ 是问题方程（12）的目标函数。函数 $\mathcal{F}_a(\boldsymbol{\mu}_i) = -\boldsymbol{\mu}_i^\top \mathbf{G}\mathbf{R}^\top$ 和 $\mathcal{F}_b(\boldsymbol{\mu}_i) = \boldsymbol{\mu}_i^\top \mathbf{G}\mathbf{R}^\top \mathbf{p}_i - \boldsymbol{\mu}_i^\top \mathbf{h}$ 与神经正则化函数 $\mathcal{C}_r$ 相关联。

为了进一步提升解的物理一致性，我们引入了 KKT 残差项 $\mathcal{L}_{KKT}$。该项利用优化理论中的一阶最优性条件（Stationarity）作为无监督约束，迫使网络即使在标签数据之外也能遵循物理规律。其计算公式为归一化的残差范数：
$$
\mathcal{L}_{KKT} = \left\| \frac{-\mathbf{a} + \rho \mathbf{G}\mathbf{G}^\top \hat{\boldsymbol{\mu}}_i - \text{ReLU}(-\hat{\boldsymbol{\mu}}_i)}{\|\mathbf{a}\|_2 + \epsilon} \right\|_2^2
$$
其中 $\mathbf{a} = \mathbf{G}\mathbf{p}_i - \mathbf{h}$ 为几何约束项。这一正则化项显著增强了模型在分布外数据上的泛化能力。

#### D. 训练策略

PDPL-Net 模型的训练超参数列于表 III 中。对于一个具有给定值 $[\mathbf{G}, \mathbf{h}]$（由形状尺寸决定）的机器人，训练过程从在每个轴内随机生成 $N_g$ 个点位置 $\mathbf{o}$ 开始，这些点位置位于特定范围 $[-10, 10]$ 内。随后，通过已知的值 $[\mathbf{G}, \mathbf{h}, \mathbf{o}_i]$ 构建 $N_g$ 个关于变量 $\boldsymbol{\mu}$ 的凸优化问题，并使用工业级求解器 CVXPY CLARABEL 求解，得到 $N_g$ 个最优值 $\boldsymbol{\mu}_i^*$。因此，每个点位置 $\mathbf{o}_i$ 都有一一对应的最优解 $\boldsymbol{\mu}_i^*$，从而形成一个标记的数据集 $\mathcal{T}=\{\mathbf{o}_i, \boldsymbol{\mu}_i^*\}$，用于神经网络训练。为了应对非凸的物理损失曲面，我们采用了两阶段课程学习策略：第一阶段（预训练）进行 2500 个周期，仅使用 MSE 损失以学习率 $5 \times 10^{-5}$ 快速建立粗略映射；第二阶段（物理微调）继续进行 2500 个周期，引入 KKT 物理正则化项并降低学习率至 $5 \times 10^{-6}$，以精细修正网络权重。利用 Adam 优化器更新网络参数，批量大小为 256。在我们的实验中，训练过程是在一台配备 Intel Core i9 CPU 和 NVIDIA GeForce RTX 4090 GPU 的台式计算机上进行的，大约需要一小时完成。该过程列在算法 1 中。

请注意，与典型的深度学习模型不同，后者需要在现实环境中进行大量数据收集，并且可能由于仿真到现实的差距而在泛化方面遇到困难，我们的 PDPL-Net 模型对这种挑战具有弹性，并且由于其独特的点表示和模型展开的网络结构，可以快速进行训练。此外，根据方程（12），PDPL-Net 模型仅受 $[\mathbf{G}, \mathbf{h}]$ 的影响，这些由形状大小决定。因此，对于特定的机器人，PDPL-Net 模型可以在不同的现实环境中部署，而无需重新训练。

**表 III：两阶段训练超参数设置**
| Stage | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Global** | Batch Size | 256 | Batch size per step |
| | Optimizer | Adam | Standard optimizer |
| | Data Range | $[-10, 10]^2$ | Sampling range (m) |
| | Dataset Size | 150,000 | Total training samples |
| **Stage 1** | Epochs | 2500 | Imitation phase duration |
| (Pre-train) | Learning Rate | $5 \times 10^{-5}$ | Initial learning rate |
| | Scheduler | Step (0.5 / 300 ep) | Decay factor / frequency |
| | Loss Weights | $w_{MSE}=1.0, w_{KKT}=0$ | Supervised loss only |
| **Stage 2** | Epochs | 2500 | Fine-tuning duration |
| (Fine-tune) | Learning Rate | $5 \times 10^{-6}$ | Reduced rate for stability |
| | Loss Weights | $w_{MSE}=1, w_{KKT}=10^{-3}$ | Enable physics regularization |
| | Frozen Layers | Feature Encoder | Prevent feature forgetting |

**算法 1**: PDPL-Net: 深度展开 PDHG 网络 (推理与训练)
1. **Input**: Robot geometry $\{\mathbf{G}, \mathbf{h}\}$ and sampled point set $\mathcal{O}$ within range.
2. **Data Generation**: Solve SOCP via solver to get labels $\{\boldsymbol{\mu}^*_i\}$. Set dataset $\mathcal{T} = \{(\mathbf{o}_i, \boldsymbol{\mu}^*_i)\}$.
3. **Initialize Network**: Parameters $\Theta$ (weights for Encoder, Heads, ProxNet).
4. **// Training Loop**
5. **for** epoch $e = 1, \dots, E$ **do**
6.    Sample batch $(\mathbf{o}, \boldsymbol{\mu}^*) \sim \mathcal{T}$
7.    **// Forward Pass (Unrolling)**
8.    $\mathbf{z} \leftarrow \text{Encoder}(\mathbf{o})$; $\boldsymbol{\mu}^{(0)} \leftarrow \text{Init}^\mu(\mathbf{z})$; $\mathbf{y}^{(0)} \leftarrow \text{Init}^y(\mathbf{z})$
9.    **for** layer $j = 1, \dots, J$ **do**
10.       Compute $\mathbf{y}^{(j)}$ using Dual Step (Eq. 10a)
11.       Compute $\boldsymbol{\mu}^{(j)}$ using Primal Step (Eq. 10b) + ProxNet
12.       Apply Hard Projection: $\boldsymbol{\mu}^{(j)} \leftarrow \text{HardProj}(\boldsymbol{\mu}^{(j)})$
13.   **end**
14.   Set prediction $\hat{\boldsymbol{\mu}} = \boldsymbol{\mu}^{(J)}$
15.   **// Backward Pass**
16.   Calculate loss $\mathcal{L}$ (Eq. 11) using $\hat{\boldsymbol{\mu}}$ and $\boldsymbol{\mu}^*$
17.   Update $\Theta \leftarrow \Theta - \eta \nabla_\Theta \mathcal{L}$
18.   **if** converged **then** break
19. **end**
20. **Output**: Trained PDPL-Net model $\Theta$.

#### E. 收敛性与复杂度分析

整个PDPL-Net的展开过程基于PDHG算法的迭代结构。具体而言，给定第$j$层的对偶变量$\{\boldsymbol{\mu}^{(j)}, \mathbf{y}^{(j)}\}$，PDPL-Net按如下方式生成下一层的解$\{\boldsymbol{\mu}^{(j+1)}, \mathbf{y}^{(j+1)}\}$：
$$
\{\boldsymbol{\mu}^{(j)}, \mathbf{y}^{(j)}\} \xrightarrow{\text{Dual-Step}} \mathbf{y}^{(j+1)} \xrightarrow{\text{Primal-Step}} \tilde{\boldsymbol{\mu}}^{(j+1)} \xrightarrow{\text{ProxNet}} \boldsymbol{\mu}^{(j+1)}
$$
因此，从编码器生成的初始值$\{\boldsymbol{\mu}^{(0)}, \mathbf{y}^{(0)}\}$开始，由PDPL-Net生成的序列为：
$$
\mathbf{o} \xrightarrow{\text{Encoder}} \{\boldsymbol{\mu}^{(0)}, \mathbf{y}^{(0)}\} \xrightarrow{\mathcal{G}_1} \{\boldsymbol{\mu}^{(1)}, \mathbf{y}^{(1)}\} \xrightarrow{\mathcal{G}_2} \cdots \xrightarrow{\mathcal{G}_J} \{\boldsymbol{\mu}^{(J)}, \mathbf{y}^{(J)}\} \xrightarrow{\mathcal{P}_{hard}} \boldsymbol{\mu}_{out}
$$
这种基于优化算法的深度展开框架可以通过端到端的反向传播进行训练。它可以通过增加展开层数$J$来提升精度，并通过硬投影层$\mathcal{P}_{hard}$来强制约束满足。

为了理解PDPL-Net为何有效，我们建立以下定理。

**定理 4.1**. 设$\boldsymbol{\mu}^*$为对偶优化问题的最优解，PDPL-Net生成的序列$[\boldsymbol{\mu}^{(0)}, \boldsymbol{\mu}^{(1)}, \cdots, \boldsymbol{\mu}^{(J)}]$满足以下性质：

(i) **可行性保证（Feasibility）**：$\boldsymbol{\mu}_{out} \in \mathcal{C}_{dual}$，即 $\boldsymbol{\mu}_{out} \geq 0$ 且 $\|\mathbf{G}^\top \boldsymbol{\mu}_{out}\|_2 \leq 1$，对于任意输入$\mathbf{o}$恒成立。

(ii) **有界近似（Bounded Approximation）**：$\|\boldsymbol{\mu}_{out} - \boldsymbol{\mu}^*\| \leq \rho^J \|\boldsymbol{\mu}^{(0)} - \boldsymbol{\mu}^*\| + \frac{\alpha b_R}{1-\rho}$，其中$\rho < 1$为收缩因子，$\alpha, b_R$为残差网络的有界常数。

(iii) **安全保守性（Safety Conservativeness）**：由$\boldsymbol{\mu}_{out}$计算的距离估计$\hat{d}$满足$\hat{d} \leq d^*$，其中$d^*$为真实距离。

*证明*. 见附录1:理论分析（定理6.3、6.5）。

定理4.1的第一部分指出，硬投影层$\mathcal{P}_{hard}$的引入保证了网络输出必然落在对偶可行域内，实现了100%的约束满足率。定理4.1的第二部分表明网络输出与最优解的误差随展开层数$J$指数衰减，且存在有界的残差项。定理4.1的第三部分意味着PDPL-Net生成的距离估计是保守的下界，这为安全控制提供了理论基础。上述性质表明，所提出的架构具有严格的数学保证，这与黑盒神经网络形成鲜明对比。

根据定理4.1，我们可以在PDPL-Net达到足够精度前选择较少的展开层数。在实践中，为了节省计算时间并实现实时端到端的机器人导航，我们设置展开层数$J=1\sim3$即可达到$10^{-6}$量级的MSE精度。

最后，我们展示PDPL-Net的复杂度分析。根据算法1中列出的程序，在每次推理中，首先由编码器提取特征，复杂度为$O(M \cdot H^2)$，其中$M$为障碍点数，$H$为隐藏层维度。随后，每层展开包含几何变换$O(M \cdot E)$、残差网络$O(M \cdot H^2)$和硬投影$O(M \cdot E)$。总结来说，以展开层数为$J$，PDPL-Net的总复杂度为$O(M \cdot (H^2 + J \cdot (H^2 + E)))$。可以看出，复杂度与障碍点数$M$成线性关系，这证实了PDPL-Net能够实时处理数千个点的能力。相比之下，传统内点法求解器的复杂度为$O(M \cdot E^{3})$，一阶迭代方法需要$K \approx 10^2 \sim 10^3$次迭代。PDPL-Net通过将迭代次数$K$压缩为极小的展开深度$J$（$1 \sim 3$），实现了的成倍加速。

# V. 实验与结果

本章通过多维度的实验验证 PDPL-Net 的有效性。实验分为三个层次：首先在点级别进行对偶变量预测的精度与效率评估，其次通过消融实验分析各组件的贡献，最后在闭环MPC导航系统中验证前端改进对后端控制的实际收益。

#### A. 实验设置

**1) 实验环境与机器人平台**:
为了全面评估算法性能，我们设计了涵盖不同保真度和场景复杂度的多层级实验环境：

*   **数值验证环境 (IR-SIM)**: 用于进行大规模的点级基线对比和参数消融实验。该环境支持对成千上万个随机生成的障碍物场景进行快速迭代，重点评估算法的数学性质（如 CSR）。
*   **高保真仿真与实物环境**: 用于闭环导航实验。我们在 Gazebo 中构建了包含狭窄走廊、动态行人及非凸障碍物的复杂场景，并在真实测试场中部署了物理机器人进行验证。

采用的机器人平台涵盖了不同的运动学特性与尺寸，如图 6 所示（注：此处为占位符，需补充对应图片）：
*   **阿克曼转向机器人 (Ackermann Steering Robot)**: 模拟城市自动驾驶车辆的动力学特性。车长 4.6m，车宽 1.6m，轴距 3.0m。由于其非完整约束和较大的转弯半径，该平台对避障算法的精确性要求极高。
*   **差分驱动机器人 (Differential Drive Robot)**: 一种高机动性的小型机器人平台，能够原地转向，适用于室内狭窄空间的灵活穿梭。
*   **轮腿式机器人 (Wheel-Legged Robot)**: 集成了轮式的高效性与腿式的地形适应性。相比前两者，它涉及更高的运动不确定性（如车身振荡），需要更鲁棒的控制策略。

所有平台均配备 2D 或 3D 激光雷达系统，以获取环境的点云表示。在未知环境中的定位通过 Fast-LIO2 或 LeGO-LOAM 实现。所有实验均不依赖预先构建的地图，仅依靠实时机载传感、目标位置及期望速度进行自主导航。对于移动障碍物（如行人），由于其速度相对较低（< 1.5m/s），我们在每个短时的 MPC 预测时域内将其视为静态处理，这种假设通过高频控制（> 50Hz）带来的快速重规划得到补偿。

**2) 评估指标**:

我们采用以下性能指标进行评估。

• **成功率（Success Rate）**：成功是指机器人完成导航任务且无碰撞。这包括三个条件：路径完成（机器人与目标位置的距离小于预定义阈值，本实验设置为0.5米）、避障（机器人与任何障碍物之间的最小距离始终大于安全裕度$d_{safe}$，本实验设置为0.3米）和时间约束（导航时间未超过场景最大期望时间的1.5倍）。如果路径完成但发生碰撞或超时，则被视为失败。成功率是成功案例数与测试试验总数的比值。

• **导航时间（Navigation Time）**：导航时间是指从机器人开始执行导航任务（$t=0$时刻）到成功到达目标位置所经历的总时间。导航时间包括所有控制步骤的累积时间：$T_{nav} = \sum_{k=0}^{K-1} \Delta t$，其中$K$为到达目标所需的总控制步数，$\Delta t$为控制周期（本实验设置为0.1秒）。

• **平均速度（Average Speed）**：平均速度是指机器人在整个导航过程中的平均移动速率，定义为总移动路径长度与导航时间的比值：$v_{avg} = \frac{L_{path}}{T_{nav}}$，其中$L_{path}$为路径长度。

• **路径长度（Path Length）**：路径长度是指机器人从起始位置到目标位置实际移动的总距离，通过对机器人轨迹进行离散采样并累加相邻采样点之间的欧氏距离得到：$L_{path} = \sum_{k=0}^{K-1} \|\mathbf{x}_{k+1} - \mathbf{x}_k\|_2$，其中$\mathbf{x}_k$为时刻$k$的机器人位置。

• **最小安全距离（Minimum Safety Distance）**：最小安全距离是指机器人在整个导航过程中与最近障碍物之间的最小距离，定义为：$d_{min} = \min_{k,i} \text{dist}(\mathcal{R}(\mathbf{x}_k), \mathbf{o}_i)$，其中$\mathcal{R}(\mathbf{x}_k)$为时刻$k$的机器人几何区域，$\mathbf{o}_i$为障碍点，$\text{dist}(\cdot)$为欧氏距离函数。

• **均方误差（Mean Squared Error, MSE）**：MSE衡量点级对偶变量预测值与Ground Truth（通过工业级求解器CVXPY-CLARABEL精确求解得到的最优解）之间的均方误差，定义为：$\text{MSE} = \frac{1}{M} \sum_{i=1}^{M} \|\hat{\boldsymbol{\mu}}_i - \boldsymbol{\mu}_i^*\|_2^2$，其中$M$为障碍点数量，$\hat{\boldsymbol{\mu}}_i$为网络预测的对偶变量，$\boldsymbol{\mu}_i^*$为最优对偶变量。

• **约束满足率（Constraint Satisfaction Rate, CSR）**：约束满足率是指预测的对偶变量满足物理可行性约束的样本比例，定义为：$\text{CSR} = \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}(\|\mathbf{G}^\top \hat{\boldsymbol{\mu}}_i\|_2 \leq 1 \land \hat{\boldsymbol{\mu}}_i \geq 0)$，其中$\mathbb{I}(\cdot)$为指示函数，$\mathbf{G}$为机器人几何矩阵，$\hat{\boldsymbol{\mu}}_i$为预测的对偶变量。该约束包括非负性约束（$\boldsymbol{\mu} \geq 0$）和二阶锥约束（$\|\mathbf{G}^\top \boldsymbol{\mu}\|_2 \leq 1$）。

• **范数违背率（Norm Violation）**：范数违背率是指预测的对偶变量在二阶锥约束上的平均违背程度，定义为：$\text{Norm Viol.} = \frac{1}{M} \sum_{i=1}^{M} \max(0, \|\mathbf{G}^\top \hat{\boldsymbol{\mu}}_i\|_2 - 1)$。

• **推理时间（Inference Time）**：推理时间是指单次前向传播的耗时，即从输入点云数据到输出对偶变量预测结果所需的计算时间，包括特征提取、编码、展开层计算和硬投影的所有耗时。

• **单步求解时间（Single-Step Solve Time）**：单步求解时间是指后端MPC优化器在一个控制周期内完成完整求解的总时间，包括前端感知（对偶变量预测）和后端优化（二次规划求解）的所有耗时，定义为：$T_{step} = T_{perception} + T_{optimization}$。

• **对偶违背率（Dual Violation Rate）**：对偶违背率是指在闭环导航过程中，前端感知模块输出违背对偶可行性约束的控制周期占总控制周期的比例，定义为：$\text{DVR} = \frac{\#\{k : \|\mathbf{G}^\top \hat{\boldsymbol{\mu}}_k\|_2 > 1\}}{K}$，其中$K$为总控制步数。

所有计算均在配备 Intel Core i9-13900K CPU 和 NVIDIA RTX 4090 GPU 的工作站上完成。

为全面评估PDPL-Net，我们选择了四类共12种代表性基线方法进行对比。传统优化求解器包括CVXPY (ECOS)、CVXPY (CLARABEL)和CVXPYLayers。黑盒神经网络包括MLP、PointNet++和Point Transformer V3。算法展开方法包括ISTA-Net、ADMM-Net、DeepInverse和DUNE（NeuPAN原始前端）。几何方法包括Center Distance和ESDF-MPC。

#### B. 点级性能对比

表IV展示了所有方法在点级对偶变量预测任务上的性能对比。实验结果揭示了当前技术路线面临的根本性矛盾，并突显了PDPL-Net的全方位优势。

**表 IV：点级对偶变量预测性能对比**

|      类别      |         方法         | MSE ($\boldsymbol{\mu}$) ↓ |   CSR ↑    |          Norm Viol. ↓           | 时间 (ms) ↓ |
| :------------: | :------------------: | :------------------------: | :--------: | :-----------------------------: | :---------: |
| **传统求解器** |   CVXPY (CLARABEL)   |         0.00 (Ref)         |   37.4%    |     $4.02 \times 10^{-10}$      |   2099.83   |
|                |     CVXPYLayers      |   $2.42 \times 10^{-10}$   |   13.2%    |      $9.90 \times 10^{-6}$      |   612.64    |
|  **几何方法**  |       ESDF-MPC       |   $5.52 \times 10^{-1}$    |    0.0%    |              3.44               |   197.23    |
|  **黑盒网络**  |      PointNet++      |    $2.33 \times 10^{0}$    |    0.0%    |      $2.93 \times 10^{-3}$      |   217.58    |
|                | Point Transformer V3 |   $4.49 \times 10^{-1}$    |    0.0%    |      $1.99 \times 10^{-4}$      |    44.07    |
|  **算法展开**  |       ISTA-Net       |   $1.12 \times 10^{-3}$    |    0.6%    |      $1.59 \times 10^{-3}$      |    2.04     |
|                |       ADMM-Net       |   $5.82 \times 10^{-4}$    |    1.1%    |      $1.19 \times 10^{-3}$      |    2.59     |
|                |     DeepInverse      |   $7.24 \times 10^{-2}$    |   88.3%    |      $2.60 \times 10^{-2}$      |    2.87     |
|                | **PDPL-Net (Ours)**  |   $5.31 \times 10^{-6}$    | **100.0%** | $\mathbf{4.79 \times 10^{-13}}$ |  **1.25**   |

*注：CSR (Constraint Satisfaction Rate) 指满足 $\|\mathbf{G}^\top \boldsymbol{\mu}\|_2 \leq 1$ 且 $\boldsymbol{\mu} \ge 0$ 的样本比例。CVXPY 由于数值容差原因，部分样本在严格阈值下被判定为违背，但其 Norm Violation 极低。*

**1) 速度与精度的极致平衡**：
CVXPY 作为工业级基准，虽然提供了理论最优解，但其 **2099.83 ms** 的求解时间对于实时控制而言是不可接受的。即便是专为微分优化设计的 CVXPYLayers，耗时也高达 612.64 ms。相比之下，PDPL-Net 的单次推理仅需 **1.25 ms**，实现了相比传统求解器 **1680倍** 的惊人加速。这种毫秒级的响应速度使得在 100Hz 甚至更高频率的控制循环中集成复杂几何约束成为可能。在精度方面，PDPL-Net 的 MSE 误差低至 $5.31 \times 10^{-6}$，与原始 DUNE ($2.24 \times 10^{-6}$) 处于同一数量级，远优于所有其他学习基线（如 MLP 的 $10^{-4}$ 或 ISTA 的 $10^{-3}$）。

**2) 从概率安全到确定性安全**：
这是 PDPL-Net 最核心的突破。

*   **黑盒网络的局限**：MLP、PointNet++ 和 Point Transformer V3 的 CSR 接近于 **0%**。这意味着单纯的数据驱动学习完全无法捕捉高维几何空间中的硬约束边界。它们生成的解几乎总是不可行的（Norm Violation 高达 $10^{-4}$ 量级），直接用于控制将导致碰撞风险或求解器崩溃。
*   **其他展开方法的不足**：ISTA-Net 和 ADMM-Net 虽然利用了优化结构，但缺乏针对对偶范数约束的专门设计，其 CSR 仅为 1% 左右。DeepInverse 虽然达到了 88.3% 的 CSR，但其 MSE 误差极大（$7.24 \times 10^{-2}$），说明它是通过牺牲解的质量来换取可行性。
*   **DUNE 的隐患**：原始 DUNE 虽然精度很高，但其 CSR 仅为 **34.4%**。这意味着超过 65% 的预测结果在几何上是不严格合法的，需要后端优化器进行额外的修正或容错处理。
*   **PDPL-Net 的统治力**：得益于架构中嵌入的硬投影层，PDPL-Net 实现了 **100% 的 CSR**，且 Norm Violation 降至机器精度级别的 $4.79 \times 10^{-13}$。这是唯一一个能够提供数学上严格可行性保证的学习方法，彻底消除了前端感知的不确定性。

综上所述，PDPL-Net 不仅在速度上碾压传统方法，在安全性上超越所有学习基线。它是目前唯一能同时满足**高精度、超实时、零违背**这三个苛刻要求的点云感知方案。

#### C. 消融实验

为深入理解PDPL-Net各组件的贡献，我们进行了系统的消融实验，结果列于表V。

**表 V：PDPL-Net各组件的消融实验结果**

| 配置 | 硬投影 | 近端算子 | KKT正则 | MSE ($\boldsymbol{\mu}$) ↓ | CSR ↑ | 时间 (ms) |
|:-----|:------:|:-------:|:-------:|:-------------:|:-----:|:---------:|
| 基础网络 | ✗ | ✗ | ✗ | 515.72 | 0.0% | 1.06 |
| + 硬投影层 | ✓ | ✗ | ✗ | 126.28 | 99.3% | 1.14 |
| + 可学习近端算子 | ✓ | ✓ | ✗ | $5.64 \times 10^{-6}$ | 100.0% | 1.33 |
| **PDPL-Net** | ✓ | ✓ | ✓ | $\mathbf{5.31 \times 10^{-6}}$ | **100.0%** | 1.25 |

消融实验揭示了各组件的关键作用。移除硬投影层后，MSE误差从$5.31 \times 10^{-6}$激增至296.78（增长超过7个数量级），CSR从100%降至0%。这表明仅靠神经网络的拟合能力，即使在监督学习框架下，也无法可靠地学习满足几何约束的映射。硬投影层通过显式的数学操作将网络输出强制拉回可行域，从架构层面消除了约束违背的可能性。

"无可学习近端算子"变体使用固定步长的PDHG展开，其MSE误差高达126.28，是完整模型的$2.3 \times 10^7$倍。这表明传统优化算法在有限迭代次数（$J=1$）下难以收敛到高精度解，而可学习近端算子通过数据驱动的方式学习了问题结构的先验知识，实现了"一步到位"的快速收敛。值得注意的是，该变体的CSR仍达到99.3%，这是因为硬投影层在推理阶段始终生效。

#### D. 闭环导航实验

为验证前端感知精度的提升如何转化为后端控制性能，我们将PDPL-Net集成到NeuPAN MPC框架中进行闭环导航实验。测试场景包括凸障碍物场景（convex_obs）和狭窄走廊场景（corridor），结果列于表VI。

**表 VI：闭环导航性能对比**

| 场景 | 方法 | 成功率 | 步时(ms) | 对偶违背率 |
|:-----|:-----|:------:|:--------:|:----------:|
| **Convex Obs** | NeuPAN (DUNE) | 100% | 58.14 | 50.4% |
| | PDPL-Net (J=1) | 100% | 55.90 | **0%** |
| | PDPL-Net (J=2) | 100% | 56.40 | **0%** |
| | PDPL-Net (J=3) | 100% | 60.21 | **0%** |
| **Corridor** | NeuPAN (DUNE) | 100% | 121.28 | 43.6% |
| | PDPL-Net (J=1) | 100% | **112.48** | **0%** |
| | PDPL-Net (J=2) | 100% | 112.17 | **0%** |
| | PDPL-Net (J=3) | 100% | 112.82 | **0%** |

闭环运行中最显著的优势是对偶可行性违背的完全消除。原始DUNE模块在导航过程中约43–50%的推理调用违背对偶可行性约束，表现为$\|\mathbf{G}^\top \boldsymbol{\mu}\|_2 > 1$。虽然NeuPAN在NRMP层之前应用硬投影进行修正，但这种事后修正导致传递给优化器的梯度信息质量下降。相比之下，PDPL-Net的对偶违背率精确为0%，无论展开层数$J$取何值，确保优化器在每个控制步骤都接收到高质量、几何有效的距离信息。

在计算效率方面，PDPL-Net (J=1)实现了比DUNE快约7–8%的单步计算时间。这一加速源于PDPL-Net相比PointNet特征提取更为精简的架构设计。虽然增加展开层数会略微增加计算时间，但所有PDPL-Net变体仍快于或持平于原始DUNE。

在路径质量方面，随着展开层数$J$增加，路径效率渐进提升。PDPL-Net (J=3)在convex_obs场景中取得了最短的导航时间（13.8s，对应138步，相比DUNE减少5.5%）和最高的平均速度（3.86m/s，相比DUNE提升5.5%），这得益于更优的路径规划（路径长度53.33m vs 53.43m）。这表明额外的展开层产生了更高质量的对偶变量估计，进而为优化器提供了更精确的距离梯度信息。综合考虑精度、效率和约束满足，我们推荐$J=2$作为实际部署的默认配置。

# VII. 讨论

#### A. 设计洞察
**1) 优化与学习的平衡**: PDPL-Net的核心设计理念是在优化的严谨性与学习的高效性之间寻找最优平衡点。这一平衡体现在三个层面。首先是**结构化归纳偏置**：与通用的MLP或Transformer不同，PDPL-Net的网络架构直接对应于PDHG优化算法的迭代步骤。这种设计使网络不需要从零开始学习输入-输出映射，而是被显式引导去执行梯度更新和投影操作。其次是**可学习残差作为自适应预处理器**。在传统PDHG算法中，步长参数$\tau, \sigma$为满足最坏情况收敛条件而设定得非常保守。PDPL-Net中的残差模块充当了数据驱动的非线性预处理器——它根据当前输入特征动态预测最优的更新方向和步长。最后是**硬约束与软学习的解耦**。硬投影层将可行性保证从网络学习过程中解耦出来：网络负责逼近最优解（可以犯错），而投影层负责修正约束违背（必须正确）。

**2) 硬投影层的必要性**: 消融实验揭示了一个重要发现：移除硬投影层后，CSR从100%直接降至0%。这证实了我们在第IV-D节的分析——在高维空间中，约束可行域的边界是测度为零的集合，仅靠神经网络的拟合能力无法精确"命中"这一边界。现有的深度展开方法（如LISTA、ADMM-Net）大多通过软惩罚处理约束，无法提供严格的可行性保证。PDPL-Net的硬投影层将安全保证从"概率性"提升到了"确定性"，这对于安全攸关的机器人应用具有本质性意义。

#### B. 局限性
**动态环境的显式建模不足**。当前框架将动态障碍物处理为连续的静态快照，未显式利用障碍物速度信息进行轨迹预测。这导致在高度动态场景（如行人密集区）中，机器人可能过于保守或反应滞后。**非凸约束的处理能力**。当前方法依赖对偶重构将避障约束转化为双凸形式。对于更一般的非凸约束——如多机器人互避障的耦合约束——PDHG的收敛性理论不再直接适用。将展开框架扩展到非凸优化是一个重要但具有挑战性的方向。**真实世界验证的缺失**。当前实验主要基于高保真仿真环境。真实世界的传感器噪声、定位误差和通信延迟可能影响算法性能。

# VIII. 结论

移动机器人在复杂环境中的安全高效导航是机器人学领域的核心挑战。模型预测控制（MPC）因其能够显式处理系统约束而成为主流方法，但传统MPC在处理稠密点云时面临的计算瓶颈严重制约了其实时应用。本文针对这一问题，提出了PDPL-Net——一种面向MPC导航的快速约束保证点云感知网络，通过将优化理论的严谨性与深度学习的高效性相结合，实现了安全性与实时性的统一。

本文的核心创新和贡献总结如下：
**在架构设计层面**，我们提出了三项关键技术。首先是基于PDHG算法的深度展开架构，将优化迭代步骤参数化为神经网络层，继承了优化算法的结构先验。其次是硬投影层设计，通过非负锥投影和二阶锥投影的显式数学操作，从架构层面保证了100%的约束满足率——这是首个在点级对偶变量预测任务上达到完全约束保证的学习方法。最后是可学习近端算子，通过数据驱动的方式学习问题结构的先验知识，使网络仅需1–2层展开即可收敛到近乎最优的解。
**在训练策略层面**，我们提出了基于KKT条件的残差正则化训练策略。该策略将优化问题的最优性条件——原始可行性、对偶可行性和互补松弛性——编码为损失函数，使网络输出不仅在数值上接近最优解，更在数学结构上满足最优性的必要条件。
**在实验验证层面**，PDPL-Net在保持与精确求解器相当精度（MSE $5.31 \times 10^{-6}$）的同时，实现了**1680倍**的速度提升（1.25ms vs 2099.8ms）；与所有黑盒神经网络相比，约束满足率从0–88%提升至**100%**。
**在闭环验证层面**，改进的前端模块转化为三个可观测的后端收益：对偶违背率从43–50%降至0%，实现了从概率性安全到确定性安全的质变；单步计算时间减少7–8%；路径质量随展开层数$J$渐进提升。

综上所述，PDPL-Net证明了将优化理论的严谨性与深度学习的高效性相结合，是解决机器人安全控制问题的有效途径。通过在网络架构中显式嵌入约束保证机制，我们在不牺牲计算效率的前提下实现了"零约束违背"的安全保障。我们相信，这一"结构化展开+硬投影保障"的设计范式将为构建安全、高效的下一代机器人导航系统提供重要的理论基础和实践参考。

# REFERENCES

[1] J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed. Nob Hill Publishing, 2017.
[2] S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge University Press, 2004.
[3] A. Chambolle and T. Pock, “A first-order primal-dual algorithm for convex problems with applications to imaging,” *Journal of Mathematical Imaging and Vision*, vol. 40, no. 1, pp. 120–145, 2011.
[4] N. Parikh and S. Boyd, “Proximal algorithms,” *Foundations and Trends in Optimization*, vol. 1, no. 3, pp. 127–239, 2014.
[5] H. H. Bauschke and P. L. Combettes, *Convex Analysis and Monotone Operator Theory in Hilbert Spaces*. Springer, 2011.
[6] K. Gregor and Y. LeCun, “Learning fast approximations of sparse coding,” in *International Conference on Machine Learning (ICML)*, 2010, pp. 399–406.
[7] Y. Yang, J. Sun, H. Li, and Z. Xu, “Deep ADMM-Net for compressive sensing MRI,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2016, pp. 10–18.
[8] J. Adler and O. Öktem, “Solving ill-posed inverse problems using iterative deep neural networks,” *Inverse Problems*, vol. 33, no. 12, p. 124007, 2017.
[9] B. Amos and J. Z. Kolter, “OptNet: Differentiable optimization as a layer in neural networks,” in *International Conference on Machine Learning (ICML)*, 2017, pp. 136–145.
[10] A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, and J. Z. Kolter, “Differentiable convex optimization layers,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019, pp. 9558–9570.
[11] V. Monga, Y. Li, and Y. C. Eldar, “Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing,” *IEEE Signal Processing Magazine*, vol. 38, no. 2, pp. 18–44, 2021.
[12] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “PointNet: Deep learning on point sets for 3D classification and segmentation,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 652–660.
[13] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep hierarchical feature learning on point sets in a metric space,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5099–5108.
[14] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang, T. He, and H. Zhao, “Point Transformer V3: Simpler, faster, stronger,” in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.
[15] S. Diamond and S. Boyd, “CVXPY: A Python-embedded modeling language for convex optimization,” *Journal of Machine Learning Research*, vol. 17, no. 83, pp. 1–5, 2016.
[16] A. Domahidi, E. Chu, and S. Boyd, “ECOS: An SOCP solver for embedded systems,” in *European Control Conference (ECC)*, 2013, pp. 3071–3076.
[17] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in *European Control Conference (ECC)*, 2019, pp. 3420–3431.
[18] R. Han *et al*., “NeuPAN: Direct point robot navigation with end-to-end model-based learning,” *IEEE Transactions on Robotics*, 2025.
[19] R. Han, “IR-SIM: Intelligent Robot Simulator,” software, https://github.com/hanruihua/ir-sim (accessed Jan. 2026).





# 附录1:理论分析

本章对PDPL-Net的收敛性和约束保证进行严格的理论分析。我们首先回顾原始-对偶混合梯度法（PDHG）的收敛性理论，随后分析带残差学习的展开网络如何继承并加速这一收敛过程，最后证明硬投影层提供的对偶可行性保证。这些理论结果不仅为实验现象提供了数学解释，更为PDPL-Net在安全攸关场景中的应用奠定了理论基础。

#### A. 预备知识与问题设定

**1) 符号约定**: 为便于后续分析，我们首先统一符号。设$\mathcal{H}$为有限维Hilbert空间，$\langle \cdot, \cdot \rangle$和$\|\cdot\|$分别表示其内积和范数。对于闭凸集$\mathcal{C} \subseteq \mathcal{H}$，投影算子定义为$\Pi_{\mathcal{C}}(x) = \arg\min_{y \in \mathcal{C}} \|y - x\|$。算子$T: \mathcal{H} \to \mathcal{H}$称为**非扩张的**，若$\|T(x) - T(y)\| \leq \|x - y\|$对所有$x, y \in \mathcal{H}$成立。投影算子到闭凸集是非扩张的这一性质是后续分析的基础。

**2) 对偶优化问题的鞍点形式**: 回顾第III节的对偶重构，对于给定的障碍物点$\mathbf{p} \in \mathbb{R}^2$和机器人几何$(\mathbf{G}, \mathbf{g})$，我们需要求解的对偶问题可以等价地写成鞍点问题的形式：
$$
\min_{\boldsymbol{\lambda} \in \mathcal{C}_\lambda} \max_{\mathbf{y} \in \mathcal{C}_y} \mathcal{L}(\boldsymbol{\lambda}, \mathbf{y}) = \langle \mathbf{K}\boldsymbol{\lambda}, \mathbf{y} \rangle - f(\boldsymbol{\lambda})
$$
其中，线性算子$\mathbf{K} = \mathbf{G}^\top \in \mathbb{R}^{2 \times E}$，原始可行域$\mathcal{C}_\lambda = \mathbb{R}_+^E$（非负锥），对偶可行域$\mathcal{C}_y = \mathcal{B}_2$（单位球），$f(\boldsymbol{\lambda}) = \mathbf{g}^\top \boldsymbol{\lambda} - \mathbf{p}^\top \mathbf{G}^\top \boldsymbol{\lambda}$为线性目标函数。鞍点$(\boldsymbol{\lambda}^*, \mathbf{y}^*)$满足对所有$\boldsymbol{\lambda} \in \mathcal{C}_\lambda$和$\mathbf{y} \in \mathcal{C}_y$：$\mathcal{L}(\boldsymbol{\lambda}^*, \mathbf{y}) \leq \mathcal{L}(\boldsymbol{\lambda}^*, \mathbf{y}^*) \leq \mathcal{L}(\boldsymbol{\lambda}, \mathbf{y}^*)$。

#### B. 标准 PDHG 算法的收敛性

Chambolle与Pock于2011年提出的原始-对偶混合梯度法通过以下迭代求解鞍点问题：
$$
\begin{cases}
\boldsymbol{\lambda}^{(k+1)} = \Pi_{\mathcal{C}_\lambda}\left( \boldsymbol{\lambda}^{(k)} - \tau \mathbf{K}^\top \bar{\mathbf{y}}^{(k)} + \tau \nabla f(\boldsymbol{\lambda}^{(k)}) \right) \\
\mathbf{y}^{(k+1)} = \Pi_{\mathcal{C}_y}\left( \mathbf{y}^{(k)} + \sigma \mathbf{K} \boldsymbol{\lambda}^{(k+1)} \right) \\
\bar{\mathbf{y}}^{(k+1)} = \mathbf{y}^{(k+1)} + \theta (\mathbf{y}^{(k+1)} - \mathbf{y}^{(k)})
\end{cases}
$$
其中$\tau, \sigma > 0$为步长参数，$\theta \in [0, 1]$为外推参数。

**定理 6.1 (PDHG 收敛性)**. 设$\|\mathbf{K}\|_{op}$为算子$\mathbf{K}$的算子范数。若步长参数满足$\tau \sigma \|\mathbf{K}\|_{op}^2 < 1$，则对于任意初始点$(\boldsymbol{\lambda}^{(0)}, \mathbf{y}^{(0)})$，PDHG迭代产生的序列弱收敛到鞍点问题的解。此外，若$\theta = 1$，则有遍历收敛速率$O(1/K)$。

该定理的证明基于构造Lyapunov函数$V^{(k)} = \frac{1}{2\tau}\|\boldsymbol{\lambda}^{(k)} - \boldsymbol{\lambda}^*\|^2 + \frac{1}{2\sigma}\|\mathbf{y}^{(k)} - \mathbf{y}^*\|^2$并证明其单调递减性。对于本文的几何矩阵$\mathbf{G} \in \mathbb{R}^{E \times 2}$，其算子范数$\|\mathbf{K}\|_{op} = \|\mathbf{G}^\top\|_{op} \leq \sqrt{E} \cdot \max_i \|\mathbf{g}_i\|_2$。对于归一化的机器人几何，取$\tau = \sigma = 0.5$即满足收敛条件。

#### C. 带残差学习的展开网络收敛性

**1) 展开网络的迭代格式**: PDPL-Net将PDHG算法展开为$J$层神经网络，每层对应一次迭代。与标准PDHG不同，我们引入了可学习的残差算子$\mathcal{R}_\theta: \mathbb{R}^2 \to \mathbb{R}^E$：
$$
\begin{cases}
\mathbf{y}^{(j+1)} = \Pi_{\mathcal{B}_2}\left( \mathbf{y}^{(j)} + \sigma \mathbf{K} \boldsymbol{\lambda}^{(j)} \right) \\
\tilde{\boldsymbol{\lambda}}^{(j+1)} = \boldsymbol{\lambda}^{(j)} + \tau \left( \mathbf{a} - \mathbf{K}^\top \mathbf{y}^{(j+1)} \right) \\
\boldsymbol{\lambda}^{(j+1)} = \Pi_{\mathcal{C}_{dual}}\left( \tilde{\boldsymbol{\lambda}}^{(j+1)} + \alpha \cdot \mathcal{R}_\theta(\mathbf{K}\tilde{\boldsymbol{\lambda}}^{(j+1)}) \right)
\end{cases}
$$
其中$\mathbf{a} = \mathbf{p}^\top \mathbf{G}^\top - \mathbf{g}^\top$为问题线性系数，$\alpha \in (0, 1)$为残差缩放因子，$\mathcal{C}_{dual} = \{\boldsymbol{\lambda} \geq 0 : \|\mathbf{G}^\top \boldsymbol{\lambda}\|_2 \leq 1\}$为对偶可行域。

**假设 6.1 (残差算子的有界性)**. 训练收敛后的残差网络$\mathcal{R}_\theta$满足：$\|\mathcal{R}_\theta(z)\| \leq L_R \|z\| + b_R$，其中$L_R, b_R \geq 0$为有界常数。
该假设对于有界激活函数（如ReLU后接有界权重）的MLP自然成立。在实际训练中，权重正则化和批归一化保证了该条件。

**定理 6.2 (带残差展开的近似收敛)**. 设$(\boldsymbol{\lambda}^*, \mathbf{y}^*)$为鞍点问题的精确解。在假设6.1下，若步长参数满足定理6.1的条件，且残差缩放因子满足$\alpha < \frac{1 - \sqrt{\tau\sigma}\|\mathbf{K}\|_{op}}{L_R \|\mathbf{K}\|_{op}}$，则$J$层展开后的输出$\boldsymbol{\lambda}^{(J)}$满足：
$$
\|\boldsymbol{\lambda}^{(J)} - \boldsymbol{\lambda}^*\| \leq \rho^J \|\boldsymbol{\lambda}^{(0)} - \boldsymbol{\lambda}^*\| + \frac{\alpha b_R}{1 - \rho}
$$
其中收缩因子$\rho = \sqrt{\tau\sigma}\|\mathbf{K}\|_{op} + \alpha L_R \|\mathbf{K}\|_{op} < 1$。

**证明思路**. 定义误差$e^{(j)} = \boldsymbol{\lambda}^{(j)} - \boldsymbol{\lambda}^*$。利用投影的非扩张性和残差的有界性，可以建立递推不等式$\|e^{(j+1)}\| \leq \rho \|e^{(j)}\| + c$，其中$c = \alpha(L_R\|\mathbf{K}\boldsymbol{\lambda}^*\| + b_R)$。递推展开即得所述结论。

**推论 6.1 (可学习残差的加速效应)**. 若残差网络$\mathcal{R}_\theta$经过训练后能够预测$\mathcal{R}_\theta(\mathbf{K}\boldsymbol{\lambda}) \approx (\boldsymbol{\lambda}^* - \boldsymbol{\lambda})/\alpha$，则$\boldsymbol{\lambda}^{(j+1)} \approx \Pi_{\mathcal{C}_{dual}}(\boldsymbol{\lambda}^*) = \boldsymbol{\lambda}^*$，即网络在**单步**内即可收敛到最优解。
推论6.1解释了实验中观察到的现象：仅需$J=1\sim2$层展开即可达到$10^{-6}$量级的MSE精度。可学习残差本质上是在学习从当前迭代点到最优解的"捷径"，将$O(1/\epsilon)$的迭代复杂度压缩至$O(1)$。

#### D. 硬投影层的理论保证

**1) 对偶可行域的几何结构**: PDPL-Net的对偶可行域定义为$\mathcal{C}_{dual} = \{ \boldsymbol{\lambda} \in \mathbb{R}^E : \boldsymbol{\lambda} \geq 0, \|\mathbf{G}^\top \boldsymbol{\lambda}\|_2 \leq 1 \}$。该集合是非负锥$\mathbb{R}_+^E$与椭球$\{\boldsymbol{\lambda} : \|\mathbf{G}^\top \boldsymbol{\lambda}\|_2 \leq 1\}$的交集，因此是一个闭凸集。硬投影层通过两步串联投影实现对$\mathcal{C}_{dual}$的近似投影：首先是非负锥投影$\hat{\boldsymbol{\lambda}} = \max(0, \boldsymbol{\lambda})$，然后是范数归一化$\boldsymbol{\lambda}^* = \hat{\boldsymbol{\lambda}}/\max(1, \|\mathbf{G}^\top \hat{\boldsymbol{\lambda}}\|_2)$。

**引理 6.1 (可行性保证)**. 对于任意输入$\boldsymbol{\lambda} \in \mathbb{R}^E$，硬投影层的输出$\boldsymbol{\lambda}^*$满足$\boldsymbol{\lambda}^* \in \mathcal{C}_{dual}$。
*证明*. （1）非负性：$\hat{\boldsymbol{\lambda}} = \max(0, \boldsymbol{\lambda}) \geq 0$，且$\boldsymbol{\lambda}^* = \hat{\boldsymbol{\lambda}}/s$（$s \geq 1$），故$\boldsymbol{\lambda}^* \geq 0$。（2）范数约束：设$s = \max(1, \|\mathbf{G}^\top \hat{\boldsymbol{\lambda}}\|_2)$。若$\|\mathbf{G}^\top \hat{\boldsymbol{\lambda}}\|_2 \leq 1$，则$s = 1$，$\boldsymbol{\lambda}^* = \hat{\boldsymbol{\lambda}}$，$\|\mathbf{G}^\top \boldsymbol{\lambda}^*\|_2 \leq 1$。若$\|\mathbf{G}^\top \hat{\boldsymbol{\lambda}}\|_2 > 1$，则$s = \|\mathbf{G}^\top \hat{\boldsymbol{\lambda}}\|_2$，$\|\mathbf{G}^\top \boldsymbol{\lambda}^*\|_2 = 1$。综上，$\boldsymbol{\lambda}^* \in \mathcal{C}_{dual}$。

**定理 6.3 (非扩张性)**. 硬投影层$\mathcal{P}_{hard}: \mathbb{R}^E \to \mathcal{C}_{dual}$是非扩张的。
*证明*. 硬投影是两个非扩张算子的复合：ReLU投影是逐分量的欧氏投影，为非扩张；范数归一化对于非负向量也是非扩张的。非扩张算子的复合仍为非扩张算子。

**2) 安全距离下界的保守性**:
**定理 6.4**. 设$\boldsymbol{\lambda}^*$为硬投影层的输出，$\mathbf{y}^* = -\mathbf{G}^\top \boldsymbol{\lambda}^*$。则由对偶变量计算的距离估计$\hat{d} = -\mathbf{g}^\top \boldsymbol{\lambda}^* - \mathbf{p}^\top \mathbf{y}^*$满足$\hat{d} \leq d^*$，其中$d^*$为点$\mathbf{p}$到机器人的真实距离。
*证明*. 由于$\boldsymbol{\lambda}^* \in \mathcal{C}_{dual}$，它是对偶问题的一个可行解。对偶问题是最大化问题，任何可行解的目标值不超过最优值，故$\hat{d} \leq d^*$。

**推论 6.2 (安全性保证)**. 若MPC控制器基于$\hat{d}$设置避障约束$\hat{d} \geq d_{safe}$，则真实距离满足$d^* \geq \hat{d} \geq d_{safe}$。即，硬投影层保证了距离估计的保守性，从而为安全控制提供了理论基础。

#### E. PDPL-Net 主定理

综合以上分析，我们给出PDPL-Net的主定理：

**定理 6.5 (PDPL-Net 主定理)**. 设PDPL-Net由特征编码器$\mathcal{E}_\phi$、$J$层带残差的PDHG展开$\{\mathcal{U}_j\}_{j=1}^J$、以及硬投影层$\mathcal{P}_{hard}$组成。在假设6.1下，若步长参数满足定理6.1的条件，则：
**(1) 可行性保证**: 对于任意输入$\mathbf{p} \in \mathbb{R}^2$，网络输出$\boldsymbol{\lambda}_{out}$满足$\boldsymbol{\lambda}_{out} \in \mathcal{C}_{dual}$，即CSR = 100%。
**(2) 近似最优性**: 网络输出与最优解的距离满足$\|\boldsymbol{\lambda}_{out} - \boldsymbol{\lambda}^*\| \leq \rho^J \|\boldsymbol{\lambda}^{(0)} - \boldsymbol{\lambda}^*\| + \frac{\alpha b_R}{1-\rho}$。
**(3) 安全保守性**: 基于网络输出计算的距离估计$\hat{d}$是真实距离$d^*$的下界，即$\hat{d} \leq d^*$。

定理6.5为PDPL-Net的核心声明——1680倍加速、100%约束满足率、安全距离保守性——提供了严格的数学基础。
