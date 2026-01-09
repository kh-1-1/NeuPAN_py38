# 2. 问题建模

本章建立移动机器人导航的数学模型框架。我们首先给出机器人的运动学描述，随后深入分析传统基于距离的避障约束的非凸性本质，并引出基于对偶变量的凸重构形式。这种重构不仅是数学上的技巧，更是本文能够实现快速且严格约束求解的理论基石。最后，我们将问题纳入模型预测控制（MPC）框架，并讨论传统方法面临的计算瓶颈。需要强调的是：在系统/算法层面，我们沿用NeuPAN的MPC/NRMP整体框架，仅替换其深度展开神经编码器（DUNE）为PDPL-Net；本文的问题建模与NeuPAN在“问题陈述（Problem Statement）”中的核心思想一致，差异主要来自符号选择与展开程度（本文进一步给出Ackermann运动学线性化、对偶形式及其在MPC中的嵌入），因此呈现形式会与NeuPAN的写法略有不同。

## 2.1 移动机器人运动学模型

### 2.1.1 阿克曼转向模型

本文考虑在二维平面上运动的非完整约束轮式机器人。以阿克曼转向（Ackermann Steering）模型为例，这是乘用车和许多服务机器人采用的运动学模型。该模型通过前轮转向角控制转弯半径，后轮作为驱动轮，从而避免了侧滑并提供了良好的运动学特性。连续时间下的状态方程可表示为：

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) = \begin{bmatrix} v \cos \theta \\ v \sin \theta \\ \frac{v \tan \delta}{L} \\ a \\ \omega \end{bmatrix}
$$

其中，状态向量$\mathbf{x} = [x, y, \theta, v, \delta]^\top$包含机器人在世界坐标系下的位置$(x, y)$、航向角$\theta$、线速度$v$和前轮转向角$\delta$。控制输入$\mathbf{u} = [a, \omega]^\top$分别表示线加速度和转向角速度。参数$L$为车辆轴距，即前后轮轴之间的距离。这一运动学模型准确描述了阿克曼转向车辆的运动约束：车辆只能沿其车身方向移动（非完整约束），且转弯半径由转向角和轴距共同决定。

### 2.1.2 线性化与离散化

为了适应数字控制系统的需求并利用成熟的凸优化工具，我们需要对上述非线性系统进行离散化与线性化处理。给定采样周期$\Delta t$，对连续时间模型进行前向欧拉离散化可得：

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)\Delta t
$$

在参考轨迹点$(\mathbf{x}_k^{ref}, \mathbf{u}_k^{ref})$处进行一阶泰勒展开，可得到线性时变（Linear Time-Varying, LTV）模型：

$$
\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{c}_k
$$

其中，$\mathbf{A}_k = \mathbf{I} + \Delta t \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\big|_{(\mathbf{x}_k^{ref}, \mathbf{u}_k^{ref})}$为状态转移矩阵，$\mathbf{B}_k = \Delta t \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{u}}\big|_{(\mathbf{x}_k^{ref}, \mathbf{u}_k^{ref})}$为控制输入矩阵，$\mathbf{c}_k$为线性化产生的常数项，$\Delta t$为离散采样时间间隔。这种线性化处理使得我们能够利用成熟的凸优化求解器（如OSQP、ECOS、CLARABEL）来高效处理动力学约束。然而，需要注意的是线性化引入了近似误差，这种误差在曲率变化剧烈的轨迹段会更加显著。在MPC框架中，由于每个控制周期都会根据实际状态重新线性化并求解优化问题，线性化误差能够得到有效抑制。

### 2.1.3 控制约束与可行域

考虑执行器饱和与控制平滑性，控制输入$\mathbf{u}_k$通常需要同时满足幅值约束与变化率约束（逐元素不等式）：

$$
\mathbf{u}_{min} \preceq \mathbf{u}_k \preceq \mathbf{u}_{max}, \qquad \Delta \mathbf{u}_{min} \preceq \mathbf{u}_{k+1} - \mathbf{u}_k \preceq \Delta \mathbf{u}_{max}
$$

在MPC框架下，初始状态$\mathbf{x}_0$由里程计或定位系统提供，随后在滚动时域内迭代求解受上述动力学与控制约束限制的最优控制序列。

## 2.2 点级碰撞避障约束

### 2.2.1 机器人几何表示

精确的碰撞检测需要对机器人的几何形状进行数学建模。本文将机器人在车体坐标系下近似为凸多边形（如矩形），这是实际应用中最常见的情形。设车体坐标系下的紧致凸集为：

$$
\mathcal{C} = \{ \mathbf{c} \in \mathbb{R}^2 \mid \mathbf{G}\mathbf{c} \leq \mathbf{g} \}
$$

其中，$\mathbf{G} \in \mathbb{R}^{E \times 2}$为边界法向量矩阵，$\mathbf{g} \in \mathbb{R}^E$为边界偏移向量，$E$为多边形的边数（对于矩形机器人，$E=4$）。矩阵$\mathbf{G}$的第$i$行记为$\mathbf{G}_i^\top$，表示第$i$条边的外法向量；$g_i$表示该边到车体坐标系原点的有符号距离。

给定状态$\mathbf{x} = [x, y, \theta, v, \delta]^\top$，机器人在世界坐标系下的占据空间可写为SE(2)刚体变换下的像：

$$
\mathcal{R}(\mathbf{x}) = \{ \mathbf{R}(\theta)\mathbf{c} + \mathbf{t}(\mathbf{x}) \mid \mathbf{c} \in \mathcal{C} \}
$$

其中旋转与平移为

$$
\mathbf{R}(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}, \qquad \mathbf{t}(\mathbf{x})=\begin{bmatrix}x\\y\end{bmatrix}.
$$

等价地，$\mathcal{R}(\mathbf{x})$也可以写成随姿态变化的H-表示形式：

$$
\mathcal{R}(\mathbf{x}) = \{ \mathbf{z} \in \mathbb{R}^2 \mid \mathbf{G}(\mathbf{x}) \mathbf{z} \leq \mathbf{g}(\mathbf{x}) \}, \quad
\mathbf{G}(\mathbf{x}) = \mathbf{G}\mathbf{R}(\theta)^\top,\quad
\mathbf{g}(\mathbf{x}) = \mathbf{g} + \mathbf{G}\mathbf{R}(\theta)^\top \mathbf{t}(\mathbf{x})
$$

这种表示方法的优点在于其简洁性和计算效率：判断一个点是否在多边形内只需进行$E$次线性不等式检验，复杂度为$O(E)$。

### 2.2.2 障碍物点云表示

在复杂的非结构化环境中，障碍物通常以离散点云$\mathcal{O} = \{ \mathbf{p}_i \}_{i=1}^N$的形式被激光雷达或深度相机感知。每个障碍物点$\mathbf{p}_i \in \mathbb{R}^2$表示环境中一个被占据的位置。为了保证安全，机器人占据的几何区域$\mathcal{R}(\mathbf{x})$与所有障碍物点必须保持足够的安全距离。这通常被建模为以下约束：

$$
\text{dist}(\mathcal{R}(\mathbf{x}), \mathbf{p}_i) \geq d_{safe}, \quad \forall i \in \{1, \dots, N\}
$$

其中，$\text{dist}(\cdot, \cdot)$表示点到凸多边形的欧几里得距离，$d_{safe}$为预设的安全距离阈值。然而，这种约束形式在优化视角下存在两个致命缺陷。首先是**非凸性**：距离函数$\text{dist}(\mathcal{R}(\mathbf{x}), \mathbf{p})$关于状态$\mathbf{x}$通常是非凸的——绕过障碍物可以选择左侧或右侧，这构成了非凸的可行域，导致优化问题存在多个局部最优解。其次是**梯度不连续**：当通过ESDF场或解析几何计算距离时，在多边形顶点和边界处会出现梯度跳变，严重影响基于梯度的优化算法的收敛稳定性。

### 2.2.3 基于对偶变量的双凸重构

为了克服上述困难，我们引入凸分析中的对偶原理对碰撞约束进行重构。根据支撑函数理论和强对偶定理，点$\mathbf{p}_i$到凸多边形$\mathcal{R}(\mathbf{x})$的距离可以等价地表示为以下对偶优化问题的最优值：

$$
d(\mathbf{x}, \mathbf{p}_i) = \max_{\mu, \lambda} \left( -\mathbf{g}(\mathbf{x})^\top \mu - \mathbf{p}_i^\top \lambda \right)
$$

$$
\text{s.t.} \quad \mathbf{G}(\mathbf{x})^\top \mu + \lambda = \mathbf{0}, \quad \| \lambda \|_2 \leq 1, \quad \mu \geq \mathbf{0}
$$

其中，$\mu \in \mathbb{R}^E$和$\lambda \in \mathbb{R}^2$是引入的对偶变量。$\mu$可以理解为各边界约束的拉格朗日乘子，而$\lambda = -\mathbf{G}(\mathbf{x})^\top \mu$对应从障碍物点指向机器人最近点的单位方向向量（与NeuPAN中最小平移向量$\mathbf{e}$的方向相反）。约束$\|\lambda\|_2 \leq 1$确保方向向量的范数有界，$\mu \geq 0$反映了拉格朗日乘子的非负性要求。

这种对偶重构带来了深刻的物理意义和计算优势。首先是**双凸性质（Biconvexity）**：该对偶形式在固定$(\mu, \lambda)$时关于$\mathbf{x}$是线性的，从而保持了MPC问题的凸性；而在固定$\mathbf{x}$时关于$(\mu, \lambda)$也是凸的。这使得我们可以通过交替优化（Alternating Minimization）来高效求解。其次是**可微性（Differentiability）**：最优对偶变量$\mu^*$直接提供了距离函数关于$\mathbf{g}(\mathbf{x})$的梯度信息，为MPC优化提供了高质量的一阶导数。最后是**并行性**：不同障碍物点的对偶问题相互独立，可以高效并行求解。因此，问题的核心转化为：**如何快速、准确地求解上述对偶问题，并获得满足约束的$\mu^*$？** 这正是本文PDPL-Net要解决的关键子问题。

## 2.3 MPC优化框架

### 2.3.1 滚动时域优化

基于上述运动学模型和对偶重构的避障约束，我们构建模型预测控制优化问题。MPC的核心思想是在每个控制周期求解一个有限时域的最优控制问题（Optimal Control Problem, OCP），执行第一个控制动作，然后在下一周期根据新的状态反馈重新规划。这种滚动时域策略能够有效处理模型不确定性和外界干扰，是当前机器人控制领域的主流方法。

设预测时域为$H$步，采样时间为$\Delta t$，在每个控制周期$t$，我们求解以下优化问题：

$$
\begin{aligned}
\min_{\mathbf{X}, \mathbf{U}, \boldsymbol{\mu}, \boldsymbol{\lambda}} \quad & \sum_{k=0}^{H-1} \left[ q_s \|\mathbf{x}_k - \mathbf{x}_k^{ref}\|^2 + p_u \|\mathbf{u}_k - \mathbf{u}_k^{nom}\|^2 \right] + q_N \|\mathbf{x}_H - \mathbf{x}_H^{ref}\|^2 \\
\text{s.t.} \quad & \mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{c}_k, \quad k = 0, \dots, H-1 \\
& \mathbf{u}_{min} \leq \mathbf{u}_k \leq \mathbf{u}_{max}, \quad k = 0, \dots, H-1 \\
& \Delta \mathbf{u}_{min} \leq \mathbf{u}_{k+1} - \mathbf{u}_k \leq \Delta \mathbf{u}_{max}, \quad k = 0, \dots, H-2 \\
& -\mathbf{g}(\mathbf{x}_k)^\top \mu_{i,k} - \mathbf{p}_i^\top \lambda_{i,k} \geq d_{safe}, \quad \forall i \in \mathcal{O}_{active}, \forall k \\
& \mathbf{G}(\mathbf{x}_k)^\top \mu_{i,k} + \lambda_{i,k} = \mathbf{0}, \quad \|\lambda_{i,k}\|_2 \leq 1, \quad \mu_{i,k} \geq \mathbf{0}, \quad \forall i \in \mathcal{O}_{active}, \forall k
\end{aligned}
$$

其中，$\mathbf{X} = [\mathbf{x}_0, \dots, \mathbf{x}_H]$和$\mathbf{U} = [\mathbf{u}_0, \dots, \mathbf{u}_{H-1}]$分别为状态和控制输入序列；$q_s$、$p_u$、$q_N$为代价函数权重；$\mathbf{x}^{ref}$为参考轨迹；$\mathbf{u}^{nom}$为标称控制输入；$\mathcal{O}_{active}$为当前激活的障碍物点集合（通过感知范围筛选）。

### 2.3.2 两层交替优化

上述MPC问题涉及状态$\mathbf{x}$和对偶变量$(\mu, \lambda)$的联合优化。由于双凸性质，可以采用两层交替优化策略求解。外层（NRMP层）固定对偶变量求解状态和控制序列的二次规划问题；内层（DUNE层）固定状态求解每个障碍物点对应的对偶变量。传统方法在内层采用凸优化求解器（如ECOS、CLARABEL）精确求解对偶问题，但这带来了显著的计算开销——单次MPC优化可能需要数百毫秒。

本文的核心创新在于用训练好的神经网络PDPL-Net替代内层的精确求解器。PDPL-Net直接将障碍物点映射到满足约束的对偶变量，将内层优化的时间复杂度从迭代求解的$O(K)$（$K$为求解器迭代次数）降低到神经网络推理的$O(1)$，同时通过架构设计保证约束的严格满足。

## 2.4 传统方法的计算瓶颈

传统MPC方法在处理点云避障时面临多重计算瓶颈。首先，ESDF地图的构建和维护需要$O(N \cdot M)$的计算复杂度和$O(M)$的存储空间，在大规模环境中成为性能瓶颈。其次，即使采用对偶重构避免了ESDF构建，精确求解每个障碍物点的对偶问题仍然耗时——使用CVXPY+ECOS求解器，单个点的求解时间约为0.2–0.3毫秒，对于包含数百个点的场景，内层优化可能需要数十至数百毫秒。最后，传统方法中的超参数（如代价权重$q_s$、$p_u$、安全距离$d_{safe}$等）需要针对不同场景手工调整，缺乏自适应能力。

这些计算瓶颈的根本原因在于传统方法将感知和优化视为独立模块，无法进行端到端的联合学习。PDPL-Net通过将优化算法展开为可训练的神经网络架构，打破了这一割裂，实现了感知与优化的深度融合。

