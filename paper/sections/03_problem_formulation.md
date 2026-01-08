# 3. 问题建模

本章将建立移动机器人导航的数学模型。我们首先给出机器人的运动学描述，随后深入分析传统基于距离的避障约束的非凸性本质，并引出基于对偶变量的凸重构形式。这种重构不仅是数学上的技巧，更是本文能够实现快速且严格约束求解的理论基石。

## 3.1 机器人运动学模型

本文考虑在二维平面上运动的非完整约束轮式机器人。以阿克曼转向（Ackermann Steering）模型为例，其连续时间状态方程为：

$$
\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t)) = \begin{bmatrix} v \cos \theta \\ v \sin \theta \\ \frac{v \tan \delta}{L} \\ a \\ \omega \end{bmatrix}
$$

其中，状态向量 $\mathbf{x} = [x, y, \theta, v, \delta]^\top$ 包含位姿及速度信息，控制输入 $\mathbf{u} = [a, \omega]^\top$ 为线加速度和转向角速度。$L$ 为车辆轴距。为了适应数字控制系统的需求，我们在工作点 $\mathbf{x}_k$ 处对其进行线性化和离散化（采样时间 $\Delta t$），得到线性时变（LTV）模型：

$$
\mathbf{x}_{k+1} = \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{c}_k
$$

这一线性化处理使得我们能够利用成熟的凸优化求解器（如OSQP, ECOS）来处理动力学约束，但环境几何约束的处理依然是难点。

## 3.2 避障约束的非凸性与对偶重构

### 3.2.1 原始距离约束的困境

在复杂的非结构化环境中，障碍物通常以离散点云 $\mathcal{O} = \{ \mathbf{p}_i \}_{i=1}^N$ 的形式被感知。为了保证安全，机器人占据的几何区域 $\mathcal{R}(\mathbf{x})$ 与障碍物点集必须互不相交。这通常被建模为距离约束：

$$
\text{dist}(\mathcal{R}(\mathbf{x}), \mathbf{p}_i) \geq d_{safe}, \quad \forall i \in \{1, \dots, N\}
$$

然而，该约束在优化视角下存在两个致命缺陷：
1.  **非凸性（Non-convexity）**：距离函数 $\text{dist}(\cdot)$ 关于状态 $\mathbf{x}$ 通常是非凸的（例如，绕过障碍物可以选择左侧或右侧，这构成了非凸的可行域）。
2.  **梯度不连续（Gradient Discontinuity）**：当通过ESDF场计算距离时，网格插值和边界处理会导致梯度信息的跳变，极大地影响了基于梯度的优化算法（如SQP, iLQR）的收敛稳定性。

### 3.2.2 基于对偶原理的凸重构

为了克服上述困难，我们引入凸分析中的对偶原理。假设机器人被近似为凸多边形（如矩形），其几何区域可以表示为一组线性不等式的交集：$\mathcal{R}(\mathbf{x}) = \{ \mathbf{z} \in \mathbb{R}^2 \mid \mathbf{G}(\mathbf{x}) \mathbf{z} \leq \mathbf{g}(\mathbf{x}) \}$。

根据Farkas引理及强对偶定理，点 $\mathbf{p}_i$ 到凸多边形 $\mathcal{R}(\mathbf{x})$ 的距离等价于以下对偶优化问题的最优值：

$$
d(\mathbf{x}, \mathbf{p}_i) = \max_{\mu, \lambda} \left( -\mathbf{g}(\mathbf{x})^\top \mu + \mathbf{p}_i^\top \lambda \right)
$$

$$
\text{s.t.} \quad \mathbf{G}(\mathbf{x})^\top \mu + \lambda = \mathbf{0}, \quad \| \lambda \|_2 \leq 1, \quad \mu \geq \mathbf{0}
$$

其中，$\mu \in \mathbb{R}^m$ 和 $\lambda \in \mathbb{R}^2$ 是引入的对偶变量。

**这种重构带来了深刻的物理意义和计算优势：**
1.  **双凸性质（Biconvexity）**：该对偶形式在固定 $\mu, \lambda$ 时关于 $\mathbf{x}$ 是线性的（从而保持了MPC问题的凸性），而在固定 $\mathbf{x}$ 时关于 $\mu, \lambda$ 也是凸的。这使得我们可以通过交替优化（Alternating Minimization）来高效求解。
2.  **可微性（Differentiability）**：最优对偶变量 $\mu^*$ 直接提供了距离函数关于 $\mathbf{g}(\mathbf{x})$ 的梯度（即灵敏度分析），这为MPC优化提供了高质量的一阶导数信息。

因此，问题的核心转化为：**如何快速、准确地求解上述对偶问题，并获得满足约束的 $\mu^*$？** 这正是本文PDPL-Net要解决的关键子问题。

## 3.3 模型预测控制 (MPC) 框架

基于上述重构，我们构建了基于对偶变量的MPC优化问题。在每一时刻，我们求解以下有限时域最优控制问题（OCP）：

$$
\begin{aligned}
\min_{\mathbf{X}, \mathbf{U}, \boldsymbol{\mu}, \boldsymbol{\lambda}} \quad & \sum_{k=0}^{H-1} \ell(\mathbf{x}_k, \mathbf{u}_k) + \ell_N(\mathbf{x}_H) \\
\text{s.t.} \quad & \mathbf{x}_{k+1} = \Phi(\mathbf{x}_k, \mathbf{u}_k), \\
& -\mathbf{g}(\mathbf{x}_k)^\top \mu_{i,k} + \mathbf{p}_i^\top \lambda_{i,k} \geq d_{safe}, \quad \forall i \in \mathcal{O}_{active} \\
& (\mu_{i,k}, \lambda_{i,k}) = \mathcal{F}_{\theta}(\mathbf{p}_i, \mathbf{x}_k) \quad \text{// 神经网络预测}
\end{aligned}
$$

其中，我们将对偶变量的求解过程抽象为映射 $\mathcal{F}_{\theta}$，即PDPL-Net。传统的NeuPAN方法采用两层交替优化，而本文通过训练PDPL-Net，旨在将内层优化（求解 $\mu, \lambda$）的时间复杂度从迭代求解的 $O(K)$ 降低到神经网络推理的 $O(1)$，同时保证约束的严格满足。
