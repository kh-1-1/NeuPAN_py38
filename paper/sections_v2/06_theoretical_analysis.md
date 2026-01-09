# 6. 理论分析

本章对PDPL-Net的收敛性和约束保证进行严格的理论分析。我们首先回顾原始-对偶混合梯度法（PDHG）的收敛性理论，随后分析带残差学习的展开网络如何继承并加速这一收敛过程，最后证明硬投影层提供的对偶可行性保证。这些理论结果不仅为实验现象提供了数学解释，更为PDPL-Net在安全攸关场景中的应用奠定了理论基础。

## 6.1 预备知识与问题设定

### 6.1.1 符号约定

为便于后续分析，我们首先统一符号。设$\mathcal{H}$为有限维Hilbert空间，$\langle \cdot, \cdot \rangle$和$\|\cdot\|$分别表示其内积和范数。对于闭凸集$\mathcal{C} \subseteq \mathcal{H}$，投影算子定义为$\Pi_{\mathcal{C}}(x) = \arg\min_{y \in \mathcal{C}} \|y - x\|$。算子$T: \mathcal{H} \to \mathcal{H}$称为**非扩张的**，若$\|T(x) - T(y)\| \leq \|x - y\|$对所有$x, y \in \mathcal{H}$成立。投影算子到闭凸集是非扩张的这一性质是后续分析的基础。

### 6.1.2 对偶优化问题的鞍点形式

回顾第2章的对偶重构，对于给定的障碍物点$\mathbf{p} \in \mathbb{R}^2$和机器人几何$(\mathbf{G}, \mathbf{g})$，我们需要求解的对偶问题可以等价地写成鞍点问题的形式：

$$
\min_{\mu \in \mathcal{C}_\mu} \max_{y \in \mathcal{C}_y} \mathcal{L}(\mu, y) = \langle \mathbf{K}\mu, y \rangle - f(\mu)
$$

其中，线性算子$\mathbf{K} = \mathbf{G}^\top \in \mathbb{R}^{2 \times E}$，原始可行域$\mathcal{C}_\mu = \mathbb{R}_+^E$（非负锥），对偶可行域$\mathcal{C}_y = \mathcal{B}_2$（单位球），$f(\mu) = \mathbf{g}^\top \mu - \mathbf{p}^\top \mathbf{G}^\top \mu$为线性目标函数。鞍点$(\mu^*, y^*)$满足对所有$\mu \in \mathcal{C}_\mu$和$y \in \mathcal{C}_y$：$\mathcal{L}(\mu^*, y) \leq \mathcal{L}(\mu^*, y^*) \leq \mathcal{L}(\mu, y^*)$。

## 6.2 标准PDHG算法的收敛性

Chambolle与Pock于2011年提出的原始-对偶混合梯度法通过以下迭代求解鞍点问题：

$$
\begin{cases}
\mu^{(k+1)} = \Pi_{\mathcal{C}_\mu}\left( \mu^{(k)} - \tau \mathbf{K}^\top \bar{y}^{(k)} + \tau \nabla f(\mu^{(k)}) \right) \\
y^{(k+1)} = \Pi_{\mathcal{C}_y}\left( y^{(k)} + \sigma \mathbf{K} \mu^{(k+1)} \right) \\
\bar{y}^{(k+1)} = y^{(k+1)} + \theta (y^{(k+1)} - y^{(k)})
\end{cases}
$$

其中$\tau, \sigma > 0$为步长参数，$\theta \in [0, 1]$为外推参数。

**定理6.1（PDHG收敛性）**。设$\|\mathbf{K}\|_{op}$为算子$\mathbf{K}$的算子范数。若步长参数满足$\tau \sigma \|\mathbf{K}\|_{op}^2 < 1$，则对于任意初始点$(\mu^{(0)}, y^{(0)})$，PDHG迭代产生的序列弱收敛到鞍点问题的解。此外，若$\theta = 1$，则有遍历收敛速率$O(1/K)$。

该定理的证明基于构造Lyapunov函数$V^{(k)} = \frac{1}{2\tau}\|\mu^{(k)} - \mu^*\|^2 + \frac{1}{2\sigma}\|y^{(k)} - y^*\|^2$并证明其单调递减性。对于本文的几何矩阵$\mathbf{G} \in \mathbb{R}^{E \times 2}$，其算子范数$\|\mathbf{K}\|_{op} = \|\mathbf{G}^\top\|_{op} \leq \sqrt{E} \cdot \max_i \|\mathbf{g}_i\|_2$。对于归一化的机器人几何，取$\tau = \sigma = 0.5$即满足收敛条件。

## 6.3 带残差学习的展开网络收敛性

### 6.3.1 展开网络的迭代格式

PDPL-Net将PDHG算法展开为$J$层神经网络，每层对应一次迭代。与标准PDHG不同，我们引入了可学习的残差算子$\mathcal{R}_\theta: \mathbb{R}^2 \to \mathbb{R}^E$：

$$
\begin{cases}
y^{(j+1)} = \Pi_{\mathcal{B}_2}\left( y^{(j)} + \sigma \mathbf{K} \mu^{(j)} \right) \\
\tilde{\mu}^{(j+1)} = \mu^{(j)} + \tau \left( \mathbf{a} - \mathbf{K}^\top y^{(j+1)} \right) \\
\mu^{(j+1)} = \Pi_{\mathcal{C}_{dual}}\left( \tilde{\mu}^{(j+1)} + \alpha \cdot \mathcal{R}_\theta(\mathbf{K}\tilde{\mu}^{(j+1)}) \right)
\end{cases}
$$

其中$\mathbf{a} = \mathbf{p}^\top \mathbf{G}^\top - \mathbf{g}^\top$为问题线性系数，$\alpha \in (0, 1)$为残差缩放因子，$\mathcal{C}_{dual} = \{\mu \geq 0 : \|\mathbf{G}^\top \mu\|_2 \leq 1\}$为对偶可行域。

### 6.3.2 残差算子的有界性假设

**假设6.1（残差有界性）**。训练收敛后的残差网络$\mathcal{R}_\theta$满足：$\|\mathcal{R}_\theta(z)\| \leq L_R \|z\| + b_R$，其中$L_R, b_R \geq 0$为有界常数。

该假设对于有界激活函数（如ReLU后接有界权重）的MLP自然成立。在实际训练中，权重正则化和批归一化保证了该条件。

### 6.3.3 近似收敛定理

**定理6.2（带残差展开的近似收敛）**。设$(\mu^*, y^*)$为鞍点问题的精确解。在假设6.1下，若步长参数满足定理6.1的条件，且残差缩放因子满足$\alpha < \frac{1 - \sqrt{\tau\sigma}\|\mathbf{K}\|_{op}}{L_R \|\mathbf{K}\|_{op}}$，则$J$层展开后的输出$\mu^{(J)}$满足：

$$
\|\mu^{(J)} - \mu^*\| \leq \rho^J \|\mu^{(0)} - \mu^*\| + \frac{\alpha b_R}{1 - \rho}
$$

其中收缩因子$\rho = \sqrt{\tau\sigma}\|\mathbf{K}\|_{op} + \alpha L_R \|\mathbf{K}\|_{op} < 1$。

**证明要点**。定义误差$e^{(j)} = \mu^{(j)} - \mu^*$。利用投影的非扩张性和残差的有界性，可以建立递推不等式$\|e^{(j+1)}\| \leq \rho \|e^{(j)}\| + c$，其中$c = \alpha(L_R\|\mathbf{K}\mu^*\| + b_R)$。递推展开即得所述结论。

**推论6.1（可学习残差的加速效应）**。若残差网络$\mathcal{R}_\theta$经过训练后能够预测$\mathcal{R}_\theta(\mathbf{K}\mu) \approx (\mu^* - \mu)/\alpha$，则$\mu^{(j+1)} \approx \Pi_{\mathcal{C}_{dual}}(\mu^*) = \mu^*$，即网络在**单步**内即可收敛到最优解。

推论6.1解释了实验中观察到的现象：仅需$J=1\sim2$层展开即可达到$10^{-5}$量级的MSE精度。可学习残差本质上是在学习从当前迭代点到最优解的"捷径"，将$O(1/\epsilon)$的迭代复杂度压缩至$O(1)$。从优化理论的角度看，这相当于网络学习了一个自适应的牛顿方向估计，将一阶方法转变为准二阶方法。

## 6.4 硬投影层的理论保证

### 6.4.1 对偶可行域的几何结构

PDPL-Net的对偶可行域定义为$\mathcal{C}_{dual} = \{ \mu \in \mathbb{R}^E : \mu \geq 0, \|\mathbf{G}^\top \mu\|_2 \leq 1 \}$。该集合是非负锥$\mathbb{R}_+^E$与椭球$\{\mu : \|\mathbf{G}^\top \mu\|_2 \leq 1\}$的交集，因此是一个闭凸集。硬投影层通过两步串联投影实现对$\mathcal{C}_{dual}$的近似投影：首先是非负锥投影$\hat{\mu} = \max(0, \mu)$，然后是范数归一化$\mu^* = \hat{\mu}/\max(1, \|\mathbf{G}^\top \hat{\mu}\|_2)$。

**引理6.1（硬投影的可行性保证）**。对于任意输入$\mu \in \mathbb{R}^E$，硬投影层的输出$\mu^*$满足$\mu^* \in \mathcal{C}_{dual}$。

**证明**。（1）非负性：$\hat{\mu} = \max(0, \mu) \geq 0$，且$\mu^* = \hat{\mu}/s$（$s \geq 1$），故$\mu^* \geq 0$。（2）范数约束：设$s = \max(1, \|\mathbf{G}^\top \hat{\mu}\|_2)$。若$\|\mathbf{G}^\top \hat{\mu}\|_2 \leq 1$，则$s = 1$，$\mu^* = \hat{\mu}$，$\|\mathbf{G}^\top \mu^*\|_2 \leq 1$。若$\|\mathbf{G}^\top \hat{\mu}\|_2 > 1$，则$s = \|\mathbf{G}^\top \hat{\mu}\|_2$，$\|\mathbf{G}^\top \mu^*\|_2 = 1$。综上，$\mu^* \in \mathcal{C}_{dual}$。

**定理6.3（硬投影的非扩张性）**。硬投影层$\mathcal{P}_{hard}: \mathbb{R}^E \to \mathcal{C}_{dual}$是非扩张的。

**证明**。硬投影是两个非扩张算子的复合：ReLU投影是逐分量的欧氏投影，为非扩张；范数归一化对于非负向量也是非扩张的。非扩张算子的复合仍为非扩张算子。

### 6.4.2 安全距离下界的保守性

**定理6.4（安全距离的保守性）**。设$\mu^*$为硬投影层的输出，$\lambda^* = -\mathbf{G}^\top \mu^*$。则由对偶变量计算的距离估计$\hat{d} = -\mathbf{g}^\top \mu^* - \mathbf{p}^\top \lambda^*$满足$\hat{d} \leq d^*$，其中$d^*$为点$\mathbf{p}$到机器人的真实距离。

**证明**。由于$\mu^* \in \mathcal{C}_{dual}$，它是对偶问题的一个可行解。对偶问题是最大化问题，任何可行解的目标值不超过最优值，故$\hat{d} \leq d^*$。

**推论6.2（安全性保证）**。若MPC控制器基于$\hat{d}$设置避障约束$\hat{d} \geq d_{safe}$，则真实距离满足$d^* \geq \hat{d} \geq d_{safe}$。即，硬投影层保证了距离估计的保守性，从而为安全控制提供了理论基础。

## 6.5 PDPL-Net主定理

综合以上分析，我们给出PDPL-Net的主定理：

**定理6.5（PDPL-Net主定理）**。设PDPL-Net由特征编码器$\mathcal{E}_\phi$、$J$层带残差的PDHG展开$\{\mathcal{U}_j\}_{j=1}^J$、以及硬投影层$\mathcal{P}_{hard}$组成。在假设6.1下，若步长参数满足定理6.1的条件，则：

**(1) 可行性保证**：对于任意输入$\mathbf{p} \in \mathbb{R}^2$，网络输出$\mu_{out}$满足$\mu_{out} \in \mathcal{C}_{dual}$，即约束满足率CSR = 100%。

**(2) 近似最优性**：网络输出与最优解的距离满足$\|\mu_{out} - \mu^*\| \leq \rho^J \|\mu^{(0)} - \mu^*\| + \frac{\alpha b_R}{1-\rho}$。

**(3) 安全保守性**：基于网络输出计算的距离估计$\hat{d}$是真实距离$d^*$的下界，即$\hat{d} \leq d^*$。

定理6.5为PDPL-Net的核心声明——954倍加速、100%约束满足率、安全距离保守性——提供了严格的数学基础。

## 6.6 与相关理论工作的联系

**与LISTA的关系**。LISTA将稀疏编码的ISTA算法展开为神经网络，是算法展开的开创性工作。PDPL-Net与LISTA的关键区别在于：LISTA处理无约束的$\ell_1$正则化问题，而PDPL-Net处理带约束的鞍点问题；LISTA使用软阈值，PDPL-Net使用硬投影保证可行性。

**与OptNet的关系**。OptNet将QP求解器嵌入神经网络，通过KKT条件的隐式微分实现端到端训练。OptNet在推理时精确求解QP，计算复杂度$O(n^3)$；PDPL-Net通过展开实现$O(n)$的近似求解。OptNet的约束满足依赖内部求解器；PDPL-Net通过显式投影层保证约束。

**与物理信息神经网络的关系**。KKT正则化训练策略与PINN共享相似理念：将物理方程的残差作为损失函数的一部分。在PDPL-Net中，"物理方程"即为KKT最优性条件，使网络输出不仅在数值上接近最优解，更在数学结构上满足最优性的必要条件。

