# 图片说明文档

本文档列出了论文中所需的关键插图及其说明。请根据描述生成或插入相应的图片文件。

## Figure 1: System Overview (系统总览)
- **描述**: 展示NeuPAN-PDPL整体框架的流程图。
- **内容**:
    1.  **左侧**: 输入部分（Lidar点云 + 参考轨迹）。
    2.  **中间**: PDPL-Net网络结构（Feature Encoder -> Unrolled PDHG Layers -> Hard Projection -> Output $\mu, \lambda$）。
    3.  **右侧**: MPC优化器（接收 $\mu$ 梯度，输出控制量 $\mathbf{u}$）。
- **说明**: 突出显示“硬投影层”和“可学习近端算子”这两个核心创新点。

## Figure 2: Unrolled PDHG Architecture (PDHG展开架构)
- **描述**: 详细展示单个PDHG展开层的内部结构。
- **内容**:
    -   显示 $\mu^{(k)}$ 和 $y^{(k)}$ 的数据流。
    -   展示 ResNet Block ($\mathcal{N}_\mu, \mathcal{N}_y$) 如何作为残差项加入到更新公式中。
    -   展示可学习参数 $\tau_k, \sigma_k$ 的位置。

## Figure 3: Hard Projection Geometric Interpretation (硬投影几何解释)
- **描述**: 解释硬投影层如何保证对偶可行性。
- **内容**:
    -   画一个单位球（Unit Ball）表示可行域 $\|\mathbf{G}^\top \mu\|_2 \le 1$。
    -   展示一个网络输出点 $\hat{\mu}$ 落在单位球外部。
    -   展示投影操作将其拉回到单位球表面，形成 $\mu^*$。
    -   配文说明：无论网络输出如何，最终结果总是安全的。

## Figure 4: Point-level Evaluation Results (点级评测结果)
- **描述**: 展示不同方法在点级距离预测上的误差分布。
- **内容**:
    -   箱线图（Box Plot）或小提琴图（Violin Plot）。
    -   横轴：不同算法 (CVXPY, CVXPYLayers, ESDF-MPC, PointNet++, Point Transformer V3, DeepInverse, PDPL-Net)。
    -   纵轴：MSE Error (对数坐标，范围 $10^{-6}$ 至 $10^{1}$)。
    -   **重点**:
        -   PDPL-Net的MSE为 $1.07 \times 10^{-5}$，接近CVXPY基准（参考值0.00）。
        -   远优于其他学习方法：PointNet++ ($2.33$)、ESDF-MPC ($0.552$)、Point Transformer V3 ($0.449$)、DeepInverse ($0.0724$)。
        -   CVXPYLayers的MSE为 $2.42 \times 10^{-10}$，但推理时间长达612.6 ms（CPU）。

## Figure 5: Constraint Violation Analysis (约束违背分析)
- **描述**: 直观展示约束满足率（CSR）。
- **内容**:
    -   柱状图，展示各方法的约束满足率。
    -   对比 Black-box Methods 和 PDPL-Net：
        -   **PDPL-Net**: 100.0% 满足率（绿色柱，0% Violation）
        -   **No KKT Loss**: 100.0% 满足率（消融实验变体）
        -   **No Learned Prox**: 99.3% 满足率
        -   **DeepInverse**: 88.3% 满足率（11.7% Violation）
        -   **CVXPYLayers**: 13.2% 满足率（86.8% Violation，红色柱）
        -   **ESDF-MPC, PointNet++, Point Transformer V3, No Projection**: 0.0% 满足率（100% Violation，深红色柱）
    -   **重点**: PDPL-Net通过硬投影层确保零违背率，而其他学习方法有显著的约束违背。

## Figure 6: Closed-loop Trajectory Comparison (闭环轨迹对比)
- **描述**: 在复杂场景（如Dense Forest或Narrow Gap）中的机器人轨迹对比。
- **内容**:
    -   背景：障碍物点云。
    -   线条1（绿色虚线）：Ground Truth (CVXPY MPC)。
    -   线条2（蓝色实线）：PDPL-Net MPC (本文方法)。
    -   线条3（红色实线）：PointNet++ MPC (基线)。
    -   **现象**: PointNet++可能发生碰撞或轨迹抖动，PDPL-Net应紧贴最优轨迹。

## Figure 7: Inference Time vs Accuracy (速度-精度权衡)
- **描述**: 散点图展示各方法的帕累托前沿。
- **内容**:
    -   横轴：Inference Time (ms，对数坐标，范围 1-10000）。
    -   纵轴：Prediction Error (MSE，对数坐标，范围 $10^{-11}$ 至 $10^{3}$)。
    -   数据点（设备标注）：
        -   **CVXPY** (CPU): 2099.8 ms, MSE=0.00（参考基准，右下角）
        -   **CVXPYLayers** (CPU): 612.6 ms, MSE=$2.42 \times 10^{-10}$（中间偏右）
        -   **ESDF-MPC** (GPU): 197.2 ms, MSE=0.552（中间偏上）
        -   **PointNet++** (GPU): 217.6 ms, MSE=2.33（右上角）
        -   **Point Transformer V3** (GPU): 44.1 ms, MSE=0.449（中间）
        -   **DeepInverse** (GPU): 2.9 ms, MSE=0.0724（左侧中间）
        -   **PDPL-Net (Ours)** (GPU): **2.2 ms, MSE=$1.07 \times 10^{-5}$**（左下角，帕累托最优）
    -   **重点**: PDPL-Net位于左下角，实现了速度最快（2.2 ms）且精度第二高（仅次于CVXPYLayers的CPU解法）的帕累托最优平衡。比CVXPY快约954倍，比CVXPYLayers快约278倍。

## Figure 8: Ablation Study on Unrolling Steps (展开层数消融)
- **描述**: 展示随着展开层数 $J$ 增加，性能的变化。
- **内容**:
    -   折线图，双纵轴设计。
    -   横轴：Unrolling Steps $J$ (1, 2, 3, 4, 5)。
    -   左纵轴（蓝色）：MSE Error（对数坐标，范围 $10^{-6}$ 至 $10^{-4}$）。
    -   右纵轴（橙色）：Inference Time (ms，范围 2-10)。
    -   趋势：
        -   $J=1$: MSE=$1.07 \times 10^{-5}$, Time=2.2 ms（Full PDPL-Net）
        -   随着 $J$ 增加，MSE逐渐降低（精度提升）
        -   推理时间线性增加（每层约2-3 ms）
    -   **结论**: $J=1$ 或 $J=2$ 是最佳平衡点，在保证实时性（<5 ms）的同时达到足够精度（MSE $< 10^{-5}$）。$J \geq 3$ 带来的精度增益有限，但显著增加推理时间。
