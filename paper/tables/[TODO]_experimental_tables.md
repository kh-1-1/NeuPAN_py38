# [TODO] 实验数据表

本文件包含从CSV文件自动提取的实验结果表。请将这些表格内容复制到主论文的相应位置。

## Table 4: 点级性能评测 (Point-level Evaluation)

| 方法 (Method) | MSE ($\mu$) | KKT Residual | Constraint Sat. Rate (CSR) | Time (ms) | 设备 (Device) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CVXPY** (Solver) | 0.00 (Ref) | 0.00 | - | 2099.8 | CPU |
| **CVXPYLayers** | $2.42 \times 10^{-10}$ | 0.00 | 13.2% | 612.6 | CPU |
| **ESDF-MPC** | $5.52 \times 10^{-1}$ | 27.77 | 0.0% | 197.2 | GPU |
| **PointNet++** | $2.33 \times 10^{0}$ | 566.5 | 0.0% | 217.6 | GPU |
| **Point Transformer V3** | $4.49 \times 10^{-1}$ | 0.19 | 0.0% | 44.1 | GPU |
| **DeepInverse** | $7.24 \times 10^{-2}$ | 6469 | 88.3% | 2.9 | GPU |
| **PDPL-Net (Ours)** | **$1.07 \times 10^{-5}$** | **$6.21 \times 10^{-5}$** | **100.0%** | **2.2** | GPU |

## Table 5: 消融实验 (Ablation Study)

| 变体 (Variant) | MSE ($\mu$) | KKT Rel. Mean | Constraint Sat. Rate | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **No Projection** | 296.8 | 12.25 | 0.0% | 移除硬投影层 |
| **No Learned Prox** | 126.3 | 0.92 | 99.3% | 移除可学习参数 |
| **No KKT Loss** | $5.64 \times 10^{-6}$ | 0.93 | 100.0% | 移除KKT正则化 |
| **Full (PDPL-Net)** | **$5.48 \times 10^{-6}$** | **0.93** | **100.0%** | 完整模型 ($J=1$) |

**注**：
- MSE ($\mu$): 对偶变量 $\mu$ 的均方误差。
- KKT Residual: KKT条件的相对残差。
- Constraint Sat. Rate: 满足对偶可行性约束的样本比例。
- Time: 单次推理的平均耗时。
