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
