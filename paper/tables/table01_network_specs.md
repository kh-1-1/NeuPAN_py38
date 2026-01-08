| 模块 (Module) | 组件 (Component) | 维度/配置 (Configuration) | 参数量 (Parameters) | 说明 (Description) |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Encoder** | MLP Layer 1 | Input: $2+N_{geo}$, Hidden: 32 | $\approx 200$ | 提取相对几何特征 |
| | MLP Layer 2 | Hidden: 32, Output: 32 | $\approx 1000$ | ReLU激活 |
| | Init Head ($\mu$) | Input: 32, Output: $N_{\mu}$ | $\approx 150$ | 预测初始对偶变量 |
| | Init Head ($y$) | Input: 32, Output: $N_{y}$ | $\approx 150$ | 预测初始辅助变量 |
| **PDHG Unrolling** | Primal Update ($\mu$) | ResNet Block (Hidden: 16) | $\approx 100 \times J$ | 可学习近端算子 $\mathcal{N}_{\mu}$ |
| ($J$ Layers) | Dual Update ($y$) | ResNet Block (Hidden: 16) | $\approx 100 \times J$ | 可学习近端算子 $\mathcal{N}_{y}$ |
| | Step Sizes | $\tau_j, \sigma_j, \theta_j$ | $3 \times J$ | 每层独立的可学习步长 |
| **Projection** | Hard Projection | Non-parametric | 0 | 强制对偶可行性 |
| **Total** | | **$J=2$** | **$\approx 1600$** | **极其轻量级** |
