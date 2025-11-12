# ISTA_UNROLLING - ISTA算法展开

## 📋 概述

**ISTA_UNROLLING** 是一个基于算法展开的方法,将ISTA(迭代收缩/阈值算法)展开成神经网络。

## 🎯 核心思想

ISTA是一个经典的一阶优化算法:

```
x_{k+1} = soft_threshold(x_k - step_size * grad, threshold)
```

通过将每次迭代展开成一个网络层,可以得到可训练的神经网络:

```
Layer 1: x_1 = soft_threshold(x_0 - α_1 * grad, λ_1)
Layer 2: x_2 = soft_threshold(x_1 - α_2 * grad, λ_2)
...
Layer K: x_K = soft_threshold(x_{K-1} - α_K * grad, λ_K)
```

其中 α_i 和 λ_i 是可学习的参数。

## 📊 输入输出

### 输入
- **point_cloud**: Tensor, shape `(N, 2)` - 点云坐标

### 输出
- **mu**: Tensor, shape `(E, N)` - 对偶变量 (E=4 for square robot)
- **lam**: Tensor, shape `(3, N)` - 辅助变量

## 🚀 使用方法

### 基本使用

```python
import torch
from baseline_methods.implementations.ista_unrolling import ISTAUnrolling

# 初始化模型
model = ISTAUnrolling(
    edge_dim=4,
    state_dim=3,
    num_layers=10,
    hidden_dim=32,
    learnable_step=True
)

# 创建点云数据
point_cloud = torch.randn(100, 2)  # 100个点

# 前向传播
mu, lam = model(point_cloud)
print(f"mu shape: {mu.shape}")  # (4, 100)
print(f"lam shape: {lam.shape}")  # (3, 100)
```

### 使用DeepInverse版本

```python
from baseline_methods.implementations.ista_unrolling import ISTAUnrollingWithDeepInverse

# 初始化模型(使用DeepInverse库)
model = ISTAUnrollingWithDeepInverse(
    edge_dim=4,
    state_dim=3,
    num_layers=10,
    hidden_dim=32
)

mu, lam = model(point_cloud)
```

### 训练示例

```python
import torch.optim as optim

model = ISTAUnrolling(edge_dim=4, state_dim=3, num_layers=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    point_cloud = torch.randn(256, 2)
    mu, lam = model(point_cloud)
    
    # 定义损失函数
    loss = torch.norm(mu) + torch.norm(lam)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 📦 依赖项

### 基本版本
```
torch>=2.0.0
numpy>=1.20.0
```

### DeepInverse版本
```
torch>=2.0.0
numpy>=1.20.0
deepinv>=0.2.0  # 可选
```

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `edge_dim` | int | 4 | 机器人几何的边数 |
| `state_dim` | int | 3 | 状态维度 |
| `num_layers` | int | 10 | 展开层数(迭代次数) |
| `hidden_dim` | int | 32 | 隐藏维度 |
| `learnable_step` | bool | True | 步长是否可学习 |

## 🏗️ 架构详情

### 基本版本 (ISTAUnrolling)

```
Input (2)
  ↓
Encoder: Linear(2, 32) -> ReLU -> Linear(32, 32) -> ReLU
  ↓
ISTA Unrolling (10 layers):
  Layer 1: x_1 = soft_threshold(x_0 - α_1 * grad, λ_1)
  Layer 2: x_2 = soft_threshold(x_1 - α_2 * grad, λ_2)
  ...
  Layer 10: x_10 = soft_threshold(x_9 - α_10 * grad, λ_10)
  ↓
Output Layer: Linear(32, 7)  # E=4, state_dim=3
  ↓
Output (7)
```

### DeepInverse版本 (ISTAUnrollingWithDeepInverse)

```
Input (2)
  ↓
Encoder: Linear(2, 32) -> ReLU -> Linear(32, 32) -> ReLU
  ↓
DeepInverse ISTA (10 layers)
  ↓
Output (7)
```

## 🔧 算法细节

### 软阈值函数 (Soft Thresholding)

```python
soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)
```

### ISTA迭代

```python
for k in range(num_layers):
    grad = -2 * (x - features)
    x = x - step_size[k] * grad
    x = soft_threshold(x, shrinkage[k])
```

### 可学习参数

- **step_sizes**: 每层的步长 α_i (可学习)
- **shrinkage_params**: 每层的阈值 λ_i (可学习)

## ⏱️ 性能特点

| 指标 | 值 |
|------|-----|
| **推理时间** | 快 (5-10ms per batch) |
| **精度** | 中等 (需要训练) |
| **约束满足率** | 100% (通过后处理) |
| **可微分** | ✅ 是 |
| **可训练** | ✅ 是 |
| **参数量** | ~2K (基本版本) |

## 📊 对比分析

| 方法 | 优点 | 缺点 |
|------|------|------|
| **ISTA** | 收敛理论好 | 需要10-15层 |
| **FISTA** | 收敛更快 | 需要6-8层 |
| **ADMM** | 处理约束好 | 需要8-10层 |
| **PDHG** | 最快收敛 | 需要4层 |

## 🔍 实现细节

### 约束处理

1. **非负约束**: `mu >= 0`
   - 使用ReLU激活函数

2. **对偶可行性**: `||G^T @ mu||_2 <= 1`
   - 通过缩放满足

3. **辅助变量**: `||lam||_2 <= 1`
   - 通过L2归一化满足

### 数值稳定性

- 使用 `1e-8` 作为分母的最小值
- 步长限制在 [0.01, 1.0]
- 阈值限制在 [0.001, 0.1]

## 📚 参考资源

- Gregor & LeCun, "Learning Fast Approximations of Sparse Coding", ICML 2010
- DeepInverse文档: https://deepinv.github.io/
- ISTA算法: https://en.wikipedia.org/wiki/Proximal_gradient_method

## ✅ 测试

```bash
cd baseline_methods/implementations/ista_unrolling
python -m pytest test_model.py
```

## 📄 许可证

GNU General Public License v3.0

