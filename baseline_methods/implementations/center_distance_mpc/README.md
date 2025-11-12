# CENTER_DISTANCE_MPC - 中心点距离方法

## 📋 概述

**CENTER_DISTANCE_MPC** 是一个传统的基于距离的方法,用于近似计算对偶变量。它通过计算每个点到点云中心的距离来估计对偶变量。

## 🎯 核心思想

基本算法:
1. 计算点云中心: `center = mean(point_cloud)`
2. 计算每个点到中心的距离: `distance = ||point - center||_2`
3. 使用距离来近似对偶变量

关键假设:
- 离中心越近的点,约束越松(mu较小)
- 离中心越远的点,约束越紧(mu较大)

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
from baseline_methods.implementations.center_distance_mpc import CenterDistanceMPC

# 初始化求解器
solver = CenterDistanceMPC(
    edge_dim=4,
    state_dim=3,
    distance_scale=1.0
)

# 创建点云数据
point_cloud = torch.randn(100, 2)  # 100个点

# 求解
mu, lam = solver(point_cloud)
print(f"mu shape: {mu.shape}")  # (4, 100)
print(f"lam shape: {lam.shape}")  # (3, 100)
```

### 使用高级版本

```python
from baseline_methods.implementations.center_distance_mpc import CenterDistanceMPCAdvanced

# 初始化高级求解器(使用多个距离度量)
solver = CenterDistanceMPCAdvanced(
    edge_dim=4,
    state_dim=3,
    distance_scale=1.0,
    use_max_distance=True
)

mu, lam = solver(point_cloud)
```

### 自定义参数

```python
import numpy as np

# 定义自定义的G和h矩阵
G = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)

solver = CenterDistanceMPC(
    edge_dim=4,
    state_dim=3,
    G=G,
    distance_scale=2.0  # 增加距离缩放因子
)
```

## 📦 依赖项

```
torch>=2.0.0
numpy>=1.20.0
```

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `edge_dim` | int | 4 | 机器人几何的边数 |
| `state_dim` | int | 3 | 状态维度 |
| `G` | np.ndarray | None | 边约束矩阵 (E, state_dim) |
| `h` | np.ndarray | None | 边约束向量 (E,) |
| `distance_scale` | float | 1.0 | 距离缩放因子 |
| `use_max_distance` | bool | True | (高级版本)使用最大距离 |

## 🔧 算法细节

### 基本版本 (CenterDistanceMPC)

1. **中心点计算**:
   ```
   center = mean(point_cloud)
   ```

2. **距离计算**:
   ```
   distance = ||point - center||_2
   normalized_distance = distance / max(distance)
   ```

3. **mu计算**:
   ```
   mu[i, :] = normalized_distance * distance_scale
   ```

4. **约束满足**:
   ```
   mu >= 0 (自动满足)
   ||G^T @ mu||_2 <= 1 (通过缩放)
   ```

5. **lam计算**:
   ```
   direction = (point - center) / ||point - center||_2
   lam = [direction_x, direction_y, 0]
   lam = lam / ||lam||_2
   ```

### 高级版本 (CenterDistanceMPCAdvanced)

使用多个距离度量:
- 欧几里得距离: `||point - center||_2`
- 曼哈顿距离: `|x| + |y|`
- 组合距离: `(euclidean + manhattan) / 2`

非线性缩放:
```
mu[i, :] = sqrt(normalized_distance) * distance_scale
```

## ⏱️ 性能特点

| 指标 | 值 |
|------|-----|
| **推理时间** | 很快 (< 1ms per batch) |
| **精度** | 低-中等 (启发式方法) |
| **约束满足率** | 100% (通过后处理) |
| **可微分** | ✅ 是 |
| **可训练** | ❌ 否 |
| **参数量** | 0 |

## 📊 对比分析

| 方法 | 优点 | 缺点 |
|------|------|------|
| **基本版本** | 简单快速 | 精度较低 |
| **高级版本** | 更鲁棒 | 计算量稍大 |
| **CVXPY** | 精度最高 | 计算慢 |
| **MLP** | 可训练 | 需要数据 |

## 🔍 实现细节

### 约束处理

1. **非负约束**: `mu >= 0`
   - 自动满足(距离非负)

2. **对偶可行性**: `||G^T @ mu||_2 <= 1`
   - 通过缩放满足

3. **辅助变量**: `||lam||_2 <= 1`
   - 通过L2归一化满足

### 数值稳定性

- 使用 `1e-8` 作为分母的最小值
- 避免除以零

## 📚 参考资源

- Zhou et al., "Real-time Collision Avoidance for Autonomous Vehicles", RA-L 2020
- 距离度量: https://en.wikipedia.org/wiki/Distance

## ✅ 测试

```bash
cd baseline_methods/implementations/center_distance_mpc
python -m pytest test_solver.py
```

## 📄 许可证

GNU General Public License v3.0

