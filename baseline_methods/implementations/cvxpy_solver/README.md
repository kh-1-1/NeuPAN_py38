# CVXPY_SOLVER - 凸优化求解器

## 📋 概述

**CVXPY_SOLVER** 是一个基于凸优化的求解器,用于求解对偶变量预测问题。它作为**真值基线**,用于与其他方法进行对比。

## 🎯 核心思想

通过求解以下凸优化问题来获得最优的对偶变量:

```
minimize: ||mu||^2 + ||lambda||^2
subject to: mu >= 0
           ||G^T @ mu||_2 <= 1
           distance_constraint(lambda, point_cloud)
```

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
from baseline_methods.implementations.cvxpy_solver import CVXPYSolver

# 初始化求解器
solver = CVXPYSolver(
    edge_dim=4,
    state_dim=3,
    solver='CLARABEL',  # 或 'ECOS', 'SCS'
    verbose=False
)

# 创建点云数据
point_cloud = torch.randn(100, 2)  # 100个点

# 求解
mu, lam = solver(point_cloud)
print(f"mu shape: {mu.shape}")  # (4, 100)
print(f"lam shape: {lam.shape}")  # (3, 100)
```

### 自定义机器人几何

```python
import numpy as np

# 定义自定义的G和h矩阵
G = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)

h = np.ones(4, dtype=np.float32)

solver = CVXPYSolver(
    edge_dim=4,
    state_dim=3,
    G=G,
    h=h,
    solver='CLARABEL'
)
```

## 📦 依赖项

```
cvxpy>=1.3.0
numpy>=1.20.0
torch>=2.0.0
```

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `edge_dim` | int | 4 | 机器人几何的边数 |
| `state_dim` | int | 3 | 状态维度 |
| `G` | np.ndarray | None | 边约束矩阵 (E, state_dim) |
| `h` | np.ndarray | None | 边约束向量 (E,) |
| `solver` | str | 'CLARABEL' | CVXPY求解器名称 |
| `verbose` | bool | False | 是否打印求解器输出 |

## 🔧 求解器选择

- **CLARABEL**: 推荐,速度快,精度高
- **ECOS**: 备选,较稳定
- **SCS**: 备选,处理大规模问题

## ⏱️ 性能特点

| 指标 | 值 |
|------|-----|
| **推理时间** | 较慢 (100-500ms per batch) |
| **精度** | 最高 (真值基线) |
| **约束满足率** | 100% |
| **可微分** | ❌ 否 |
| **可训练** | ❌ 否 |

## 📝 注意事项

1. **计算复杂度**: 对每个点独立求解,计算量较大
2. **求解失败处理**: 如果求解失败,返回零解
3. **数值稳定性**: 使用CLARABEL求解器以获得最佳稳定性
4. **GPU支持**: 输入支持GPU,但求解在CPU上进行

## 🔍 实现细节

### 约束处理

1. **非负约束**: `mu >= 0`
2. **对偶可行性**: `||G^T @ mu||_2 <= 1`
3. **距离约束**: 通过优化目标隐式处理

### 求解策略

- 对每个点独立求解
- 使用CVXPY的DCP规则确保凸性
- 自动选择合适的求解器

## 📚 参考资源

- CVXPY文档: https://www.cvxpy.org/
- 凸优化理论: Boyd & Vandenberghe, "Convex Optimization"

## ✅ 测试

```bash
cd baseline_methods/implementations/cvxpy_solver
python -m pytest test_solver.py
```

## 📄 许可证

GNU General Public License v3.0

