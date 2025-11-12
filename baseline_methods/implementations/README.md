# 对比方法实现框架

## 📋 概述

本文件夹包含12个对比方法的完整实现框架。所有方法遵循统一的接口规范。

## 🎯 统一接口

所有方法都实现以下接口:

```python
class BaselineMethod:
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: Tensor, shape (N, 2) - 点云坐标
        
        Returns:
            mu: Tensor, shape (E, N) - 对偶变量
            lam: Tensor, shape (3, N) - 辅助变量
        """
        pass
```

## 📁 文件夹结构

```
implementations/
├── __init__.py
├── README.md (本文件)
│
├── cvxpy_solver/
│   ├── __init__.py
│   ├── solver.py
│   └── README.md
│
├── mlp_baseline/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
├── center_distance_mpc/
│   ├── __init__.py
│   ├── solver.py
│   └── README.md
│
├── ista_unrolling/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
├── admm_unrolling/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
├── point_transformer_v3/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
├── pointnet_plusplus/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
├── cvxpylayers_solver/
│   ├── __init__.py
│   ├── solver.py
│   └── README.md
│
├── fista_unrolling/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
├── physics_informed_projection/
│   ├── __init__.py
│   ├── model.py
│   └── README.md
│
└── (其他方法...)
```

## ✅ 已实现的方法 (第一优先级)

### 1. CVXPY_SOLVER ✅
- **类型**: 凸优化求解器
- **文件**: `cvxpy_solver/`
- **特点**: 真值基线,精度最高
- **推理时间**: 较慢 (100-500ms)
- **可训练**: ❌ 否
- **状态**: ✅ 完成

### 2. MLP_BASELINE ✅
- **类型**: 黑盒神经网络
- **文件**: `mlp_baseline/`
- **特点**: 简单快速,可训练
- **推理时间**: 很快 (< 1ms)
- **可训练**: ✅ 是
- **状态**: ✅ 完成

### 3. CENTER_DISTANCE_MPC ✅
- **类型**: 传统启发式方法
- **文件**: `center_distance_mpc/`
- **特点**: 基于距离的近似
- **推理时间**: 很快 (< 1ms)
- **可训练**: ❌ 否
- **状态**: ✅ 完成

### 4. ISTA_UNROLLING ✅
- **类型**: 算法展开
- **文件**: `ista_unrolling/`
- **特点**: 经典一阶优化算法展开
- **推理时间**: 快 (5-10ms)
- **可训练**: ✅ 是
- **状态**: ✅ 完成

### 5. ADMM_UNROLLING ✅
- **类型**: 算法展开
- **文件**: `admm_unrolling/`
- **特点**: 约束优化算法展开
- **推理时间**: 快 (5-10ms)
- **可训练**: ✅ 是
- **状态**: ✅ 完成

## 🚀 待实现的方法 (第二优先级)

### 6. POINT_TRANSFORMER_V3 ⏳
- **类型**: Transformer方法
- **文件**: `point_transformer_v3/`
- **特点**: 2024最新方法
- **推理时间**: 中等 (10-20ms)
- **可训练**: ✅ 是
- **状态**: ⏳ 待实现

### 7. POINTNET_PLUSPLUS ⏳
- **类型**: 点云神经网络
- **文件**: `pointnet_plusplus/`
- **特点**: 经典点云处理方法
- **推理时间**: 中等 (10-20ms)
- **可训练**: ✅ 是
- **状态**: ⏳ 待实现

### 8. CVXPYLAYERS_SOLVER ⏳
- **类型**: 可微分凸优化
- **文件**: `cvxpylayers_solver/`
- **特点**: 端到端可微分
- **推理时间**: 中等 (20-50ms)
- **可训练**: ✅ 是
- **状态**: ⏳ 待实现

### 9. FISTA_UNROLLING ⏳
- **类型**: 算法展开
- **文件**: `fista_unrolling/`
- **特点**: 加速版ISTA
- **推理时间**: 快 (5-10ms)
- **可训练**: ✅ 是
- **状态**: ⏳ 待实现

### 10. PHYSICS_INFORMED_PROJECTION ⏳
- **类型**: 物理约束方法
- **文件**: `physics_informed_projection/`
- **特点**: 硬投影参考
- **推理时间**: 快 (< 1ms)
- **可训练**: ❌ 否
- **状态**: ⏳ 待实现

## 📊 方法对比表

| # | 方法 | 类型 | 推理时间 | 精度 | 可训练 | 状态 |
|---|------|------|---------|------|--------|------|
| 1 | CVXPY_SOLVER | 凸优化 | 慢 | ⭐⭐⭐⭐⭐ | ❌ | ✅ |
| 2 | MLP_BASELINE | 神经网络 | 很快 | ⭐⭐⭐ | ✅ | ✅ |
| 3 | CENTER_DISTANCE_MPC | 启发式 | 很快 | ⭐⭐ | ❌ | ✅ |
| 4 | ISTA_UNROLLING | 展开 | 快 | ⭐⭐⭐ | ✅ | ✅ |
| 5 | ADMM_UNROLLING | 展开 | 快 | ⭐⭐⭐ | ✅ | ✅ |
| 6 | POINT_TRANSFORMER_V3 | Transformer | 中等 | ⭐⭐⭐⭐ | ✅ | ⏳ |
| 7 | POINTNET_PLUSPLUS | 点云 | 中等 | ⭐⭐⭐ | ✅ | ⏳ |
| 8 | CVXPYLAYERS_SOLVER | 可微分 | 中等 | ⭐⭐⭐⭐ | ✅ | ⏳ |
| 9 | FISTA_UNROLLING | 展开 | 快 | ⭐⭐⭐ | ✅ | ⏳ |
| 10 | PHYSICS_INFORMED_PROJECTION | 物理 | 很快 | ⭐⭐⭐ | ❌ | ⏳ |

## 🚀 快速开始

### 导入方法

```python
from baseline_methods.implementations import (
    CVXPYSolver,
    MLPBaseline,
    CenterDistanceMPC,
    ISTAUnrolling,
    ADMMUnrolling,
)

# 创建点云数据
import torch
point_cloud = torch.randn(100, 2)

# 使用各种方法
cvxpy_solver = CVXPYSolver(edge_dim=4, state_dim=3)
mu1, lam1 = cvxpy_solver(point_cloud)

mlp_model = MLPBaseline(edge_dim=4, state_dim=3)
mu2, lam2 = mlp_model(point_cloud)

center_solver = CenterDistanceMPC(edge_dim=4, state_dim=3)
mu3, lam3 = center_solver(point_cloud)

ista_model = ISTAUnrolling(edge_dim=4, state_dim=3, num_layers=10)
mu4, lam4 = ista_model(point_cloud)

admm_model = ADMMUnrolling(edge_dim=4, state_dim=3, num_layers=8)
mu5, lam5 = admm_model(point_cloud)
```

### 批量测试

```python
import torch
from baseline_methods.implementations import (
    CVXPYSolver, MLPBaseline, CenterDistanceMPC,
    ISTAUnrolling, ADMMUnrolling
)

# 创建测试数据
point_cloud = torch.randn(256, 2)

# 初始化所有方法
methods = {
    'CVXPY': CVXPYSolver(edge_dim=4, state_dim=3),
    'MLP': MLPBaseline(edge_dim=4, state_dim=3),
    'CenterDistance': CenterDistanceMPC(edge_dim=4, state_dim=3),
    'ISTA': ISTAUnrolling(edge_dim=4, state_dim=3, num_layers=10),
    'ADMM': ADMMUnrolling(edge_dim=4, state_dim=3, num_layers=8),
}

# 测试所有方法
for name, method in methods.items():
    mu, lam = method(point_cloud)
    print(f"{name}: mu shape {mu.shape}, lam shape {lam.shape}")
```

## 📚 文档

每个方法都有详细的README文档:
- 算法原理
- 使用方法
- 参数说明
- 性能特点
- 参考资源

## 🔧 开发指南

### 添加新方法

1. 创建新文件夹: `baseline_methods/implementations/new_method/`
2. 创建 `__init__.py` 和 `model.py` (或 `solver.py`)
3. 实现统一接口
4. 创建 `README.md` 文档
5. 在 `implementations/__init__.py` 中导入

### 接口规范

```python
class NewMethod(nn.Module):
    def __init__(self, edge_dim=4, state_dim=3, **kwargs):
        super().__init__()
        self.edge_dim = edge_dim
        self.state_dim = state_dim
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (N, 2)
        Returns:
            mu: (E, N)
            lam: (3, N)
        """
        # 实现逻辑
        return mu, lam
```

## 📊 性能基准

| 方法 | 推理时间 | 内存 | 精度 |
|------|---------|------|------|
| CVXPY_SOLVER | 100-500ms | 低 | 最高 |
| MLP_BASELINE | < 1ms | 低 | 中等 |
| CENTER_DISTANCE_MPC | < 1ms | 低 | 低 |
| ISTA_UNROLLING | 5-10ms | 低 | 中等 |
| ADMM_UNROLLING | 5-10ms | 低 | 中等 |

## 📝 许可证

GNU General Public License v3.0

## 🤝 贡献

欢迎提交新的方法实现!

---

**最后更新**: 2025-11-12

