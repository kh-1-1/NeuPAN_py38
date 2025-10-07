# DF-DUNE 技术实现指南

> **文档状态**: ✅ 已更新（2025-01-XX）
> **实现进度**: A-1 ✅ | A-2 ✅ | A-3 ✅ | B-1 ✅ | C ✅ | 配置链路 ✅
> **本指南基于**: 已完成的代码实现，提供使用说明与验证方法

## 目录
1. [快速开始](#快速开始)
2. [核心模块实现状态](#核心模块实现状态)
3. [训练流程配置](#训练流程配置)
4. [评测框架](#评测框架)
5. [调试与优化](#调试与优化)

---

## 一、快速开始

### 1.1 环境准备

```bash
# 已有环境
cd NeuPAN-py38
conda activate neupan  # 或你的环境名

# 可选: 安装额外依赖(用于高级功能)
pip install tensorboard  # 训练可视化
pip install pytest       # 单元测试
```

### 1.2 最小可行验证（使用已实现模块）

**目标**: 验证所有 DF-DUNE 模块已正确集成

#### 方法1: 加载已有配置并检查参数

```python
from neupan import neupan

# 加载完整 DF-DUNE 配置（M5）
planner = neupan.init_from_yaml('example/dune_train/dune_train_acker_kkt_se2.yaml')

# 验证所有开关
dune = planner.pan.dune_layer
print(f"✅ 硬投影 (A-1): projection = {dune.projection}")
print(f"✅ KKT正则 (A-3): use_kkt = {getattr(dune, 'use_kkt', 'N/A')}")
print(f"✅ PDHG展开 (B-1): unroll_J = {dune.unroll_J}")
print(f"✅ SE(2)编码 (C): se2_embed = {dune.se2_embed}")
print(f"✅ ObsPointNet输入维度: {dune.model.MLP[0].in_features} (应为3)")
```

#### 方法2: 完整 YAML 配置示例（M5）

```yaml
# example/dune_train/dune_train_acker_kkt_se2.yaml (已存在)
robot:
  kinematics: 'acker'
  length: 4.6
  width: 1.6
  wheelbase: 3

device: cuda

pan:
  dune_checkpoint: example/dune_train/model/acker_learned_prox_robot/model_2500.pth

train:
  # 完整 DF-DUNE 配置
  direct_train: true
  model_name: acker_learned_prox_kkt_se2_robot

  projection: hard          # A-1: 硬投影
  use_kkt: true             # A-3: KKT 正则
  w_kkt: 1e-3               # KKT 损失权重
  kkt_rho: 0.5              # KKT 惩罚参数
  use_lconstr: true         # 对偶约束损失
  w_constr: 0.1             # 约束损失权重
  unroll_J: 3               # B-1: PDHG 展开 3 步
  se2_embed: true           # C: SE(2) 极坐标编码

  # 训练超参数
  data_size: 100000
  data_range: [-25, -25, 25, 25]
  batch_size: 256
  epoch: 2500
  valid_freq: 1
  save_freq: 500
  lr: 2.5e-6
  lr_decay: 0.5
  decay_freq: 250
```

**验证**: 运行推理并检查时延与对偶范数

```bash
python -c "
from neupan import neupan
import numpy as np

planner = neupan.init_from_yaml('example/dune_train/dune_train_acker_kkt_se2.yaml')
start = np.array([0, 0, 0])
goal = np.array([10, 10, 0])
obs_list = [np.array([[5, 5], [5, 6], [6, 6], [6, 5]])]

info = planner.plan(start, goal, obs_list)
print(f'PDHG时延: {info.get(\"pdhg_profile\", {}).get(\"total\", \"N/A\")}')
print(f'对偶范数: {info.get(\"dual_norm_post\", \"N/A\")}')
"
```

---

## 二、核心模块实现状态

### 2.1 A-1: 硬投影（Hard Projection）✅ 已实现

#### 实现位置
`neupan/blocks/dune.py` 第 109-119 行

#### 核心代码
```python
def hard_projection(self, mu: torch.Tensor) -> torch.Tensor:
    """
    零参数硬投影: μ ≥ 0, ||G^T μ||_2 ≤ 1

    Args:
        mu: [E, N] 原始对偶变量
    Returns:
        mu_proj: [E, N] 投影后的对偶变量
    """
    # 1. 非负投影
    mu = mu.clamp(min=0.0)

    # 2. 对偶范数约束（列向量投影）
    v = self.G.t() @ mu  # [2, N]
    v_norm = torch.norm(v, dim=0, keepdim=True)  # [1, N]
    scale = torch.clamp(v_norm, min=1.0)  # max(1, ||v||)
    mu = mu / scale  # 缩放到可行域

    return mu
```

#### YAML 配置
```yaml
train:
  projection: hard  # 启用硬投影
```

#### 验证方法
```python
from neupan import neupan
planner = neupan.init_from_yaml('your_config.yaml')
print(f"投影模式: {planner.pan.dune_layer.projection}")  # 应输出 'hard'
```

---

### 2.2 A-3: KKT 残差正则（KKT Residual Regularization）✅ 已实现

#### 实现位置
`neupan/blocks/dune_train.py` 第 398-440 行

#### 核心代码（已实现）
```python
# 在 train_one_epoch 方法中（第 431-440 行）
L_kkt = torch.zeros((), device=mu_vec.device)
if self.use_kkt:
    ip = input_point.unsqueeze(-1)             # [B,2,1] or [2,1]
    a = self.G @ ip - self.h                  # [B,E,1] or [E,1]
    Gy = self.G @ (self.G.t() @ mu_reg_be1)   # [B,E,1] or [E,1]
    s = torch.relu(-mu_reg_be1)               # [B,E,1] or [E,1]
    r = -a + self.kkt_rho * Gy - s            # [B,E,1] or [E,1]

    # 相对/归一化 KKT 残差（减少尺度主导）
    # L_kkt = mean( ( ||r||_2 / (||a||_2 + eps) )^2 )
    r_norm = torch.norm(r.squeeze(-1), dim=-1)
    a_norm = torch.norm(a.squeeze(-1), dim=-1).clamp(min=1e-6)
    L_kkt = ((r_norm / a_norm) ** 2).mean()

# 总损失（第 447 行）
loss = mse_mu + mse_distance + mse_fa + mse_fb + \
       self.w_constr * L_constr + self.w_kkt * L_kkt
```

#### YAML 配置
```yaml
train:
  use_kkt: true       # 启用 KKT 正则
  w_kkt: 1e-3         # KKT 损失权重（Acker: 1e-3, Diff: 1e-4）
  kkt_rho: 0.5        # KKT 惩罚参数（Acker: 0.5, Diff: 0.1）
  use_lconstr: true   # 启用对偶约束损失
  w_constr: 0.1       # 约束损失权重
```

#### 验证方法
```python
import pickle
with open('path/to/train_dict.pkl', 'rb') as f:
    d = pickle.load(f)
    print(f"use_kkt: {d['use_kkt']}")
    print(f"w_kkt: {d['w_kkt']}")
    print(f"kkt_rho: {d['kkt_rho']}")
```

---

### 2.3 B-1: PDHG-Unroll（PDHG 展开）✅ 已实现

#### 实现位置
`neupan/blocks/pdhg_unroll.py` 完整模块（245 行）

#### 核心代码（已实现）
```python
class PDHGUnroll(nn.Module):
    """
    J-step PDHG 迭代优化对偶变量

    Args:
        E: 边数（约束数）
        J: 展开步数（推荐 1-3）
        tau: 原始步长（默认 0.5）
        sigma: 对偶步长（默认 0.5）
        learnable_steps: 是否可学习步长
    """
    def forward(self, mu, a, G):
        y = torch.zeros(2, N)
        for _ in range(self.J):
            # 1. y-update: 投影到单位球
            y = y + sigma * (G.t() @ mu)
            y = y / torch.clamp(torch.norm(y, dim=0, keepdim=True), min=1.0)

            # 2. μ-update: 非负投影
            mu = mu + tau * (a - G @ y)
            mu = mu.clamp(min=0.0)

            # 3. 安全投影: ||G^T μ||_2 ≤ 1
            v = G.t() @ mu
            mu = mu / torch.clamp(torch.norm(v, dim=0, keepdim=True), min=1.0)

        return mu
```

#### YAML 配置
```yaml
train:
  unroll_J: 3  # PDHG 展开步数（0=禁用, 1-3=推荐, >3=高精度场景）
```

#### 验证方法
```python
from neupan import neupan
planner = neupan.init_from_yaml('your_config.yaml')
dune = planner.pan.dune_layer
print(f"PDHG 步数: {dune.unroll_J}")
print(f"PDHG 模块: {dune.pdhg_unroll}")

# 推理后检查时延
info = planner.plan(start, goal, obs_list)
if 'pdhg_profile' in info:
    print(f"PDHG 总时延: {info['pdhg_profile']['total']:.4f}s")
    print(f"每步时延: {info['pdhg_profile']['per_step']}")
```

---

### 2.4 C: SE(2) 等变编码（SE(2) Equivariant Encoding）✅ 已实现

#### 实现位置
`neupan/blocks/obs_point_net.py` 第 26-65 行

#### 核心代码（已实现）
```python
class ObsPointNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=4, se2_embed=False):
        super().__init__()
        self.se2_embed = se2_embed
        actual_in = 3 if se2_embed else input_dim  # 动态调整输入维度

        self.MLP = nn.Sequential(
            nn.Linear(actual_in, 32),  # 输入维度: 2 或 3
            # ... 其余层
        )

    def polar_embed(self, x):
        """
        极坐标编码: (x, y) → (r, cos(θ), sin(θ))

        Args:
            x: [N, 2] Cartesian 坐标
        Returns:
            [N, 3] 极坐标编码
        """
        r = torch.norm(x, dim=1, keepdim=True)
        theta = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1)
        return torch.cat([r, torch.cos(theta), torch.sin(theta)], dim=1)

    def forward(self, x):
        if self.se2_embed:
            x = self.polar_embed(x)  # [N, 2] → [N, 3]
        return self.MLP(x)
```

#### YAML 配置
```yaml
train:
  se2_embed: true  # 启用 SE(2) 极坐标编码
```

#### 验证方法
```python
from neupan import neupan
planner = neupan.init_from_yaml('your_config.yaml')
model = planner.pan.dune_layer.model
print(f"SE(2) 编码: {model.se2_embed}")
print(f"输入维度: {model.MLP[0].in_features}")  # 应为 3（启用时）

# 验证极坐标变换
import torch
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
polar = model.polar_embed(x)
print(f"Cartesian: {x}")
print(f"Polar: {polar}")  # [[1, 1, 0], [1, 0, 1]]
```

---

### 2.5 A-2: Learned-Prox（可学习近端头）✅ 已实现

#### 实现位置
`neupan/blocks/learned_prox.py` 完整模块（70 行）

#### 核心代码（已实现）
```python
class ProxHead(nn.Module):
    """
    轻量级可学习近端头：通过小型 MLP 优化对偶变量

    给定 mu ∈ R^{E×N}，计算 z = (G^T mu)^T ∈ R^{N×2}，
    通过 MLP 预测残差 delta ∈ R^{N×E}，
    返回优化后的 mu' = clamp_min(mu + delta^T, 0)

    Args:
        E: 边数（约束数）
        hidden: 隐藏层维度（默认 32）
    """
    def __init__(self, E: int, hidden: int = 32):
        super().__init__()
        self.E = int(E)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.E),
        )

    def forward(self, mu: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        支持两种布局:
        - 列布局 [E, N]: z = (G^T mu)^T
        - 行布局 [N, E]: z = mu @ G
        返回优化后的 mu（保持输入布局）
        """
        # 自动检测布局并计算
        if mu.shape[0] == self.E:  # [E, N]
            z = (G.t() @ mu).t()  # [N, 2]
            delta = self.mlp(z)   # [N, E]
            mu_ref = (mu.t() + delta).t()  # [E, N]
        elif mu.shape[1] == self.E:  # [N, E]
            z = mu @ G            # [N, 2]
            delta = self.mlp(z)   # [N, E]
            mu_ref = mu + delta   # [N, E]

        return mu_ref.clamp_min(0.0)
```

#### YAML 配置
```yaml
train:
  projection: learned  # 启用 Learned-Prox
  use_kkt: true        # 推荐与 KKT 正则配合
  w_kkt: 1e-3
```

#### 验证方法
```python
from neupan import neupan

# 加载 Learned-Prox 配置
planner = neupan.init_from_yaml('example/dune_train/dune_train_acker_kkt.yaml')
dune = planner.pan.dune_layer

print(f"投影模式: {dune.projection}")  # 应为 'learned'
print(f"ProxHead: {dune.prox_head}")   # 应为 ProxHead(E=4, hidden=32)

# 检查参数量
if dune.prox_head:
    n_params = sum(p.numel() for p in dune.prox_head.parameters())
    print(f"ProxHead 参数量: {n_params}")  # 约 200 参数（轻量级）
```

#### 推理流程集成
```python
# 在 dune.py forward() 中（L157-163）
if self.projection == 'learned' and self.prox_head is not None:
    try:
        total_mu = self.prox_head(total_mu, self.G)  # 应用 Learned-Prox
    except Exception:
        pass  # 回退：跳过 prox

# 后续仍会应用硬投影（L165-180）
if self.projection in ('hard', 'learned'):
    total_mu = self.hard_projection(total_mu)
```

#### 训练流程集成
```python
# 在 dune_train.py train_one_epoch() 中（L404-407）
mu_reg = mu_vec
if (self.prox_head is not None) and (self.projection_mode == 'learned'):
    mu_reg = self.prox_head(mu_reg, self.G)  # 应用 Learned-Prox

# 后续计算 KKT 残差与约束损失（L424-440）
```

#### 模型保存与加载
```python
# 保存（dune.py L304-316）
state = {
    'model': self.model.state_dict(),
    'prox_head': self.prox_head.state_dict() if self.prox_head else None,
    'pdhg_unroll': self.pdhg_unroll.state_dict() if self.pdhg_unroll else None,
}
torch.save(state, checkpoint_path)

# 加载（dune.py L306-310）
if self.projection == 'learned' and self.prox_head and 'prox_head' in state:
    self.prox_head.load_state_dict(state['prox_head'])
```

#### 性能特点
- **参数量**: ~200 参数（2→32→E 的小型 MLP）
- **计算开销**: 极小（<1ms）
- **优势**: 可学习的对偶变量优化，比硬投影更灵活
- **适用场景**: 需要更精细对偶可行性控制的场景
        distance_loss / len(train_dataloader),
        fa_loss / len(train_dataloader),
        fb_loss / len(train_dataloader),
        kkt_loss / len(train_dataloader),
    )
```

#### 配置参数

```yaml
train:
  # ... 其他参数
  kkt_weight: 0.1    # KKT损失权重,建议范围[0.01, 1.0]
  kkt_rho: 1.0       # 惩罚系数,建议范围[0.5, 2.0]
```

#### 调试检查点

1. **损失下降**: `kkt_loss` 应在前1000个epoch内下降至1e-3量级
2. **不影响主损失**: `mse_mu` 和 `mse_distance` 不应显著上升
3. **违反率改善**: 训练后的模型,`dual_norm_violation_rate` 应<1%

---

### 2.2 PDHG-Unroll(B-1) ⭐⭐ 核心创新

#### 实现位置
`neupan/blocks/dune.py` 新增 `PDHGLayer` 类

#### 完整实现

```python
# 在 neupan/blocks/ 下新建 pdhg_layer.py
import torch
import torch.nn as nn

class PDHGLayer(nn.Module):
    """
    原始-对偶混合梯度展开层

    求解问题:
        max_{μ≥0} μ^T * a
        s.t. ||G^T * μ||_2 ≤ 1

    其中 a = G*p - h
    """

    def __init__(self, G, h, num_iters=3, learn_steps=True):
        """
        Args:
            G: 机器人几何矩阵 [E, 2]
            h: 偏移向量 [E, 1]
            num_iters: PDHG迭代次数
            learn_steps: 是否学习步长参数
        """
        super().__init__()
        self.G = G
        self.h = h
        self.num_iters = num_iters

        # 理论步长: σ*τ*||G||^2 < 1
        G_norm = torch.norm(G, p=2).item()
        init_sigma = 0.9 / G_norm
        init_tau = 0.9 / G_norm

        if learn_steps:
            self.sigma = nn.Parameter(torch.tensor(init_sigma))
            self.tau = nn.Parameter(torch.tensor(init_tau))
        else:
            self.register_buffer('sigma', torch.tensor(init_sigma))
            self.register_buffer('tau', torch.tensor(init_tau))

    def forward(self, mu_init, p):
        """
        Args:
            mu_init: 初始对偶变量 [E, N] (来自ObsPointNet)
            p: 点云 [2, N]

        Returns:
            mu: 优化后的对偶变量 [E, N]
            y: 对偶变量(用于调试) [2, N]
        """
        # 初始化
        mu = mu_init.clamp(min=0.0)  # 确保非负
        y = torch.zeros(2, mu.shape[1], device=mu.device)

        # 计算 a = G*p - h
        a = self.G @ p - self.h  # [E, N]

        # PDHG迭代
        for _ in range(self.num_iters):
            # y-update: 投影到L2单位球
            y_temp = y + self.sigma * (self.G.T @ mu)
            y_norm = torch.norm(y_temp, dim=0, keepdim=True).clamp(min=1.0)
            y = y_temp / y_norm

            # μ-update: 非负投影
            mu_temp = mu + self.tau * (a - self.G @ y)
            mu = mu_temp.clamp(min=0.0)

            # 额外投影: 确保 ||G^T*μ|| ≤ 1 (可选,增强鲁棒性)
            v = self.G.T @ mu
            v_norm = torch.norm(v, dim=0, keepdim=True)
            scale = torch.where(v_norm > 1.0, 1.0 / v_norm, torch.ones_like(v_norm))
            mu = mu * scale

        return mu, y

    def get_convergence_metric(self, mu, mu_prev):
        """计算收敛指标(用于早停)"""
        return torch.norm(mu - mu_prev, dim=0).mean()
```

#### 集成到DUNE

```python
# 修改 neupan/blocks/dune.py

from neupan.blocks.pdhg_layer import PDHGLayer

class DUNE(torch.nn.Module):
    def __init__(self, ...):
        # ... 现有代码

        # 新增: PDHG层
        if self.unroll_J > 0:
            self.pdhg_layer = PDHGLayer(
                self.G, self.h,
                num_iters=self.unroll_J,
                learn_steps=True  # 学习步长
            )
        else:
            self.pdhg_layer = None

    def forward(self, point_flow, R_list, obs_points_list=[]):
        # ... 现有代码到第90行

        with torch.no_grad():
            total_mu = self.model(total_points.T).T

            # 新增: PDHG优化
            if self.pdhg_layer is not None:
                total_mu, _ = self.pdhg_layer(total_mu, total_points)

            # 监控与投影(保持原逻辑)
            if self.monitor_dual_norm or (self.projection == 'hard'):
                # ... 现有监控代码
                pass

        # ... 后续代码不变
```

#### 训练配置

```yaml
train:
  unroll_J: 3           # PDHG迭代次数,建议1/2/3
  learn_pdhg_steps: true  # 学习步长参数
```

#### 性能优化

1. **早停机制** (推理时):
```python
def forward(self, mu_init, p, tol=1e-3):
    mu = mu_init.clamp(min=0.0)
    y = torch.zeros(...)

    for i in range(self.num_iters):
        mu_prev = mu.clone()
        # ... PDHG更新

        # 早停
        if self.get_convergence_metric(mu, mu_prev) < tol:
            break
    return mu, y
```

2. **JIT编译** (加速推理):
```python
# 在DUNE.__init__中
if self.pdhg_layer is not None:
    self.pdhg_layer = torch.jit.script(self.pdhg_layer)
```

---

### 2.3 SE(2)等变编码(C) ⭐ 提升泛化

#### 实现位置
`neupan/blocks/obs_point_net.py` 修改输入层

#### 代码补丁

```python
# 修改 ObsPointNet 类

class ObsPointNet(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 4, se2_embed: bool = False):
        super().__init__()

        self.se2_embed = se2_embed

        # 如果启用SE(2)编码,输入维度变为3
        actual_input_dim = 3 if se2_embed else input_dim

        hidden_dim = 32
        self.MLP = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def polar_embed(self, x):
        """
        将笛卡尔坐标转换为极坐标编码

        Args:
            x: [N, 2] - (x, y)

        Returns:
            [N, 3] - (r, cos(θ), sin(θ))
        """
        r = torch.norm(x, dim=1, keepdim=True)  # [N, 1]
        theta = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1)  # [N, 1]

        return torch.cat([r, torch.cos(theta), torch.sin(theta)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 2] - 点云坐标

        Returns:
            [N, output_dim] - 对偶权重μ
        """
        if self.se2_embed:
            x = self.polar_embed(x)

        return self.MLP(x)
```

#### 配置与使用

```yaml
train:
  se2_embed: true  # 启用SE(2)等变编码
```

```python
# 在 DUNE.__init__ 中传递参数
self.model = to_device(ObsPointNet(2, self.edge_dim, se2_embed=self.se2_embed))
```

---

## 三、训练流程改造

### 3.1 损失函数权重自适应

实现Kendall风格的不确定性加权:

```python
class DUNETrain:
    def __init__(self, ...):
        # ... 现有代码

        # 可学习的损失权重(log空间)
        self.log_var_mu = nn.Parameter(torch.zeros(1))
        self.log_var_dist = nn.Parameter(torch.zeros(1))
        self.log_var_fa = nn.Parameter(torch.zeros(1))
        self.log_var_fb = nn.Parameter(torch.zeros(1))
        self.log_var_kkt = nn.Parameter(torch.zeros(1))

        # 将这些参数加入优化器
        self.optimizer = Adam(
            list(self.model.parameters()) + [
                self.log_var_mu, self.log_var_dist,
                self.log_var_fa, self.log_var_fb, self.log_var_kkt
            ],
            lr=1e-4, weight_decay=1e-4
        )

    def adaptive_loss(self, losses_dict):
        """
        自适应加权损失

        L = Σ exp(-log_var_i) * L_i + Σ log_var_i
        """
        total_loss = 0
        total_loss += torch.exp(-self.log_var_mu) * losses_dict['mu'] + self.log_var_mu
        total_loss += torch.exp(-self.log_var_dist) * losses_dict['dist'] + self.log_var_dist
        total_loss += torch.exp(-self.log_var_fa) * losses_dict['fa'] + self.log_var_fa
        total_loss += torch.exp(-self.log_var_fb) * losses_dict['fb'] + self.log_var_fb
        total_loss += torch.exp(-self.log_var_kkt) * losses_dict['kkt'] + self.log_var_kkt

        return total_loss
```

### 3.2 课程学习

逐步增加约束难度:

```python
def get_curriculum_weight(epoch, total_epochs, mode='linear'):
    """
    课程学习权重调度

    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        mode: 'linear' | 'cosine' | 'step'
    """
    progress = epoch / total_epochs

    if mode == 'linear':
        return progress
    elif mode == 'cosine':
        return 0.5 * (1 - np.cos(np.pi * progress))
    elif mode == 'step':
        if progress < 0.3:
            return 0.1
        elif progress < 0.6:
            return 0.5
        else:
            return 1.0

# 在训练循环中使用
kkt_weight_current = kkt_weight * get_curriculum_weight(i, epoch, mode='cosine')
```

---

## 四、评测框架

### 4.1 模块级指标

创建 `neupan/evaluation/dune_metrics.py`:

```python
import torch
import numpy as np

class DUNEMetrics:
    """DUNE模块评测指标"""

    @staticmethod
    def dual_feasibility_rate(mu, G, threshold=1.0):
        """
        对偶可行性: ||G^T μ||_2 ≤ 1 的比例
        """
        v = G.T @ mu  # [2, N]
        norms = torch.norm(v, dim=0)
        return (norms <= threshold).float().mean().item()

    @staticmethod
    def dual_norm_percentile(mu, G, percentile=95):
        """
        对偶范数的百分位数
        """
        v = G.T @ mu
        norms = torch.norm(v, dim=0)
        return torch.quantile(norms, percentile/100.0).item()

    @staticmethod
    def distance_error(d_pred, d_true):
        """
        距离预测误差
        """
        return {
            'mae': torch.mean(torch.abs(d_pred - d_true)).item(),
            'rmse': torch.sqrt(torch.mean((d_pred - d_true)**2)).item(),
            'max': torch.max(torch.abs(d_pred - d_true)).item(),
        }

    @staticmethod
    def safety_margin_tightness(d_pred, d_true):
        """
        安全上界紧致度: max(0, d_pred - d_true)

        理想情况: d_pred ≈ d_true (紧致)
        危险情况: d_pred < d_true (乐观估计)
        """
        slack = d_pred - d_true
        return {
            'mean_slack': torch.mean(torch.clamp(slack, min=0)).item(),
            'p95_slack': torch.quantile(torch.clamp(slack, min=0), 0.95).item(),
            'optimistic_rate': (slack < 0).float().mean().item(),  # 应接近0
        }
```

### 4.2 闭环评测脚本

创建 `example/evaluation/closed_loop_test.py`:

```python
import numpy as np
from neupan import neupan
from neupan.evaluation.dune_metrics import DUNEMetrics

def run_closed_loop_test(config_path, num_episodes=50):
    """
    闭环测试

    Returns:
        metrics: {
            'success_rate': float,
            'min_clearance_mean': float,
            'collision_rate': float,
            'planning_time_mean': float,
            ...
        }
    """
    planner = neupan(**load_config(config_path))

    results = []
    for ep in range(num_episodes):
        # 生成随机场景
        start, goal, obstacles = generate_random_scenario()

        # 运行规划
        trajectory, info = planner.plan(start, goal, obstacles)

        # 评估
        results.append({
            'success': info['reached_goal'],
            'min_clearance': info['min_distance_to_obstacle'],
            'collision': info['collision_occurred'],
            'planning_time': info['total_time'],
            'path_length': compute_path_length(trajectory),
        })

    # 聚合统计
    return aggregate_results(results)
```


### 4.3 多因素消融与统计（统一因子化设计）

- 因子/水平：projection ∈ {none, hard, learned} × use_kkt ∈ {0,1} × unroll_J ∈ {0,3}（核心网格），se2_embed 作为扩展在优胜组合上开启
- 控制方式：全部通过 YAML `train:` 段开关控制，无需改代码
- 运行建议：每组≥3 个随机种子；先跑核心 12 组，再对前 2–3 个候选补充 J=1 与 se2_embed=true
- 统计分析：对模块级/闭环级关键指标做多因素 ANOVA 分析主效应与二阶交互；事后比较使用 Bonferroni/Holm 校正；报告效应量（ηp² 或 Cohen’s d）

---

## 五、调试与优化

### 5.1 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| KKT损失不下降 | ρ设置不当 | 尝试ρ∈[0.1, 10] |
| PDHG不收敛 | 步长过大 | 减小σ,τ或增加迭代次数 |
| 训练loss震荡 | 学习率过高 | 降低lr或使用warmup |
| 推理时延过高 | PDHG迭代过多 | 启用早停或减少J |

### 5.2 性能分析工具

```python
# 使用PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    mu_list, lam_list, _ = dune_layer(point_flow, R_list, obs_points_list)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 六、检查清单

实施前请确认:

- [ ] 代码备份完成
- [ ] 环境依赖已安装
- [ ] 单元测试通过
- [ ] 配置文件正确
- [ ] 监控指标已设置
- [ ] 实验记录工具就绪(如TensorBoard)

实施后请验证:

- [ ] 训练损失正常下降
- [ ] 违反率<1%
- [ ] 推理时延<2ms
- [ ] 闭环成功率≥baseline
- [ ] 最小间距≥baseline

---

**下一步**: 参考 `DF-DUNE创新方案分析报告.md` 中的路线图,按阶段实施

