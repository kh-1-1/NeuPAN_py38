# DF-DUNE 技术实现指南

## 目录
1. [快速开始](#快速开始)
2. [核心模块实现](#核心模块实现)
3. [训练流程改造](#训练流程改造)
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

### 1.2 最小可行实现(MVP)

**目标**: 在不修改现有代码的情况下,通过配置开关启用硬投影

```yaml
# example/dune_train/dune_train_acker_df.yaml
robot:
  kinematics: 'acker'
  length: 4.6
  width: 1.6
  wheelbase: 3

train:
  direct_train: true
  data_size: 100000
  data_range: [-25, -25, 25, 25]
  batch_size: 256
  epoch: 5000
  valid_freq: 250
  save_freq: 500
  lr: 5e-5
  lr_decay: 0.5
  decay_freq: 1500
  
  # DF-DUNE 配置
  projection: 'hard'           # 启用硬投影
  monitor_dual_norm: true      # 监控违反率
  unroll_J: 0                  # 暂不启用PDHG
  se2_embed: false             # 暂不启用等变
```

**验证**: 运行训练并检查日志中的违反率

```bash
python example/dune_train/dune_train_acker.py --config dune_train_acker_df.yaml
# 查看输出中的 dual_norm_violation_rate 和 dual_norm_p95
```

---

## 二、核心模块实现

### 2.1 KKT残差正则(A-3) ⭐ 优先级最高

#### 实现位置
`neupan/blocks/dune_train.py` 的 `train_one_epoch` 方法

#### 代码补丁

```python
# 在 DUNETrain 类中添加新方法
def cal_loss_kkt(self, output_mu, input_point, rho=1.0):
    """
    计算KKT残差损失
    
    KKT条件(惩罚化形式):
    ∇f(μ) = -a + ρ*G*G^T*μ - s ≈ 0
    其中 s = ReLU(-μ) 是互补松弛项
    
    Args:
        output_mu: [batch, E, 1]
        input_point: [batch, 2]
        rho: 惩罚系数
    
    Returns:
        L_KKT: KKT残差的L2范数
    """
    batch_size = output_mu.shape[0]
    
    # 计算 a = G*p - h
    ip = input_point.unsqueeze(2)  # [batch, 2, 1]
    a = self.G @ ip - self.h       # [batch, E, 1]
    
    # 计算梯度: -a + ρ*G*G^T*μ
    GTmu = self.G.T @ output_mu    # [batch, 2, 1]
    grad_f = -a + rho * (self.G @ GTmu)  # [batch, E, 1]
    
    # 互补松弛项: s ≈ ReLU(-μ)
    s = torch.relu(-output_mu)
    
    # KKT残差
    kkt_residual = grad_f - s      # [batch, E, 1]
    
    # L2损失
    loss_kkt = torch.mean(torch.sum(kkt_residual ** 2, dim=1))
    
    return loss_kkt


# 修改 train_one_epoch 方法
def train_one_epoch(self, train_dataloader, validate=False, kkt_weight=0.1, kkt_rho=1.0):
    """
    新增参数:
        kkt_weight: KKT损失的权重
        kkt_rho: KKT惩罚系数
    """
    mu_loss, distance_loss, fa_loss, fb_loss, kkt_loss = 0, 0, 0, 0, 0

    for input_point, label_mu, label_distance in train_dataloader:
        self.optimizer.zero_grad()

        input_point = torch.squeeze(input_point)
        output_mu = self.model(input_point)
        output_mu = torch.unsqueeze(output_mu, 2)

        distance = self.cal_distance(output_mu, input_point)

        mse_mu = self.loss_fn(output_mu, label_mu)
        mse_distance = self.loss_fn(distance, label_distance)
        mse_fa, mse_fb = self.cal_loss_fab(output_mu, label_mu, input_point)
        
        # 新增: KKT残差损失
        mse_kkt = self.cal_loss_kkt(output_mu, input_point, rho=kkt_rho)

        # 总损失
        loss = mse_mu + mse_distance + mse_fa + mse_fb + kkt_weight * mse_kkt

        if not validate:
            loss.backward()
            self.optimizer.step()

        mu_loss += mse_mu.item()
        distance_loss += mse_distance.item()
        fa_loss += mse_fa.item()
        fb_loss += mse_fb.item()
        kkt_loss += mse_kkt.item()

    return (
        mu_loss / len(train_dataloader),
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

