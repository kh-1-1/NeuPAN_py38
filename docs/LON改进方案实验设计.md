# LON 改进 NeuPAN 方案实验设计与评估

## 目录
1. [实验总体设计](#1-实验总体设计)
2. [对比实验详细设计](#2-对比实验详细设计)
3. [消融实验设计](#3-消融实验设计)
4. [泛化能力测试](#4-泛化能力测试)
5. [评估指标体系](#5-评估指标体系)
6. [数据收集与分析](#6-数据收集与分析)
7. [实验环境配置](#7-实验环境配置)

---

## 1. 实验总体设计

### 1.1 实验目标

| 目标 | 描述 | 验证方法 |
|------|------|---------|
| **有效性** | 验证 LON 方法能否提升 NeuPAN 性能 | 对比实验 |
| **鲁棒性** | 验证方法在不同场景下的稳定性 | 泛化测试 |
| **高效性** | 验证训练和推理的计算效率 | 时间分析 |
| **可解释性** | 理解参数优化的内在机制 | 消融实验 |

### 1.2 实验假设

**H1**: 在线学习能够自动找到优于手动调参的参数组合
**H2**: 多目标损失函数能够平衡安全性、效率和舒适性
**H3**: 课程学习能够加速训练收敛
**H4**: 优化更多参数（7个 vs 3个）能够提升性能
**H5**: 学到的参数具有跨场景泛化能力
**H6**: 固定 DUNE 网络不会限制整体性能提升

### 1.3 实验变量

#### 自变量（Independent Variables）
- 学习方法：Baseline, LON-Basic, LON-Extended, Adaptive-NeuPAN, Adaptive-NeuPAN-Full
- 场景复杂度：Easy, Medium, Hard
- 噪声水平：0.0, 0.1, 0.2
- 优化参数数量：0 (固定), 3, 5, 7
- 损失函数类型：单一距离损失, 多目标损失

#### 因变量（Dependent Variables）
- 成功率（Success Rate）
- 完成时间（Completion Time）
- 碰撞率（Collision Rate）
- 轨迹平滑度（Trajectory Smoothness）
- 能量消耗（Energy Consumption）

#### 控制变量（Control Variables）
- 机器人模型：差速驱动（diff）
- 预测时域：10 步
- 时间步长：0.1 秒
- 随机种子：固定为 [42, 123, 456, 789, 1024]

---

## 2. 对比实验详细设计

### 2.1 实验组设置

#### Group 1: Baseline (手动调参 NeuPAN)
```yaml
method: manual_tuning
parameters:
  q_s: 0.1
  p_u: 2.0
  eta: 10.0
  d_max: 0.2
  d_min: 0.01
learning: false
description: "专家手动调参的基线方法"
```

#### Group 2: LON-Basic (基础 LON)
```yaml
method: lon_basic
learnable_params: [p_u, eta, d_max]
fixed_params:
  q_s: 0.1
  d_min: 0.01
optimizer:
  type: Adam
  lr: 5e-3
loss_function: distance_only
description: "仅优化 3 个参数，单一距离损失"
```

#### Group 3: LON-Extended (扩展 LON)
```yaml
method: lon_extended
learnable_params: [q_s, p_u, eta, d_max, d_min]
optimizer:
  type: Adam
  lr: 5e-3
loss_function: distance_only
description: "优化 5 个参数，单一距离损失"
```

#### Group 4: Adaptive-NeuPAN (完整方案)
```yaml
method: adaptive_neupan
learnable_params: [q_s, p_u, eta, d_max, d_min]
optimizer:
  type: Adam
  lr: 5e-3
loss_function: multi_objective
  weights:
    distance: 10.0
    smoothness: 1.0
    energy: 0.5
    time: 1.0
    tracking: 2.0
curriculum_learning: true
description: "完整自适应方案，多目标损失 + 课程学习"
```

#### Group 5: Adaptive-NeuPAN-Full (完整 7 参数优化)
```yaml
method: adaptive_neupan_full
learnable_params: [q_s, p_u, eta, d_max, d_min, ro_obs, bk]
optimizer:
  type: Adam
  lr: 5e-3
loss_function: multi_objective
  weights:
    distance: 10.0
    smoothness: 1.0
    energy: 0.5
    time: 1.0
    tracking: 2.0
curriculum_learning: true
dune_fixed: true  # DUNE 网络权重固定
description: "完整方案：优化所有 7 个 NRMP 参数，DUNE 固定"
```

### 2.2 实验流程

```
对于每个实验组 (Group 1-5):
    对于每个随机种子 (5 个):
        对于每个场景 (LON_corridor, LON_corridor_01, LON_corridor_02):
            1. 初始化环境和规划器
            2. 训练阶段 (150 epochs)
                - 记录训练损失
                - 记录参数演化
                - 保存检查点
            3. 测试阶段 (100 episodes)
                - 记录成功率
                - 记录完成时间
                - 记录碰撞次数
                - 记录轨迹数据
            4. 保存结果
```

### 2.3 评估指标

| 指标 | 计算方法 | 期望值 |
|------|---------|--------|
| **成功率** | `成功次数 / 总次数` | > 95% |
| **平均完成时间** | `Σ完成时间 / 成功次数` | < 40s |
| **碰撞率** | `碰撞次数 / 总次数` | < 2% |
| **平均平滑度** | `1 / (1 + Σ‖加加速度‖)` | > 0.9 |
| **平均能量** | `Σv² * dt` | < 90 |
| **训练收敛速度** | `达到 90% 成功率的 epoch 数` | < 100 |

### 2.4 统计检验

- **显著性检验**: 使用 t-test 或 Mann-Whitney U test
- **显著性水平**: α = 0.05
- **效应量**: Cohen's d
- **置信区间**: 95%

---

## 3. 消融实验设计

### 3.1 实验目的

验证各个组件对整体性能的贡献度。

### 3.2 消融实验组

#### Ablation-1: 移除平滑度损失
```yaml
name: "No Smoothness Loss"
loss_weights:
  distance: 10.0
  smoothness: 0.0  # 移除
  energy: 0.5
  time: 1.0
  tracking: 2.0
expected_impact: "轨迹抖动增加"
```

#### Ablation-2: 移除能量损失
```yaml
name: "No Energy Loss"
loss_weights:
  distance: 10.0
  smoothness: 1.0
  energy: 0.0  # 移除
  time: 1.0
  tracking: 2.0
expected_impact: "能量消耗增加"
```

#### Ablation-3: 移除时间损失
```yaml
name: "No Time Loss"
loss_weights:
  distance: 10.0
  smoothness: 1.0
  energy: 0.5
  time: 0.0  # 移除
  tracking: 2.0
expected_impact: "完成时间增加"
```

#### Ablation-4: 移除跟踪损失
```yaml
name: "No Tracking Loss"
loss_weights:
  distance: 10.0
  smoothness: 1.0
  energy: 0.5
  time: 1.0
  tracking: 0.0  # 移除
expected_impact: "路径偏离增加"
```

#### Ablation-5: 固定损失权重
```yaml
name: "Fixed Loss Weights"
adaptive_weights: false  # 禁用自适应权重
expected_impact: "适应性下降"
```

#### Ablation-6: 移除课程学习
```yaml
name: "No Curriculum Learning"
curriculum_learning: false
expected_impact: "训练收敛变慢"
```

#### Ablation-7: 移除经验回放
```yaml
name: "No Experience Replay"
experience_replay: false
expected_impact: "训练稳定性下降"
```

#### Ablation-8: 移除梯度裁剪
```yaml
name: "No Gradient Clipping"
gradient_clipping: false
expected_impact: "训练可能发散"
```

### 3.3 消融实验分析

对于每个消融实验：
1. 运行 50 个 epochs
2. 记录关键指标
3. 与完整方案对比
4. 计算性能下降百分比

**性能贡献度计算**：
```
贡献度 = (完整方案性能 - 消融方案性能) / 完整方案性能 × 100%
```

---

## 4. 泛化能力测试

### 4.1 场景变体测试

#### 4.1.1 走廊宽度变化
```yaml
scenarios:
  - name: "Narrow Corridor"
    width: 2.0m
    difficulty: hard
  - name: "Medium Corridor"
    width: 4.0m
    difficulty: medium
  - name: "Wide Corridor"
    width: 6.0m
    difficulty: easy
```

#### 4.1.2 障碍物密度变化
```yaml
scenarios:
  - name: "Sparse Obstacles"
    density: 0.1
    num_obstacles: 3
  - name: "Medium Obstacles"
    density: 0.3
    num_obstacles: 6
  - name: "Dense Obstacles"
    density: 0.5
    num_obstacles: 10
```

#### 4.1.3 噪声水平变化
```yaml
scenarios:
  - name: "No Noise"
    lidar_noise_std: 0.0
  - name: "Low Noise"
    lidar_noise_std: 0.1
  - name: "Medium Noise"
    lidar_noise_std: 0.2
  - name: "High Noise"
    lidar_noise_std: 0.5
```

### 4.2 跨场景测试

#### 训练场景
- LON_corridor (走廊场景)

#### 测试场景
1. **dune_train** (开阔区域)
2. **dyna_obs** (动态障碍物)
3. **u_turn** (U 型转弯)
4. **parking** (停车场)
5. **reverse** (倒车入库)

### 4.3 迁移学习测试

#### 零样本迁移（Zero-shot Transfer）
```python
# 在走廊场景训练
train_on_scenario("LON_corridor")

# 直接在新场景测试（无微调）
test_on_scenario("dune_train", finetune=False)
```

#### 少样本适应（Few-shot Adaptation）
```python
# 在走廊场景训练
train_on_scenario("LON_corridor")

# 在新场景微调 10 个 episodes
finetune_on_scenario("dune_train", num_episodes=10)

# 测试
test_on_scenario("dune_train")
```

#### 跨域泛化（Cross-domain Generalization）
```python
# 在多个场景训练
train_on_scenarios(["LON_corridor", "dune_train", "u_turn"])

# 在未见过的场景测试
test_on_scenario("parking")
```

---

## 5. 评估指标体系

### 5.1 主要指标（Primary Metrics）

#### 5.1.1 成功率（Success Rate）
```python
def calculate_success_rate(results):
    """
    成功定义：到达目标且无碰撞
    """
    success_count = sum(1 for r in results if r['arrived'] and not r['collision'])
    return success_count / len(results)
```

#### 5.1.2 完成时间（Completion Time）
```python
def calculate_avg_completion_time(results):
    """
    仅统计成功案例的时间
    """
    success_times = [r['time'] for r in results if r['arrived']]
    return np.mean(success_times) if success_times else float('inf')
```

#### 5.1.3 碰撞率（Collision Rate）
```python
def calculate_collision_rate(results):
    """
    碰撞定义：最小距离 < 碰撞阈值
    """
    collision_count = sum(1 for r in results if r['collision'])
    return collision_count / len(results)
```

### 5.2 次要指标（Secondary Metrics）

#### 5.2.1 轨迹平滑度（Trajectory Smoothness）
```python
def calculate_smoothness(trajectory):
    """
    基于加加速度（jerk）计算平滑度
    """
    velocities = np.diff(trajectory['positions'], axis=0) / dt
    accelerations = np.diff(velocities, axis=0) / dt
    jerks = np.diff(accelerations, axis=0) / dt
    
    jerk_magnitude = np.linalg.norm(jerks, axis=1)
    smoothness = 1.0 / (1.0 + np.sum(jerk_magnitude))
    
    return smoothness
```

#### 5.2.2 能量消耗（Energy Consumption）
```python
def calculate_energy(trajectory):
    """
    基于速度平方积分计算能量
    """
    velocities = trajectory['velocities']
    energy = np.sum(velocities ** 2) * dt
    return energy
```

#### 5.2.3 路径长度（Path Length）
```python
def calculate_path_length(trajectory):
    """
    计算实际行驶路径长度
    """
    positions = trajectory['positions']
    segments = np.diff(positions, axis=0)
    length = np.sum(np.linalg.norm(segments, axis=1))
    return length
```

#### 5.2.4 最小安全距离（Minimum Safety Distance）
```python
def calculate_min_safety_distance(trajectory):
    """
    记录整个轨迹中的最小安全距离
    """
    return np.min(trajectory['distances'])
```

### 5.3 训练指标（Training Metrics）

#### 5.3.1 收敛速度（Convergence Speed）
```python
def calculate_convergence_speed(training_log, threshold=0.9):
    """
    达到目标成功率所需的 epoch 数
    """
    for epoch, success_rate in enumerate(training_log['success_rates']):
        if success_rate >= threshold:
            return epoch + 1
    return len(training_log['success_rates'])
```

#### 5.3.2 参数稳定性（Parameter Stability）
```python
def calculate_parameter_stability(param_history):
    """
    参数变化的标准差（越小越稳定）
    """
    param_changes = np.diff(param_history, axis=0)
    stability = 1.0 / (1.0 + np.std(param_changes))
    return stability
```

### 5.4 综合评分（Overall Score）

```python
def calculate_overall_score(metrics, weights=None):
    """
    加权综合评分
    """
    if weights is None:
        weights = {
            'success_rate': 0.3,
            'completion_time': 0.2,
            'collision_rate': 0.2,
            'smoothness': 0.15,
            'energy': 0.15
        }
    
    # 归一化指标
    normalized = {
        'success_rate': metrics['success_rate'],
        'completion_time': 1.0 / (1.0 + metrics['avg_time'] / 40.0),
        'collision_rate': 1.0 - metrics['collision_rate'],
        'smoothness': metrics['avg_smoothness'],
        'energy': 1.0 / (1.0 + metrics['avg_energy'] / 100.0)
    }
    
    # 加权求和
    score = sum(weights[k] * normalized[k] for k in weights)
    
    return score
```

---

## 6. 数据收集与分析

### 6.1 数据收集协议

#### 6.1.1 训练数据
```python
training_data = {
    'epoch': [],
    'total_loss': [],
    'loss_components': {
        'distance': [],
        'smoothness': [],
        'energy': [],
        'time': [],
        'tracking': []
    },
    'parameters': {
        'q_s': [],
        'p_u': [],
        'eta': [],
        'd_max': [],
        'd_min': []
    },
    'metrics': {
        'success_rate': [],
        'avg_time': [],
        'collision_rate': []
    }
}
```

#### 6.1.2 测试数据
```python
test_data = {
    'episode': [],
    'success': [],
    'collision': [],
    'time': [],
    'path_length': [],
    'min_distance': [],
    'smoothness': [],
    'energy': [],
    'trajectory': []  # 完整轨迹数据
}
```

### 6.2 数据分析方法

#### 6.2.1 描述性统计
```python
def descriptive_statistics(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }
```

#### 6.2.2 假设检验
```python
from scipy import stats

def compare_methods(method1_data, method2_data):
    """
    比较两种方法的性能差异
    """
    # t-test
    t_stat, p_value = stats.ttest_ind(method1_data, method2_data)
    
    # Mann-Whitney U test (非参数)
    u_stat, u_p_value = stats.mannwhitneyu(method1_data, method2_data)
    
    # Cohen's d (效应量)
    cohens_d = (np.mean(method1_data) - np.mean(method2_data)) / \
               np.sqrt((np.std(method1_data)**2 + np.std(method2_data)**2) / 2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'u_statistic': u_stat,
        'u_p_value': u_p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
```

#### 6.2.3 可视化分析
```python
def visualize_results(results):
    """
    生成可视化图表
    """
    # 1. 箱线图：对比不同方法
    plt.figure(figsize=(12, 6))
    plt.boxplot([results[m]['success_rate'] for m in methods])
    plt.title('Success Rate Comparison')
    plt.savefig('success_rate_boxplot.png')
    
    # 2. 折线图：训练曲线
    plt.figure(figsize=(12, 6))
    for method in methods:
        plt.plot(results[method]['training_loss'], label=method)
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig('training_curves.png')
    
    # 3. 热力图：参数演化
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['parameter_evolution'], cmap='viridis')
    plt.title('Parameter Evolution Heatmap')
    plt.savefig('parameter_heatmap.png')
```

---

## 7. 实验环境配置

### 7.1 硬件环境

```yaml
hardware:
  cpu: Intel i7-10700K (8 cores, 16 threads)
  ram: 32 GB DDR4
  gpu: NVIDIA RTX 3080 (10 GB VRAM) [可选]
  storage: 500 GB SSD
```

### 7.2 软件环境

```yaml
software:
  os: Ubuntu 20.04 LTS / Windows 10
  python: 3.8.10
  pytorch: 1.12.0
  cuda: 11.3 [可选]
  dependencies:
    - numpy==1.21.0
    - matplotlib==3.4.2
    - scipy==1.7.0
    - pandas==1.3.0
    - tensorboard==2.6.0
    - ir-sim==1.0.0
    - cvxpy==1.2.0
    - cvxpylayers==0.1.5
```

### 7.3 实验配置管理

使用配置文件管理实验参数：

```yaml
# experiments/config/experiment_001.yaml
experiment:
  name: "Adaptive-NeuPAN vs Baseline"
  description: "对比自适应方法与基线方法"
  date: "2025-01-XX"
  
  groups:
    - baseline
    - lon_basic
    - adaptive_neupan
  
  scenarios:
    - LON_corridor
    - LON_corridor_01
    - LON_corridor_02
  
  num_seeds: 5
  num_train_epochs: 150
  num_test_episodes: 100
  
  save_dir: "results/experiment_001"
  log_interval: 10
  checkpoint_interval: 50
```

---

## 附录：实验检查清单

### 训练前检查
- [ ] 环境配置正确
- [ ] 数据目录创建
- [ ] 配置文件验证
- [ ] 随机种子设置
- [ ] 日志系统初始化

### 训练中监控
- [ ] 损失是否收敛
- [ ] 参数是否在合理范围
- [ ] 成功率是否提升
- [ ] 内存使用是否正常
- [ ] 定期保存检查点

### 训练后分析
- [ ] 收集所有实验数据
- [ ] 生成可视化图表
- [ ] 进行统计检验
- [ ] 撰写实验报告
- [ ] 归档实验结果

---

**文档版本**: v1.0  
**创建日期**: 2025-01-XX  
**作者**: AI Assistant

