# DF-DUNE 实验设计方案

## 目录

1. [实验目标](#实验目标)
2. [对比方法](#对比方法)
3. [评测指标体系](#评测指标体系)
4. [实验场景设计](#实验场景设计)
5. [消融研究](#消融研究)
6. [统计分析方法](#统计分析方法)

---

## 一、实验目标

### 1.1 核心研究问题(RQ)

**RQ1**: 硬约束神经化(A-1/A-2/A-3)能否显著降低对偶可行性违反率?

- **假设**: 违反率从baseline的~10%降至<1%
- **关键指标**: `dual_norm_violation_rate`, `dual_norm_p95`

**RQ2**: PDHG展开(B-1)能否在保持实时性的同时提升距离估计精度?

- **假设**: 距离MAE降低20%,时延增加<100%
- **关键指标**: `distance_mae`, `inference_time_p95`

**RQ3**: SE(2)等变编码(C)能否提升旋转鲁棒性?

- **假设**: OOD旋转场景下,性能下降<5%
- **关键指标**: `rotation_robustness_score`

**RQ4**: DF-DUNE能否在闭环导航中提升安全性与成功率?

- **假设**: 成功率+5%, 最小间距+20%
- **关键指标**: `success_rate`, `min_clearance_mean`

### 1.2 实验层次

```
Level 1: 模块级评测 (DUNE单独)
    ├─ 几何精度: d, λ的误差
    ├─ 约束满足: 违反率, 上界紧致度
    └─ 计算效率: 时延, 吞吐

Level 2: 系统级评测 (DUNE + NRMP)
    ├─ 优化质量: 轨迹代价, 收敛速度
    └─ 安全性: 最小间距, 碰撞率

Level 3: 闭环评测 (完整NeuPAN)
    ├─ 任务成功: 到达率, 路径长度
    ├─ 实时性: 规划频率, 总时间
    └─ 鲁棒性: OOD场景, 动态障碍
```

---

## 二、对比方法

### 2.1 内部变体(消融)

| 方法ID       | 描述                  | 配置                     |
| ------------ | --------------------- | ------------------------ |
| **M0** | Baseline (原始NeuPAN) | `projection='none'`    |
| **M1** | +硬投影               | `projection='hard'`    |
| **M2** | M1 + KKT正则          | `+kkt_weight=0.1`      |
| **M3** | M2 + PDHG(J=1)        | `+unroll_J=1`          |
| **M4** | M2 + PDHG(J=3)        | `+unroll_J=3`          |
| **M5** | M4 + SE(2)等变        | `+se2_embed=True`      |
| **M6** | M2 + Learned-Prox     | `projection='learned'` |

### 2.2 外部基线(可选)

| 方法                   | 类型      | 优势       | 劣势              |
| ---------------------- | --------- | ---------- | ----------------- |
| **NVBlox-ESDF**  | GPU ESDF  | 工业级实时 | 需要体素化,内存大 |
| **Neural-SDF**   | 学习型SDF | 平滑梯度   | 训练慢,泛化差     |
| **传统MPC+ESDF** | 优化+地图 | 成熟稳定   | 计算慢,难处理动态 |

**建议**: 优先完成内部消融(M0-M6),外部基线作为附录或未来工作

---

## 三、评测指标体系

### 3.1 模块级指标(DUNE)

#### 几何精度

```python
# 距离误差
distance_metrics = {
    'mae': mean_absolute_error(d_pred, d_true),
    'rmse': root_mean_squared_error(d_pred, d_true),
    'max_error': max_absolute_error(d_pred, d_true),
    'r2_score': r2_score(d_pred, d_true),
}

# 方向误差
direction_metrics = {
    'angle_error_mean': mean(angle_between(λ_pred, λ_true)),
    'angle_error_p95': percentile(angle_between(λ_pred, λ_true), 95),
    'cosine_similarity': mean(cosine_sim(λ_pred, λ_true)),
}
```

#### 约束满足

```python
# 对偶可行性
feasibility_metrics = {
    'violation_rate': mean(||G^T μ||_2 > 1),
    'violation_p95': percentile(||G^T μ||_2, 95),
    'violation_max': max(||G^T μ||_2),
    'nonneg_violation_rate': mean(μ < 0),
}

# 安全上界紧致度
tightness_metrics = {
    'slack_mean': mean(max(0, d_pred - d_true)),
    'slack_p95': percentile(max(0, d_pred - d_true), 95),
    'optimistic_rate': mean(d_pred < d_true),  # 危险!应接近0
    'optimistic_gap_mean': mean(max(0, d_true - d_pred)),
}
```

#### 计算效率

```python
efficiency_metrics = {
    'inference_time_mean': mean(forward_time),  # μs/点
    'inference_time_p95': percentile(forward_time, 95),
    'throughput': points_per_second,
    'memory_peak': max_gpu_memory_mb,
}
```

### 3.2 系统级指标(DUNE+NRMP)

```python
system_metrics = {
    # 优化质量
    'trajectory_cost': final_objective_value,
    'convergence_iters': num_iterations_to_converge,
    'qp_solve_time': cvxpylayer_time,
  
    # 安全性
    'min_clearance': min_distance_to_obstacles,
    'clearance_p5': percentile(distance_to_obstacles, 5),
    'constraint_violation_count': num_violated_constraints,
}
```

### 3.3 闭环指标(完整NeuPAN)

```python
closed_loop_metrics = {
    # 任务成功
    'success_rate': reached_goal_within_threshold / total_episodes,
    'path_length_mean': mean(trajectory_length),
    'path_efficiency': mean(trajectory_length / straight_line_distance),
  
    # 实时性
    'planning_freq_mean': mean(1 / planning_cycle_time),
    'planning_freq_p5': percentile(1 / planning_cycle_time, 5),
    'total_time_mean': mean(episode_duration),
  
    # 鲁棒性
    'collision_rate': collision_occurred / total_episodes,
    'timeout_rate': timeout_occurred / total_episodes,
    'recovery_success_rate': recovered_from_stuck / stuck_situations,
}
```

---

## 四、实验场景设计

### 4.1 模块级测试集

#### 数据集A: 标准分布(In-Distribution)

```python
# 与训练集同分布
test_set_A = {
    'size': 10000,
    'data_range': [-25, -25, 25, 25],  # 与训练一致
    'robot': 'acker',
    'purpose': '验证基本性能'
}
```

#### 数据集B: 边界情况(Boundary Cases)

```python
test_set_B = {
    'size': 5000,
    'scenarios': [
        {'type': 'near_robot', 'distance_range': [0.1, 1.0]},  # 靠近机器人
        {'type': 'far_robot', 'distance_range': [20, 50]},     # 远离机器人
        {'type': 'on_edge', 'constraint_margin': 0.05},        # 接近约束边界
    ],
    'purpose': '测试极端情况'
}
```

#### 数据集C: 分布外(Out-of-Distribution)

```python
test_set_C = {
    'size': 5000,
    'scenarios': [
        {'type': 'larger_range', 'data_range': [-50, -50, 50, 50]},
        {'type': 'different_robot', 'robot': 'diff'},  # 不同机器人
        {'type': 'rotated', 'rotation_angles': np.linspace(0, 2*np.pi, 36)},
    ],
    'purpose': '测试泛化能力'
}
```

### 4.2 闭环场景库

#### 场景1: 静态障碍(基础)

```python
scenario_static = {
    'name': 'Static Obstacles',
    'num_episodes': 50,
    'obstacles': {
        'type': 'random_polygons',
        'count': [5, 15],
        'size_range': [1.0, 5.0],
    },
    'start_goal': {
        'distance_range': [20, 40],
        'clearance_min': 2.0,
    },
    'difficulty': 'easy',
}
```

#### 场景2: 狭窄通道(挑战)

```python
scenario_corridor = {
    'name': 'Narrow Corridor',
    'num_episodes': 30,
    'corridor': {
        'width': [2.5, 4.0],  # 机器人宽度1.6m
        'length': [15, 25],
        'turns': [1, 3],
    },
    'difficulty': 'hard',
}
```

#### 场景3: 动态障碍(高级)

```python
scenario_dynamic = {
    'name': 'Dynamic Obstacles',
    'num_episodes': 40,
    'obstacles': {
        'static_count': [3, 8],
        'dynamic_count': [2, 5],
        'dynamic_velocity': [0.5, 2.0],  # m/s
    },
    'difficulty': 'very_hard',
}
```

#### 场景4: 真实环境(验证)

```python
scenario_real = {
    'name': 'Real-world Maps',
    'source': ['warehouse', 'parking_lot', 'campus'],
    'num_episodes_per_map': 20,
    'difficulty': 'realistic',
}
```

---

## 五、消融研究

### 5.1 消融矩阵

| 实验ID | 硬投影 | KKT正则 | PDHG | SE(2) | 目的           |
| ------ | ------ | ------- | ---- | ----- | -------------- |
| Exp-1  | ❌     | ❌      | ❌   | ❌    | Baseline       |
| Exp-2  | ✅     | ❌      | ❌   | ❌    | 硬投影单独效果 |
| Exp-3  | ✅     | ✅      | ❌   | ❌    | +KKT正则       |
| Exp-4  | ✅     | ✅      | J=1  | ❌    | +PDHG(轻量)    |
| Exp-5  | ✅     | ✅      | J=3  | ❌    | +PDHG(完整)    |
| Exp-6  | ✅     | ✅      | J=3  | ✅    | 完整DF-DUNE    |
| Exp-7  | ✅     | ❌      | J=3  | ❌    | 验证KKT必要性  |
| Exp-8  | ✅     | ✅      | ❌   | ✅    | 验证PDHG必要性 |

### 5.2 超参数敏感性分析

#### KKT权重

```python
kkt_weight_sweep = {
    'values': [0.01, 0.05, 0.1, 0.5, 1.0],
    'metric': 'violation_rate',
    'expected': '0.1附近最优',
}
```

#### PDHG迭代次数

```python
pdhg_iters_sweep = {
    'values': [1, 2, 3, 5],
    'metrics': ['distance_mae', 'inference_time'],
    'tradeoff': 'J=3时性价比最高',
}
```

#### PDHG步长

```python
pdhg_steps_sweep = {
    'sigma': [0.5, 0.7, 0.9, 1.1] / ||G||,
    'tau': [0.5, 0.7, 0.9, 1.1] / ||G||,
    'metric': 'convergence_rate',
}
```

---

## 六、统计分析方法

### 6.1 显著性检验

```python
from scipy.stats import ttest_rel, wilcoxon

def statistical_test(baseline_results, method_results, metric='success_rate'):
    """
    配对t检验(参数) 或 Wilcoxon符号秩检验(非参数)
    """
    # 正态性检验
    _, p_normal = shapiro(baseline_results[metric])
  
    if p_normal > 0.05:
        # 使用t检验
        t_stat, p_value = ttest_rel(method_results[metric], baseline_results[metric])
        test_name = 'Paired t-test'
    else:
        # 使用Wilcoxon检验
        t_stat, p_value = wilcoxon(method_results[metric], baseline_results[metric])
        test_name = 'Wilcoxon signed-rank test'
  
    return {
        'test': test_name,
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

### 6.2 效应量(Effect Size)

```python
def cohens_d(group1, group2):
    """Cohen's d: 标准化均值差"""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std

# 解释:
# |d| < 0.2: 小效应
# 0.2 ≤ |d| < 0.5: 中等效应
# |d| ≥ 0.5: 大效应
```

### 6.3 多重比较校正

```python
from statsmodels.stats.multitest import multipletests

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni校正: 控制家族错误率"""
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    return reject, p_corrected
```

### 6.4 置信区间

```python
from scipy.stats import t

def confidence_interval(data, confidence=0.95):
    """计算均值的置信区间"""
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    margin = std_err * t.ppf((1 + confidence) / 2, n - 1)
  
    return (mean - margin, mean + margin)
```

---

## 七、实验执行计划

### 7.1 时间线(4周)

**Week 1: 模块级评测**

- Day 1-2: 准备测试集A/B/C
- Day 3-4: 运行M0-M6,收集几何精度指标
- Day 5-6: 约束满足与效率分析
- Day 7: 数据整理与初步分析

**Week 2: 消融研究**

- Day 1-3: 执行Exp-1至Exp-8
- Day 4-5: 超参数扫描(KKT, PDHG)
- Day 6-7: 统计检验与效应量计算

**Week 3: 闭环评测**

- Day 1-2: 场景1(静态障碍)
- Day 3: 场景2(狭窄通道)
- Day 4: 场景3(动态障碍)
- Day 5: 场景4(真实地图)
- Day 6-7: 闭环指标汇总

**Week 4: 分析与可视化**

- Day 1-3: 生成图表(见下节)
- Day 4-5: 撰写实验章节
- Day 6-7: 审查与补充实验

### 7.2 计算资源需求

- **GPU**: 1x RTX 3060
- **训练时间**: ~3小时/方法 × 7方法 = 21小时
- **评测时间**: ~10小时(模块级) + 20小时(闭环) = 30小时
- **总计**: ~50 GPU小时

---

## 八、可视化方案

### 8.1 核心图表清单

1. **Figure 1: 违反率对比** (柱状图)

   - X轴: M0-M6
   - Y轴: violation_rate (%)
   - 误差棒: 95% CI
2. **Figure 2: 距离误差vs时延** (散点图)

   - X轴: inference_time (ms)
   - Y轴: distance_mae
   - 点: 不同方法,大小表示成功率
3. **Figure 3: PDHG收敛曲线** (折线图)

   - X轴: 迭代次数
   - Y轴: ||μ^k - μ*||
   - 多条线: 不同初始化
4. **Figure 4: 消融热图** (heatmap)

   - 行: Exp-1至Exp-8
   - 列: 各项指标
   - 颜色: 归一化性能
5. **Figure 5: 闭环成功率** (箱线图)

   - X轴: 场景1-4
   - Y轴: success_rate
   - 分组: M0, M5(DF-DUNE)
6. **Figure 6: 轨迹可视化** (2D路径图)

   - 典型成功/失败案例
   - 对比M0 vs M5

### 8.2 表格清单

- **Table 1**: 方法配置总结
- **Table 2**: 模块级指标全表
- **Table 3**: 闭环指标全表
- **Table 4**: 统计检验结果(p-value, Cohen's d)
- **Table 5**: 计算效率对比

---

## 九、实验检查清单

### 执行前

- [ ] 所有方法代码已实现并通过单元测试
- [ ] 测试集已生成并验证
- [ ] 随机种子已固定(建议5个: 42, 123, 456, 789, 2024)
- [ ] 日志与检查点保存路径已配置
- [ ] GPU资源已预留

### 执行中

- [ ] 每个实验运行≥5次(不同种子)
- [ ] 实时监控训练曲线,及时发现异常
- [ ] 定期备份中间结果
- [ ] 记录意外情况(如OOM, 不收敛)

### 执行后

- [ ] 所有指标已计算并保存
- [ ] 统计检验已完成
- [ ] 图表已生成并审查
- [ ] 结果已与预期假设对比
- [ ] 异常结果已分析原因

---

## 十、预期结果与备选方案

### 10.1 乐观情景(Best Case)

- 违反率: 10% → <1% ✅
- 距离MAE: -20% ✅
- 成功率: +5% ✅
- 时延: +50% (可接受) ✅

**行动**: 直接投稿顶会,强调所有改进

### 10.2 中性情景(Expected Case)

- 违反率: 10% → 2-3% ✅
- 距离MAE: -10% ✅
- 成功率: +2-3% ⚠️
- 时延: +80% ⚠️

**行动**: 强调约束满足与安全性,淡化成功率;优化时延

### 10.3 悲观情景(Worst Case)

- 违反率改善不明显 ❌
- 闭环性能下降 ❌

**行动**:

1. 回退到M2(硬投影+KKT),放弃PDHG
2. 重新定位为"约束满足"而非"性能提升"
3. 考虑投稿workshop或arxiv

---

**下一步**: 根据本方案执行实验,结果记录在 `experiments/results/` 目录
