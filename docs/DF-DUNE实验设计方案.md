# DF-DUNE 实验设计方案

> **文档状态**: ✅ 已更新（2025-01-XX）
> **实现进度**: A-1 ✅ | A-2 ✅ | A-3 ✅ | B-1 ✅ | C ✅ | 配置链路 ✅
> **待实施**: B-2 ❌（未来工作）

## 目录

1. [实验目标](#实验目标)
2. [对比方法](#对比方法)
3. [评测指标体系](#评测指标体系)
4. [实验场景设计](#实验场景设计)
5. [消融研究](#消融研究)
6. [统计分析方法](#统计分析方法)
7. [已实现模块状态](#已实现模块状态)

---

## 一、实验目标

### 1.1 核心研究问题(RQ)

**RQ1**: 硬约束神经化(A-1/A-3)能否显著降低对偶可行性违反率?

- **假设**: 违反率从baseline的~10%降至<1%
- **关键指标**: `dual_norm_violation_rate`, `dual_norm_p95`
- **实现状态**: ✅ A-1（硬投影）已实现 | ✅ A-3（KKT正则）已实现

**RQ2（核心创新）**: Learned-Prox (A-2) 相比于硬投影/PDHG，是否能以极低开销(<1ms)进一步降低违反率并提升几何精度？

- **假设**: M6 对比 M2（硬投影+KKT）可降低 violation_rate 和 distance_mae；M6+PDHG 对比 M4 进一步提升
- **关键指标**: `dual_norm_violation_rate`, `distance_mae`, `inference_time_mean`
- **实现状态**: ✅ A-2（Learned-Prox）已实现（`neupan/blocks/learned_prox.py`）

**RQ3**: PDHG展开(B-1)能否在保持实时性的同时提升距离估计精度?

- **假设**: 距离MAE降低20%, 时延增加<100%
- **关键指标**: `distance_mae`, `inference_time_p95`, `avg_pdhg_time_ms`
- **实现状态**: ✅ B-1（PDHG-Unroll）已实现，含时延埋点

**RQ4**: SE(2)等变编码(C)能否提升旋转鲁棒性?

- **假设**: OOD旋转场景下, 性能下降<5%
- **关键指标**: `rotation_robustness_score`
- **实现状态**: ✅ C（极坐标编码）已实现

**RQ5**: DF-DUNE能否在闭环导航中提升安全性与成功率?

- **假设**: 成功率+5%, 最小间距+20%
- **关键指标**: `success_rate`, `min_clearance_mean`
- **实现状态**: ✅ 所有核心模块已就绪，可直接评测

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

### 2.1 内部变体(消融) - 已更新配置映射

| 方法ID | 描述 | YAML 配置 | 实现状态 |
|--------|------|-----------|----------|
| **M0** | Baseline (原始NeuPAN) | `projection: none` | ✅ 可用 |
| **M1** | +硬投影 | `projection: hard` | ✅ 可用 |
| **M2** | M1 + KKT正则 | `projection: hard`<br>`use_kkt: true`<br>`w_kkt: 1e-3`<br>`kkt_rho: 0.5` | ✅ 可用 |
| **M3** | M2 + PDHG(J=1) | M2配置 + `unroll_J: 1` | ✅ 可用 |
| **M4** | M2 + PDHG(J=3) | M2配置 + `unroll_J: 3` | ✅ 可用 |
| **M5** | M4 + SE(2)等变 | M4配置 + `se2_embed: true` | ✅ 可用 |
| **M6** | Learned-Prox + KKT | `projection: learned`<br>`use_kkt: true`<br>`w_kkt: 1e-3` | ✅ 可用 |

**配置说明**:
- 所有开关统一在 YAML 的 `train:` 段管理
- **M0-M6 全部可直接运行**，无需额外代码修改
- M6 使用 `ProxHead` 模块（`neupan/blocks/learned_prox.py`，70 行轻量级 MLP）

**完整 YAML 示例**（M5 配置）:
```yaml
train:
  projection: hard          # {none|hard|learned}
  use_kkt: true             # 启用 KKT 正则
  w_kkt: 1e-3               # KKT 损失权重
  kkt_rho: 0.5              # KKT 惩罚参数
  unroll_J: 3               # PDHG 展开步数（0=禁用）
  se2_embed: true           # SE(2) 极坐标编码
  use_lconstr: true         # 启用对偶约束损失
  w_constr: 0.1             # 约束损失权重
```

### 2.2 外部基线(可选)

| 方法 | 类型 | 优势 | 劣势 |
|------|------|------|------|
| **NVBlox-ESDF** | GPU ESDF | 工业级实时 | 需要体素化,内存大 |
| **Neural-SDF** | 学习型SDF | 平滑梯度 | 训练慢,泛化差 |
| **传统MPC+ESDF** | 优化+地图 | 成熟稳定 | 计算慢,难处理动态 |

**建议**: 优先完成内部消融(核心网格),外部基线作为附录或未来工作

### 2.3 统一因子化消融设计（核心网格与扩展）

- 因子与水平（统一纳管所有创新点）
  - projection ∈ {none, hard, learned}
  - use_kkt ∈ {false, true}
  - unroll_J ∈ {0, 3}（核心网格；J=1 放在扩展）
  - se2_embed ∈ {false}（核心网格；true 放在扩展）
- 核心网格：3 × 2 × 2 = 12 组，全部可通过 YAML 的 train 段开关直接运行（零代码改动）
- 扩展补充（在核心网格优胜组合上加测）
  - 为前 2–3 个候选补充 unroll_J: 1（形成 J 的收益曲线）
  - 为最佳 hard/learned 轨道各加 se2_embed: true（旋转 OOD 验证）
  - 可选：在 learned + KKT 路线下补充 unroll_J: 3, se2_embed: true 的“全栈”组（论文主方法）

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

### 5.1 统一因子化消融（核心网格 12 组，可直接运行）

| 实验ID | projection | use_kkt | unroll_J | se2_embed | 目的 |
|--------|------------|---------|----------|-----------|------|
| Exp-1  | none       | false   | 0        | false     | Baseline |
| Exp-2  | none       | true    | 0        | false     | KKT 对 baseline 的作用 |
| Exp-3  | none       | false   | 3        | false     | 仅 PDHG 的作用 |
| Exp-4  | none       | true    | 3        | false     | PDHG × KKT 交互 |
| Exp-5  | hard       | false   | 0        | false     | 硬投影单独作用 |
| Exp-6  | hard       | true    | 0        | false     | 硬投影 × KKT |
| Exp-7  | hard       | false   | 3        | false     | 硬投影 × PDHG |
| Exp-8  | hard       | true    | 3        | false     | 硬投影 × KKT × PDHG |
| Exp-9  | learned    | false   | 0        | false     | Learned-Prox 独立作用 |
| Exp-10 | learned    | true    | 0        | false     | Learned-Prox × KKT |
| Exp-11 | learned    | false   | 3        | false     | Learned-Prox × PDHG |
| Exp-12 | learned    | true    | 3        | false     | Learned-Prox × KKT × PDHG |

- 以上 12 组即“核心网格”，全部通过 YAML `train:` 段开关控制，无需任何代码修改。
- 建议每组至少 3 次不同随机种子重复，以支持后续 ANOVA 与事后多重比较。

**扩展补充（小成本补齐关键交互）**
- 为得分最高的 hard/learned 轨道各加 `se2_embed: true`（旋转 OOD 评估）
- 为前 2–3 个候选加 `unroll_J: 1`（补齐 J 的收益曲线）

**YAML 片段示例**（以 Exp-12 为例）
```yaml
train:
  projection: learned
  use_kkt: true
  w_kkt: 1e-3
  kkt_rho: 0.5
  unroll_J: 3
  se2_embed: false
```

### 5.2 超参数敏感性分析

#### KKT权重（w_kkt）

```python
kkt_weight_sweep = {
    'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],  # 更新推荐范围
    'metric': 'violation_rate',
    'expected': '1e-3 附近最优（Acker）, 1e-4（Diff）',
    'yaml_key': 'train.w_kkt',
}
```

#### PDHG迭代次数（unroll_J）

```python
pdhg_iters_sweep = {
    'values': [0, 1, 2, 3, 5],  # 0=禁用
    'metrics': ['distance_mae', 'inference_time', 'avg_pdhg_time_ms'],
    'tradeoff': 'J=2~3 时性价比最高（精度提升 vs 时延增加）',
    'yaml_key': 'train.unroll_J',
}
```

#### PDHG步长（tau/sigma）

```python
pdhg_steps_sweep = {
    'sigma': [0.3, 0.5, 0.7, 0.9],  # 默认 0.5
    'tau': [0.3, 0.5, 0.7, 0.9],    # 默认 0.5
    'constraint': 'sigma * tau * ||G||^2 < 1',  # 收敛条件
    'metric': 'convergence_rate',
    'note': '当前实现为固定步长，可选 learnable=True 启用可学习步长',
}
```

#### KKT惩罚参数（kkt_rho）

```python
kkt_rho_sweep = {
    'values': [0.1, 0.3, 0.5, 0.7, 1.0],
    'metric': 'kkt_residual_norm',
    'expected': '0.5 附近最优（Acker）, 0.1（Diff）',
    'yaml_key': 'train.kkt_rho',
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

2b. **Figure 2b: Learned-Prox 对比（M2 vs M6, M4 vs M6+PDHG）**

   - X轴: 方法组（M2, M6, M4, M6+PDHG）
   - Y轴: violation_rate / distance_mae（双轴）
   - 备注: 柱状+折线组合，标注时延开销


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

## 九、实验检查清单（已更新实现状态）

### 执行前

- [x] ✅ 核心方法代码已实现（A-1, A-3, B-1, C）
- [x] ✅ YAML 配置链路已打通（train: 段统一管理）
- [x] ✅ 时延埋点已集成（PDHG timing, dual_norm monitoring）
- [ ] ⚠️ 测试集已生成并验证（需执行数据生成脚本）
- [ ] 随机种子已固定(建议5个: 42, 123, 456, 789, 2024)
- [ ] 日志与检查点保存路径已配置
- [ ] GPU资源已预留

### 训练阶段（如需重新训练）

- [ ] 准备 Baseline 模型（M0: projection=none）
- [ ] 训练 M1-M5 模型（使用对应 YAML 配置）
- [ ] 每个配置训练 ≥3 次（不同随机种子）
- [ ] 记录训练元数据（train_dict.pkl 包含所有开关）
- [ ] 验证 se2_embed、use_kkt 等参数正确保存

**训练时长估算**（基于文档）:
- 单个模型（Acker）: ~2500 epochs × 100k samples → 约 4-6 小时
- 单个模型（Diff）: ~1000 epochs × 100k samples → 约 2-3 小时
- M0-M5 全量训练: 约 20-30 GPU 小时

### 评测执行中

- [ ] 每个实验运行≥5次(不同种子)
- [ ] 实时监控评测指标（violation_rate, distance_mae）
- [ ] 定期备份中间结果（CSV/JSON）
- [ ] 记录意外情况(如OOM, 不收敛, 卡死)
- [ ] 验证 PDHG 时延数据正确记录（info['pdhg_profile']）

### 执行后

- [ ] 所有指标已计算并保存
- [ ] 统计检验已完成（t-test / Wilcoxon）
- [ ] 图表已生成并审查（违反率柱状图、时延散点图、收敛曲线）
- [ ] 结果已与预期假设对比（RQ1-RQ5）
- [ ] 异常结果已分析原因（如 PDHG 未降低违反率 → 检查步长/迭代次数）

---

## 十、已实现模块状态总结

### 10.1 核心模块实现清单

| 模块 | 组件 | 实现文件 | 状态 | 配置参数 |
|------|------|----------|------|----------|
| **A-1** | 硬投影 | `neupan/blocks/dune.py` (L109-119) | ✅ 完整 | `projection: hard` |
| **A-2** | Learned-Prox | `neupan/blocks/learned_prox.py` (70行) | ✅ 完整 | `projection: learned` |
| **A-3** | KKT 正则 | `neupan/blocks/dune_train.py` (L398-440) | ✅ 完整 | `use_kkt: true, w_kkt: 1e-3, kkt_rho: 0.5` |
| **B-1** | PDHG-Unroll | `neupan/blocks/pdhg_unroll.py` (245行) | ✅ 完整 | `unroll_J: 1/2/3` |
| **C** | SE(2) 编码 | `neupan/blocks/obs_point_net.py` (L26-65) | ✅ 完整 | `se2_embed: true` |
| **配置链路** | YAML 传递 | `neupan/neupan.py` → `PAN` → `DUNE` | ✅ 完整 | `train:` 段统一管理 |
| **时延埋点** | PDHG Timing | `pdhg_unroll.py` (L121-149) | ✅ 完整 | 自动记录 `total/per_step` |
| **训练元数据** | train_dict | `dune_train.py` (L181-201) | ✅ 完整 | 自动保存所有开关 |

### 10.2 YAML 配置完整示例

#### M0 (Baseline)
```yaml
train:
  projection: none
  use_kkt: false
  unroll_J: 0
  se2_embed: false
```

#### M2 (Hard + KKT)
```yaml
train:
  projection: hard
  use_kkt: true
  w_kkt: 1e-3
  kkt_rho: 0.5
  use_lconstr: true
  w_constr: 0.1
  unroll_J: 0
  se2_embed: false
```

#### M5 (完整 DF-DUNE)
```yaml
train:
  projection: hard
  use_kkt: true
  w_kkt: 1e-3
  kkt_rho: 0.5
  use_lconstr: true
  w_constr: 0.1
  unroll_J: 3
  se2_embed: true

  # PDHG 可选参数（当前为固定步长）
  # pdhg_learnable: false  # 可选：启用可学习步长
  # pdhg_per_step: false   # 可选：每步独立参数
```

### 10.3 训练与评测流程

#### 训练流程（如需重新训练）
```bash
# 1. 准备训练配置（已提供模板）
# - example/dune_train/dune_train_acker_kkt_se2.yaml
# - example/dune_train/dune_train_diff_kkt_se2.yaml

# 2. 执行训练（示例）
python example/dune_train/dune_train_acker.py \
  --config example/dune_train/dune_train_acker_kkt_se2.yaml

# 3. 验证训练元数据
python -c "
import pickle
with open('example/dune_train/model/acker_learned_prox_kkt_se2_robot/train_dict.pkl', 'rb') as f:
    d = pickle.load(f)
    print(f'se2_embed: {d[\"se2_embed\"]}')
    print(f'use_kkt: {d[\"use_kkt\"]}')
    print(f'projection: {d[\"projection\"]}')
"
```

#### 评测流程（三阶段）
```bash
# 阶段1: 消融研究（2-3小时）
python test/batch_projection_evaluation.py \
  --config test/configs/ablation_study_config.yaml

# 阶段2: 时延分析（1-2小时）
python test/timing_analysis.py \
  --config test/configs/timing_analysis_config.yaml

# 阶段3: 完整评测（6-8小时）
python test/full_evaluation.py \
  --config test/configs/full_evaluation_config.yaml
```

### 10.4 关键验证点

#### 验证 SE(2) 编码生效
```python
from neupan import neupan
planner = neupan.init_from_yaml('example/dune_train/dune_train_acker_kkt_se2.yaml')
print(f"SE(2) enabled: {planner.pan.dune_layer.se2_embed}")
print(f"ObsPointNet input_dim: {planner.pan.dune_layer.model.MLP[0].in_features}")  # 应为 3
```

#### 验证 PDHG 时延记录
```python
# 在推理后检查
info = planner.plan(start, goal, obs_list)
if 'pdhg_profile' in info:
    print(f"PDHG total time: {info['pdhg_profile']['total']:.4f}s")
    print(f"Per-step times: {info['pdhg_profile']['per_step']}")
```

#### 验证 KKT 正则生效
```python
# 检查训练日志中是否包含 L_kkt 项
# 或查看 train_dict.pkl
with open('path/to/train_dict.pkl', 'rb') as f:
    d = pickle.load(f)
    assert d['use_kkt'] == True
    assert d['w_kkt'] > 0
```

---

## 十一、预期结果与备选方案

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
