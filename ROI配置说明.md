# ROI (神经聚焦) 配置说明

## 概述

ROI (Region of Interest) / 神经聚焦功能受 Neural Informed RRT* (ICRA 2024) 启发,在将障碍点云送入 DUNE/Flexible PDHG 前端之前,先过滤到相关走廊区域,从而:
- **提升实时性**: 减少点云数量 (通常降至 20-50%),缩短特征抽取、排序/选K、NRMP 计算时间
- **提高稳定性**: 屏蔽无关侧向/后向"伪障碍",减少过度保守绕行、卡顿与失败
- **改善路径质量**: 减少"锯齿"路径,提升平滑度与非保守性
- **增强鲁棒性**: 冷启动与恢复能力更强,自适应切换策略

## 快速开始

### 1. 启用 ROI 功能

在 `planner.yaml` 中设置:

```yaml
roi:
  enabled: true  # 设为 true 启用,false 禁用(默认)
```

### 2. 默认配置

如果只设置 `enabled: true`,其余参数将使用默认值:

```yaml
roi:
  enabled: true
  strategy_order: [ellipse, tube, wedge]
  wedge: { fov_deg: 60.0, r_max_m: 8.0 }
  ellipse: { safety_scale: 1.08 }
  tube: { r0_m: 0.5, kappa_gain: 0.4, v_tau_m: 0.3 }
  guardrail: { n_min: 30, n_max: 500, relax_step: 1.15, tighten_step: 0.9 }
  always_keep: { near_radius_m: 1.5, goal_radius_m: 1.5 }
```

### 3. 查看 ROI 统计

ROI 统计信息会写入 `info` 字典:

```python
action, info = planner.forward(state, points)

if 'roi' in info:
    print(f"ROI 策略: {info['roi']['strategy']}")
    print(f"输入点数: {info['roi']['n_in']}")
    print(f"过滤后点数: {info['roi']['n_roi']}")
    print(f"压缩比: {info['roi']['n_roi'] / info['roi']['n_in']:.2%}")
```

## 三种 ROI 策略

### 1. Ellipse (椭圆走廊)

**原理**: 基于 Informed RRT* 的椭圆可行域,使用起点、终点和当前最佳路径长度 c_best 定义椭圆,只保留椭圆内的点。

**适用场景**: 直道、缓弯路径

**参数**:
- `safety_scale`: 椭圆轴长的安全系数 (默认 1.08,即放大 8%)

**触发条件**: 有参考路径且 c_best 可用

**数学公式**:
- 长轴 a = c_best / 2 × safety_scale
- 短轴 b = sqrt(c_best² - c_min²) / 2 × safety_scale
- c_min 为起终点直线距离

### 2. Tube (路径管道)

**原理**: 对参考路径做形态学膨胀,保留路径周围自适应半径内的点。

**适用场景**: 弯曲路径、高曲率路段

**参数**:
- `r0_m`: 基础半径 (米,默认 0.5)
- `v_tau_m`: 速度相关增益 (米,默认 0.3)
- `kappa_gain`: 曲率相关增益 (默认 0.4)

**自适应半径**: r = r0 + v_tau × |v| + kappa_gain × |κ|

**触发条件**: 有参考路径且至少 2 个点

### 3. Wedge (前向楔形)

**原理**: 以机器人为中心的扇形区域,保留前向视野内、最大距离内的点。

**适用场景**: 冷启动、无可靠路径时的回退策略

**参数**:
- `fov_deg`: 视野角度 (度,默认 60)
- `r_max_m`: 最大距离 (米,默认 8.0)

**触发条件**: 有机器人朝向信息

## 策略选择逻辑

每个周期按 `strategy_order` 顺序尝试策略,直到某个策略成功(过滤后点数满足 guardrail 要求):

```yaml
strategy_order: [ellipse, tube, wedge]
```

**推荐顺序**:
- 直道场景优先: `[ellipse, tube, wedge]`
- 弯道场景优先: `[tube, ellipse, wedge]`
- 保守策略: `[wedge]` (仅使用前向楔形)

## Guardrail (护栏机制)

自适应调节 ROI 参数,防止过度过滤或点数过多:

```yaml
guardrail:
  n_min: 30          # 最少点数阈值
  n_max: 500         # 最多点数阈值
  relax_step: 1.15   # 放宽倍数
  tighten_step: 0.9  # 收紧倍数
```

**工作原理**:
- 若过滤后点数 < n_min: 放宽当前策略参数 (×1.15),尝试下一策略
- 若过滤后点数 > n_max: 确定性下采样到 n_max
- 参数放宽仅在本周期有效,避免长期膨胀

## Always Keep (安全保留集)

无论选择哪种 ROI 策略,始终保留以下区域的点:

```yaml
always_keep:
  near_radius_m: 1.5  # 机器人近身圆盘
  goal_radius_m: 1.5  # 目标附近圆盘
```

**作用**: 防止误删关键近场障碍,确保安全性

## 参数调优建议

### 场景 1: 开阔直道

```yaml
roi:
  enabled: true
  strategy_order: [ellipse, wedge]
  ellipse: { safety_scale: 1.05 }  # 可适当收紧
  guardrail: { n_min: 20, n_max: 300 }
```

### 场景 2: 狭窄弯道

```yaml
roi:
  enabled: true
  strategy_order: [tube, wedge]
  tube: { r0_m: 0.8, kappa_gain: 0.6 }  # 加宽管道
  guardrail: { n_min: 50, n_max: 500 }
```

### 场景 3: 动态障碍物

```yaml
roi:
  enabled: true
  strategy_order: [tube, wedge]
  tube: { r0_m: 1.0, v_tau_m: 0.5 }  # 速度相关扩展
  always_keep: { near_radius_m: 2.0 }  # 扩大近场保留
```

### 场景 4: 高密度点云

```yaml
roi:
  enabled: true
  guardrail: { n_min: 50, n_max: 300 }  # 更激进的下采样
```

## 性能影响

### 预期收益

- **周期时间**: 通常降低 20-50% (取决于场景密度)
- **点云压缩**: N_in → N_roi 约 20-50%
- **路径质量**: 减少"锯齿"与过度保守绕行
- **成功率**: 冷启动与狭窄通道场景提升

### 开销

- **ROI 计算**: 每周期约 0.1-0.5 ms (Python 实现)
- **内存**: 可忽略 (仅存储配置与上一周期 c_best)

## 调试与诊断

### 1. 查看策略命中率

```python
strategy_counts = {}
for step in range(num_steps):
    action, info = planner.forward(state, points)
    if 'roi' in info:
        strategy = info['roi']['strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

print("策略命中率:", strategy_counts)
```

### 2. 监控点数变化

```python
if 'roi' in info:
    compression = info['roi']['n_roi'] / max(info['roi']['n_in'], 1)
    print(f"压缩比: {compression:.2%}, 策略: {info['roi']['strategy']}")
```

### 3. 检查放宽/收紧次数

```python
if 'roi' in info:
    print(f"放宽次数: {info['roi']['relax_count']}")
    print(f"收紧次数: {info['roi']['tighten_count']}")
```

## 常见问题

### Q1: ROI 会影响训练好的模型吗?

**A**: 不会。ROI 仅在推理时过滤点云,不改变 DUNE/Flexible PDHG 的网络结构与权重。建议训练时关闭 ROI (`enabled: false`),推理时开启。

### Q2: 如何确保 ROI 不会误删关键障碍?

**A**: 通过以下机制保障:
1. `always_keep` 永久保留近身/目标区域
2. `guardrail.n_min` 确保最少点数
3. 策略回退机制 (ellipse 失败→tube→wedge)

### Q3: 为什么有时 ROI 策略是 'none'?

**A**: 表示所有策略都未能满足 guardrail 要求,回退到全局点云 (可能经过下采样)。检查:
- 参考路径是否可用
- n_min 是否设置过高
- 点云是否过于稀疏

### Q4: 如何针对特定场景调参?

**A**: 建议流程:
1. 先用默认参数跑一遍,记录策略命中率与压缩比
2. 若 ellipse 命中率低,考虑放宽 `safety_scale` 或优先 tube
3. 若点数经常触发 n_min,降低阈值或放宽策略参数
4. 若周期时间仍高,降低 `n_max` 做更激进下采样

## 参考文献

- **Neural Informed RRT*** (ICRA 2024): [arXiv:2309.14595](https://arxiv.org/abs/2309.14595)
- **Informed RRT***: Gammell et al., "Informed RRT*: Optimal Sampling-based Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic", IROS 2014

## 完整配置示例

参见 `example/LON/planner.yaml` 中的 `roi` 段,包含所有参数的详细注释。

