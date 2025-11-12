# IR-SIM 快速参考卡片

> 一页纸速查手册 - 最常用的 IR-SIM 功能

---

## 🚀 快速开始

### 安装

```bash
pip install ir-sim
```

### 最小示例

```python
import irsim
import numpy as np

# 创建环境
env = irsim.make('example/corridor/diff/env.yaml', display=True)

# 主循环
for i in range(1000):
    # 获取状态
    robot_state = env.get_robot_state()  # [x, y, theta, v, w]
    lidar_scan = env.get_lidar_scan()    # [d1, d2, ..., dn]
    
    # 计算动作（示例：简单控制）
    action = np.array([1.0, 0.0])  # [v, w]
    
    # 执行动作
    env.step(action)
    env.render()
    
    # 检查终止
    if env.done():
        break

# 结束
env.end(delay=3)
```

---

## 📋 核心 API

### 环境管理

| 功能 | 代码 |
|------|------|
| **创建环境** | `env = irsim.make('env.yaml', display=True)` |
| **结束环境** | `env.end(delay=3, ani_name='animation')` |

### 状态获取

| 功能 | 代码 | 返回值 |
|------|------|--------|
| **机器人状态** | `env.get_robot_state()` | `[x, y, θ, v, ω]` (diff) |
| **激光雷达** | `env.get_lidar_scan()` | `[d1, d2, ..., dn]` |

### 动作执行

| 功能 | 代码 |
|------|------|
| **执行动作** | `env.step(np.array([v, w]))` |
| **检查终止** | `env.done()` → `bool` |

### 可视化

| 功能 | 代码 |
|------|------|
| **绘制点云** | `env.draw_points(points, s=25, c="g", refresh=True)` |
| **绘制轨迹** | `env.draw_trajectory(traj, "r", refresh=True)` |
| **渲染** | `env.render()` |

---

## ⚙️ YAML 配置速查

### 最小配置

```yaml
world:
  height: 20
  width: 20
  step_time: 0.1

robot:
  - kinematics: {name: 'diff'}
    shape: {name: 'circle', radius: 0.2}
    state: [1, 1, 0]
    goal: [9, 9, 0]
    sensors:
      - type: 'lidar2d'
        range_max: 10
        number: 100

obstacle:
  - number: 5
    distribution: {name: 'random', range_low: [2, 2, 0], range_high: [8, 8, 0]}
    shape:
      - {name: 'circle', radius: 0.5}
```

### 常用配置选项

#### 运动学模型

```yaml
kinematics: {name: 'diff'}   # 差速驱动 [v, w]
kinematics: {name: 'omni'}   # 全向移动 [vx, vy]
kinematics: {name: 'acker'}  # 阿克曼转向 [v, δ]
```

#### 障碍物形状

```yaml
shape:
  - {name: 'circle', radius: 1.0}
  - {name: 'rectangle', length: 5, width: 2}
  - {name: 'polygon', vertices: [[x1,y1], [x2,y2], ...]}
```

#### 障碍物分布

```yaml
# 手动分布
distribution: {name: 'manual'}
state: [[x1, y1, θ1], [x2, y2, θ2], ...]

# 随机分布
distribution:
  name: 'random'
  range_low: [10, 10, -3.14]
  range_high: [40, 40, 3.14]
```

#### 随机形状

```yaml
shape:
  - name: 'polygon'
    random_shape: true
    avg_radius_range: [0.5, 1.0]
    irregularity_range: [0.9, 1.0]
```

#### 动态障碍物

```yaml
obstacle:
  - number: 20
    distribution: {name: 'random', ...}
    kinematics: {name: 'diff'}
    behavior:
      - name: 'rvo'
        wander: True
        vxmax: 0.5
        vymax: 0.5
```

#### 传感器配置

```yaml
sensors:
  - type: 'lidar2d'
    range_min: 0
    range_max: 10
    angle_range: 3.1415926  # 180度
    number: 100             # 扫描点数
    noise: False            # 是否添加噪声
    std: 0.1                # 噪声标准差
```

---

## 🎨 可视化速查

### 绘制点云

```python
# 基本用法
env.draw_points(points, s=25, c="g", refresh=True)

# 参数说明
# points: numpy.ndarray, shape=(N, 2) 或 List[[x, y], ...]
# s: 点大小 (默认 25)
# c: 颜色 'r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'
# refresh: 是否清除之前的点 (默认 True)
```

### 绘制轨迹

```python
# 基本用法
env.draw_trajectory(trajectory, "r", show_direction=False, refresh=True)

# 参数说明
# trajectory: List[numpy.ndarray], 每个 shape=(3,1) 或 (4,1)
# traj_type: 轨迹类型/颜色 "r", "b", "g", "-k", etc.
# show_direction: 是否显示方向箭头
# refresh: 是否清除之前的轨迹
```

### 颜色代码

| 代码 | 颜色 | 常用场景 |
|------|------|---------|
| `'r'` | 红色 | 优化轨迹、NRMP 点 |
| `'g'` | 绿色 | DUNE 点、机器人 |
| `'b'` | 蓝色 | 参考轨迹 |
| `'y'` | 黄色 | 警告区域 |
| `'c'` | 青色 | ROI 区域 |
| `'m'` | 品红 | 特殊标记 |
| `'k'` | 黑色 | 初始路径 |
| `'gray'` | 灰色 | 障碍物 |

---

## 🔧 常用代码片段

### 1. 环境随机化

```python
import yaml
import numpy as np

# 加载基础配置
with open('base_env.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 随机化障碍物尺寸
for obs in config['obstacle']:
    for shape in obs['shape']:
        if shape['name'] == 'circle':
            shape['radius'] *= np.random.uniform(0.8, 1.2)

# 保存并创建环境
with open('temp_env.yaml', 'w') as f:
    yaml.dump(config, f)
env = irsim.make('temp_env.yaml', display=True)
```

### 2. 批量评估

```python
def evaluate(env_file, planner, num_runs=10):
    success_count = 0
    for run in range(num_runs):
        env = irsim.make(env_file, display=False)
        success = run_episode(planner, env)
        if success:
            success_count += 1
        env.end(delay=0)
    return success_count / num_runs
```

### 3. 保存失败场景

```python
if env.done() and not arrived:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env.end(delay=3, ani_name=f'failure_{timestamp}')
```

### 4. 关闭可视化加速训练

```python
# 训练时
env = irsim.make(env_file, display=False)
# 不调用 env.render()
# 不调用 env.draw_*()

# 评估时
env = irsim.make(env_file, display=True)
env.render()
```

---

## 🐛 常见问题

| 问题 | 解决方案 |
|------|---------|
| **机器人不动** | 检查 `vel_max` 参数 |
| **频繁碰撞** | 减少障碍物数量或增大空间 |
| **激光雷达无数据** | 检查 `sensors` 配置 |
| **动画保存失败** | 创建 `example/animation/` 目录 |
| **可视化卡顿** | 减少 `draw_points` 调用频率 |
| **随机形状不变** | 调用 `np.random.seed(None)` |

---

## 📊 性能优化

| 优化项 | 方法 | 加速比 |
|--------|------|--------|
| **关闭显示** | `display=False` | 2-3x |
| **减少扫描点** | `number: 50` (从 100) | 1.5x |
| **增大步长** | `step_time: 0.2` (从 0.1) | 2x |
| **不绘制** | 注释掉 `draw_*()` | 1.5x |

---

## 📚 更多资源

| 资源 | 链接 |
|------|------|
| **完整文档** | `docs/IR-SIM完整能力分析.md` |
| **官方文档** | https://ir-sim.readthedocs.io/ |
| **GitHub** | https://github.com/hanruihua/ir-sim |
| **NeuPAN 集成** | `example/run_exp.py` |

---

**版本**: v1.0  
**最后更新**: 2025-01-XX  
**打印提示**: 建议打印此页作为速查手册

