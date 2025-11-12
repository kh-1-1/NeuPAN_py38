# IR-SIM 完整能力分析

> 基于官网文档、GitHub 仓库和 NeuPAN 项目示例的综合分析

## 📋 目录

1. [IR-SIM 简介](#1-ir-sim-简介)
2. [核心功能](#2-核心功能)
3. [API 完整列表](#3-api-完整列表)
4. [YAML 配置能力](#4-yaml-配置能力)
5. [高级特性](#5-高级特性)
6. [在 NeuPAN 中的应用](#6-在-neupan-中的应用)
7. [与其他仿真器对比](#7-与其他仿真器对比)

---

## 1. IR-SIM 简介

### 1.1 定位

**IR-SIM** (Intelligent Robot Simulator) 是一个：
- ✅ **开源**的 Python 机器人仿真器
- ✅ **轻量级**（基于 matplotlib，无需 GPU）
- ✅ **易用**（YAML 配置，无需复杂编程）
- ✅ **快速**（适合算法原型开发）

### 1.2 官方资源

| 资源 | 链接 |
|------|------|
| **GitHub** | https://github.com/hanruihua/ir-sim |
| **文档** | https://ir-sim.readthedocs.io/en/stable/ |
| **PyPI** | `pip install ir-sim` |
| **版本** | 2.7.5 (2025-10-26) |
| **Stars** | 613+ ⭐ |

### 1.3 应用项目

| 项目 | 会议/期刊 | 说明 |
|------|----------|------|
| **rl-rvo-nav** | RAL & ICRA 2023 | 强化学习 + RVO 多机器人导航 |
| **RDA_planner** | RAL & IROS 2023 | 加速无碰撞运动规划 |
| **NeuPAN** | T-RO 2025 | 端到端模型学习导航 |

---

## 2. 核心功能

### 2.1 机器人运动学模型

| 模型 | 名称 | 状态维度 | 控制输入 | 适用场景 |
|------|------|---------|---------|---------|
| **diff** | 差速驱动 | [x, y, θ, v, ω] | [v, ω] | 圆形移动机器人 |
| **omni** | 全向移动 | [x, y, θ, vx, vy] | [vx, vy] | 全向轮机器人 |
| **acker** | 阿克曼转向 | [x, y, θ, v, ω, δ] | [v, δ] | 类车机器人 |

### 2.2 传感器支持

| 传感器 | 类型 | 输出 | 配置参数 |
|--------|------|------|---------|
| **lidar2d** | 2D 激光雷达 | 距离数组 | `range_min`, `range_max`, `angle_range`, `number`, `noise`, `std` |
| **FOV** | 视野检测器 | 视野内物体 | `range`, `angle` |
| **has_velocity** | 速度传感器 | 障碍物速度 | `has_velocity: True` |

### 2.3 障碍物类型

| 类型 | 参数 | 示例 |
|------|------|------|
| **circle** | `radius` | `{name: 'circle', radius: 1.5}` |
| **rectangle** | `length`, `width` | `{name: 'rectangle', length: 5, width: 2}` |
| **polygon** | `vertices` | `{name: 'polygon', vertices: [[x1,y1], [x2,y2], ...]}` |
| **random polygon** | `random_shape`, `avg_radius_range`, `irregularity_range` | `{name: 'polygon', random_shape: true, ...}` |
| **linestring** | `points` | 线段障碍物 |
| **grid map** | 二值栅格地图 | 从图像加载 |

### 2.4 障碍物行为模式

| 行为 | 说明 | 参数 | 适用场景 |
|------|------|------|---------|
| **dash** | 直接冲向目标 | `range_low`, `range_high` | 简单导航 |
| **rvo** | Reciprocal Velocity Obstacles | `vxmax`, `vymax`, `wander` | 多机器人避障 |
| **orca** | Optimal Reciprocal Collision Avoidance | 通过 pyrvo 实现 | 大规模多智能体 |
| **wander** | 随机游走 | `range_low`, `range_high` | 动态环境 |

### 2.5 碰撞模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **stop** | 碰撞后停止 | 安全测试 |
| **unobstructed** | 穿透障碍物 | 调试 |
| **reactive** | 碰撞后反弹 | 物理仿真 |
| **unobstructed_obstacles** | 机器人与障碍物不碰撞，障碍物之间碰撞 | 动态障碍物场景 |

---

## 3. API 完整列表

### 3.1 环境创建与管理

```python
import irsim

# 创建环境
env = irsim.make(
    env_file='path/to/env.yaml',  # YAML 配置文件
    save_ani=False,                # 是否保存动画
    full=False,                    # 是否全屏显示
    display=True                   # 是否显示可视化
)

# 结束环境
env.end(
    delay=3,                       # 延迟时间（秒）
    ani_name='animation'           # 动画文件名（保存为 GIF）
)
```

### 3.2 状态获取

```python
# 获取机器人状态
robot_state = env.get_robot_state()
# 返回: numpy.ndarray
# diff: [x, y, theta, v, w]
# omni: [x, y, theta, vx, vy]
# acker: [x, y, theta, v, w, delta]

# 获取激光雷达扫描
lidar_scan = env.get_lidar_scan()
# 返回: numpy.ndarray, shape=(num_beams,)
# 每个元素是该方向的障碍物距离
```

### 3.3 动作执行

```python
# 执行控制动作
action = np.array([[v], [w]])  # shape=(2, 1)
env.step(action)

# 或者
action = np.array([v, w])  # shape=(2,)
env.step(action)
```

### 3.4 终止条件检查

```python
# 检查是否结束（碰撞或到达目标）
is_done = env.done()
# 返回: bool
```

### 3.5 可视化

```python
# 绘制点云
env.draw_points(
    points,              # numpy.ndarray, shape=(N, 2) 或 List[[x, y], ...]
    s=25,                # 点大小
    c="g",               # 颜色: 'r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'
    marker='o',          # 标记类型: 'o', 's', '^', 'v', etc.
    alpha=1.0,           # 透明度 (0.0-1.0)
    refresh=True         # 是否刷新之前的点
)

# 绘制轨迹
env.draw_trajectory(
    trajectory,          # List[numpy.ndarray], 每个 shape=(3,1) 或 (4,1)
    traj_type="r",       # 轨迹类型/颜色: "r", "b", "g", "-k", etc.
    show_direction=False,  # 是否显示方向箭头
    refresh=True         # 是否刷新之前的轨迹
)

# 绘制线段
env.draw_line(
    start_point,         # [x, y]
    end_point,           # [x, y]
    color='r',           # 颜色
    linewidth=2          # 线宽
)

# 渲染（更新显示）
env.render()
```

### 3.6 控制模式

```python
# YAML 配置
world:
  control_mode: 'auto'  # 'auto' 或 'keyboard'

# keyboard 模式需要安装额外依赖
# pip install ir-sim[keyboard]
```

---

## 4. YAML 配置能力

### 4.1 世界配置

```yaml
world:
  height: 42              # 世界高度（米）
  width: 42               # 世界宽度（米）
  step_time: 0.1          # 仿真步长（秒）10Hz
  sample_time: 0.1        # 采样时间（秒）10Hz
  offset: [5, 5]          # 显示偏移
  collision_mode: 'stop'  # 碰撞模式
  control_mode: 'auto'    # 控制模式
```

### 4.2 机器人配置

```yaml
robot:
  - kinematics: {name: 'diff'}  # 运动学模型
    shape: {name: 'circle', radius: 0.2}  # 形状
    state: [1, 1, 0]            # 初始状态 [x, y, theta]
    goal: [9, 9, 0]             # 目标位置
    vel_min: [-8, -3.14]        # 最小速度
    vel_max: [8, 3.14]          # 最大速度
    goal_threshold: 0.3         # 到达阈值（米）
    arrive_mode: 'position'     # 到达模式: 'position' 或 'state'
    behavior: {name: 'dash'}    # 行为模式
    color: 'g'                  # 颜色
    description: 'robot.png'    # 图片描述
    
    plot:                       # 绘图选项
      show_trail: True          # 显示轨迹
      show_goal: True           # 显示目标
      show_arrow: True          # 显示方向箭头
      traj_color: 'g'           # 轨迹颜色
    
    sensors:                    # 传感器列表
      - type: 'lidar2d'
        range_min: 0
        range_max: 10
        angle_range: 3.1415926  # 180度
        number: 100             # 扫描点数
        noise: False            # 是否添加噪声
        std: 0.1                # 噪声标准差
        has_velocity: False     # 是否检测速度
```

### 4.3 障碍物配置

#### 4.3.1 静态障碍物（手动分布）

```yaml
obstacle:
  - number: 6
    distribution: {name: 'manual'}
    shape:
      - {name: 'rectangle', length: 70, width: 2}
      - {name: 'circle', radius: 1.5}
      - {name: 'polygon', vertices: [[x1,y1], [x2,y2], ...]}
    state: [
      [30, 25, 0],      # [x, y, theta]
      [20, 15, 0],
      ...
    ]
    color: 'gray'       # 障碍物颜色
```

#### 4.3.2 静态障碍物（随机分布）

```yaml
obstacle:
  - number: 10
    distribution:
      name: 'random'
      range_low: [10, 10, -3.14]
      range_high: [40, 40, 3.14]
    shape:
      - {name: 'circle', radius: 1.0}
```

#### 4.3.3 随机形状障碍物

```yaml
obstacle:
  - number: 11
    distribution: {name: 'manual'}
    shape:
      - name: 'polygon'
        random_shape: true              # 启用随机形状
        center_range: [0, 0, 0, 0]      # 中心偏移
        avg_radius_range: [0.5, 1.0]    # 平均半径范围
        irregularity_range: [0.9, 1.0]  # 不规则度 (1.0=正多边形)
    state: [[20, 34], [31, 38], ...]
```

#### 4.3.4 动态障碍物

```yaml
obstacle:
  - number: 20
    distribution: {name: 'random', ...}
    kinematics: {name: 'diff'}          # 运动学模型
    shape:
      - {name: 'circle', radius: 0.5}
    behavior:                            # 行为模式
      - name: 'rvo'
        range_low: [10, 10, -3.14]
        range_high: [40, 40, 3.14]
        wander: True                     # 随机游走
        vxmax: 0.5
        vymax: 0.5
    vel_min: [-1.0, -3.14]
    vel_max: [1.0, 3.14]
    arrive_mode: 'position'
    goal_threshold: 0.3
    plot:
      show_goal: False
      show_arrow: True
```

---

## 5. 高级特性

### 5.1 环境随机化能力

| 特性 | 支持情况 | 实现方式 |
|------|---------|---------|
| **障碍物位置随机** | ✅ 内置 | `distribution: random` |
| **障碍物形状随机** | ✅ 内置 | `random_shape: true` |
| **障碍物尺寸随机** | ❌ 需自己实现 | 动态生成 YAML |
| **障碍物数量随机** | ❌ 需自己实现 | 动态生成 YAML |
| **传感器噪声** | ✅ 内置 | `noise: True, std: 0.1` |
| **动态障碍物** | ✅ 内置 | `kinematics` + `behavior` |

### 5.2 多机器人支持

```yaml
robot:
  - kinematics: {name: 'diff'}
    state: [0, 0, 0]
    goal: [10, 10, 0]
    # ... 机器人 1 配置
  
  - kinematics: {name: 'acker'}
    state: [5, 5, 0]
    goal: [15, 15, 0]
    # ... 机器人 2 配置
```

**注意**: NeuPAN 项目中主要使用单机器人场景。

### 5.3 栅格地图支持

```yaml
world:
  map:
    file: 'path/to/map.png'  # 二值图像
    resolution: 0.05         # 米/像素
    origin: [0, 0]           # 地图原点
```

**应用场景**:
- 从 HM3D、MatterPort3D、Gibson 等 3D 数据集生成 2D 地图
- 室内导航场景

### 5.4 大规模多智能体（ORCA）

```python
# 需要安装 pyrvo
# pip install pyrvo

# 支持 200+ 智能体的高效避障
```

### 5.5 动画保存

```python
env = irsim.make(env_file, save_ani=True)
# ...
env.end(delay=3, ani_name='my_animation')
# 保存为: example/animation/my_animation.gif
```

---

## 6. 在 NeuPAN 中的应用

### 6.1 使用的 API

| API | 用途 | 调用频率 |
|-----|------|---------|
| `irsim.make()` | 创建环境 | 每个实验 1 次 |
| `env.get_robot_state()` | 获取状态 | 每步 1 次 |
| `env.get_lidar_scan()` | 获取传感器数据 | 每步 1 次 |
| `env.step(action)` | 执行动作 | 每步 1 次 |
| `env.render()` | 渲染 | 每步 1 次 |
| `env.done()` | 检查终止 | 每步 1 次 |
| `env.draw_points()` | 绘制点云 | 每步 3-4 次 |
| `env.draw_trajectory()` | 绘制轨迹 | 每步 2-3 次 |
| `env.end()` | 结束环境 | 每个实验 1 次 |

### 6.2 典型使用模式

```python
# 1. 创建环境
env = irsim.make(env_file, save_ani=False, display=True)
neupan_planner = neupan.init_from_yaml(planner_file)
neupan_planner.set_env_reference(env)

# 2. 主循环
for i in range(max_steps):
    # 获取状态
    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()
    
    # 规划
    points = neupan_planner.scan_to_point(robot_state, lidar_scan)
    action, info = neupan_planner(robot_state, points)
    
    # 可视化
    env.draw_points(neupan_planner.dune_points, s=25, c="g", refresh=True)
    env.draw_points(neupan_planner.nrmp_points, s=13, c="r", refresh=True)
    env.draw_trajectory(neupan_planner.opt_trajectory, "r", refresh=True)
    env.draw_trajectory(neupan_planner.ref_trajectory, "b", refresh=True)
    
    # 执行
    env.render()
    env.step(action)
    
    # 检查终止
    if info["arrive"] or env.done():
        break

# 3. 结束
env.end(delay=3, ani_name="animation")
```

### 6.3 NeuPAN 使用的场景

| 场景 | 障碍物 | 动态性 | 用途 |
|------|--------|--------|------|
| **LON** | 矩形（6个） | 静态 | 在线参数学习 |
| **corridor** | 矩形（6个） | 静态 | 走廊导航测试 |
| **convex_obs** | 圆形+多边形（11个） | 静态 | 凸障碍物避障 |
| **non_obs** | 随机多边形（11个） | 静态 | 非凸障碍物 |
| **dyna_obs** | 圆形（15-20个） | 动态 | 动态避障 |
| **dyna_non_obs** | 多边形（11个） | 动态 | 动态非凸避障 |

---

## 7. 与其他仿真器对比

| 特性 | IR-SIM | Gazebo | PyBullet | CARLA |
|------|--------|--------|----------|-------|
| **安装难度** | ⭐ 简单 | ⭐⭐⭐ 困难 | ⭐⭐ 中等 | ⭐⭐⭐⭐ 很难 |
| **学习曲线** | ⭐ 平缓 | ⭐⭐⭐ 陡峭 | ⭐⭐ 中等 | ⭐⭐⭐⭐ 陡峭 |
| **配置方式** | YAML | XML/SDF | Python | Python |
| **可视化** | matplotlib | 3D | 3D | 3D |
| **物理仿真** | 简化 | 完整 | 完整 | 完整 |
| **传感器** | 2D LiDAR, FOV | 丰富 | 丰富 | 非常丰富 |
| **性能** | ⭐⭐⭐⭐ 快 | ⭐⭐ 慢 | ⭐⭐⭐ 中等 | ⭐ 很慢 |
| **适用场景** | 算法原型 | 完整仿真 | 机器人学习 | 自动驾驶 |

### IR-SIM 的优势

✅ **轻量级**: 无需 GPU，CPU 即可运行  
✅ **快速**: 适合快速迭代算法  
✅ **易用**: YAML 配置，无需复杂编程  
✅ **开源**: MIT 许可证  
✅ **Python**: 纯 Python 实现，易于集成

### IR-SIM 的局限

❌ **2D 仿真**: 不支持 3D 环境  
❌ **简化物理**: 物理仿真不如 Gazebo/PyBullet  
❌ **传感器有限**: 主要是 2D LiDAR  
❌ **可视化简单**: 基于 matplotlib

---

## 8. 总结

### 8.1 核心能力

| 类别 | 能力 |
|------|------|
| **运动学** | diff, omni, acker |
| **传感器** | 2D LiDAR, FOV, 速度检测 |
| **障碍物** | 圆形、矩形、多边形、随机形状 |
| **行为** | dash, rvo, orca, wander |
| **随机化** | 位置随机、形状随机、传感器噪声 |
| **可视化** | 点云、轨迹、线段 |
| **动画** | GIF 保存 |

### 8.2 适用场景

✅ **算法原型开发**  
✅ **强化学习训练**  
✅ **多机器人协调**  
✅ **导航算法测试**  
✅ **教学演示**

### 8.3 不适用场景

❌ **高保真物理仿真**  
❌ **3D 环境导航**  
❌ **复杂传感器仿真**  
❌ **真实机器人部署前的最终测试**

---

## 9. 实用技巧与最佳实践

### 9.1 环境随机化实现

虽然 IR-SIM 内置了位置随机和形状随机，但如果需要更多随机化（如尺寸、数量），可以动态生成 YAML：

```python
import yaml
import numpy as np
import irsim

def create_randomized_env(base_config_path, randomization_params):
    """动态生成随机化环境"""
    # 1. 加载基础配置
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 随机化障碍物尺寸
    for obs_group in config.get('obstacle', []):
        for shape in obs_group.get('shape', []):
            if shape['name'] == 'circle':
                # 随机化半径
                base_radius = shape['radius']
                shape['radius'] = base_radius * np.random.uniform(0.8, 1.2)
            elif shape['name'] == 'rectangle':
                # 随机化长宽
                shape['length'] *= np.random.uniform(0.8, 1.2)
                shape['width'] *= np.random.uniform(0.8, 1.2)

    # 3. 随机化障碍物数量
    if 'num_obstacles_range' in randomization_params:
        num_range = randomization_params['num_obstacles_range']
        config['obstacle'][0]['number'] = np.random.randint(*num_range)

    # 4. 随机化传感器噪声
    if 'sensor_noise_range' in randomization_params:
        noise_std = np.random.uniform(*randomization_params['sensor_noise_range'])
        for robot in config.get('robot', []):
            for sensor in robot.get('sensors', []):
                sensor['noise'] = True
                sensor['std'] = noise_std

    # 5. 保存临时配置
    temp_path = 'temp_env.yaml'
    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 6. 创建环境
    env = irsim.make(temp_path, display=True)
    return env

# 使用示例
randomization_params = {
    'num_obstacles_range': [10, 20],
    'sensor_noise_range': [0.0, 0.2]
}
env = create_randomized_env('example/corridor/diff/env.yaml', randomization_params)
```

### 9.2 多场景训练设置

```python
# 定义场景库
SCENE_LIBRARY = {
    'corridor': 'example/corridor/diff/env.yaml',
    'convex_obs': 'example/convex_obs/diff/env.yaml',
    'non_obs': 'example/non_obs/diff/env.yaml',
    'dyna_obs': 'example/dyna_obs/diff/env.yaml',
}

# 课程学习：渐进式难度
CURRICULUM = {
    'easy': ['corridor'],
    'medium': ['corridor', 'convex_obs'],
    'hard': ['corridor', 'convex_obs', 'non_obs', 'dyna_obs']
}

def train_with_curriculum(planner, curriculum_level='easy'):
    """课程学习训练"""
    scenes = CURRICULUM[curriculum_level]

    for epoch in range(num_epochs):
        # 随机选择场景
        scene_name = np.random.choice(scenes)
        env_file = SCENE_LIBRARY[scene_name]

        # 创建环境
        env = irsim.make(env_file, display=False)

        # 训练一个 episode
        train_episode(planner, env)

        env.end(delay=0)
```

### 9.3 性能优化技巧

#### 9.3.1 关闭可视化加速训练

```python
# 训练时关闭显示
env = irsim.make(env_file, display=False)

# 不调用 render()
# env.render()  # 注释掉

# 不绘制点云和轨迹
# env.draw_points(...)  # 注释掉
# env.draw_trajectory(...)  # 注释掉
```

#### 9.3.2 调整仿真步长

```yaml
world:
  step_time: 0.2  # 从 0.1 增加到 0.2（5Hz）
  sample_time: 0.2
```

**注意**: 步长过大可能导致碰撞检测不准确。

#### 9.3.3 减少激光雷达扫描点数

```yaml
sensors:
  - type: 'lidar2d'
    number: 50  # 从 100 减少到 50
```

### 9.4 调试技巧

#### 9.4.1 可视化 ROI 区域

```python
# 在 run_exp.py 中添加 -vr 参数
python example/run_exp.py -e corridor -d diff -vr

# 或在代码中
neupan_planner.visualize_roi_region(env)
```

#### 9.4.2 打印详细信息

```python
# 打印机器人状态
print(f"Robot state: {robot_state}")

# 打印激光雷达数据
print(f"LiDAR scan: min={lidar_scan.min()}, max={lidar_scan.max()}")

# 打印规划器信息
print(f"Action: {action}, Info: {info}")
```

#### 9.4.3 保存失败场景

```python
if env.done() and not info.get('arrive'):
    # 碰撞失败，保存场景
    env.end(delay=3, ani_name=f'failure_{timestamp}')
```

### 9.5 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **机器人不动** | 速度限制过小 | 检查 `vel_max` 参数 |
| **频繁碰撞** | 障碍物过密 | 减少障碍物数量或增大空间 |
| **激光雷达无数据** | 传感器配置错误 | 检查 `sensors` 配置 |
| **动画保存失败** | 路径不存在 | 创建 `example/animation/` 目录 |
| **可视化卡顿** | 绘制点过多 | 减少 `draw_points` 调用频率 |
| **随机形状不变** | 未设置随机种子 | 每次运行前调用 `np.random.seed(None)` |

---

## 10. 高级应用示例

### 10.1 强化学习集成

```python
import gym
from gym import spaces
import irsim

class IRSimEnv(gym.Env):
    """将 IR-SIM 包装为 OpenAI Gym 环境"""

    def __init__(self, env_file):
        super().__init__()
        self.env_file = env_file
        self.env = None

        # 定义动作空间（连续）
        self.action_space = spaces.Box(
            low=np.array([-1.0, -3.14]),
            high=np.array([1.0, 3.14]),
            dtype=np.float32
        )

        # 定义观测空间（激光雷达 + 机器人状态）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(105,),  # 100 (lidar) + 5 (robot state)
            dtype=np.float32
        )

    def reset(self):
        """重置环境"""
        if self.env is not None:
            self.env.end(delay=0)

        self.env = irsim.make(self.env_file, display=False)

        robot_state = self.env.get_robot_state()
        lidar_scan = self.env.get_lidar_scan()

        obs = np.concatenate([robot_state, lidar_scan])
        return obs

    def step(self, action):
        """执行动作"""
        self.env.step(action)

        robot_state = self.env.get_robot_state()
        lidar_scan = self.env.get_lidar_scan()
        obs = np.concatenate([robot_state, lidar_scan])

        # 计算奖励
        goal = np.array([9, 9])  # 从配置文件读取
        distance = np.linalg.norm(robot_state[:2] - goal)
        reward = -distance

        # 检查终止
        done = self.env.done()
        if distance < 0.3:
            reward += 100
            done = True
        elif done:
            reward -= 100

        info = {}
        return obs, reward, done, info

    def close(self):
        if self.env is not None:
            self.env.end(delay=0)

# 使用示例
env = IRSimEnv('example/corridor/diff/env.yaml')
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

### 10.2 批量评估

```python
def batch_evaluate(planner, env_files, num_runs=10):
    """批量评估规划器性能"""
    results = []

    for env_file in env_files:
        scene_name = env_file.split('/')[-2]
        success_count = 0
        total_steps = []

        for run in range(num_runs):
            # 设置随机种子（可选）
            np.random.seed(run)

            env = irsim.make(env_file, display=False)

            success, steps = run_episode(planner, env)

            if success:
                success_count += 1
                total_steps.append(steps)

            env.end(delay=0)

        # 统计结果
        success_rate = success_count / num_runs
        avg_steps = np.mean(total_steps) if total_steps else 0

        results.append({
            'scene': scene_name,
            'success_rate': success_rate,
            'avg_steps': avg_steps
        })

    return results

# 使用示例
env_files = [
    'example/corridor/diff/env.yaml',
    'example/convex_obs/diff/env.yaml',
    'example/non_obs/diff/env.yaml',
]

results = batch_evaluate(neupan_planner, env_files, num_runs=10)

# 打印结果
for r in results:
    print(f"{r['scene']}: Success Rate={r['success_rate']:.2%}, Avg Steps={r['avg_steps']:.1f}")
```

### 10.3 在线参数优化（LON 风格）

```python
import torch
import torch.optim as optim

def online_parameter_tuning(planner, env_file, num_episodes=100):
    """在线参数优化"""
    # 定义可学习参数
    params = {
        'p_u': torch.tensor([1.0], requires_grad=True),
        'eta': torch.tensor([10.0], requires_grad=True),
        'd_max': torch.tensor([1.0], requires_grad=True),
    }

    optimizer = optim.Adam(params.values(), lr=5e-3)

    for episode in range(num_episodes):
        env = irsim.make(env_file, display=False)

        # 更新规划器参数
        planner.update_adjust_parameters(
            p_u=params['p_u'].item(),
            eta=params['eta'].item(),
            d_max=params['d_max'].item()
        )

        # 运行一个 episode
        total_loss = 0
        for step in range(1000):
            robot_state = env.get_robot_state()
            lidar_scan = env.get_lidar_scan()
            points = planner.scan_to_point(robot_state, lidar_scan)

            action, info = planner(robot_state, points)

            # 计算损失
            loss = compute_loss(robot_state, action, info)
            total_loss += loss

            env.step(action)

            if env.done() or info.get('arrive'):
                break

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        env.end(delay=0)

        print(f"Episode {episode}: Loss={total_loss.item():.4f}, "
              f"p_u={params['p_u'].item():.4f}, "
              f"eta={params['eta'].item():.4f}, "
              f"d_max={params['d_max'].item():.4f}")

    return params
```

---

## 11. 未来发展方向

### 11.1 IR-SIM 可能的改进

| 改进方向 | 优先级 | 难度 |
|---------|-------|------|
| **3D 可视化** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **更多传感器** (RGB 相机、深度相机) | ⭐⭐⭐ | ⭐⭐⭐ |
| **物理引擎集成** | ⭐⭐ | ⭐⭐⭐⭐ |
| **ROS 集成** | ⭐⭐⭐⭐ | ⭐⭐ |
| **GPU 加速** | ⭐⭐ | ⭐⭐⭐ |
| **更多行为模式** | ⭐⭐⭐ | ⭐⭐ |

### 11.2 社区贡献方向

- 📚 **文档**: 补充更多示例和教程
- 🧪 **测试**: 增加单元测试和集成测试
- 🎨 **可视化**: 改进 matplotlib 渲染性能
- 🔧 **工具**: 开发 YAML 配置生成器
- 📦 **集成**: 与 ROS、Gym 等框架集成

---

## 12. 参考资源

### 12.1 官方资源

| 资源 | 链接 |
|------|------|
| **GitHub 仓库** | https://github.com/hanruihua/ir-sim |
| **官方文档** | https://ir-sim.readthedocs.io/ |
| **PyPI 包** | https://pypi.org/project/ir-sim/ |
| **问题反馈** | https://github.com/hanruihua/ir-sim/issues |

### 12.2 相关论文

1. **rl-rvo-nav** (RAL & ICRA 2023)
   - 标题: "Learning-based Reciprocal Velocity Obstacles for Multi-Robot Navigation"
   - 链接: https://github.com/hanruihua/rl-rvo-nav

2. **RDA_planner** (RAL & IROS 2023)
   - 标题: "Accelerating Collision-Free Motion Planning via Reinforcement Learning"
   - 链接: https://github.com/hanruihua/RDA_planner

3. **NeuPAN** (T-RO 2025)
   - 标题: "NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning"
   - 链接: https://github.com/hanruihua/NeuPAN

### 12.3 教程与示例

- **IR-SIM 快速入门**: https://ir-sim.readthedocs.io/en/stable/get_started.html
- **YAML 配置语法**: https://ir-sim.readthedocs.io/en/stable/yaml_syntax.html
- **NeuPAN 集成示例**: `example/run_exp.py`
- **LON 在线学习示例**: `example/LON/LON_corridor.py`

### 12.4 社区与支持

- **GitHub Discussions**: https://github.com/hanruihua/ir-sim/discussions
- **作者邮箱**: hanrh@connect.hku.hk
- **Star 项目**: 如果觉得有用，请给项目点个 ⭐！

---

## 附录 A: 完整 API 速查表

### A.1 环境管理

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `irsim.make()` | `env_file`, `save_ani`, `full`, `display` | `env` | 创建环境 |
| `env.end()` | `delay`, `ani_name` | None | 结束环境 |

### A.2 状态与传感器

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `env.get_robot_state()` | None | `numpy.ndarray` | 获取机器人状态 |
| `env.get_lidar_scan()` | None | `numpy.ndarray` | 获取激光雷达数据 |

### A.3 动作与控制

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `env.step()` | `action` | None | 执行动作 |
| `env.done()` | None | `bool` | 检查是否结束 |

### A.4 可视化

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `env.draw_points()` | `points`, `s`, `c`, `marker`, `alpha`, `refresh` | None | 绘制点云 |
| `env.draw_trajectory()` | `trajectory`, `traj_type`, `show_direction`, `refresh` | None | 绘制轨迹 |
| `env.draw_line()` | `start_point`, `end_point`, `color`, `linewidth` | None | 绘制线段 |
| `env.render()` | None | None | 渲染显示 |

---

## 附录 B: YAML 配置模板

### B.1 最小配置

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

### B.2 完整配置（包含所有选项）

```yaml
world:
  height: 42
  width: 42
  step_time: 0.1
  sample_time: 0.1
  offset: [5, 5]
  collision_mode: 'stop'  # 'stop', 'unobstructed', 'reactive', 'unobstructed_obstacles'
  control_mode: 'auto'    # 'auto', 'keyboard'

robot:
  - kinematics: {name: 'diff'}  # 'diff', 'omni', 'acker'
    shape: {name: 'circle', radius: 0.2}
    state: [1, 1, 0]
    goal: [9, 9, 0]
    vel_min: [-8, -3.14]
    vel_max: [8, 3.14]
    goal_threshold: 0.3
    arrive_mode: 'position'  # 'position', 'state'
    behavior: {name: 'dash'}
    color: 'g'
    description: 'robot.png'

    plot:
      show_trail: True
      show_goal: True
      show_arrow: True
      traj_color: 'g'

    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 10
        angle_range: 3.1415926
        number: 100
        noise: False
        std: 0.1
        has_velocity: False

obstacle:
  # 静态障碍物（手动分布）
  - number: 6
    distribution: {name: 'manual'}
    shape:
      - {name: 'rectangle', length: 70, width: 2}
      - {name: 'circle', radius: 1.5}
      - {name: 'polygon', vertices: [[0,0], [1,0], [1,1], [0,1]]}
    state: [[30, 25, 0], [20, 15, 0], ...]
    color: 'gray'

  # 静态障碍物（随机分布）
  - number: 10
    distribution:
      name: 'random'
      range_low: [10, 10, -3.14]
      range_high: [40, 40, 3.14]
    shape:
      - {name: 'circle', radius: 1.0}

  # 随机形状障碍物
  - number: 11
    distribution: {name: 'manual'}
    shape:
      - name: 'polygon'
        random_shape: true
        center_range: [0, 0, 0, 0]
        avg_radius_range: [0.5, 1.0]
        irregularity_range: [0.9, 1.0]
    state: [[20, 34], [31, 38], ...]

  # 动态障碍物
  - number: 20
    distribution: {name: 'random', range_low: [10, 10, -3.14], range_high: [40, 40, 3.14]}
    kinematics: {name: 'diff'}
    shape:
      - {name: 'circle', radius: 0.5}
    behavior:
      - name: 'rvo'
        range_low: [10, 10, -3.14]
        range_high: [40, 40, 3.14]
        wander: True
        vxmax: 0.5
        vymax: 0.5
    vel_min: [-1.0, -3.14]
    vel_max: [1.0, 3.14]
    arrive_mode: 'position'
    goal_threshold: 0.3
    plot:
      show_goal: False
      show_arrow: True
```

---

**文档版本**: v1.0
**最后更新**: 2025-01-XX
**作者**: NeuPAN 项目组
**参考资源**:
- IR-SIM GitHub: https://github.com/hanruihua/ir-sim
- IR-SIM 文档: https://ir-sim.readthedocs.io/
- NeuPAN 项目: https://github.com/hanruihua/NeuPAN

