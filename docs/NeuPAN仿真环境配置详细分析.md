# NeuPAN-py38 仿真环境配置详细分析

## 📋 目录
1. [example 目录结构分析](#1-example-目录结构分析)
2. [障碍物配置分析](#2-障碍物配置分析)
3. [IR-SIM 仿真器使用](#3-ir-sim-仿真器使用)
4. [动态变化机制分析](#4-动态变化机制分析)
5. [环境随机化实现建议](#5-环境随机化实现建议)

---

## 1. example 目录结构分析

### 1.1 目录概览

```
example/
├── LON/                    # Learning-based Online tuNing 实验
├── corridor/               # 走廊场景（静态障碍物）
├── convex_obs/             # 凸障碍物场景（圆形+多边形）
├── dune_train/             # DUNE 模型训练场景
├── dyna_obs/               # 动态障碍物场景（移动障碍物）
├── dyna_non_obs/           # 动态非凸障碍物场景
├── non_obs/                # 非凸障碍物场景（多边形）
├── pf/                     # 势场法测试场景（无障碍物）
├── pf_obs/                 # 势场法测试场景（有障碍物）
├── polygon_robot/          # 多边形机器人测试
├── reverse/                # 倒车场景
├── model/                  # 预训练模型存储
├── animation/              # 动画输出
├── animation_buffer/       # 动画帧缓存
└── run_exp.py              # 统一实验运行脚本
```

### 1.2 场景分类表

| 场景目录 | 机器人类型 | 障碍物类型 | 障碍物数量 | 动态性 | 主要用途 |
|---------|-----------|-----------|-----------|--------|---------|
| **LON** | diff | 矩形（静态） | 6 | 静态 | 在线参数学习 |
| **corridor** | diff/acker | 矩形（静态） | 6 | 静态 | 走廊导航测试 |
| **convex_obs** | diff/acker | 圆形+多边形 | 11 | 静态 | 凸障碍物避障 |
| **dune_train** | diff/acker | 无 | 0 | - | DUNE 模型训练 |
| **dyna_obs** | diff/acker | 圆形（动态） | 15-20 | 动态 | 动态避障测试 |
| **dyna_non_obs** | diff/acker | 多边形（动态） | 11 | 动态 | 动态非凸避障 |
| **non_obs** | diff/acker | 随机多边形 | 11 | 静态 | 非凸障碍物 |
| **pf** | diff/acker | 无 | 0 | - | 势场法基准 |
| **pf_obs** | diff/acker | 圆形+多边形 | 11 | 静态 | 势场法对比 |
| **polygon_robot** | diff | 自定义 | 变化 | 静态 | 多边形机器人 |
| **reverse** | diff/acker | 自定义 | 变化 | 静态 | 倒车测试 |

### 1.3 配置文件结构

每个场景目录（除 LON 和 dune_train）通常包含：
```
scene_name/
├── diff/
│   ├── env.yaml        # 环境配置（世界、机器人、障碍物）
│   └── planner.yaml    # 规划器配置（MPC、DUNE、NRMP 参数）
└── acker/
    ├── env.yaml
    └── planner.yaml
```

---

## 2. 障碍物配置分析

### 2.1 障碍物定义方式

#### 2.1.1 基本结构

```yaml
obstacle:
  - number: 6                              # 障碍物数量
    distribution: {name: 'manual'}         # 分布方式：manual/random
    shape:                                 # 形状列表
      - {name: 'rectangle', length: 70, width: 2}
      - {name: 'circle', radius: 1.5}
      - {name: 'polygon', vertices: [[x1,y1], [x2,y2], ...]}
    state: [[x, y, theta], ...]            # 位置和姿态
    kinematics: {name: 'diff'}             # 运动学模型（动态障碍物）
    behavior: {name: 'rvo', ...}           # 行为模式（动态障碍物）
```

### 2.2 障碍物类型详解

#### 2.2.1 矩形障碍物（Rectangle）

**示例：LON_corridor.yaml**
```yaml
obstacle:
  - number: 6
    distribution: {name: 'manual'}
    shape:
      - {name: 'rectangle', length: 70, width: 2}   # 走廊墙壁
      - {name: 'rectangle', length: 70, width: 2}
      - {name: 'rectangle', length: 5, width: 2}    # 小障碍物
      - {name: 'rectangle', length: 5, width: 2}
      - {name: 'rectangle', length: 6, width: 2}
      - {name: 'rectangle', length: 5, width: 2}
    state: [
      [30, 25, 0],      # [x, y, theta]
      [30, 15, 0],
      [10, 18.5, 1.57], # 旋转 90°
      [23, 21.5, 1.57],
      [36, 17, 2.1],
      [50, 22, 4.3]
    ]
```

**参数说明**：
- `length`: 矩形长度（米）
- `width`: 矩形宽度（米）
- `state`: `[x, y, theta]` - 中心位置和旋转角度（弧度）

#### 2.2.2 圆形障碍物（Circle）

**示例：convex_obs/diff/env.yaml**
```yaml
obstacle:
  - number: 10
    distribution: {name: 'manual'}
    state: [[20, 34], [31, 38], [10, 20], ...]
    shape:
      - {name: 'circle', radius: 1.5}
      - {name: 'circle', radius: 1.0}
```

**参数说明**：
- `radius`: 圆形半径（米）
- `state`: `[x, y]` - 圆心位置（圆形无需旋转角度）

#### 2.2.3 多边形障碍物（Polygon）

**方式 1：手动指定顶点**
```yaml
obstacle:
  - number: 1
    distribution: {name: 'manual'}
    shape:
      - {name: 'polygon', vertices: [[31, 24], [33, 24], [33, 28], [31, 28]]}
    state: [[0, 0, 0]]  # 相对于顶点的偏移
```

**方式 2：随机生成多边形**
```yaml
obstacle:
  - number: 11
    distribution: {name: 'manual'}
    shape:
      - name: 'polygon'
        random_shape: true
        center_range: [0, 0, 0, 0]
        avg_radius_range: [0.5, 1.0]      # 平均半径范围
        irregularity_range: [0.9, 1.0]    # 不规则度（0-1，1为正多边形）
    state: [[20, 34], [31, 38], ...]
```

**参数说明**：
- `vertices`: 顶点坐标列表 `[[x1,y1], [x2,y2], ...]`
- `random_shape`: 是否随机生成
- `avg_radius_range`: 平均半径范围
- `irregularity_range`: 不规则度（1.0 = 正多边形，0.0 = 高度不规则）

### 2.3 障碍物分布方式

#### 2.3.1 手动分布（Manual）

```yaml
distribution: {name: 'manual'}
state: [[x1, y1, theta1], [x2, y2, theta2], ...]
```

- 精确控制每个障碍物的位置
- 适用于固定场景设计

#### 2.3.2 随机分布（Random）

```yaml
distribution:
  name: 'random'
  range_low: [10, 10, -3.14]   # [x_min, y_min, theta_min]
  range_high: [40, 40, 3.14]   # [x_max, y_max, theta_max]
```

- 在指定范围内随机生成障碍物位置
- 每次运行环境时位置不同
- 适用于泛化性测试

### 2.4 障碍物形状随机化 ⚠️ **重要发现**

#### 2.4.1 随机形状生成

IR-SIM 支持在**位置固定**的情况下，**形状随机变化**：

```yaml
obstacle:
  - number: 11
    distribution: {name: 'manual'}  # 位置固定
    shape:
      - name: 'polygon'
        random_shape: true            # 形状随机！
        avg_radius_range: [0.5, 1.0]  # 平均半径范围
        irregularity_range: [0.9, 1.0]  # 不规则度范围
    state: [[20, 34], [31, 38], ...]  # 固定位置
```

**效果**：
- ✅ 障碍物位置不变
- 🎲 **每次运行时形状重新生成**
- 🎯 提供天然的环境随机化

#### 2.4.2 使用随机形状的场景

| 场景 | 位置 | 形状 | 配置文件 |
|------|------|------|---------|
| **non_obs/diff** | 固定 | **随机** | `example/non_obs/diff/env.yaml` |
| **dyna_non_obs/diff** | 随机 | **部分随机** | `example/dyna_non_obs/diff/env.yaml` |

#### 2.4.3 控制随机形状

**方法 1：设置随机种子**
```python
import numpy as np

# 在创建环境前设置种子
np.random.seed(42)
env = irsim.make("example/non_obs/diff/env.yaml")
# 相同种子 → 相同形状
```

**方法 2：修改为固定形状**
```yaml
# 将 random_shape: true 改为固定顶点
shape:
  - {name: 'polygon', vertices: [[0, 1], [0.9, 0.3], [0.5, -0.8], [-0.5, -0.8], [-0.9, 0.3]]}
```

**推荐策略**：
- **训练时**：不设置种子，利用形状随机性提升泛化能力
- **评估时**：设置固定种子，确保公平对比
- **调试时**：使用固定形状场景（如 LON_corridor）

> 📖 **详细分析**：参见 `docs/障碍物随机形状机制分析.md`

### 2.4 动态障碍物配置

#### 2.4.1 基本配置

**示例：dyna_obs/diff/env.yaml**
```yaml
obstacle:
  - number: 20
    distribution: {name: 'random', range_low: [10, 10, -3.14], range_high: [40, 40, 3.14]}
    kinematics: {name: 'diff'}              # 差速驱动
    shape:
      - {name: 'circle', radius: 0.5}
      - {name: 'circle', radius: 1.0}
    
    behavior:                                # 行为模式
      - name: 'rvo'                          # Reciprocal Velocity Obstacles
        range_low: [10, 10, -3.14]
        range_high: [40, 40, 3.14]
        wander: True                         # 随机游走
        vxmax: 0.5                           # 最大线速度
        vymax: 0.5
    
    vel_min: [-1.0, -3.14]                  # 速度限制
    vel_max: [1.0, 3.14]
    arrive_mode: position
    goal_threshold: 0.3
```

#### 2.4.2 行为模式

| 行为模式 | 说明 | 参数 |
|---------|------|------|
| **rvo** | Reciprocal Velocity Obstacles | `vxmax`, `vymax`, `wander` |
| **dash** | 冲刺行为 | `range_low`, `range_high`, `wander` |
| **wander** | 随机游走 | `range_low`, `range_high` |

#### 2.4.3 运动学模型

| 模型 | 说明 | 适用场景 |
|------|------|---------|
| **diff** | 差速驱动 | 圆形移动机器人 |
| **omni** | 全向移动 | 全向轮机器人 |
| **acker** | 阿克曼转向 | 类车机器人 |

---

## 3. IR-SIM 仿真器使用

### 3.1 IR-SIM 核心 API

#### 3.1.1 环境创建

**文件路径**: `example/run_exp.py` (第 20 行)

```python
import irsim

# 创建仿真环境
env = irsim.make(
    env_file,           # YAML 配置文件路径
    save_ani=False,     # 是否保存动画
    full=False,         # 是否全屏显示
    display=True        # 是否显示可视化界面
)
```

#### 3.1.2 环境交互

**文件路径**: `example/run_exp.py` (第 30-73 行)

```python
# 主循环
for i in range(max_steps):
    # 1. 获取机器人状态
    robot_state = env.get_robot_state()
    # 返回: [x, y, theta, v, w] (位置、姿态、速度)
    
    # 2. 获取激光雷达扫描
    lidar_scan = env.get_lidar_scan()
    # 返回: 激光雷达距离数组
    
    # 3. 转换为障碍物点云
    points = neupan_planner.scan_to_point(robot_state, lidar_scan)
    
    # 4. 规划器计算动作
    action, info = neupan_planner(robot_state, points)
    
    # 5. 可视化
    env.draw_points(points, s=25, c="g", refresh=True)
    env.draw_trajectory(trajectory, "r", refresh=True)
    env.render()
    
    # 6. 执行动作
    env.step(action)
    
    # 7. 检查终止条件
    if env.done():
        break

# 8. 结束仿真
env.end(delay=3, ani_name="animation")
```

### 3.2 IR-SIM 关键功能

#### 3.2.1 状态获取

```python
# 机器人状态
robot_state = env.get_robot_state()
# 返回: numpy.ndarray, shape=(5,) 或 (6,)
# diff: [x, y, theta, v, w]
# acker: [x, y, theta, v, w, delta]

# 激光雷达数据
lidar_scan = env.get_lidar_scan()
# 返回: numpy.ndarray, shape=(num_beams,)
```

#### 3.2.2 碰撞检测

**文件路径**: `neupan/neupan.py` (第 251-266 行)

```python
class neupan:
    def set_env_reference(self, env):
        """设置 IR-SIM 环境引用"""
        self._env = env
    
    def check_stop(self):
        """检查是否需要停止规划（使用 IR-SIM 碰撞检测）"""
        return self._env.done() if self._env else False
```

#### 3.2.3 可视化

```python
# 绘制点云
env.draw_points(
    points,          # numpy.ndarray, shape=(N, 2)
    s=25,            # 点大小
    c="g",           # 颜色 ('r', 'g', 'b', ...)
    refresh=True     # 是否刷新
)

# 绘制轨迹
env.draw_trajectory(
    trajectory,      # List[numpy.ndarray], 每个元素 shape=(3,1) 或 (4,1)
    traj_type="r",   # 轨迹类型/颜色
    show_direction=False,  # 是否显示方向箭头
    refresh=True
)

# 渲染
env.render()
```

### 3.3 IR-SIM 配置文件结构

#### 3.3.1 世界配置

```yaml
world:
  height: 42              # 世界高度（米）
  width: 42               # 世界宽度（米）
  step_time: 0.1          # 仿真步长（秒）
  sample_time: 0.1        # 采样时间（秒）
  offset: [5, 5]          # 显示偏移
  collision_mode: 'stop'  # 碰撞模式：stop/unobstructed/reactive
  control_mode: 'auto'    # 控制模式：auto/keyboard
```

#### 3.3.2 机器人配置

```yaml
robot:
  - kinematics: {name: 'diff'}                    # 运动学模型
    shape: {name: 'rectangle', length: 1.6, width: 2.0}
    state: [-5, 20, 0]                            # 初始状态 [x, y, theta]
    goal: [40, 40, 0]                             # 目标位置
    vel_min: [-8, -3.14]                          # 最小速度 [v, w]
    vel_max: [8, 3.14]                            # 最大速度
    goal_threshold: 0.3                           # 到达阈值（米）
    
    sensors:
      - type: 'lidar2d'                           # 传感器类型
        range_min: 0                              # 最小距离
        range_max: 10                             # 最大距离
        angle_range: 3.1415926                    # 扫描角度（弧度）
        number: 100                               # 扫描点数
        noise: False                              # 是否添加噪声
        std: 0.1                                  # 噪声标准差
```

---

## 4. 动态变化机制分析

### 4.1 现有动态机制

#### 4.1.1 动态障碍物（IR-SIM 内置）

**支持情况**: ✅ **已支持**

**实现方式**: 通过 YAML 配置

```yaml
obstacle:
  - number: 20
    distribution: {name: 'random', ...}
    kinematics: {name: 'diff'}
    behavior: {name: 'rvo', wander: True, ...}
```

**特点**：
- 障碍物在仿真过程中自主移动
- 支持多种行为模式（RVO、Dash、Wander）
- 位置随时间动态变化

#### 4.1.2 随机初始化（IR-SIM 内置）

**支持情况**: ✅ **已支持**

**实现方式**:
```yaml
distribution: 
  name: 'random'
  range_low: [10, 10, -3.14]
  range_high: [40, 40, 3.14]
```

**特点**：
- 每次 `env.reset()` 时障碍物位置随机
- 适用于泛化性测试

### 4.2 缺失的动态机制

#### 4.2.1 环境随机化（Domain Randomization）

**支持情况**: ❌ **未实现**

**需求**：
- 动态调整障碍物数量
- 动态调整障碍物尺寸
- 动态调整传感器噪声
- 动态调整走廊宽度

#### 4.2.2 课程学习环境调整

**支持情况**: ❌ **未实现**

**需求**：
- 根据训练进度调整环境难度
- 从简单场景逐步过渡到复杂场景

---

## 5. 环境随机化实现建议

### 5.1 方案设计

由于 IR-SIM 通过 YAML 文件加载环境，我们需要：
1. **动态生成 YAML 配置文件**
2. **在每个 episode 开始时重新加载环境**

### 5.2 实现代码

#### 5.2.1 环境随机化类

```python
# example/adaptive_LON/environment_randomizer.py

import yaml
import numpy as np
from pathlib import Path
import irsim

class EnvironmentRandomizer:
    """环境随机化器"""
    
    def __init__(self, base_config_path):
        """
        Args:
            base_config_path: 基础配置文件路径
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        self.temp_config_path = Path("temp_env_config.yaml")
    
    def randomize(self, randomization_config):
        """
        生成随机化的环境配置
        
        Args:
            randomization_config: 随机化配置字典
                {
                    'obstacle_position': (-2.0, 2.0),
                    'obstacle_size': (0.7, 1.3),
                    'corridor_width': (2.0, 6.0),
                    'noise_std': (0.0, 0.2),
                    'num_obstacles': (2, 10)
                }
        """
        config = self.base_config.copy()
        
        # 1. 随机化障碍物位置
        if 'obstacle_position' in randomization_config:
            offset_range = randomization_config['obstacle_position']
            for obs_group in config.get('obstacle', []):
                if obs_group.get('distribution', {}).get('name') == 'manual':
                    states = obs_group['state']
                    for i in range(len(states)):
                        offset_x = np.random.uniform(*offset_range)
                        offset_y = np.random.uniform(*offset_range)
                        states[i][0] += offset_x
                        states[i][1] += offset_y
        
        # 2. 随机化障碍物尺寸
        if 'obstacle_size' in randomization_config:
            scale_range = randomization_config['obstacle_size']
            for obs_group in config.get('obstacle', []):
                for shape in obs_group.get('shape', []):
                    scale = np.random.uniform(*scale_range)
                    if shape['name'] == 'rectangle':
                        shape['length'] *= scale
                        shape['width'] *= scale
                    elif shape['name'] == 'circle':
                        shape['radius'] *= scale
        
        # 3. 随机化传感器噪声
        if 'noise_std' in randomization_config:
            noise_range = randomization_config['noise_std']
            noise_std = np.random.uniform(*noise_range)
            for robot in config.get('robot', []):
                for sensor in robot.get('sensors', []):
                    sensor['noise'] = (noise_std > 0)
                    sensor['std'] = noise_std
        
        # 4. 随机化障碍物数量
        if 'num_obstacles' in randomization_config:
            num_range = randomization_config['num_obstacles']
            num_obstacles = np.random.randint(*num_range)
            # 这里需要根据具体场景调整
        
        return config
    
    def save_and_create_env(self, config, display=True):
        """保存配置并创建环境"""
        # 保存临时配置文件
        with open(self.temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 创建环境
        env = irsim.make(str(self.temp_config_path), display=display)
        return env
    
    def cleanup(self):
        """清理临时文件"""
        if self.temp_config_path.exists():
            self.temp_config_path.unlink()
```

#### 5.2.2 课程学习管理器

```python
# example/adaptive_LON/curriculum_manager.py

class CurriculumManager:
    """课程学习管理器"""
    
    def __init__(self):
        self.stages = {
            'easy': {
                'obstacle_position': (-1.0, 1.0),
                'obstacle_size': (0.9, 1.1),
                'corridor_width': (5.0, 6.0),
                'noise_std': (0.0, 0.05),
                'num_obstacles': (2, 4)
            },
            'medium': {
                'obstacle_position': (-2.0, 2.0),
                'obstacle_size': (0.8, 1.2),
                'corridor_width': (3.0, 5.0),
                'noise_std': (0.0, 0.1),
                'num_obstacles': (4, 7)
            },
            'hard': {
                'obstacle_position': (-3.0, 3.0),
                'obstacle_size': (0.7, 1.3),
                'corridor_width': (2.0, 4.0),
                'noise_std': (0.0, 0.2),
                'num_obstacles': (6, 10)
            }
        }
        
        self.current_stage = 'easy'
        self.success_history = []
    
    def get_current_config(self):
        """获取当前阶段的随机化配置"""
        return self.stages[self.current_stage]
    
    def update(self, success):
        """更新课程学习状态"""
        self.success_history.append(success)
        
        # 计算最近 20 个 episode 的成功率
        if len(self.success_history) >= 20:
            recent_success_rate = np.mean(self.success_history[-20:])
            
            # 根据成功率调整难度
            if recent_success_rate > 0.8:
                if self.current_stage == 'easy':
                    self.current_stage = 'medium'
                    print("📈 课程学习：进入 Medium 阶段")
                elif self.current_stage == 'medium':
                    self.current_stage = 'hard'
                    print("📈 课程学习：进入 Hard 阶段")
```

#### 5.2.3 训练循环集成

```python
# example/adaptive_LON/train_with_randomization.py

import irsim
from neupan import neupan
from environment_randomizer import EnvironmentRandomizer
from curriculum_manager import CurriculumManager

def train_with_randomization():
    """使用环境随机化的训练"""
    
    # 初始化
    base_config = "example/LON/LON_corridor.yaml"
    planner_config = "example/LON/planner.yaml"
    
    randomizer = EnvironmentRandomizer(base_config)
    curriculum = CurriculumManager()
    planner = neupan.init_from_yaml(planner_config)
    
    for epoch in range(150):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/150 - Stage: {curriculum.current_stage}")
        print(f"{'='*60}")
        
        for episode in range(400):
            # 1. 获取当前课程阶段的随机化配置
            random_config = curriculum.get_current_config()
            
            # 2. 生成随机化环境
            env_config = randomizer.randomize(random_config)
            env = randomizer.save_and_create_env(env_config, display=False)
            planner.set_env_reference(env)
            
            # 3. 运行一个 episode
            success = run_one_episode(env, planner)
            
            # 4. 更新课程学习
            curriculum.update(success)
            
            # 5. 清理环境
            env.end(delay=0)
        
        # 每个 epoch 结束后保存模型
        save_checkpoint(planner, epoch)
    
    # 清理
    randomizer.cleanup()

def run_one_episode(env, planner):
    """运行一个 episode"""
    for step in range(400):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()
        points = planner.scan_to_point(robot_state, lidar_scan)
        
        action, info = planner(robot_state, points)
        env.step(action)
        
        if info['arrive']:
            return True
        if env.done():
            return False
    
    return False
```

### 5.3 使用示例

```python
# 快速测试
if __name__ == "__main__":
    randomizer = EnvironmentRandomizer("example/LON/LON_corridor.yaml")
    
    # 生成 5 个随机环境
    for i in range(5):
        config = randomizer.randomize({
            'obstacle_position': (-2.0, 2.0),
            'obstacle_size': (0.8, 1.2),
            'noise_std': (0.0, 0.2)
        })
        
        env = randomizer.save_and_create_env(config, display=True)
        print(f"环境 {i+1} 已创建")
        
        # 运行几步查看效果
        for _ in range(50):
            env.render()
        
        env.end(delay=1)
    
    randomizer.cleanup()
```

---

## 6. 总结

### 6.1 现有能力

✅ **已支持**：
- 多种障碍物类型（矩形、圆形、多边形）
- 动态障碍物（移动、行为模式）
- 随机初始化（位置随机）
- 完整的 IR-SIM 仿真环境
- 丰富的场景库（11 种场景）

### 6.2 需要实现

❌ **待实现**：
- 环境随机化（Domain Randomization）
- 课程学习环境调整
- 动态障碍物数量调整
- 动态障碍物尺寸调整
- 传感器噪声动态调整

### 6.3 实现路径

1. **短期**（1 周）：
   - 实现 `EnvironmentRandomizer` 类
   - 实现 `CurriculumManager` 类
   - 集成到训练循环

2. **中期**（2 周）：
   - 测试不同随机化策略
   - 优化课程学习阈值
   - 验证泛化性能

3. **长期**（1 个月）：
   - 扩展到更多场景
   - 实现元学习
   - 在线适应机制

---

## 附录 A: 场景障碍物配置对比表

### A.1 静态场景障碍物配置

| 场景 | 障碍物类型 | 数量 | 分布方式 | 位置示例 | 用途 |
|------|-----------|------|---------|---------|------|
| **LON_corridor** | 矩形 | 6 | manual | `[30,25,0]`, `[30,15,0]`, ... | 走廊导航 + 参数学习 |
| **corridor/diff** | 矩形 | 6 | manual | `[30,25,0]`, `[30,15,0]`, ... | 走廊导航测试 |
| **convex_obs/diff** | 圆形 + 多边形 | 11 | manual | `[20,34]`, `[31,38]`, ... | 凸障碍物避障 |
| **non_obs/diff** | 随机多边形 | 11 | manual | `[20,34]`, `[31,38]`, ... | 非凸障碍物避障 |
| **pf_obs/diff** | 圆形 + 多边形 | 11 | manual | `[20,34]`, `[31,38]`, ... | 势场法对比 |

### A.2 动态场景障碍物配置

| 场景 | 障碍物类型 | 数量 | 分布方式 | 运动学 | 行为模式 | 速度范围 |
|------|-----------|------|---------|--------|---------|---------|
| **dyna_obs/diff** | 圆形 | 20 | random | diff | rvo (wander) | v: [-1, 1], w: [-3.14, 3.14] |
| **dyna_obs/acker** | 圆形 | 15 | random | diff | dash (wander) | v: [-0.5, 0.5], w: [-3.14, 3.14] |
| **dyna_non_obs/diff** | 多边形 + 圆形 | 11 | random | omni | rvo (wander) | vx: [-4, 4], vy: [-4, 4] |
| **dyna_non_obs/acker** | 多边形 + 圆形 | 11 | random | omni | rvo (wander) | vx: [-4, 4], vy: [-4, 4] |

### A.3 传感器配置对比

| 场景 | 传感器类型 | 扫描范围 | 扫描角度 | 扫描点数 | 噪声 | 噪声标准差 |
|------|-----------|---------|---------|---------|------|-----------|
| **LON_corridor** | lidar2d | 10m | 180° | 100 | False | 0.1 |
| **LON_corridor_01** | lidar2d | 10m | 180° | 100 | False | 0.1 |
| **LON_corridor_02** | lidar2d | 10m | 180° | 100 | True | 0.2 |
| **corridor/diff** | lidar2d | 10m | 180° | 100 | False | - |
| **dyna_obs/diff** | lidar2d | 10m | 180° | 100 | False | - |
| **dyna_non_obs/diff** | lidar2d | 10m | 180° | 100 | False | - |

---

## 附录 B: 完整配置文件示例

### B.1 LON_corridor.yaml（完整）

```yaml
world:
  height: 22
  width: 90
  step_time: 0.1
  sample_time: 0.1
  offset: [-10, 9]
  collision_mode: 'stop'
  control_mode: 'auto'

robot:
  - kinematics: {name: 'diff'}
    shape: {name: 'rectangle', length: 1.6, width: 2.0}
    state: [-5, 20, 0]
    goal: [80, 40, 0]
    vel_min: [-8, -3.14]
    vel_max: [8, 3.14]
    goal_threshold: 0.3
    description: diff_robot0.png
    plot:
      show_trail: True
      show_goal: False

    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 10
        angle_range: 3.1415926
        number: 100
        noise: False
        std: 0.1

obstacle:
  - number: 6
    distribution: {name: 'manual'}
    shape:
      - {name: 'rectangle', length: 70, width: 2}
      - {name: 'rectangle', length: 70, width: 2}
      - {name: 'rectangle', length: 5, width: 2}
      - {name: 'rectangle', length: 5, width: 2}
      - {name: 'rectangle', length: 6, width: 2}
      - {name: 'rectangle', length: 5, width: 2}
    state: [
      [30, 25, 0],
      [30, 15, 0],
      [10, 18.5, 1.57],
      [23, 21.5, 1.57],
      [36, 17, 2.1],
      [50, 22, 4.3]
    ]
```

### B.2 dyna_obs/diff/env.yaml（完整）

```yaml
world:
  height: 42
  width: 42
  step_time: 0.1
  sample_time: 0.1
  offset: [5, 5]
  collision_mode: 'stop'
  control_mode: 'auto'

robot:
  - kinematics: {name: 'diff'}
    shape: {name: 'rectangle', length: 1.6, width: 2.0}
    state: [10, 42, 1.57]
    goal: [40, 40, 0]
    vel_min: [-8, -3.14]
    vel_max: [8, 3.14]
    goal_threshold: 0.8
    description: diff_robot0.png
    plot:
      show_goal: True
      show_trail: True

    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 10
        angle_range: 3.1415926
        number: 100
        noise: False

obstacle:
  - number: 20
    distribution:
      name: 'random'
      range_low: [10, 10, -3.14]
      range_high: [40, 40, 3.14]
    kinematics: {name: 'diff'}
    shape:
      - {name: 'circle', radius: 0.5}
      - {name: 'circle', radius: 1.0}
      - {name: 'circle', radius: 1.0}
      - {name: 'circle', radius: 0.4}
    behavior:
      - name: 'rvo'
        range_low: [10, 10, -3.14]
        range_high: [40, 40, 3.14]
        wander: True
        vxmax: 0.5
        vymax: 0.5
    vel_min: [-1.0, -3.14]
    vel_max: [1.0, 3.14]
    arrive_mode: position
    goal_threshold: 0.3
    plot:
      show_goal: False
      show_arrow: True
```

---

## 附录 C: IR-SIM API 完整参考

### C.1 环境创建与管理

```python
import irsim

# 创建环境
env = irsim.make(
    env_file='path/to/env.yaml',
    save_ani=False,      # 是否保存动画
    full=False,          # 是否全屏
    display=True,        # 是否显示
    ani_name='animation' # 动画文件名
)

# 环境重置（如果支持）
# env.reset()  # 注意：当前版本可能不支持

# 环境结束
env.end(
    delay=3,             # 延迟时间（秒）
    ani_name='my_animation'  # 动画保存名称
)
```

### C.2 状态获取

```python
# 获取机器人状态
robot_state = env.get_robot_state()
# 返回: numpy.ndarray
# diff: [x, y, theta, v, w]
# acker: [x, y, theta, v, w, delta]

# 获取激光雷达扫描
lidar_scan = env.get_lidar_scan()
# 返回: numpy.ndarray, shape=(num_beams,)
# 每个元素是该方向的障碍物距离

# 获取机器人位置
position = robot_state[:2]  # [x, y]

# 获取机器人姿态
theta = robot_state[2]

# 获取机器人速度
velocity = robot_state[3:5]  # [v, w]
```

### C.3 动作执行

```python
# 执行动作
action = np.array([[v], [w]])  # shape=(2, 1)
env.step(action)

# 或者
action = np.array([v, w])  # shape=(2,)
env.step(action)
```

### C.4 可视化

```python
# 绘制点云
env.draw_points(
    points,              # numpy.ndarray, shape=(N, 2)
    s=25,                # 点大小
    c="g",               # 颜色：'r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'
    marker='o',          # 标记类型
    alpha=1.0,           # 透明度
    refresh=True         # 是否刷新之前的点
)

# 绘制轨迹
env.draw_trajectory(
    trajectory,          # List[numpy.ndarray], 每个 shape=(3,1) 或 (4,1)
    traj_type="r",       # 轨迹类型/颜色
    show_direction=False,  # 是否显示方向箭头
    refresh=True         # 是否刷新之前的轨迹
)

# 绘制线段
env.draw_line(
    start_point,         # [x, y]
    end_point,           # [x, y]
    color='r',
    linewidth=2
)

# 渲染（更新显示）
env.render()
```

### C.5 碰撞检测与终止条件

```python
# 检查是否结束（到达目标或碰撞）
is_done = env.done()
# 返回: bool

# 检查是否到达目标
# 需要通过 info 字典获取（由规划器提供）
action, info = neupan_planner(robot_state, points)
if info['arrive']:
    print("到达目标！")

# 检查是否碰撞
if env.done() and not info['arrive']:
    print("发生碰撞！")
```

---

## 附录 D: 环境随机化高级示例

### D.1 完整的随机化训练脚本

```python
# example/adaptive_LON/train_with_full_randomization.py

import irsim
import torch
import numpy as np
from neupan import neupan
from pathlib import Path
import yaml

class AdvancedEnvironmentRandomizer:
    """高级环境随机化器"""

    def __init__(self, base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        self.temp_path = Path("temp_env.yaml")

    def randomize_full(self, stage='medium'):
        """完整随机化"""
        config = self._deep_copy(self.base_config)

        # 根据阶段设置随机化强度
        if stage == 'easy':
            pos_range = (-1.0, 1.0)
            size_range = (0.9, 1.1)
            noise_range = (0.0, 0.05)
        elif stage == 'medium':
            pos_range = (-2.0, 2.0)
            size_range = (0.8, 1.2)
            noise_range = (0.0, 0.1)
        else:  # hard
            pos_range = (-3.0, 3.0)
            size_range = (0.7, 1.3)
            noise_range = (0.0, 0.2)

        # 1. 随机化障碍物位置
        for obs_group in config.get('obstacle', []):
            if obs_group.get('distribution', {}).get('name') == 'manual':
                for state in obs_group['state']:
                    state[0] += np.random.uniform(*pos_range)
                    state[1] += np.random.uniform(*pos_range)
                    if len(state) > 2:
                        state[2] += np.random.uniform(-0.3, 0.3)

        # 2. 随机化障碍物尺寸
        for obs_group in config.get('obstacle', []):
            for shape in obs_group.get('shape', []):
                scale = np.random.uniform(*size_range)
                if shape['name'] == 'rectangle':
                    shape['length'] *= scale
                    shape['width'] *= scale
                elif shape['name'] == 'circle':
                    shape['radius'] *= scale

        # 3. 随机化传感器噪声
        noise_std = np.random.uniform(*noise_range)
        for robot in config.get('robot', []):
            for sensor in robot.get('sensors', []):
                sensor['noise'] = (noise_std > 0)
                sensor['std'] = noise_std

        # 4. 随机化初始位置
        for robot in config.get('robot', []):
            robot['state'][0] += np.random.uniform(-2, 2)
            robot['state'][1] += np.random.uniform(-2, 2)

        # 5. 随机化目标位置
        for robot in config.get('robot', []):
            if 'goal' in robot:
                robot['goal'][0] += np.random.uniform(-2, 2)
                robot['goal'][1] += np.random.uniform(-2, 2)

        return config

    def _deep_copy(self, obj):
        """深拷贝"""
        import copy
        return copy.deepcopy(obj)

    def save_and_load(self, config, display=False):
        """保存并加载环境"""
        with open(self.temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        return irsim.make(str(self.temp_path), display=display)

    def cleanup(self):
        """清理临时文件"""
        if self.temp_path.exists():
            self.temp_path.unlink()


def train_with_full_randomization():
    """完整随机化训练"""

    # 初始化
    randomizer = AdvancedEnvironmentRandomizer("example/LON/LON_corridor.yaml")
    planner = neupan.init_from_yaml("example/LON/planner.yaml")

    # 优化器
    optimizer = torch.optim.Adam(planner.pan.nrmp_layer.adjust_parameters, lr=5e-3)

    # 训练循环
    stages = ['easy'] * 50 + ['medium'] * 50 + ['hard'] * 50

    for epoch, stage in enumerate(stages):
        print(f"\nEpoch {epoch+1}/150 - Stage: {stage}")

        epoch_loss = 0.0
        success_count = 0

        for episode in range(10):  # 每个 epoch 10 个 episodes
            # 生成随机环境
            config = randomizer.randomize_full(stage)
            env = randomizer.save_and_load(config, display=False)
            planner.set_env_reference(env)

            # 运行 episode
            episode_loss = 0.0
            for step in range(400):
                robot_state = env.get_robot_state()
                lidar_scan = env.get_lidar_scan()
                points = planner.scan_to_point(robot_state, lidar_scan)

                action, info = planner(robot_state, points)

                # 计算损失
                loss = calculate_loss(info)
                episode_loss += loss.item()

                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 执行动作
                env.step(action)

                if info['arrive']:
                    success_count += 1
                    break
                if env.done():
                    break

            epoch_loss += episode_loss
            env.end(delay=0)

        # 打印统计
        avg_loss = epoch_loss / 10
        success_rate = success_count / 10
        print(f"  Loss: {avg_loss:.4f}, Success Rate: {success_rate:.2%}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'params': {
                    'q_s': planner.pan.nrmp_layer.q_s.item(),
                    'p_u': planner.pan.nrmp_layer.p_u.item(),
                    'eta': planner.pan.nrmp_layer.eta.item(),
                    'd_max': planner.pan.nrmp_layer.d_max.item(),
                    'd_min': planner.pan.nrmp_layer.d_min.item(),
                }
            }, f'checkpoint_epoch_{epoch+1}.pth')

    randomizer.cleanup()


def calculate_loss(info):
    """计算损失"""
    # 距离损失
    distance = info.get('distance', torch.tensor(0.0))
    min_distance = torch.min(distance) if len(distance) > 0 else torch.tensor(10.0)

    if min_distance < 0.2:
        loss = 50 - torch.sum(distance)
    else:
        loss = torch.tensor(0.0, requires_grad=True)

    return loss


if __name__ == "__main__":
    train_with_full_randomization()
```

---

**文档版本**: v1.0
**创建日期**: 2025-01-XX
**作者**: AI Assistant

