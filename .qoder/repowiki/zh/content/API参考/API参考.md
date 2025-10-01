# API参考

<cite>
**本文档中引用的文件**  
- [neupan.py](file://neupan/neupan.py)
- [pan.py](file://neupan/blocks/pan.py)
- [nrmp.py](file://neupan/blocks/nrmp.py)
- [dune.py](file://neupan/blocks/dune.py)
- [initial_path.py](file://neupan/blocks/initial_path.py)
- [robot.py](file://neupan/robot/robot.py)
</cite>

## 目录
1. [简介](#简介)
2. [NeuPAN类接口](#neupan类接口)
3. [PAN类接口](#pan类接口)
4. [robot类接口](#robot类接口)
5. [实际使用示例](#实际使用示例)
6. [属性说明](#属性说明)

## 简介
NeuPAN是一个基于模型预测控制（MPC）框架的路径规划算法，结合了神经网络与优化方法，用于机器人在动态环境中的实时避障与路径跟踪。本API文档详细说明了`neupan`、`PAN`和`robot`三个核心类的主要公共接口，包括初始化、前向传播、点云处理、路径设置等关键功能。

**本文档中引用的文件**  
- [neupan.py](file://neupan/neupan.py)
- [pan.py](file://neupan/blocks/pan.py)
- [robot.py](file://neupan/robot/robot.py)

## NeuPAN类接口

### init_from_yaml
从YAML配置文件初始化NeuPAN规划器。

**参数**：
- `yaml_file`：YAML配置文件路径
- `**kwargs`：可选的覆盖参数

**返回值**：初始化后的`neupan`实例

**调用顺序**：通常作为创建规划器的入口方法，在控制循环开始前调用。

**使用场景**：适用于通过配置文件快速构建规划器，便于参数调整和实验复现。

[SPEC SYMBOL](file://neupan/neupan.py#L73-L73)

### forward
执行一次规划前向传播，计算最优控制输入。

**参数**：
- `state`：机器人当前状态，形状为(3, 1)的矩阵，包含x, y, theta
- `points`：障碍物点云位置，形状为(2, N)的矩阵
- `velocities`：障碍物点云速度，形状为(2, N)的矩阵（可选）

**返回值**：
- `action`：最优速度控制输入，形状为(2, 1)的矩阵
- `info`：包含规划状态信息的字典

**调用顺序**：在控制循环中每一步调用，接收传感器数据后立即执行。

**使用场景**：实时控制循环中的核心规划步骤，用于生成下一时刻的速度指令。

[SPEC SYMBOL](file://neupan/neupan.py#L100-L150)

### scan_to_point
将激光雷达扫描数据转换为全局坐标系下的点云。

**参数**：
- `state`：机器人当前状态 [x, y, theta]
- `scan`：扫描数据字典，包含ranges, angle_min, angle_max等字段
- `scan_offset`：传感器相对于机器人坐标系的偏移
- `angle_range`：角度范围过滤
- `down_sample`：降采样步长

**返回值**：转换后的点云数据，形状为(2, n)

**使用场景**：将原始传感器数据预处理为规划器可接受的障碍物点云格式。

[SPEC SYMBOL](file://neupan/neupan.py#L160-L200)

### set_initial_path
设置初始路径。

**参数**：
- `path`：路径点列表，每个点为[x, y, theta, gear]的4x1向量，gear表示前进或后退档位

**使用场景**：当需要手动指定初始路径而非自动生成时使用。

[SPEC SYMBOL](file://neupan/neupan.py#L280-L290)

**本文档中引用的文件**  
- [neupan.py](file://neupan/neupan.py)
- [initial_path.py](file://neupan/blocks/initial_path.py)

## PAN类接口

### forward
PAN模块的前向传播，整合NRMP和DUNE进行优化求解。

**参数**：
- `nom_s`：名义状态，形状(3, receding+1)
- `nom_u`：名义控制，形状(2, receding)
- `ref_s`：参考轨迹，形状(3, receding+1)
- `ref_us`：参考速度数组，形状(receding,)
- `obs_points`：障碍物点云，形状(2, N)
- `point_velocities`：点云速度，形状(2, N)

**返回值**：
- `opt_state`：最优状态序列
- `opt_vel`：最优速度序列
- `nom_distance`：名义距离

**处理流程**：迭代执行DUNE特征提取和NRMP优化，直至收敛。

[SPEC SYMBOL](file://neupan/blocks/pan.py#L100-L140)

### generate_point_flow
生成每个预测步长下的点流、旋转矩阵和障碍物点列表。

**参数**：
- `nom_s`：名义状态序列
- `obs_points`：障碍物点云
- `point_velocities`：点云速度

**返回值**：
- `point_flow_list`：机器人坐标系下的点流列表
- `R_list`：旋转矩阵列表
- `obs_points_list`：全局坐标系下的障碍物点列表

[SPEC SYMBOL](file://neupan/blocks/pan.py#L150-L180)

**本文档中引用的文件**  
- [pan.py](file://neupan/blocks/pan.py)
- [dune.py](file://neupan/blocks/dune.py)
- [nrmp.py](file://neupan/blocks/nrmp.py)

## robot类接口

### __init__
初始化机器人模型。

**参数**：
- `receding`：预测时域步数
- `step_time`：时间步长
- `kinematics`：运动学模型类型（'acker'或'diff'）
- `vertices`：机器人顶点坐标
- `max_speed`：最大速度限制
- `max_acce`：最大加速度限制
- `wheelbase`：轴距（仅适用于阿克曼转向）
- `length`：长度
- `width`：宽度

**功能**：定义机器人几何形状、运动学约束和优化变量。

[SPEC SYMBOL](file://neupan/robot/robot.py#L10-L50)

### define_variable
定义优化问题中的变量。

**参数**：
- `no_obs`：是否无避障
- `indep_dis`：独立距离变量

**返回值**：变量列表，包含状态和速度变量。

[SPEC SYMBOL](file://neupan/robot/robot.py#L60-L80)

### state_parameter_define
定义状态相关参数。

**返回值**：参数列表，包括名义状态、参考状态、参考速度等。

[SPEC SYMBOL](file://neupan/robot/robot.py#L90-L110)

**本文档中引用的文件**  
- [robot.py](file://neupan/robot/robot.py)

## 实际使用示例

### 从YAML配置初始化规划器
```python
planner = neupan.init_from_yaml('example/LON/planner.yaml')
```

### 在控制循环中调用forward方法
```python
while not arrived:
    # 获取当前状态和传感器数据
    state = get_current_state()
    scan_data = get_lidar_data()
    
    # 转换点云
    points = planner.scan_to_point(state, scan_data)
    
    # 执行规划
    action, info = planner.forward(state, points)
    
    # 执行控制
    apply_control(action)
    
    if info['stop'] or info['arrive']:
        break
```

**本文档中引用的文件**  
- [neupan.py](file://neupan/neupan.py)
- [example/LON/planner.yaml](file://example/LON/planner.yaml)

## 属性说明

### min_distance
当前最小障碍物距离，用于碰撞检测。

[SPEC SYMBOL](file://neupan/neupan.py#L300-L305)

### dune_points
DUNE模块考虑的障碍物点。

[SPEC SYMBOL](file://neupan/blocks/pan.py#L250-L260)

### nrmp_points
NRMP模块考虑的障碍物点。

[SPEC SYMBOL](file://neupan/blocks/pan.py#L260-L270)

### initial_path
当前使用的初始路径。

[SPEC SYMBOL](file://neupan/neupan.py#L310-L315)

### waypoints
用于生成初始路径的航点。

[SPEC SYMBOL](file://neupan/blocks/initial_path.py#L450-L460)

### opt_trajectory
MPC预测时域内的最优轨迹。

[SPEC SYMBOL](file://neupan/neupan.py#L320-L330)

### ref_trajectory
初始路径上的参考轨迹。

[SPEC SYMBOL](file://neupan/neupan.py#L335-L345)

**本文档中引用的文件**  
- [neupan.py](file://neupan/neupan.py)
- [pan.py](file://neupan/blocks/pan.py)
- [initial_path.py](file://neupan/blocks/initial_path.py)