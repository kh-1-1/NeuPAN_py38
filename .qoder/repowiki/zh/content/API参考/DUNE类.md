# DUNE类

<cite>
**本文档引用的文件**   
- [dune.py](file://neupan/blocks/dune.py)
- [dune_train.py](file://neupan/blocks/dune_train.py)
- [obs_point_net.py](file://neupan/blocks/obs_point_net.py)
- [pan.py](file://neupan/blocks/pan.py)
</cite>

## 目录
1. [简介](#简介)
2. [核心组件](#核心组件)
3. [初始化参数说明](#初始化参数说明)
4. [前向传播流程](#前向传播流程)
5. [模型加载与训练机制](#模型加载与训练机制)
6. [最小距离计算逻辑](#最小距离计算逻辑)
7. [高密度点云优化策略](#高密度点云优化策略)

## 简介
DUNE（Deep Unfolded Neural Encoder）是NeuPAN算法中的核心编码模块，负责将原始点云数据编码为NRMP优化器所需的碰撞特征（μ和λ）。该模块通过深度展开网络结构，实现对障碍物点云的高效特征提取，为后续的运动规划提供关键的几何约束信息。

## 核心组件

DUNE类作为深度展开神经编码器，集成了点云特征提取、模型加载、训练与推理功能。其主要职责包括：
- 将机器人坐标系下的点流映射到潜在距离特征空间
- 计算并输出碰撞特征μ和λ
- 维护最近障碍物距离状态
- 支持模型检查点加载与现场训练

**本节来源**
- [dune.py](file://neupan/blocks/dune.py#L28-L209)

## 初始化参数说明

DUNE类的`__init__`方法接受多个关键参数，用于配置模型行为和训练选项：

- **receding**: 预测时域长度，决定前向传播的时间步数
- **checkpoint**: 预训练模型检查点路径，用于加载已训练的DUNE模型权重
- **robot**: 机器人实例，包含机器人的几何信息（G和h矩阵）
- **dune_max_num**: 最大处理点数，限制单次处理的点云数量
- **train_kwargs**: 训练参数字典，包含模型名称、数据大小、训练轮数等配置

当提供`checkpoint`时，模型会从指定路径加载预训练权重；若未提供且`train_kwargs`配置有效，则会触发交互式训练流程。

**本节来源**
- [dune.py](file://neupan/blocks/dune.py#L28-L64)

## 前向传播流程

`forward`方法处理点流数据并生成碰撞特征，其处理流程如下：

1. **输入参数**：
   - `point_flow`: 机器人坐标系下的点流列表，形状为(state_dim, num_points)，列表长度为T+1
   - `R_list`: 旋转矩阵列表，用于从μ生成λ，形状为(2, 2)，列表长度为T
   - `obs_points_list`: 全局坐标系下的障碍物点列表

2. **处理流程**：
   - 将所有时间步的点流水平拼接为单个张量
   - 通过ObsPointNet模型批量计算所有时间步的μ特征
   - 按时间步切分总μ特征，分别计算每个时间步的λ特征
   - 基于目标函数距离对点进行排序，确保最近点优先处理

3. **多时间步预测应用**：
   - 在每个预测时间步独立计算碰撞特征
   - 保持时间序列一致性，支持滚动时域预测
   - 输出排序后的特征列表，便于后续优化器处理

**本节来源**
- [dune.py](file://neupan/blocks/dune.py#L66-L109)

## 模型加载与训练机制

DUNE通过`load_model`方法实现灵活的模型加载与训练机制：

- **检查点加载**：当提供有效`checkpoint`路径时，直接加载预训练模型权重
- **交互式训练**：当未找到检查点时，提示用户是否立即训练新模型
- **训练参数传递**：通过`train_kwargs`字典传递训练配置，如数据大小、批次大小、学习率等
- **训练流程**：使用DUNETrain类生成训练数据集，通过优化问题求解获取标签，执行端到端训练

训练过程采用多损失函数组合，包括μ预测损失、距离损失、fa和fb几何一致性损失，确保模型学习到准确的碰撞特征。

**本节来源**
- [dune.py](file://neupan/blocks/dune.py#L111-L170)
- [dune_train.py](file://neupan/blocks/dune_train.py#L61-L543)

## 最小距离计算逻辑

`min_distance`属性记录当前场景中的最小障碍物距离，其计算逻辑如下：

- 在`forward`方法中，对每个时间步计算目标函数距离
- 使用`cal_objective_distance`方法计算距离值，公式为μᵀ(Gp₀ - h)
- 仅在第一个时间步（index == 0）更新`min_distance`属性
- 通过`torch.min(distance)`获取当前帧的最小距离值
- 该属性为只读属性，通过`@property`装饰器实现

此最小距离值可用于碰撞检测和安全距离监控，是系统安全性的重要指标。

**本节来源**
- [dune.py](file://neupan/blocks/dune.py#L104-L107)
- [pan.py](file://neupan/blocks/pan.py#L243-L249)

## 高密度点云优化策略

针对高密度点云的性能优化主要通过`downsample_decimation`策略实现：

- 当输入点云数量超过`dune_max_num`时自动触发降采样
- 在PAN类的`generate_point_flow`方法中执行降采样
- 使用`downsample_decimation`工具函数进行均匀降采样
- 保持点云的空间分布特性，避免信息过度损失
- 通过`print_once`机制提示用户降采样操作

此策略有效控制了计算复杂度，确保系统在高密度点云场景下的实时性能，同时保持足够的环境感知精度。

**本节来源**
- [pan.py](file://neupan/blocks/pan.py#L155-L165)
- [dune.py](file://neupan/blocks/dune.py#L34-L35)