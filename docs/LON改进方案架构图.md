# LON 改进 NeuPAN 方案架构图与流程图

## 1. 整体系统架构

```mermaid
graph TB
    subgraph "环境层 Environment Layer"
        ENV[IR-SIM 仿真环境]
        SENSOR[传感器数据<br/>激光雷达/状态]
    end
    
    subgraph "感知层 Perception Layer"
        SCAN[激光扫描数据]
        POINTS[障碍物点云]
        STATE[机器人状态]
    end
    
    subgraph "规划层 Planning Layer - Adaptive NeuPAN"
        subgraph "核心模块 Core Modules"
            IPATH[初始路径生成<br/>Initial Path]
            PAN[PAN 交替优化<br/>Proximal Alternating]
            
            subgraph "DUNE 模块 (固定)"
                DUNE_NET[FlexiblePDHGFront<br/>离线监督学习]
            end
            
            subgraph "NRMP 模块"
                NRMP_OPT[凸优化求解器<br/>Convex Optimizer]
                NRMP_PARAMS[可学习参数<br/>Learnable Params]
            end
        end
        
        subgraph "自适应学习模块 Adaptive Learning"
            PARAM_MGR[参数管理器<br/>Param Manager]
            LOSS_CALC[损失计算器<br/>Loss Calculator]
            OPTIMIZER[优化器<br/>Adam/SGD]
            CURRICULUM[课程学习<br/>Curriculum]
        end
    end
    
    subgraph "执行层 Execution Layer"
        ACTION[控制动作<br/>Velocity/Steering]
        FEEDBACK[环境反馈<br/>Reward/Penalty]
    end
    
    ENV --> SENSOR
    SENSOR --> SCAN
    SENSOR --> STATE
    SCAN --> POINTS
    STATE --> IPATH
    POINTS --> DUNE_NET
    DUNE_NET --> PAN
    IPATH --> PAN
    PAN --> NRMP_OPT
    NRMP_PARAMS --> NRMP_OPT
    NRMP_OPT --> ACTION
    ACTION --> ENV
    ENV --> FEEDBACK
    FEEDBACK --> LOSS_CALC
    LOSS_CALC --> OPTIMIZER
    OPTIMIZER --> PARAM_MGR
    PARAM_MGR --> NRMP_PARAMS
    CURRICULUM --> PARAM_MGR
    
    style ENV fill:#e1f5ff
    style PAN fill:#fff4e1
    style DUNE_NET fill:#ffe1f5
    style NRMP_OPT fill:#e1ffe1
    style OPTIMIZER fill:#ffe1e1
```

## 2. LON 在线学习流程

```mermaid
flowchart TD
    START([开始训练]) --> INIT[初始化环境和规划器]
    INIT --> EPOCH_START{开始新 Epoch}
    
    EPOCH_START --> CURRICULUM[获取课程学习阶段配置]
    CURRICULUM --> RESET[重置环境和规划器]
    RESET --> STEP_START{开始新 Step}
    
    STEP_START --> GET_STATE[获取机器人状态]
    GET_STATE --> GET_LIDAR[获取激光雷达数据]
    GET_LIDAR --> CONVERT[转换为点云]
    
    CONVERT --> FORWARD[NeuPAN 前向传播]
    
    subgraph "NeuPAN 前向传播"
        FORWARD --> DUNE[DUNE: 特征提取]
        DUNE --> NRMP[NRMP: 轨迹优化]
        NRMP --> OUTPUT[输出控制动作]
    end
    
    OUTPUT --> EXECUTE[执行动作]
    EXECUTE --> RENDER{需要渲染?}
    RENDER -->|是| DRAW[绘制轨迹]
    RENDER -->|否| CALC_LOSS
    DRAW --> CALC_LOSS[计算多目标损失]
    
    subgraph "损失计算"
        CALC_LOSS --> DIST_LOSS[距离损失]
        CALC_LOSS --> SMOOTH_LOSS[平滑度损失]
        CALC_LOSS --> ENERGY_LOSS[能量损失]
        CALC_LOSS --> TIME_LOSS[时间损失]
        CALC_LOSS --> TRACK_LOSS[跟踪损失]
        
        DIST_LOSS --> WEIGHTED_SUM[加权求和]
        SMOOTH_LOSS --> WEIGHTED_SUM
        ENERGY_LOSS --> WEIGHTED_SUM
        TIME_LOSS --> WEIGHTED_SUM
        TRACK_LOSS --> WEIGHTED_SUM
    end
    
    WEIGHTED_SUM --> BACKWARD[反向传播]
    BACKWARD --> CLIP_GRAD[梯度裁剪]
    CLIP_GRAD --> UPDATE_PARAMS[更新参数]
    UPDATE_PARAMS --> APPLY_CONSTRAINT[应用参数约束]
    
    APPLY_CONSTRAINT --> CHECK_TERM{检查终止条件}
    CHECK_TERM -->|到达目标| SUCCESS[记录成功]
    CHECK_TERM -->|碰撞| COLLISION[记录碰撞]
    CHECK_TERM -->|卡住| STUCK[记录卡住]
    CHECK_TERM -->|继续| STEP_START
    
    SUCCESS --> EPOCH_END
    COLLISION --> EPOCH_END
    STUCK --> EPOCH_END
    
    EPOCH_END[Epoch 结束] --> LOG[记录性能指标]
    LOG --> SAVE_CKPT{需要保存检查点?}
    SAVE_CKPT -->|是| SAVE[保存模型]
    SAVE_CKPT -->|否| CHECK_EARLY
    SAVE --> CHECK_EARLY{检查早停条件}
    
    CHECK_EARLY -->|满足| FINISH([训练完成])
    CHECK_EARLY -->|不满足| CHECK_EPOCH{达到最大 Epoch?}
    CHECK_EPOCH -->|是| FINISH
    CHECK_EPOCH -->|否| EPOCH_START
    
    style START fill:#90EE90
    style FINISH fill:#FFB6C1
    style FORWARD fill:#FFE4B5
    style CALC_LOSS fill:#E0BBE4
    style UPDATE_PARAMS fill:#FFA07A
```

## 3. 参数优化流程

```mermaid
flowchart LR
    subgraph "参数空间 Parameter Space"
        P1[q_s: 状态权重]
        P2[p_u: 控制权重]
        P3[eta: 避障权重]
        P4[d_max: 最大距离]
        P5[d_min: 最小距离]
    end
    
    subgraph "优化过程 Optimization Process"
        INIT_PARAMS[初始参数<br/>从配置文件]
        FORWARD_PASS[前向传播<br/>生成轨迹]
        LOSS_COMP[损失计算<br/>多目标]
        BACKWARD_PASS[反向传播<br/>计算梯度]
        GRADIENT[梯度信息<br/>∂L/∂θ]
        OPTIMIZER_STEP[优化器更新<br/>Adam]
        NEW_PARAMS[新参数值]
        CONSTRAINT[约束投影<br/>参数范围]
    end
    
    subgraph "约束条件 Constraints"
        C1[q_s ∈ [0.01, 5.0]]
        C2[p_u ∈ [0.1, 10.0]]
        C3[eta ∈ [1.0, 50.0]]
        C4[d_max ∈ [0.1, 2.0]]
        C5[d_min ∈ [0.01, 0.5]]
    end
    
    P1 --> INIT_PARAMS
    P2 --> INIT_PARAMS
    P3 --> INIT_PARAMS
    P4 --> INIT_PARAMS
    P5 --> INIT_PARAMS
    
    INIT_PARAMS --> FORWARD_PASS
    FORWARD_PASS --> LOSS_COMP
    LOSS_COMP --> BACKWARD_PASS
    BACKWARD_PASS --> GRADIENT
    GRADIENT --> OPTIMIZER_STEP
    OPTIMIZER_STEP --> NEW_PARAMS
    NEW_PARAMS --> CONSTRAINT
    
    C1 --> CONSTRAINT
    C2 --> CONSTRAINT
    C3 --> CONSTRAINT
    C4 --> CONSTRAINT
    C5 --> CONSTRAINT
    
    CONSTRAINT -.更新.-> P1
    CONSTRAINT -.更新.-> P2
    CONSTRAINT -.更新.-> P3
    CONSTRAINT -.更新.-> P4
    CONSTRAINT -.更新.-> P5
    
    style INIT_PARAMS fill:#B0E0E6
    style OPTIMIZER_STEP fill:#FFB6C1
    style CONSTRAINT fill:#98FB98
```

## 4. 多目标损失函数结构

```mermaid
graph TD
    subgraph "输入信息 Input Information"
        INFO[规划器输出 info]
        STATE[状态序列]
        VEL[速度序列]
        DIST[最小距离]
        REF[参考轨迹]
    end
    
    subgraph "损失计算 Loss Calculation"
        L_DIST[距离损失<br/>L_distance]
        L_SMOOTH[平滑度损失<br/>L_smoothness]
        L_ENERGY[能量损失<br/>L_energy]
        L_TIME[时间损失<br/>L_time]
        L_TRACK[跟踪损失<br/>L_tracking]
    end
    
    subgraph "损失公式 Loss Formulas"
        F_DIST["L_d = max(0, threshold - d_min)"]
        F_SMOOTH["L_s = Σ‖Δs‖² + Σ‖Δv‖²"]
        F_ENERGY["L_e = Σv²"]
        F_TIME["L_t = -10 if arrive else 1"]
        F_TRACK["L_tr = ‖s - s_ref‖²"]
    end
    
    subgraph "权重系数 Weights"
        W_DIST[w_d = 10.0]
        W_SMOOTH[w_s = 1.0]
        W_ENERGY[w_e = 0.5]
        W_TIME[w_t = 1.0]
        W_TRACK[w_tr = 2.0]
    end
    
    TOTAL[总损失<br/>L_total]
    
    INFO --> STATE
    INFO --> VEL
    INFO --> DIST
    INFO --> REF
    
    DIST --> L_DIST
    STATE --> L_SMOOTH
    VEL --> L_SMOOTH
    VEL --> L_ENERGY
    INFO --> L_TIME
    STATE --> L_TRACK
    REF --> L_TRACK
    
    L_DIST --> F_DIST
    L_SMOOTH --> F_SMOOTH
    L_ENERGY --> F_ENERGY
    L_TIME --> F_TIME
    L_TRACK --> F_TRACK
    
    F_DIST --> W_DIST
    F_SMOOTH --> W_SMOOTH
    F_ENERGY --> W_ENERGY
    F_TIME --> W_TIME
    F_TRACK --> W_TRACK
    
    W_DIST --> TOTAL
    W_SMOOTH --> TOTAL
    W_ENERGY --> TOTAL
    W_TIME --> TOTAL
    W_TRACK --> TOTAL
    
    style TOTAL fill:#FF6B6B
    style L_DIST fill:#FFE66D
    style L_SMOOTH fill:#4ECDC4
    style L_ENERGY fill:#95E1D3
    style L_TIME fill:#F38181
    style L_TRACK fill:#AA96DA
```

## 5. 课程学习策略

```mermaid
stateDiagram-v2
    [*] --> Easy: 开始训练
    
    state Easy {
        [*] --> Training_Easy
        Training_Easy --> Evaluation_Easy
        Evaluation_Easy --> Check_Easy
        Check_Easy --> Training_Easy: 成功率 < 80%
    }
    
    Easy --> Medium: 成功率 ≥ 80% & Epoch ≥ 50
    
    state Medium {
        [*] --> Training_Medium
        Training_Medium --> Evaluation_Medium
        Evaluation_Medium --> Check_Medium
        Check_Medium --> Training_Medium: 成功率 < 80%
    }
    
    Medium --> Hard: 成功率 ≥ 80% & Epoch ≥ 100
    
    state Hard {
        [*] --> Training_Hard
        Training_Hard --> Evaluation_Hard
        Evaluation_Hard --> Check_Hard
        Check_Hard --> Training_Hard: 成功率 < 90%
    }
    
    Hard --> [*]: 成功率 ≥ 90%
    
    note right of Easy
        简单阶段
        - 障碍物密度: 10%
        - 噪声标准差: 0.0
        - 走廊宽度: 6m
    end note
    
    note right of Medium
        中等阶段
        - 障碍物密度: 30%
        - 噪声标准差: 0.1
        - 走廊宽度: 4m
    end note
    
    note right of Hard
        困难阶段
        - 障碍物密度: 50%
        - 噪声标准差: 0.2
        - 走廊宽度: 2m
    end note
```

## 6. NRMP 参数优化架构

```mermaid
graph TB
    subgraph "NRMP 参数空间"
        INPUT[环境状态 + 障碍物点云]

        subgraph "固定模块 Fixed Modules"
            DUNE[DUNE Layer<br/>FlexiblePDHGFront<br/>离线监督学习]
            CVXPY[CvxpyLayer<br/>凸优化求解器]
        end

        subgraph "可学习参数 Learnable Parameters"
            P1[q_s: 状态权重]
            P2[p_u: 控制权重]
            P3[eta: 避障权重]
            P4[d_max: 最大距离]
            P5[d_min: 最小距离]
            P6[ro_obs: 障碍惩罚]
            P7[bk: 后退惩罚]
        end

        OUTPUT[优化轨迹<br/>控制序列]
    end

    subgraph "优化策略 Optimization Strategy"
        INIT[初始参数<br/>从配置文件]
        LOSS[多目标损失<br/>5 个损失项]
        GRAD[梯度计算<br/>反向传播]
        ADAM[Adam 优化器<br/>lr=5e-3]
        CONSTRAINT[参数约束<br/>投影到可行域]
    end

    INPUT --> DUNE
    DUNE -.无梯度.-> CVXPY

    P1 --> CVXPY
    P2 --> CVXPY
    P3 --> CVXPY
    P4 --> CVXPY
    P5 --> CVXPY
    P6 --> CVXPY
    P7 --> CVXPY

    CVXPY --> OUTPUT
    OUTPUT --> LOSS
    LOSS --> GRAD
    GRAD -.梯度流.-> P1
    GRAD -.梯度流.-> P2
    GRAD -.梯度流.-> P3
    GRAD -.梯度流.-> P4
    GRAD -.梯度流.-> P5
    GRAD -.梯度流.-> P6
    GRAD -.梯度流.-> P7

    INIT --> P1
    INIT --> P2
    INIT --> P3
    INIT --> P4
    INIT --> P5
    INIT --> P6
    INIT --> P7

    ADAM --> CONSTRAINT
    CONSTRAINT -.更新.-> P1
    CONSTRAINT -.更新.-> P2
    CONSTRAINT -.更新.-> P3
    CONSTRAINT -.更新.-> P4
    CONSTRAINT -.更新.-> P5
    CONSTRAINT -.更新.-> P6
    CONSTRAINT -.更新.-> P7

    style DUNE fill:#D3D3D3
    style CVXPY fill:#D3D3D3
    style P1 fill:#FFE4B5
    style P2 fill:#FFE4B5
    style P3 fill:#FFE4B5
    style P4 fill:#FFE4B5
    style P5 fill:#FFE4B5
    style P6 fill:#FFE4B5
    style P7 fill:#FFE4B5
```

## 7. 性能监控与可视化

```mermaid
graph LR
    subgraph "数据收集 Data Collection"
        TRAIN[训练过程]
        METRICS[性能指标]
        PARAMS[参数值]
        LOSSES[损失值]
    end
    
    subgraph "实时监控 Real-time Monitoring"
        LOGGER[日志记录器]
        TENSORBOARD[TensorBoard]
        CONSOLE[控制台输出]
    end
    
    subgraph "离线分析 Offline Analysis"
        PLOT_LOSS[损失曲线图]
        PLOT_PARAMS[参数演化图]
        PLOT_SUCCESS[成功率图]
        PLOT_COMPARE[方法对比图]
    end
    
    subgraph "报告生成 Report Generation"
        JSON_REPORT[JSON 报告]
        MD_REPORT[Markdown 报告]
        PDF_REPORT[PDF 报告]
    end
    
    TRAIN --> METRICS
    TRAIN --> PARAMS
    TRAIN --> LOSSES
    
    METRICS --> LOGGER
    PARAMS --> LOGGER
    LOSSES --> LOGGER
    
    LOGGER --> TENSORBOARD
    LOGGER --> CONSOLE
    LOGGER --> JSON_REPORT
    
    JSON_REPORT --> PLOT_LOSS
    JSON_REPORT --> PLOT_PARAMS
    JSON_REPORT --> PLOT_SUCCESS
    JSON_REPORT --> PLOT_COMPARE
    
    PLOT_LOSS --> MD_REPORT
    PLOT_PARAMS --> MD_REPORT
    PLOT_SUCCESS --> MD_REPORT
    PLOT_COMPARE --> MD_REPORT
    
    MD_REPORT --> PDF_REPORT
    
    style LOGGER fill:#87CEEB
    style TENSORBOARD fill:#98FB98
    style JSON_REPORT fill:#FFB6C1
```

## 8. 对比：传统 NeuPAN vs Adaptive NeuPAN

```mermaid
graph TB
    subgraph "传统 NeuPAN Traditional"
        T_CONFIG[手动配置参数]
        T_FIXED[固定参数值]
        T_PLAN[规划执行]
        T_RESULT[规划结果]
        T_MANUAL[人工评估]
        T_ADJUST[手动调整]
        
        T_CONFIG --> T_FIXED
        T_FIXED --> T_PLAN
        T_PLAN --> T_RESULT
        T_RESULT --> T_MANUAL
        T_MANUAL --> T_ADJUST
        T_ADJUST -.反馈.-> T_CONFIG
    end
    
    subgraph "自适应 NeuPAN Adaptive"
        A_INIT[初始参数]
        A_LEARN[可学习参数]
        A_PLAN[规划执行]
        A_LOSS[损失计算]
        A_BACKWARD[反向传播]
        A_UPDATE[自动更新]
        
        A_INIT --> A_LEARN
        A_LEARN --> A_PLAN
        A_PLAN --> A_LOSS
        A_LOSS --> A_BACKWARD
        A_BACKWARD --> A_UPDATE
        A_UPDATE -.实时反馈.-> A_LEARN
    end
    
    COMPARE[对比优势]
    
    T_MANUAL -.耗时长.-> COMPARE
    T_ADJUST -.需专家.-> COMPARE
    A_UPDATE -.自动化.-> COMPARE
    A_BACKWARD -.快速.-> COMPARE
    
    style T_CONFIG fill:#FFB6C1
    style A_LEARN fill:#90EE90
    style COMPARE fill:#FFD700
```

---

**说明**：
- 以上图表使用 Mermaid 语法绘制，可在支持 Mermaid 的 Markdown 渲染器中查看
- 建议使用 Typora、VS Code (Markdown Preview Enhanced) 或 GitHub 查看
- 图表颜色编码：
  - 🟦 蓝色：输入/数据层
  - 🟨 黄色：处理/计算层
  - 🟩 绿色：输出/结果层
  - 🟥 红色：关键/核心模块

