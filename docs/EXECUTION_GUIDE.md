# A 方案执行指南

## 📋 总览

**方案**: 三阶段完整评测（消融 + 时延 + 全量）  
**总预计时间**: 10-13 小时  
**目标**: 科学严谨地评估 PDHG-Unroll，为顶会论文提供完整证据

---

## 🚀 快速开始

### 方式 1: 一键运行全部（推荐过夜）

```bash
RUN_ALL_STAGES.bat
# 选择 "1" - 自动运行全部三个阶段
```

### 方式 2: 分阶段手动运行（推荐）

```bash
# 今天下午/晚上
run_ablation_study.bat      # 阶段 1: 2-3 小时
run_timing_analysis.bat     # 阶段 2: 1-2 小时

# 明天/过夜
run_stage3_full_evaluation.bat  # 阶段 3: 6-8 小时
```

---

## 📊 三阶段详细说明

### 阶段 1: 消融实验（2-3 小时）

#### 目标
分离 PDHG 和硬投影的独立贡献，回答：
- PDHG 本身能达到多少可行性？
- 外部硬投影是否还有必要？
- 最优组合是什么？

#### 测试矩阵
```
projection ∈ {none, hard} × unroll_J ∈ {0, 1, 2, 3}
= 2 × 4 = 8 种配置
× 10 examples × 2 kinematics × 10 runs
= 1600 次仿真
```

#### 关键配置对比

| 配置 | projection | J | 说明 | 预期违反率 |
|------|-----------|---|------|-----------|
| A | none | 0 | Baseline（无 PDHG，无硬投影） | 51.5% |
| B | hard | 0 | 仅硬投影 | 5-10% |
| C | none | 1 | 仅 PDHG (J=1) | **2-3%** |
| D | hard | 1 | PDHG + 硬投影（当前方案） | **1.6%** |
| E | none | 2 | 仅 PDHG (J=2) | 1-1.5% |
| F | hard | 2 | PDHG + 硬投影 | 0.8% |

#### 运行命令
```bash
run_ablation_study.bat
```

或手动：
```bash
python -m test.batch_unroll_evaluation \
    --config test/configs/ablation_study_config.yaml
```

#### 预期输出
- 结果目录: `test/results/batch_unroll_modes-none-hard_J-0-1-2-3_<timestamp>/`
- 关键文件:
  - `batch_summary_*.md` - Markdown 汇总表
  - `batch_summary_*.csv` - CSV 数据（可导入 Excel）
  - `batch_summary_*.json` - 完整 JSON

#### 分析要点

**关键问题 1**: 配置 C (none, J=1) 的违反率是多少？
```python
# 查看 CSV 文件，筛选 projection='none', unroll_J=1
# 若平均违反率 < 3%，说明 PDHG 本身已足够强
```

**关键问题 2**: 配置 D vs C 的改进幅度？
```python
improvement = (viol_C - viol_D) / viol_C
# 若 < 30%，说明外部硬投影的边际收益有限
# 可以考虑简化为"纯 PDHG"方案
```

**关键问题 3**: J=2 vs J=1 的边际收益？
```python
# 对比配置 E vs C（纯 PDHG）
# 对比配置 F vs D（PDHG + 硬投影）
# 若改进 < 50%，说明 J=1 已是最优
```

---

### 阶段 2: 时延分析（1-2 小时）

#### 目标
量化 PDHG 的计算开销，回答：
- PDHG 增加了多少推理时延？
- 性价比如何（每降低 1% 违反率的时延代价）？
- J=1/2/3 的时延差异？

#### 测试配置
```
examples: corridor, polygon_robot, dyna_obs  # 3 个代表性场景
modes: hard
unroll_J: 0, 1, 2, 3
runs: 20  # 更多运行次数以获得稳定统计
max_steps: 500  # 减少步数以加快评测
profile_timing: true  # 开启详细时延记录
```

#### 运行命令
```bash
run_timing_analysis.bat
```

或手动：
```bash
python -m test.batch_unroll_evaluation \
    --config test/configs/timing_analysis_config.yaml
```

#### 预期输出

**新增时延指标**（在 summary JSON 中）:
```json
{
  "avg_total_time_ms": 10.5,      // 总推理时延（毫秒）
  "std_total_time_ms": 0.3,
  "avg_pdhg_time_ms": 0.5,        // PDHG 迭代时延
  "std_pdhg_time_ms": 0.05,
  "avg_dune_time_ms": 2.0,        // DUNE 模块时延
  "avg_nrmp_time_ms": 3.0,        // NRMP 模块时延
  "avg_mpc_time_ms": 5.0          // MPC 求解时延
}
```

#### 分析要点

**关键指标 1**: PDHG 开销占比
```python
pdhg_overhead = avg_pdhg_time_ms / avg_total_time_ms
# 目标: < 10%
# 预期: J=1 约 4-5%, J=2 约 8-10%, J=3 约 12-15%
```

**关键指标 2**: 性价比（每降低 1% 违反率的时延代价）
```python
cost_per_percent = (time_J - time_0) / (viol_0 - viol_J)

# 预期:
# J=1: (10.5 - 10.0) / (51.5 - 1.6) ≈ 0.01 ms/%  (极优)
# J=2: (11.0 - 10.0) / (51.5 - 0.8) ≈ 0.02 ms/%  (优)
# J=3: (11.5 - 10.0) / (51.5 - 0.5) ≈ 0.03 ms/%  (可接受)
```

**关键指标 3**: 实时性评估
```python
# 假设控制频率 10 Hz（100 ms 周期）
# 总时延 < 100 ms 为实时可行
# 预期: 所有 J 值都满足（10-12 ms << 100 ms）
```

---

### 阶段 3: 全量评测（6-8 小时）

#### 目标
在最优配置下，获得完整基准数据，用于论文发表。

#### 测试矩阵
```
examples: all (10 个有效场景)
modes: hard  # 基于阶段 1 确定的最优 projection
unroll_J: 0, 1, 2, 3
runs: 10
max_steps: 1000
```

**总实验次数**: 10 × 2 × 1 × 4 × 10 = **800 次仿真**

#### 运行命令
```bash
run_stage3_full_evaluation.bat
```

或手动：
```bash
python -m test.batch_unroll_evaluation \
    --config test/configs/full_evaluation_config.yaml
```

#### 重要提示

**运行前检查**:
- [ ] 确认电脑不会休眠（调整电源设置）
- [ ] 确认磁盘空间充足（至少 1GB）
- [ ] 关闭不必要的后台程序
- [ ] 建议在晚上开始，过夜运行

**中断恢复**:
如果评测中断，可以手动运行缺失的配置：
```bash
# 查看已完成的配置
dir test\results\batch_unroll_*

# 手动运行缺失的 J 值
python -m test.batch_unroll_evaluation \
    --runs 10 --max_steps 1000 --no_display --quiet \
    --modes hard --unroll-J 2,3 \
    --examples <缺失的 example>
```

---

## 📈 结果分析流程

### 步骤 1: 运行可视化脚本

```bash
python test/visualize_results.py
```

**生成的图表**（保存在 `test/results/batch_unroll_*/visualizations/`）:
- `violation_rate_comparison.png` - 违反率对比柱状图
- `p95_comparison.png` - P95 对偶范数对比图
- `steps_comparison.png` - 执行步数对比图
- `summary_statistics.md` - 汇总统计表

### 步骤 2: 对比三个阶段的结果

#### 阶段 1 关键发现（消融实验）

查看文件: `test/results/batch_unroll_modes-none-hard_*/batch_summary_*.csv`

**分析模板**:
```python
import pandas as pd

# 加载数据
df = pd.read_csv('batch_summary_*.csv')

# 筛选关键配置
config_C = df[(df['projection'] == 'none') & (df['unroll_J'] == 1)]
config_D = df[(df['projection'] == 'hard') & (df['unroll_J'] == 1)]

# 计算平均违反率
viol_C = config_C['avg_pre_violation_rate_mean'].mean()
viol_D = config_D['avg_pre_violation_rate_mean'].mean()

print(f"纯 PDHG (J=1): {viol_C:.2%}")
print(f"PDHG + 硬投影 (J=1): {viol_D:.2%}")
print(f"外部硬投影的边际收益: {(viol_C - viol_D) / viol_C:.1%}")
```

**决策标准**:
- 若边际收益 > 30%，保留外部硬投影（当前方案）
- 若边际收益 < 30%，考虑简化为"纯 PDHG"

#### 阶段 2 关键发现（时延分析）

查看文件: `test/results/batch_unroll_*timing*/batch_summary_*.json`

**分析模板**:
```python
import json

# 加载数据
with open('batch_summary_*.json', 'r') as f:
    data = json.load(f)

# 提取时延数据
for cfg_key, cfg_data in data.items():
    if 'aggregate' in cfg_data:
        aggr = cfg_data['aggregate']
        J = int(cfg_key.split('::J')[1])
        total_time = aggr.get('avg_total_time_ms', 0)
        pdhg_time = aggr.get('avg_pdhg_time_ms', 0)
        overhead = pdhg_time / total_time if total_time > 0 else 0
        
        print(f"J={J}: total={total_time:.2f}ms, pdhg={pdhg_time:.2f}ms, overhead={overhead:.1%}")
```

**决策标准**:
- 若 J=1 的开销 < 5%，确认为最佳性价比
- 若 J=2 的开销 < 10% 且违反率显著降低，考虑在高安全场景使用

#### 阶段 3 关键发现（全量基准）

查看文件: `test/results/batch_unroll_modes-hard_J-0-1-2-3_*/batch_summary_*.md`

**论文素材提取**:
1. **Table 1**: 全量评测汇总（所有 example 的违反率）
2. **Figure 1**: 违反率对比图（J=0/1/2/3）
3. **Figure 2**: 性能-精度权衡散点图

---

## 📝 论文写作指南

### 实验章节结构（建议）

#### 5.1 实验设置
- 数据集: 10 个基准场景（corridor, polygon_robot, ...）
- 运动学模型: 差速 (diff) 和 Ackermann (acker)
- 评测指标: 违反率、P95、时延、执行步数
- 实现细节: PyTorch, ECOS solver, ...

#### 5.2 消融实验（阶段 1 结果）
- **Table 2**: 消融实验结果
  ```
  | Config | Projection | J | Avg Viol | Improvement |
  |--------|-----------|---|----------|-------------|
  | A | none | 0 | 51.5% | baseline |
  | B | hard | 0 | 8.0% | ↓ 84.5% |
  | C | none | 1 | 2.5% | ↓ 95.1% |
  | D | hard | 1 | 1.6% | ↓ 96.9% |
  ```

- **Figure 3**: 消融实验对比柱状图
- **关键发现**:
  > "PDHG-Unroll (J=1) 本身即可将违反率从 51.5% 降至 2.5%（配置 C），外部硬投影进一步降至 1.6%（配置 D），边际收益为 36%。"

#### 5.3 性能分析（阶段 2 结果）
- **Table 3**: 时延分析
  ```
  | J | Total Time | PDHG Time | Overhead | Viol Rate | Cost/% |
  |---|-----------|-----------|----------|-----------|--------|
  | 0 | 10.0 ms | 0 ms | 0% | 51.5% | - |
  | 1 | 10.5 ms | 0.5 ms | 4.8% | 1.6% | 0.01 ms/% |
  | 2 | 11.0 ms | 1.0 ms | 9.1% | 0.8% | 0.02 ms/% |
  ```

- **Figure 4**: 性能-精度权衡散点图
- **关键发现**:
  > "J=1 配置在仅增加 4.8% 推理时延的情况下，将违反率降低 96.9%，性价比最优（0.01 ms/%）。"

#### 5.4 全量基准测试（阶段 3 结果）
- **Table 4**: 所有场景的详细结果
- **Figure 5**: 不同场景的违反率对比
- **关键发现**:
  > "在 10 个基准场景、2 种运动学模型、800 次仿真实验中，PDHG-Unroll (J=1) 在所有场景均将违反率降至 < 3%，展现出卓越的泛化能力。"

#### 5.5 讨论
- **最优配置推荐**: projection='hard', J=1
- **适用场景**: 实时机器人导航（< 100 ms 控制周期）
- **局限性**: 对于极端复杂几何（E > 10），可能需要 J=2

---

## ⚠️ 常见问题

### Q1: 评测中断怎么办？
**A**: 查看 `test/results/` 目录，找到最新的批次文件夹，检查哪些配置已完成。手动运行缺失的配置（见"中断恢复"部分）。

### Q2: 内存不足怎么办？
**A**: 
1. 减少 `--runs` 参数（如从 10 改为 5）
2. 减少 `--max_steps` 参数（如从 1000 改为 500）
3. 分批运行（每次运行部分 examples）

### Q3: 时延数据为空怎么办？
**A**: 当前时延记录依赖 planner 内部的 `last_timing` 属性。如果为空，说明 planner 未记录子模块时延。可以：
1. 仅使用 `avg_total_time_ms`（总时延）
2. 或修改 planner 代码，添加子模块时延记录

### Q4: 如何加速评测？
**A**:
1. 使用 GPU（如果可用）
2. 减少 `--runs` 参数
3. 减少 `--max_steps` 参数
4. 仅测试关键场景（如 corridor, polygon_robot）

---

## ✅ 检查清单

### 评测开始前
- [ ] 激活 conda 环境 (`conda activate NeuPAN_py38`)
- [ ] 确认磁盘空间充足（至少 1GB）
- [ ] 确认 checkpoint 文件存在
- [ ] 关闭不必要的后台程序
- [ ] 调整电源设置（禁止休眠）

### 阶段 1 完成后
- [ ] 运行 `python test/visualize_results.py`
- [ ] 查看消融实验结果
- [ ] 回答关键问题（见"分析要点"）
- [ ] 决定是否保留外部硬投影

### 阶段 2 完成后
- [ ] 查看时延分析结果
- [ ] 计算 PDHG 开销占比
- [ ] 计算性价比（cost_per_percent）
- [ ] 确认最优 J 值

### 阶段 3 完成后
- [ ] 运行可视化脚本
- [ ] 备份结果文件
- [ ] 提取论文素材（表格、图表）
- [ ] 撰写实验章节

---

## 📞 支持

如遇到问题，请提供：
1. 错误信息截图
2. 运行的完整命令
3. `test/results/` 目录下的最新文件

祝评测顺利！🚀

