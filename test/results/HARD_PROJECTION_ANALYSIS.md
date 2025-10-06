# 硬投影与 PDHG-Unroll 的交互分析

## 问题发现

### 当前实现的"双重硬投影"

#### 1. PDHG 内部硬投影（`pdhg_unroll.py`）

每次 PDHG 迭代都执行：
```python
# 第 130-137 行
mu = mu + tau * (a - G @ y)
mu = mu.clamp(min=0.0)  # 非负投影

v = G.t() @ mu
v_norm = torch.norm(v, dim=0, keepdim=True)
v_norm_clamped = torch.clamp(v_norm, min=1.0)
mu = mu / v_norm_clamped  # ||G^T μ|| ≤ 1 投影
```

#### 2. DUNE 外部硬投影（`dune.py`）

在 PDHG 之后，如果 `projection='hard'`：
```python
# 第 159-170 行
if self.projection in ('hard', 'learned'):
    total_mu.clamp_(min=0.0)
    v = (self.G.T @ total_mu)
    v_norm = torch.norm(v, dim=0, keepdim=True)
    mask = (v_norm > 1.0).float()
    denom = v_norm.clamp(min=1.0)
    scale = mask / denom + (1.0 - mask)
    total_mu = total_mu * scale
```

### 问题分析

#### 问题 1: PDHG 改进被"覆盖"

**现象**：
- PDHG 迭代 J 步后，μ 已经接近可行（violation_rate ≈ 1.6%）
- 外部硬投影再次修正，可能"抹平"了 PDHG 的精细调整

**影响**：
- 无法真实评估 PDHG 的净效果
- J=1/2/3 的差异可能被掩盖（因为最终都被硬投影"拉回"）

#### 问题 2: 性价比评估不准确

**缺失的关键信息**：
1. **PDHG 的时延开销**：J=1/2/3 各增加多少毫秒？
2. **PDHG 的净改进**：去掉外部硬投影后，PDHG 能达到多少可行性？
3. **硬投影的必要性**：PDHG 后是否还需要硬投影？

**当前结论的局限性**：
- "J=1 是最佳性价比"基于 `projection='hard'` 的结果
- 但如果 PDHG 本身已足够可行（`projection='none'`），可能 J=0 就够了
- 或者 J=2 在纯 PDHG 模式下才能体现优势

---

## 解决方案

### 方案 1: 消融实验（推荐）

**目标**：分离 PDHG 和硬投影的独立贡献

**实验设计**：

| 配置 | projection | unroll_J | 说明 |
|------|-----------|----------|------|
| A | `none` | 0 | Baseline（无 PDHG，无硬投影） |
| B | `hard` | 0 | 仅硬投影 |
| C | `none` | 1 | 仅 PDHG (J=1) |
| D | `hard` | 1 | PDHG + 硬投影（当前方案） |
| E | `none` | 2 | 仅 PDHG (J=2) |
| F | `hard` | 2 | PDHG + 硬投影 |

**预期发现**：

1. **硬投影的贡献**：
   - 对比 A vs B：硬投影能降低多少违反率？
   - 预期：51.5% → 5-10%（硬投影有效，但不如 PDHG）

2. **PDHG 的净效果**：
   - 对比 A vs C：纯 PDHG (J=1) 的改进
   - 预期：51.5% → 2-3%（PDHG 本身已很强）

3. **双重投影的必要性**：
   - 对比 C vs D：外部硬投影是否还有额外收益？
   - 预期：2-3% → 1.6%（边际收益小，可能不必要）

4. **最优 J 值**：
   - 对比 C vs E：纯 PDHG 模式下，J=2 是否显著优于 J=1？
   - 预期：若 C 已 < 2%，E 的改进有限（< 0.5%）

**运行命令**：
```bash
# 使用消融实验配置
python -m test.batch_unroll_evaluation \
    --config test/configs/ablation_study_config.yaml
```

---

### 方案 2: 时延分析（必须）

**目标**：量化 PDHG 的计算开销

**关键指标**：

| Metric | 定义 | 目标 |
|--------|------|------|
| `avg_total_time_ms` | 总推理时延 | 基准 |
| `avg_pdhg_time_ms` | PDHG 迭代时延 | 量化 J 的开销 |
| `pdhg_overhead` | `pdhg_time / total_time` | < 10% 为优 |

**预期结果**（基于理论分析）：

| J | PDHG 时延 (ms) | 总时延 (ms) | 开销占比 | 违反率 |
|---|---------------|------------|---------|--------|
| 0 | 0 | 10 | 0% | 51.5% |
| 1 | 0.5 | 10.5 | 4.8% | 1.6% |
| 2 | 1.0 | 11.0 | 9.1% | 0.8% |
| 3 | 1.5 | 11.5 | 13.0% | 0.5% |

**性价比计算**：
```python
# 每降低 1% 违反率的时延代价
cost_per_percent = (time_J - time_0) / (viol_0 - viol_J)

# 预期：
# J=1: (10.5 - 10) / (51.5 - 1.6) ≈ 0.01 ms/% （极优）
# J=2: (11.0 - 10) / (51.5 - 0.8) ≈ 0.02 ms/% （优）
# J=3: (11.5 - 10) / (51.5 - 0.5) ≈ 0.03 ms/% （可接受）
```

**运行命令**：
```bash
# 使用时延分析配置
python -m test.batch_unroll_evaluation \
    --config test/configs/timing_analysis_config.yaml
```

---

## 修正后的评测方案

### 阶段 1: 消融实验（2-3 小时）

**目标**：理解 PDHG 和硬投影的独立贡献

```bash
# 测试 projection ∈ {none, hard} × unroll_J ∈ {0,1,2}
python -m test.batch_unroll_evaluation \
    --config test/configs/ablation_study_config.yaml
```

**决策点**：
- 若 `projection='none', J=1` 的违反率 < 3%，说明 PDHG 本身已足够
  - 可以考虑去掉外部硬投影（简化流程）
- 若 `projection='hard', J=1` 显著优于 `projection='none', J=1`
  - 保留双重投影（当前方案）

### 阶段 2: 时延分析（1-2 小时）

**目标**：量化性能-精度权衡

```bash
python -m test.batch_unroll_evaluation \
    --config test/configs/timing_analysis_config.yaml
```

**决策点**：
- 若 J=1 的 PDHG 开销 < 5%，且违反率降低 > 95%
  - **确认 J=1 为最佳性价比**
- 若 J=2 的开销 < 10%，且违反率进一步降低 > 50%
  - 考虑在高安全场景使用 J=2

### 阶段 3: 全量评测（6-8 小时）

**目标**：在最优配置下，获得完整基准

```bash
# 使用阶段 1/2 确定的最优配置
python -m test.batch_unroll_evaluation \
    --config test/configs/full_evaluation_config.yaml
```

---

## 预期论文素材

### Figure: 消融实验对比

```
┌─────────────────────────────────────────────────────────┐
│  Ablation Study: PDHG vs Hard Projection                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Violation Rate (%)                                     │
│  60 ┤                                                   │
│  50 ┤ ████ A (none, J=0)                                │
│  40 ┤                                                   │
│  30 ┤                                                   │
│  20 ┤                                                   │
│  10 ┤ ██ B (hard, J=0)                                  │
│   5 ┤ █ C (none, J=1)                                   │
│   2 ┤ █ D (hard, J=1) ← Current                         │
│   1 ┤ █ E (none, J=2)                                   │
│   0 ┤ █ F (hard, J=2)                                   │
│     └─────────────────────────────────────────────────  │
│                                                         │
│  Key Findings:                                          │
│  • PDHG (C) alone achieves 2-3% violation rate          │
│  • Hard projection (B) alone achieves 5-10%             │
│  • Combined (D) achieves 1.6% (marginal gain)           │
│  • J=2 (E/F) offers diminishing returns                 │
└─────────────────────────────────────────────────────────┘
```

### Table: 性能-精度权衡

| Config | Projection | J | Viol Rate | PDHG Time | Total Time | Cost/% |
|--------|-----------|---|-----------|-----------|------------|--------|
| A | none | 0 | 51.5% | 0 ms | 10.0 ms | - |
| B | hard | 0 | 8.0% | 0 ms | 10.2 ms | - |
| C | none | 1 | 2.5% | 0.5 ms | 10.5 ms | 0.01 ms/% |
| **D** | **hard** | **1** | **1.6%** | **0.5 ms** | **10.5 ms** | **0.01 ms/%** |
| E | none | 2 | 1.2% | 1.0 ms | 11.0 ms | 0.77 ms/% |
| F | hard | 2 | 0.8% | 1.0 ms | 11.0 ms | 0.63 ms/% |

**结论**：
- **配置 D (hard, J=1) 是最佳性价比**：
  - 违反率降低 96.9%（51.5% → 1.6%）
  - PDHG 开销仅 4.8%（0.5 / 10.5）
  - 每降低 1% 违反率仅需 0.01 ms

- **配置 C (none, J=1) 是简化方案**：
  - 若可接受 2.5% 违反率，可去掉外部硬投影
  - 代码更简洁，性能相同

---

## 立即行动项

### 选项 1: 完整消融 + 时延分析（推荐）

**理由**：彻底理解 PDHG 的价值，为论文提供完整证据

**步骤**：
1. 运行消融实验（2-3 小时）
2. 运行时延分析（1-2 小时）
3. 分析结果，确定最优配置
4. 运行全量评测（6-8 小时）

**总时间**：10-13 小时

### 选项 2: 仅时延分析 + 全量评测（快速路线）

**理由**：假设当前方案 (hard, J=1) 已是最优，仅需量化时延

**步骤**：
1. 运行时延分析（1-2 小时）
2. 确认 J=1 的开销 < 5%
3. 运行全量评测（6-8 小时）

**总时间**：7-10 小时

---

## 我的推荐

**建议选择：选项 1（完整消融 + 时延分析）**

**理由**：
1. **科学严谨性**：消融实验是顶会论文的标配
2. **意外发现**：可能发现"纯 PDHG"已足够优，简化系统
3. **时间成本可控**：额外 3 小时换取完整理解，值得
4. **论文素材丰富**：消融图表 + 时延分析表 = 2 个 figure/table

**下一步**：
请您确认是否同意此方案，我将立即准备运行脚本和分析工具。

