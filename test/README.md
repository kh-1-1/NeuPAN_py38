用于验证DUNE硬投影与监控的测试脚本

脚本功能
- 验证硬投影约束条件（μ ≥ 0 且 ||G^T μ||_2 ≤ 1）
- 通过lam报告投影前违规率/P95值和投影后最大范数
- 提供无需依赖的合成测试，以及可选参数与`example/run_exp.py`完全兼容的规划器/环境步骤测试

脚本列表
- test_projection_ab_synthetic.py
  - 无需irsim即可运行投影模式'hard'与'none'的A/B对比测试
  - 使用合成机器人+随机障碍点直接调用DUNE层

- test_projection_planner_step.py（可选）
  - 重构`example/run_exp.py`循环流程，保持命令行参数（`-e`、`-d`、`-a`、`-f`、`-n`、`-v`、`-m`）完全一致
  - 新增`--projection`（默认hard）和`--compare`（顺序执行hard/none模式）参数，实时输出每步对偶范数统计

快速使用
1) 合成测试（无需外部依赖）：
   python -m test.test_projection_ab_synthetic

2) 结合规划器/环境示例（需要irsim）：
   python -m test.test_projection_planner_step -e corridor -d acker --projection hard --compare --show
   （参数与`example/run_exp.py`完全一致，可酌情省略`--compare`或`--show`）