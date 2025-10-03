"""
整合版投影测试脚本

结合 test_projection_planner_step.py 和 dual_metrics.py
提供完整的对偶可行性评估报告
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

import irsim
from neupan import neupan
from neupan.evaluation.dual_metrics import DualFeasibilityMetrics


class ProjectionEvaluator:
    """投影效果评估器"""

    def __init__(self, projection: str):
        self.projection = projection
        self.metrics_calculator = None

        # 记录每一步的指标
        self.step_records = []

        # 累积统计
        self.all_mu_pre = []   # 投影前的所有μ
        self.all_mu_post = []  # 投影后的所有μ

    def set_dune_layer(self, dune_layer):
        """设置DUNE层,初始化metrics计算器"""
        if dune_layer is not None and self.metrics_calculator is None:
            self.metrics_calculator = DualFeasibilityMetrics(
                dune_layer.G,
                tolerance=1e-6
            )

    def record_step(self, step: int, dune_layer, lam_list):
        """记录单步的评估指标"""
        if dune_layer is None:
            return

        # 获取DUNE内部的监控指标
        pre_v = getattr(dune_layer, "dual_norm_violation_rate", None)
        pre_p95 = getattr(dune_layer, "dual_norm_p95", None)
        pre_exc = getattr(dune_layer, "dual_norm_max_excess_pre", None)
        post_exc = getattr(dune_layer, "dual_norm_max_excess_post", None)

        # 通过λ计算post_max (间接验证)
        post_max = float("nan")
        if lam_list:
            vals = []
            for lam in lam_list:
                if lam is None or lam.numel() == 0:
                    continue
                vals.append(torch.norm(lam, dim=0).max().item())
            if vals:
                post_max = max(vals)

        # 记录
        record = {
            'step': step,
            'pre_violation_rate': pre_v,
            'pre_p95': pre_p95,
            'pre_excess': pre_exc,
            'post_excess': post_exc,
            'post_max_via_lam': post_max,
        }
        self.step_records.append(record)

        # 打印实时信息（兼容 None/NaN）
        pv = "None" if pre_v is None else f"{pre_v:.4f}"
        p95 = "None" if pre_p95 is None else f"{pre_p95:.6f}"
        prex = "None" if pre_exc is None else f"{pre_exc:.6f}"
        postx = "None" if post_exc is None else f"{post_exc:.6f}"
        pm = "nan" if np.isnan(post_max) else f"{post_max:.8f}"
        print(
            f"  step {step:03d} | pre-viol {pv} | pre-p95 {p95} "
            f"| post-max {pm} | pre-exc {prex} | post-exc {postx}"
        )

    def generate_report(self) -> dict:
        """生成最终评估报告"""
        if not self.step_records:
            return {}

        # 计算平均指标
        pre_violations = [r['pre_violation_rate'] for r in self.step_records if r['pre_violation_rate'] is not None]
        pre_p95s = [r['pre_p95'] for r in self.step_records if r['pre_p95'] is not None]
        pre_excesses = [r['pre_excess'] for r in self.step_records if r['pre_excess'] is not None]
        post_excesses = [r['post_excess'] for r in self.step_records if r['post_excess'] is not None]
        post_maxs = [r['post_max_via_lam'] for r in self.step_records if not np.isnan(r['post_max_via_lam'])]

        report = {
            'projection': self.projection,
            'total_steps': len(self.step_records),
            'avg_pre_violation_rate': np.mean(pre_violations) if pre_violations else 0.0,
            'avg_pre_p95': np.mean(pre_p95s) if pre_p95s else 0.0,
            'avg_pre_excess': np.mean(pre_excesses) if pre_excesses else 0.0,
            'avg_post_excess': np.mean(post_excesses) if post_excesses else 0.0,
            'avg_post_max': np.mean(post_maxs) if post_maxs else 0.0,
            'max_post_max': np.max(post_maxs) if post_maxs else 0.0,
            'std_pre_violation_rate': np.std(pre_violations) if pre_violations else 0.0,
            'std_post_max': np.std(post_maxs) if post_maxs else 0.0,
        }

        return report

    def print_summary(self):
        """打印汇总报告"""
        report = self.generate_report()

        print(f"\n{'='*70}")
        print(f"Projection Evaluation Summary: {self.projection}")
        print(f"{'='*70}")

        print(f"\n总步数: {report['total_steps']}")

        print(f"\n投影前 (Pre-projection):")
        print(f"  平均违反率: {report['avg_pre_violation_rate']:.2%} ± {report['std_pre_violation_rate']:.2%}")
        print(f"  平均P95范数: {report['avg_pre_p95']:.6f}")
        print(f"  平均超出幅度: {report['avg_pre_excess']:.6f}")

        print(f"\n投影后 (Post-projection):")
        print(f"  平均最大范数: {report['avg_post_max']:.8f} ± {report['std_post_max']:.8f}")
        print(f"  最大范数峰值: {report['max_post_max']:.8f}")
        print(f"  平均超出幅度: {report['avg_post_excess']:.6f}")

        # 判断投影效果
        tolerance = 1e-6
        if self.projection == 'hard':
            if report['max_post_max'] <= 1.0 + tolerance:
                print(f"\n✅ 硬投影工作正常!")
                print(f"   最大违反 ({report['max_post_max'] - 1.0:.2e}) 在浮点数精度内")
            else:
                print(f"\n⚠️  硬投影可能存在问题")
                print(f"   最大违反 ({report['max_post_max'] - 1.0:.2e}) 超过容差")

        # 改进幅度
        if report['avg_pre_excess'] > 0:
            improvement = (report['avg_pre_excess'] - report['avg_post_excess']) / report['avg_pre_excess'] * 100
            print(f"\n改进幅度:")
            print(f"  超出幅度降低: {improvement:.2f}%")

        print(f"\n{'='*70}\n")

        return report


def simulate(
    example_name: str,
    kinematics: str,
    projection: str,
    save_animation: bool,
    ani_name: str,
    full: bool,
    no_display: bool,
    point_vel: bool,
    max_steps: int,
    reverse: bool,
    save_results: bool = True,
):
    """运行仿真并评估投影效果"""

    env_file = f"example/{example_name}/{kinematics}/env.yaml"
    planner_file = f"example/{example_name}/{kinematics}/planner.yaml"

    # no_display=True 表示关闭可视化；irsim.make 接收 display=True/False
    env = irsim.make(env_file, save_ani=save_animation, full=full, display=not no_display)

    train_overrides = dict(
        projection=projection,
        monitor_dual_norm=True,
        unroll_J=0,
        se2_embed=False,
    )

    planner = neupan.init_from_yaml(planner_file, train=train_overrides)

    print(f"\n{'='*70}")
    print(f"Running simulation: projection={projection}")
    print(f"{'='*70}\n")

    # 创建评估器
    evaluator = ProjectionEvaluator(projection)

    # 运行状态统计
    arrived = False
    stopped = False
    steps_executed = 0
    collided = None

    for step in range(max_steps):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()

        if point_vel:
            points, point_velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
        else:
            points = planner.scan_to_point(robot_state, lidar_scan)
            point_velocities = None

        action, info = planner(robot_state, points, point_velocities)

        # 获取DUNE层
        dune_layer = getattr(planner.pan, "dune_layer", None)
        if dune_layer is not None:
            # 初始化metrics计算器
            evaluator.set_dune_layer(dune_layer)

            # 获取λ列表
            current_values = getattr(planner.pan, "current_nom_values", None)
            lam_list = current_values[3] if current_values else []

            # 记录这一步的指标
            evaluator.record_step(step, dune_layer, lam_list)

        # 更新状态
        steps_executed = step + 1
        if info.get("stop"):
            print("NeuPAN stops because of minimum distance")
            stopped = True
        if info.get("arrive"):
            print("NeuPAN arrives at the target")
            arrived = True
            break

        env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(planner.opt_trajectory, "r", refresh=True)
        env.draw_trajectory(planner.ref_trajectory, "b", refresh=True)
        env.render()

        env.step(action)

        if env.done():
            break

        if step == 0:
            if reverse:
                for j in range(len(planner.initial_path)):
                    planner.initial_path[j][-1, 0] = -1
                    planner.initial_path[j][-2, 0] = planner.initial_path[j][-2, 0] + 3.14

                env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=True)
            else:
                env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=False)

            env.render()

    env.end(3, ani_name=ani_name)

    # 生成并打印报告
    report = evaluator.print_summary()

    # 按用户规则：未到达终点即视为碰撞
    collided = (not arrived)
    success = arrived  # 成功 = 到达终点

    report.update({
        'steps_executed': int(steps_executed),
        'arrived': bool(arrived),
        'stopped': bool(stopped),
        'collided': bool(collided),
        'success': bool(success),
    })


    # 保存结果
    if save_results:
        results_dir = Path("test/results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"{example_name}_{kinematics}_{projection}_{timestamp}.json"

        with open(result_file, 'w') as f:
            json.dump({
                'config': {
                    'example': example_name,
                    'kinematics': kinematics,
                    'projection': projection,
                    'max_steps': max_steps,
                },
                'summary': report,
                'step_records': evaluator.step_records,
            }, f, indent=2)

        print(f"Results saved to: {result_file}")

    return report


def compare_projections(reports: dict):
    """对比不同投影方法的效果"""

    print(f"\n{'='*70}")
    print("Projection Methods Comparison")
    print(f"{'='*70}\n")

    # 表头
    header = f"{'Metric':<30} {'hard':<15} {'none':<15} {'Improvement':<15}"
    print(header)
    print("=" * len(header))

    # 对比指标
    metrics_to_compare = [
        ('avg_pre_violation_rate', '平均违反率(投影前)', '%'),
        ('avg_post_max', '平均最大范数(投影后)', ''),
        ('max_post_max', '最大范数峰值', ''),
        ('avg_post_excess', '平均超出幅度(投影后)', ''),
    ]

    for key, name, unit in metrics_to_compare:
        hard_val = reports.get('hard', {}).get(key, 0)
        none_val = reports.get('none', {}).get(key, 0)

        if unit == '%':
            hard_str = f"{hard_val:.2%}"
            none_str = f"{none_val:.2%}"
        else:
            hard_str = f"{hard_val:.6f}"
            none_str = f"{none_val:.6f}"

        # 计算改进
        if none_val > 0:
            improvement = (none_val - hard_val) / none_val * 100
            imp_str = f"{improvement:.2f}%"
        else:
            imp_str = "N/A"

        print(f"{name:<30} {hard_str:<15} {none_str:<15} {imp_str:<15}")

    print("=" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser(description="投影效果评估 (整合版)")
    parser.add_argument("-e", "--example", type=str, default="polygon_robot",
                       help="example name")
    parser.add_argument("-d", "--kinematics", type=str, default="diff",
                       help="acker, diff")
    parser.add_argument("-a", "--save_animation", action="store_true",
                       help="save animation")
    parser.add_argument("-f", "--full", action="store_true",
                       help="full screen")
    parser.add_argument("-n", "--no_display", action="store_true",
                       help="no display (disable rendering)")
    parser.add_argument("-v", "--point_vel", action='store_true',
                       help="point vel")
    parser.add_argument("-m", "--max_steps", type=int, default=1000,
                       help="max steps")
    parser.add_argument("--projection", type=str, default="hard",
                       choices=["hard", "none", "learned"],
                       help="projection mode")
    parser.add_argument("--compare", action="store_true",
                       help="run hard and none sequentially and compare")
    parser.add_argument("--save_results", action="store_true",
                       help="save results to JSON file")
    args = parser.parse_args()

    projections = [args.projection]
    if args.compare:
        projections = ["hard", "none"]

    ani_base = f"{args.example}_{args.kinematics}"

    reports = {}

    for proj in projections:
        report = simulate(
            args.example,
            args.kinematics,
            proj,
            args.save_animation,
            f"{ani_base}_{proj}",
            args.full,
            args.no_display,
            args.point_vel,
            args.max_steps,
            reverse=(args.example == "reverse" and args.kinematics == "diff"),
            save_results=args.save_results,
        )
        reports[proj] = report

    # 如果是对比模式,打印对比表格
    if args.compare and len(reports) > 1:
        compare_projections(reports)


if __name__ == "__main__":
    main()

