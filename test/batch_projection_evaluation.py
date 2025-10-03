"""
批量硬投影评测脚本（不修改核心DUNE逻辑）

功能：
- 自动遍历 example 下的环境
- 对每个 环境×运动学(diff/acker)×投影(hard/none) 运行 N 次完整仿真
- 关闭可视化，加速运行；保存每次 JSON；容错不中断
- 生成多维汇总（控制台 + 保存为 Markdown/CSV/JSON）

用法示例：
  python -m test.batch_projection_evaluation --runs 50 --max_steps 1000 --save_results --quiet
  # 仅评测指定环境：
  python -m test.batch_projection_evaluation --runs 10 --examples polygon_robot,pf_obs --save_results --quiet
"""

import argparse
import os
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
from typing import List, Dict, Any, Tuple

from test.test_projection_with_metrics import simulate



def discover_examples(base_dir: str = "example", kins=("diff", "acker")) -> List[str]:
    """发现可用环境（存在对应运动学配置与 yaml 文件）"""
    base = Path(base_dir)
    if not base.exists():
        return []
    examples = []
    for d in sorted([p for p in base.iterdir() if p.is_dir()]):
        ok = False
        for kin in kins:
            env_yaml = d / kin / "env.yaml"
            planner_yaml = d / kin / "planner.yaml"
            if env_yaml.exists() and planner_yaml.exists():
                ok = True
                break
        if ok:
            examples.append(d.name)
    return examples


def safe_simulate(example: str, kin: str, proj: str, max_steps: int, no_display: bool,
                  save_results: bool, quiet: bool) -> Tuple[bool, Dict[str, Any], str]:
    """
    安全地调用单次仿真，捕获异常
    Returns: (success, summary_report, err_msg)
    """
    try:
        if quiet:
            f = StringIO()
            with redirect_stdout(f):
                rep = simulate(
                    example_name=example,
                    kinematics=kin,
                    projection=proj,
                    save_animation=False,
                    ani_name=f"{example}_{kin}_{proj}",
                    full=False,
                    no_display=no_display,
                    point_vel=False,
                    max_steps=max_steps,
                    reverse=(example == "reverse" and kin == "diff"),
                    save_results=save_results,
                )
        else:
            rep = simulate(
                example_name=example,
                kinematics=kin,
                projection=proj,
                save_animation=False,
                ani_name=f"{example}_{kin}_{proj}",
                full=False,
                no_display=no_display,
                point_vel=False,
                max_steps=max_steps,
                reverse=(example == "reverse" and kin == "diff"),
                save_results=save_results,
            )
        return True, rep, ""
    except Exception as e:
        return False, {}, str(e)


def aggregate_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对多次运行的 summary 进行聚合统计"""
    import numpy as np

    def arr(key):
        return np.array([r.get(key) for r in reports if key in r and r.get(key) is not None], dtype=float)

    out = {}
    for key in [
        'avg_pre_violation_rate', 'avg_pre_p95', 'avg_pre_excess',
        'avg_post_max', 'max_post_max', 'avg_post_excess',
        'steps_executed'
    ]:
        a = arr(key)
        out[key + '_mean'] = float(a.mean()) if a.size else None
        out[key + '_std'] = float(a.std()) if a.size else None

    # 成功率：到达率/停止率/碰撞率/成功率
    for bkey, name in [('arrived', 'arrived_rate'), ('stopped', 'stopped_rate'), ('collided', 'collided_rate'), ('success', 'success_rate')]:
        vals = [bool(r.get(bkey)) for r in reports if bkey in r]
        out[name] = float(sum(vals) / len(vals)) if vals else None

    out['num_runs'] = len(reports)
    return out


def make_improvement(hard_aggr: Dict[str, Any], none_aggr: Dict[str, Any]) -> Dict[str, Any]:
    """hard 相对 none 的改进百分比"""
    def imp(none_val, hard_val):
        if none_val is None or none_val == 0:
            return None
        return float((none_val - hard_val) / none_val * 100.0)

    return {
        'post_excess_improve_percent': imp(none_aggr.get('avg_post_excess_mean'), hard_aggr.get('avg_post_excess_mean')),
        'post_max_improve_percent': imp(none_aggr.get('avg_post_max_mean'), hard_aggr.get('avg_post_max_mean')),
        'max_peak_improve_percent': imp(none_aggr.get('max_post_max_mean'), hard_aggr.get('max_post_max_mean')),
        'pre_violation_improve_percent': imp(none_aggr.get('avg_pre_violation_rate_mean'), hard_aggr.get('avg_pre_violation_rate_mean')),
    }


def run_batch(runs: int, examples: List[str], kins: List[str], projections: List[str],
              max_steps: int, no_display: bool, save_results: bool, quiet: bool) -> Dict[str, Any]:
    results = {}
    total_configs = 0

    for ex in examples:
        for kin in kins:
            # 跳过不存在的配置
            env_yaml = Path('example') / ex / kin / 'env.yaml'
            planner_yaml = Path('example') / ex / kin / 'planner.yaml'
            if not (env_yaml.exists() and planner_yaml.exists()):
                print(f"[SKIP] example={ex}, kin={kin} (missing yaml)")
                continue

            cfg_key = f"{ex}::{kin}"
            results[cfg_key] = {}
            total_configs += 1

            for proj in projections:
                print(f"\n=== Running {runs} runs for {ex} | {kin} | projection={proj} ===")
                one_cfg_reports = []
                errors = 0
                t0 = time.time()
                for i in range(1, runs + 1):
                    ok, rep, err = safe_simulate(
                        example=ex, kin=kin, proj=proj, max_steps=max_steps,
                        no_display=no_display, save_results=save_results, quiet=quiet
                    )
                    if ok:
                        one_cfg_reports.append(rep)
                        if not quiet:
                            print(f"[OK] {ex}|{kin}|{proj} run {i}/{runs}")
                    else:
                        errors += 1
                        print(f"[ERR] {ex}|{kin}|{proj} run {i}/{runs}: {err}")
                dt = time.time() - t0
                aggr = aggregate_metrics(one_cfg_reports)
                aggr['errors'] = errors
                aggr['elapsed_sec'] = dt
                results[cfg_key][proj] = {
                    'aggregate': aggr,
                    'runs': one_cfg_reports,
                }

    # 计算改进
    for cfg_key, d in results.items():
        if 'hard' in d and 'none' in d:
            d['improvement'] = make_improvement(d['hard']['aggregate'], d['none']['aggregate'])
    return results


def save_batch_summary(batch: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"batch_summary_{timestamp}.md"
    csv_path = out_dir / f"batch_summary_{timestamp}.csv"
    json_path = out_dir / f"batch_summary_{timestamp}.json"

    # Markdown
    lines = []
    lines.append("# Batch Projection Evaluation Summary\n")
    lines.append("| Example | Kinematics | Metric | hard | none | Improve(%) |\n")
    lines.append("|---|---|---|---:|---:|---:|\n")

    # CSV
    csv_lines = [
        "example,kinematics,metric,hard,none,improve_percent"
    ]

    for cfg_key, d in batch.items():
        if not isinstance(d, dict) or 'hard' not in d or 'none' not in d:
            continue
        ex, kin = cfg_key.split("::")
        hard_aggr = d['hard']['aggregate']
        none_aggr = d['none']['aggregate']
        imp = d.get('improvement', {})

        rows = [
            ("avg_pre_violation_rate_mean", "Avg Pre Violation Rate", hard_aggr.get('avg_pre_violation_rate_mean'), none_aggr.get('avg_pre_violation_rate_mean'), imp.get('pre_violation_improve_percent')),
            ("avg_pre_p95_mean", "Avg Pre P95 Norm", hard_aggr.get('avg_pre_p95_mean'), none_aggr.get('avg_pre_p95_mean'), None),
            ("avg_post_max_mean", "Avg Post Max Norm", hard_aggr.get('avg_post_max_mean'), none_aggr.get('avg_post_max_mean'), imp.get('post_max_improve_percent')),
            ("max_post_max_mean", "Max Norm Peak", hard_aggr.get('max_post_max_mean'), none_aggr.get('max_post_max_mean'), imp.get('max_peak_improve_percent')),
            ("avg_post_excess_mean", "Avg Post Excess", hard_aggr.get('avg_post_excess_mean'), none_aggr.get('avg_post_excess_mean'), imp.get('post_excess_improve_percent')),
            ("arrived_rate", "Arrival Rate", hard_aggr.get('arrived_rate'), none_aggr.get('arrived_rate'), None),
            ("stopped_rate", "Stopped Rate", hard_aggr.get('stopped_rate'), none_aggr.get('stopped_rate'), None),
            ("collided_rate", "Collision Rate", hard_aggr.get('collided_rate'), none_aggr.get('collided_rate'), None),
        ]
        for key, metric_name, hard_val, none_val, improve in rows:
            def fmt(x, is_rate=False):
                if x is None:
                    return "NA"
                if is_rate:
                    return f"{x*100:.2f}%"
                return f"{x:.6f}"
            is_rate = metric_name.endswith("Rate")
            lines.append(f"| {ex} | {kin} | {metric_name} | {fmt(hard_val, is_rate)} | {fmt(none_val, is_rate)} | {('NA' if improve is None else f'{improve:.2f}%')} |")
            csv_lines.append(f"{ex},{kin},{metric_name},{hard_val},{none_val},{'' if improve is None else improve}")

    md_path.write_text("\n".join(lines), encoding='utf-8')
    csv_path.write_text("\n".join(csv_lines), encoding='utf-8')
    json_path.write_text(json.dumps(batch, indent=2), encoding='utf-8')
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Batch projection evaluation")
    parser.add_argument("--runs", type=int, default=50, help="runs per (example,kin,proj)")
    parser.add_argument("--examples", type=str, default="all", help="comma-separated list or 'all'")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per run")
    parser.add_argument("--no_display", action="store_true", help="disable display for faster run")
    parser.add_argument("--save_results", action="store_true", help="save each run's JSON result")
    parser.add_argument("--quiet", action="store_true", help="suppress inner per-step prints")
    args = parser.parse_args()

    kins = ["diff", "acker"]
    projections = ["hard", "none"]

    # 发现环境
    all_examples = discover_examples("example", kins)
    if args.examples != "all":
        wanted = set([s.strip() for s in args.examples.split(',') if s.strip()])
        examples = [e for e in all_examples if e in wanted]
    else:
        examples = all_examples

    if not examples:
        print("No valid examples found.")
        return

    print(f"Discovered {len(examples)} example(s): {examples}")
    print(f"Each config will run {args.runs} time(s). This may take a long time.")

    # 执行批量评测
    batch = run_batch(
        runs=args.runs,
        examples=examples,
        kins=kins,
        projections=projections,
        max_steps=args.max_steps,
        no_display=args.no_display,
        save_results=args.save_results,
        quiet=args.quiet,
    )

    # 保存汇总
    out_dir = Path("test/results")
    md_path = save_batch_summary(batch, out_dir)
    print(f"\nBatch summary saved to: {md_path}")


if __name__ == "__main__":
    main()

