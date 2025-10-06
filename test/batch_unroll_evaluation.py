"""
PDHG-Unroll batch evaluation.

Evaluates (projection_mode × unroll_J) combinations across examples.
Designed for Stage 1-4 of the PDHG-Unroll implementation roadmap.

Usage:
  # Stage 1: Fixed step verification (J=0 vs J=1)
  python -m test.batch_unroll_evaluation --runs 3 --max_steps 800 --no_display --quiet \
      --modes hard --unroll-J 0,1 \
      --ckpt-acker-hard example/model/acker_robot_default/model_5000.pth

  # Stage 2: Grid search optimal tau/sigma (run separately with different --pdhg-tau/sigma)
  python -m test.batch_unroll_evaluation --runs 5 --max_steps 800 --no_display --quiet \
      --modes hard --unroll-J 2 --pdhg-tau 0.7 --pdhg-sigma 0.7

  # Stage 3: Learnable steps
  python -m test.batch_unroll_evaluation --runs 10 --max_steps 800 --no_display --quiet \
      --modes hard --unroll-J 2 --pdhg-learnable

  # Full matrix: all modes × all J values
  python -m test.batch_unroll_evaluation --runs 10 --max_steps 1000 --no_display --quiet \
      --modes hard,learned,none --unroll-J 0,1,2,3
"""

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from test.evaluation_utils import safe_simulate, aggregate_metrics


# Default checkpoints (same as batch_projection_evaluation.py)
DEFAULT_CKPT: Dict[str, Dict[str, str]] = {
    'diff': {
        'hard':    'example/model/diff_robot_default/model_5000.pth',
        'learned': 'example/dune_train/model/diff_learned_prox_robot/model_2500.pth',
        'none':    'example/model/diff_robot_default/model_5000.pth',
    },
    'acker': {
        'hard':    'example/model/acker_robot_default/model_5000.pth',
        'learned': 'example/dune_train/model/acker_learned_prox_robot/model_2500.pth',
        'none':    'example/model/acker_robot_default/model_5000.pth',
    },
}


def discover_examples(base_dir: str = "example", kins: tuple = ("diff", "acker")) -> List[str]:
    """Discover valid examples with env.yaml and planner.yaml."""
    base = Path(base_dir)
    if not base.exists():
        return []
    out: List[str] = []
    for d in sorted([p for p in base.iterdir() if p.is_dir()]):
        ok = False
        for kin in kins:
            if (d / kin / "env.yaml").exists() and (d / kin / "planner.yaml").exists():
                ok = True
                break
        if ok:
            out.append(d.name)
    return out


def run_batch(
    runs: int,
    examples: List[str],
    kins: List[str],
    projections: List[str],
    unroll_Js: List[int],
    max_steps: int,
    no_display: bool,
    save_results: bool,
    quiet: bool,
    ckpt_map: Dict[str, Dict[str, str]],
    pdhg_tau: float,
    pdhg_sigma: float,
    pdhg_learnable: bool,
    profile_timing: bool,
    timing_warmup: int,
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Run batch evaluation over (example × kin × projection × unroll_J) Cartesian product.
    
    Returns:
        dict with structure: {cfg_key: {aggregate: {...}, runs: [...]}}
    """
    results: Dict[str, Any] = {}
    
    for ex in examples:
        for kin in kins:
            env_yaml = Path('example') / ex / kin / 'env.yaml'
            planner_yaml = Path('example') / ex / kin / 'planner.yaml'
            if not (env_yaml.exists() and planner_yaml.exists()):
                print(f"[SKIP] example={ex}, kin={kin} (missing yaml)")
                continue
            
            for proj in projections:
                for J in unroll_Js:
                    cfg_key = f"{ex}::{kin}::{proj}::J{J}"
                    print(f"\n=== Running {runs} runs for {ex} | {kin} | {proj} | J={J} ===")
                    
                    planner_config = {
                        'projection': proj,
                        'unroll_J': J,
                        'pdhg_tau': pdhg_tau,
                        'pdhg_sigma': pdhg_sigma,
                        'pdhg_learnable': pdhg_learnable,
                        'ckpt_override': ckpt_map.get(kin, {}).get(proj),
                        'profile_timing': profile_timing,
                        'timing_warmup': timing_warmup,
                    }
                    
                    one_cfg_reports: List[Dict[str, Any]] = []
                    errors = 0
                    
                    for i in range(1, runs + 1):
                        ok, rep, err = safe_simulate(
                            example=ex,
                            kin=kin,
                            planner_config=planner_config,
                            max_steps=max_steps,
                            no_display=no_display,
                            save_results=save_results,
                            quiet=quiet,
                            results_dir=results_dir,
                        )
                        if ok:
                            one_cfg_reports.append(rep)
                            if not quiet:
                                print(f"[OK] {ex}|{kin}|{proj}|J{J} run {i}/{runs}")
                        else:
                            errors += 1
                            print(f"[ERR] {ex}|{kin}|{proj}|J{J} run {i}/{runs}: {err}")
                    
                    aggr = aggregate_metrics(one_cfg_reports)
                    aggr['errors'] = errors
                    
                    results[cfg_key] = {
                        'aggregate': aggr,
                        'runs': one_cfg_reports,
                    }
    
    return results


def save_batch_summary(batch: Dict[str, Any], out_dir: Path, tag: str = "unroll") -> Path:
    """Save batch results as Markdown, CSV, and JSON."""
    import json
    
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"batch_summary_{tag}_{timestamp}.md"
    csv_path = out_dir / f"batch_summary_{tag}_{timestamp}.csv"
    json_path = out_dir / f"batch_summary_{tag}_{timestamp}.json"

    lines: List[str] = []
    lines.append(f"# PDHG-Unroll Batch Evaluation Summary ({tag})\n")
    lines.append("| Example | Kin | Projection | J | Avg Pre Viol | Avg Pre P95 | Avg Post Max | Max Post Max | Avg Post Excess |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")

    csv_lines = [
        "example,kinematics,projection,unroll_J,avg_pre_violation_rate_mean,avg_pre_p95_mean,avg_post_max_mean,max_post_max_mean,avg_post_excess_mean"
    ]

    for cfg_key, d in batch.items():
        if not isinstance(d, dict) or 'aggregate' not in d:
            continue
        
        parts = cfg_key.split("::")
        if len(parts) != 4:
            continue
        ex, kin, proj, j_str = parts
        J = j_str.replace("J", "")
        
        aggr = d['aggregate']
        
        def fmt(x):
            return f"{x:.6f}" if x is not None else "NA"
        
        lines.append(
            f"| {ex} | {kin} | {proj} | {J} | "
            f"{fmt(aggr.get('avg_pre_violation_rate_mean'))} | "
            f"{fmt(aggr.get('avg_pre_p95_mean'))} | "
            f"{fmt(aggr.get('avg_post_max_mean'))} | "
            f"{fmt(aggr.get('max_post_max_mean'))} | "
            f"{fmt(aggr.get('avg_post_excess_mean'))} |"
        )
        
        csv_lines.append(
            f"{ex},{kin},{proj},{J},"
            f"{aggr.get('avg_pre_violation_rate_mean')},"
            f"{aggr.get('avg_pre_p95_mean')},"
            f"{aggr.get('avg_post_max_mean')},"
            f"{aggr.get('max_post_max_mean')},"
            f"{aggr.get('avg_post_excess_mean')}"
        )

    md_path.write_text("\n".join(lines), encoding='utf-8')
    csv_path.write_text("\n".join(csv_lines), encoding='utf-8')
    json_path.write_text(json.dumps(batch, indent=2), encoding='utf-8')
    
    return md_path


def main():
    parser = argparse.ArgumentParser(description="PDHG-Unroll batch evaluation")

    # YAML 配置文件（优先级最高）
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")

    # Reuse parameters from batch_projection_evaluation
    parser.add_argument("--runs", type=int, default=10, help="runs per (example,kin,proj,J)")
    parser.add_argument("--examples", type=str, default="all", help="comma-separated list or 'all'")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per run")
    parser.add_argument("--no_display", action="store_true", help="disable display for faster run")
    parser.add_argument("--save_results", action="store_true", help="save each run's JSON result")
    parser.add_argument("--quiet", action="store_true", help="suppress inner per-step prints")
    parser.add_argument("--modes", type=str, default="hard", help="comma-separated projection modes")
    parser.add_argument("--profile_timing", action="store_true", help="collect per-step timing stats")
    parser.add_argument("--timing_warmup", type=int, default=3, help="number of warmup steps to skip for timing")

    # PDHG-specific parameters
    parser.add_argument("--unroll-J", type=str, default="0,1",
                       help="Comma-separated unroll steps (e.g., '0,1,2,3')")
    parser.add_argument("--pdhg-tau", type=float, default=0.5, help="PDHG primal step size")
    parser.add_argument("--pdhg-sigma", type=float, default=0.5, help="PDHG dual step size")
    parser.add_argument("--pdhg-learnable", action="store_true",
                       help="Enable learnable step sizes (Stage 3)")

    # 性能分析参数
    # Checkpoint overrides (reuse from batch_projection_evaluation)
    parser.add_argument("--ckpt-hard", type=str, default="", help="override ckpt for hard mode")
    parser.add_argument("--ckpt-learned", type=str, default="", help="override ckpt for learned mode")
    parser.add_argument("--ckpt-none", type=str, default="", help="override ckpt for none mode")
    parser.add_argument("--ckpt-diff-hard", type=str, default="")
    parser.add_argument("--ckpt-diff-learned", type=str, default="")
    parser.add_argument("--ckpt-diff-none", type=str, default="")
    parser.add_argument("--ckpt-acker-hard", type=str, default="")
    parser.add_argument("--ckpt-acker-learned", type=str, default="")
    parser.add_argument("--ckpt-acker-none", type=str, default="")
    
    args = parser.parse_args()

    # 加载 YAML 配置（如果提供）
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

        # YAML 配置覆盖命令行参数
        for key, value in yaml_config.items():
            key_arg = key.replace('_', '-')  # 转换为命令行格式
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

        print(f"Loaded config from: {args.config}")
        print(f"Effective config: {yaml_config}\n")

    kins = ["diff", "acker"]
    projections = [m.strip() for m in args.modes.split(',') if m.strip()]
    unroll_Js = [int(j.strip()) for j in args.unroll_J.split(',') if j.strip()]

    def pick(val: str, gen: str, default: str) -> str:
        return val or gen or default

    ckpt_map: Dict[str, Dict[str, str]] = {
        'diff': {
            'hard':    pick(args.ckpt_diff_hard,    args.ckpt_hard,    DEFAULT_CKPT['diff']['hard']),
            'learned': pick(args.ckpt_diff_learned, args.ckpt_learned, DEFAULT_CKPT['diff']['learned']),
            'none':    pick(args.ckpt_diff_none,    args.ckpt_none,    DEFAULT_CKPT['diff']['none']),
        },
        'acker': {
            'hard':    pick(args.ckpt_acker_hard,    args.ckpt_hard,    DEFAULT_CKPT['acker']['hard']),
            'learned': pick(args.ckpt_acker_learned, args.ckpt_learned, DEFAULT_CKPT['acker']['learned']),
            'none':    pick(args.ckpt_acker_none,    args.ckpt_none,    DEFAULT_CKPT['acker']['none']),
        },
    }

    print("CKPT mapping in use:")
    for kin, mp in ckpt_map.items():
        for proj, path in mp.items():
            print(f"  {kin:5s} | {proj:7s} -> {path} (found={os.path.exists(path)})")

    all_examples = discover_examples("example", kins)
    examples = all_examples if args.examples == "all" else [
        e for e in all_examples if e in {s.strip() for s in args.examples.split(',') if s.strip()}
    ]
    if not examples:
        print("No valid examples found.")
        return

    print(f"Discovered {len(examples)} example(s): {examples}")
    print(f"Projection modes: {projections}")
    print(f"Unroll J values: {unroll_Js}")
    print(f"PDHG config: tau={args.pdhg_tau}, sigma={args.pdhg_sigma}, learnable={args.pdhg_learnable}")
    print(f"Each config will run {args.runs} time(s). This may take a long time.")

    # Results dir per batch
    tag = f"modes-{'-'.join(projections)}_J-{'-'.join(map(str, unroll_Js))}"
    if args.pdhg_learnable:
        tag += "_learnable"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("test/results") / f"batch_unroll_{tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = run_batch(
        runs=args.runs,
        examples=examples,
        kins=kins,
        projections=projections,
        unroll_Js=unroll_Js,
        max_steps=args.max_steps,
        no_display=args.no_display,
        save_results=args.save_results,
        quiet=args.quiet,
        ckpt_map=ckpt_map,
        pdhg_tau=args.pdhg_tau,
        pdhg_sigma=args.pdhg_sigma,
        pdhg_learnable=args.pdhg_learnable,
        profile_timing=args.profile_timing,
        timing_warmup=args.timing_warmup,
        results_dir=out_dir,
    )

    md_path = save_batch_summary(batch, out_dir, tag=tag)
    print(f"\nBatch summary saved to: {md_path}")


if __name__ == "__main__":
    main()

