"""
Batch projection evaluation (single entrypoint for planner simulations)

Features
- Traverse examples under `example/` and run multiple simulations per (example, kinematics, projection)
- Support three modes: hard | learned | none, with per-mode and per-kinematics ckpt overrides
- Headless by default for speed; save per-run JSON and batch summaries (MD/CSV/JSON)
- Self-contained: includes metrics evaluator and simulate() (no other test scripts needed)

Examples
  python -m test.batch_projection_evaluation --runs 10 --max_steps 800 --no_display --quiet \
      --modes hard,learned,none \
      --ckpt-acker-hard example/model/acker_robot_default/model_5000.pth \
      --ckpt-acker-learned example/dune_train/model/acker_learned_prox_robot/model_2500.pth

  python -m test.batch_projection_evaluation --examples corridor,pf_obs --runs 5 --no_display --quiet
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
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import yaml
import torch
import irsim
from neupan import neupan

# ---------------------- Defaults ----------------------

DEFAULT_CKPT: Dict[str, Dict[str, str]] = {
    # Diff-drive defaults 
    'diff': {
        'hard':    'example/model/diff_robot_default/model_5000.pth',
        'learned': 'example/dune_train/model/diff_learned_prox_robot/model_2500.pth',  
        'none':    'example/model/diff_robot_default/model_5000.pth',
    },
    # Acker defaults
    'acker': {
        'hard':    'example/model/acker_robot_default/model_5000.pth',
        'learned': 'example/dune_train/model/acker_learned_prox_robot/model_2500.pth',
        'none':    'example/model/acker_robot_default/model_5000.pth',
    },
}


# ---------------------- Helper: discover examples ----------------------

def discover_examples(base_dir: str = "example", kins: Tuple[str, str] = ("diff", "acker")) -> List[str]:
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


# ---------------------- Metrics & Simulation ----------------------

class ProjectionEvaluator:
    """Collect per-step dual-feasibility metrics and aggregate a summary."""

    def __init__(self, projection: str):
        self.projection = projection
        self.step_records: List[Dict[str, Any]] = []

    def record_step(self, step: int, dune_layer, lam_list):
        if dune_layer is None:
            return

        pre_v = getattr(dune_layer, "dual_norm_violation_rate", None)
        pre_p95 = getattr(dune_layer, "dual_norm_p95", None)
        pre_exc = getattr(dune_layer, "dual_norm_max_excess_pre", None)
        post_exc = getattr(dune_layer, "dual_norm_max_excess_post", None)

        post_max = float("nan")
        if lam_list:
            vals = []
            for lam in lam_list:
                if lam is None or lam.numel() == 0:
                    continue
                vals.append(torch.norm(lam, dim=0).max().item())
            if vals:
                post_max = max(vals)

        self.step_records.append({
            'step': step,
            'pre_violation_rate': pre_v,
            'pre_p95': pre_p95,
            'pre_excess': pre_exc,
            'post_excess': post_exc,
            'post_max_via_lam': post_max,
        })

    def summary(self) -> Dict[str, Any]:
        if not self.step_records:
            return {}
        pre_viol = [r['pre_violation_rate'] for r in self.step_records if r['pre_violation_rate'] is not None]
        pre_p95s = [r['pre_p95'] for r in self.step_records if r['pre_p95'] is not None]
        pre_excs = [r['pre_excess'] for r in self.step_records if r['pre_excess'] is not None]
        post_excs = [r['post_excess'] for r in self.step_records if r['post_excess'] is not None]
        post_maxs = [r['post_max_via_lam'] for r in self.step_records if not np.isnan(r['post_max_via_lam'])]

        return {
            'projection': self.projection,
            'steps_executed': len(self.step_records),
            'avg_pre_violation_rate': float(np.mean(pre_viol)) if pre_viol else 0.0,
            'avg_pre_p95': float(np.mean(pre_p95s)) if pre_p95s else 0.0,
            'avg_pre_excess': float(np.mean(pre_excs)) if pre_excs else 0.0,
            'avg_post_excess': float(np.mean(post_excs)) if post_excs else 0.0,
            'avg_post_max': float(np.mean(post_maxs)) if post_maxs else 0.0,
            'max_post_max': float(np.max(post_maxs)) if post_maxs else 0.0,
        }


def simulate(example_name: str,
             kinematics: str,
             projection: str,
             save_animation: bool,
             ani_name: str,
             full: bool,
             no_display: bool,
             point_vel: bool,
             max_steps: int,
             reverse: bool,
             save_results: bool,
             ckpt_override: Optional[str] = None,
             results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run one planner simulation with metrics; optionally override ckpt via pan.dune_checkpoint."""

    env_file = f"example/{example_name}/{kinematics}/env.yaml"
    planner_file = f"example/{example_name}/{kinematics}/planner.yaml"

    env = irsim.make(env_file, save_ani=save_animation, full=full, display=not no_display)

    # Read YAML config and only override checkpoint to preserve other pan parameters
    pan_kwargs = None
    if ckpt_override:
        try:
            with open(planner_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                pan_kwargs = yaml_config.get('pan', {})
                pan_kwargs['dune_checkpoint'] = ckpt_override
        except Exception:
            pan_kwargs = {'dune_checkpoint': ckpt_override}

    train_kwargs = dict(
        projection=projection,
        monitor_dual_norm=True,
        direct_train=True,
    )

    planner = neupan.init_from_yaml(
        planner_file,
        pan=pan_kwargs if pan_kwargs else None,
        train=train_kwargs,
    )

    evaluator = ProjectionEvaluator(projection)

    # stuck detection parameters (aligned with LON_corridor_01)
    stuck_threshold = 0.01
    stuck_count = 0
    stuck_count_thresh = 5

    for step in range(max_steps):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()
        if point_vel:
            points, point_velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
        else:
            points = planner.scan_to_point(robot_state, lidar_scan)
            point_velocities = None

        action, info = planner(robot_state, points, point_velocities)
        dune_layer = getattr(planner.pan, 'dune_layer', None)
        lam_list = planner.pan.current_nom_values[3] if planner.pan.current_nom_values else []
        evaluator.record_step(step, dune_layer, lam_list)

        if not no_display:
            env.draw_points(planner.dune_points, s=25, c='g', refresh=True)
            env.draw_points(planner.nrmp_points, s=13, c='r', refresh=True)
            env.draw_trajectory(planner.opt_trajectory, 'r', refresh=True)
            env.draw_trajectory(planner.ref_trajectory, 'b', refresh=True)
            env.render()

        # record pre-step position for stuck detection
        pre_pos = env.get_robot_state()[0:2]

        env.step(action)

        # stuck detection: small displacement for consecutive steps
        cur_pos = env.get_robot_state()[0:2]
        if np.linalg.norm(cur_pos - pre_pos) < stuck_threshold:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count > stuck_count_thresh:
            print(f"stuck: True, diff_distance < {stuck_threshold}")
            break

        done = env.done()
        if done and not info.get('arrive'):
            info['arrive'] = True
        if info.get('arrive') or info.get('stop') or done:
            break

    env.end(0)
    report = evaluator.summary()

    if save_results:
        out_dir = results_dir if results_dir is not None else Path("test/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = out_dir / f"{example_name}_{kinematics}_{projection}_{ts}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'example': example_name,
                    'kinematics': kinematics,
                    'projection': projection,
                    'max_steps': max_steps,
                    'ckpt': ckpt_override,
                },
                'summary': report,
            }, f, indent=2)
        print(f"Results saved to: {result_file}")

    return report


def safe_simulate(example: str, kin: str, proj: str, max_steps: int, no_display: bool,
                  save_results: bool, quiet: bool,
                  ckpt_map: Optional[Dict[str, Dict[str, str]]] = None,
                  results_dir: Optional[Path] = None) -> Tuple[bool, Dict[str, Any], str]:
    """Wrapper that catches exceptions and returns a (success, report, err) tuple."""
    try:
        ckpt_override = (ckpt_map.get(kin, {}).get(proj) if ckpt_map else None)
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
                    point_vel=True,
                    max_steps=max_steps,
                    reverse=(example == "reverse" and kin == "diff"),
                    save_results=save_results,
                    ckpt_override=ckpt_override,
                    results_dir=results_dir,
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
                point_vel=True,
                max_steps=max_steps,
                reverse=(example == "reverse" and kin == "diff"),
                save_results=save_results,
                ckpt_override=ckpt_override,
                results_dir=results_dir,
            )
        return True, rep, ""
    except Exception as e:
        return False, {}, str(e)


def aggregate_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    def arr(key):
        return np.array([r.get(key) for r in reports if key in r and r.get(key) is not None], dtype=float)

    out: Dict[str, Any] = {}
    for key in [
        'avg_pre_violation_rate', 'avg_pre_p95', 'avg_pre_excess',
        'avg_post_max', 'max_post_max', 'avg_post_excess',
        'steps_executed'
    ]:
        a = arr(key)
        out[key + '_mean'] = float(a.mean()) if a.size else None
        out[key + '_std'] = float(a.std()) if a.size else None
    for bkey, name in [('arrived', 'arrived_rate'), ('stopped', 'stopped_rate'), ('collided', 'collided_rate'), ('success', 'success_rate')]:
        vals = [bool(r.get(bkey)) for r in reports if bkey in r]
        out[name] = float(sum(vals) / len(vals)) if vals else None
    out['num_runs'] = len(reports)
    return out


def make_improvement(hard_aggr: Dict[str, Any], none_aggr: Dict[str, Any]) -> Dict[str, Any]:
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
              max_steps: int, no_display: bool, save_results: bool, quiet: bool,
              ckpt_map: Dict[str, Dict[str, str]], results_dir: Optional[Path]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for ex in examples:
        for kin in kins:
            env_yaml = Path('example') / ex / kin / 'env.yaml'
            planner_yaml = Path('example') / ex / kin / 'planner.yaml'
            if not (env_yaml.exists() and planner_yaml.exists()):
                print(f"[SKIP] example={ex}, kin={kin} (missing yaml)")
                continue
            cfg_key = f"{ex}::{kin}"
            results[cfg_key] = {}
            for proj in projections:
                print(f"\n=== Running {runs} runs for {ex} | {kin} | projection={proj} ===")
                one_cfg_reports: List[Dict[str, Any]] = []
                errors = 0
                t0 = time.time()
                for i in range(1, runs + 1):
                    ok, rep, err = safe_simulate(
                        example=ex, kin=kin, proj=proj, max_steps=max_steps,
                        no_display=no_display, save_results=save_results, quiet=quiet,
                        ckpt_map=ckpt_map, results_dir=results_dir
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

    # improvements relative to none
    for cfg_key, d in results.items():
        if 'none' in d:
            none_aggr = d['none']['aggregate']
            d['improvement'] = {}
            if 'hard' in d:
                d['improvement']['hard_vs_none'] = make_improvement(d['hard']['aggregate'], none_aggr)
            if 'learned' in d:
                d['improvement']['learned_vs_none'] = make_improvement(d['learned']['aggregate'], none_aggr)
    return results


def save_batch_summary(batch: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"batch_summary_{timestamp}.md"
    csv_path = out_dir / f"batch_summary_{timestamp}.csv"
    json_path = out_dir / f"batch_summary_{timestamp}.json"

    lines: List[str] = []
    lines.append("# Batch Projection Evaluation Summary\n")
    lines.append("| Example | Kinematics | Metric | hard | learned | none | Improve(hard→none) | Improve(learned→none) |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|\n")

    csv_lines = [
        "example,kinematics,metric,hard,learned,none,improve_hard_vs_none,improve_learned_vs_none"
    ]

    for cfg_key, d in batch.items():
        if not isinstance(d, dict) or 'none' not in d:
            continue
        ex, kin = cfg_key.split("::")
        hard_aggr = d.get('hard', {}).get('aggregate', {})
        learned_aggr = d.get('learned', {}).get('aggregate', {})
        none_aggr = d.get('none', {}).get('aggregate', {})
        imp_all = d.get('improvement', {})
        imp_hard = imp_all.get('hard_vs_none', {}) if isinstance(imp_all, dict) else {}
        imp_learned = imp_all.get('learned_vs_none', {}) if isinstance(imp_all, dict) else {}

        rows = [
            ("avg_pre_violation_rate_mean", "Avg Pre Violation Rate", hard_aggr.get('avg_pre_violation_rate_mean'), learned_aggr.get('avg_pre_violation_rate_mean'), none_aggr.get('avg_pre_violation_rate_mean'), imp_hard.get('pre_violation_improve_percent'), imp_learned.get('pre_violation_improve_percent')),
            ("avg_pre_p95_mean", "Avg Pre P95 Norm", hard_aggr.get('avg_pre_p95_mean'), learned_aggr.get('avg_pre_p95_mean'), none_aggr.get('avg_pre_p95_mean'), None, None),
            ("avg_post_max_mean", "Avg Post Max Norm", hard_aggr.get('avg_post_max_mean'), learned_aggr.get('avg_post_max_mean'), none_aggr.get('avg_post_max_mean'), imp_hard.get('post_max_improve_percent'), imp_learned.get('post_max_improve_percent')),
            ("max_post_max_mean", "Max Norm Peak", hard_aggr.get('max_post_max_mean'), learned_aggr.get('max_post_max_mean'), none_aggr.get('max_post_max_mean'), imp_hard.get('max_peak_improve_percent'), imp_learned.get('max_peak_improve_percent')),
            ("avg_post_excess_mean", "Avg Post Excess", hard_aggr.get('avg_post_excess_mean'), learned_aggr.get('avg_post_excess_mean'), none_aggr.get('avg_post_excess_mean'), imp_hard.get('post_excess_improve_percent'), imp_learned.get('post_excess_improve_percent')),
        ]
        for key, metric_name, hard_val, learned_val, none_val, imp_h, imp_l in rows:
            def fmt(x, is_rate=False):
                if x is None:
                    return "NA"
                if is_rate:
                    return f"{x*100:.2f}%"
                return f"{x:.6f}"
            is_rate = metric_name.endswith("Rate")
            lines.append(
                f"| {ex} | {kin} | {metric_name} | {fmt(hard_val, is_rate)} | {fmt(learned_val, is_rate)} | {fmt(none_val, is_rate)} | "
                f"{('NA' if imp_h is None else f'{imp_h:.2f}%')} | {('NA' if imp_l is None else f'{imp_l:.2f}%')} |"
            )
            csv_lines.append(
                f"{ex},{kin},{metric_name},{hard_val},{learned_val},{none_val},{'' if imp_h is None else imp_h},{'' if imp_l is None else imp_l}"
            )

    md_path.write_text("\n".join(lines), encoding='utf-8')
    csv_path.write_text("\n".join(csv_lines), encoding='utf-8')
    json_path.write_text(json.dumps(batch, indent=2), encoding='utf-8')
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Batch projection evaluation")
    parser.add_argument("--runs", type=int, default=10, help="runs per (example,kin,proj)")
    parser.add_argument("--examples", type=str, default="all", help="comma-separated list or 'all'")
    parser.add_argument("--max_steps", type=int, default=1000, help="max steps per run")
    parser.add_argument("--no_display", action="store_true", help="disable display for faster run")
    parser.add_argument("--save_results", action="store_true", help="save each run's JSON result")
    parser.add_argument("--quiet", action="store_true", help="suppress inner per-step prints")
    parser.add_argument("--modes", type=str, default="hard,learned,none", help="comma-separated projection modes")
    # generic and per-kinematics ckpt overrides
    parser.add_argument("--ckpt-hard", type=str, default="", help="override ckpt path for hard mode (generic)")
    parser.add_argument("--ckpt-learned", type=str, default="", help="override ckpt path for learned mode (generic)")
    parser.add_argument("--ckpt-none", type=str, default="", help="override ckpt path for none mode (generic)")
    parser.add_argument("--ckpt-diff-hard", type=str, default="", help="override ckpt for diff+hard")
    parser.add_argument("--ckpt-diff-learned", type=str, default="", help="override ckpt for diff+learned")
    parser.add_argument("--ckpt-diff-none", type=str, default="", help="override ckpt for diff+none")
    parser.add_argument("--ckpt-acker-hard", type=str, default="", help="override ckpt for acker+hard")
    parser.add_argument("--ckpt-acker-learned", type=str, default="", help="override ckpt for acker+learned")
    parser.add_argument("--ckpt-acker-none", type=str, default="", help="override ckpt for acker+none")
    args = parser.parse_args()

    kins = ["diff", "acker"]
    projections = [m.strip() for m in args.modes.split(',') if m.strip()]

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
    examples = all_examples if args.examples == "all" else [e for e in all_examples if e in {s.strip() for s in args.examples.split(',') if s.strip()}]
    if not examples:
        print("No valid examples found.")
        return

    print(f"Discovered {len(examples)} example(s): {examples}")
    print(f"Each config will run {args.runs} time(s). This may take a long time.")

    # Results dir per batch (modes + timestamp)
    tag = "-".join(projections)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("test/results") / f"batch_{tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = run_batch(
        runs=args.runs,
        examples=examples,
        kins=kins,
        projections=projections,
        max_steps=args.max_steps,
        no_display=args.no_display,
        save_results=args.save_results,
        quiet=args.quiet,
        ckpt_map=ckpt_map,
        results_dir=out_dir,
    )

    md_path = save_batch_summary(batch, out_dir)
    print(f"\nBatch summary saved to: {md_path}")


if __name__ == "__main__":
    main()

