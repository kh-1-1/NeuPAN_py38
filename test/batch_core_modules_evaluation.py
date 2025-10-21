"""
Core modules closed-loop batch evaluation (final version).

Runs a matrix over configurations (baseline/flex_no_learned/flex_learned/flex_roi),
examples (corridor, pf_obs, dyna_obs, convex_obs), and kinematics (diff, acker).

Notes
- No front_J: we do not pass PDHG unroll steps here.
- Front selection via train kwargs: front=obs_point|flex_pdhg, front_learned=True/False.
- ROI support: inject roi kwargs when enabled (n_min defaults to 10 via template YAML).
- Checkpoints: use example/model/... paths; overridable via config YAML.

CLI Quick examples
  python -m test.batch_core_modules_evaluation -c baseline -e corridor -k diff -r 3 -nd
  python -m test.batch_core_modules_evaluation \
      --config-file test/configs/core_modules_evaluation.yaml \
      --output-dir test/results/core_modules
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml
import irsim
from neupan import neupan


# ---------------- Defaults ----------------

DEFAULT_CKPT = {
    'baseline': {
        'diff':  'example/model/diff_CLARABEL_robot/model_5000.pth',
        'acker': 'example/model/acker_CLARABEL_robot/model_5000.pth',
    },
    'flex_no_learned': {
        'diff':  'example/model/diff_flex_pdhg-nolearned_robot/model_best.pth',
        'acker': 'example/model/acker_flex_pdhg-nolearned_robot/model_best.pth',
    },
    'flex_learned': {
        'diff':  'example/model/diff_flex_pdhg_robot/model_5000.pth',
        'acker': 'example/model/acker_flex_pdhg_robot/model_5000.pth',
    },
    'flex_roi': {
        'diff':  'example/model/diff_flex_pdhg_robot/model_5000.pth',
        'acker': 'example/model/acker_flex_pdhg_robot/model_5000.pth',
    },
}


CONFIG_TO_FRONT = {
    'baseline':        dict(front='obs_point', front_learned=False, roi=False),
    'flex_no_learned': dict(front='flex_pdhg',  front_learned=False, roi=False),
    'flex_learned':    dict(front='flex_pdhg',  front_learned=True,  roi=False),
    'flex_roi':        dict(front='flex_pdhg',  front_learned=True,  roi=True),
}


def discover_examples(base_dir: str = 'example', kins: Tuple[str, str] = ('diff', 'acker')) -> List[str]:
    base = Path(base_dir)
    if not base.exists():
        return []
    out: List[str] = []
    for d in sorted([p for p in base.iterdir() if p.is_dir()]):
        ok = False
        for kin in kins:
            if (d / kin / 'env.yaml').exists() and (d / kin / 'planner.yaml').exists():
                ok = True
                break
        if ok:
            out.append(d.name)
    return out


@dataclass
class RunMetrics:
    steps: int
    arrive: bool
    stop: bool
    min_distance: float
    path_length: float
    avg_step_time_ms: float
    max_v: float
    avg_v: float
    roi_strategy_counts: Dict[str, int]
    roi_avg_n_in: Optional[float]
    roi_avg_n_roi: Optional[float]
    roi_avg_reduction_ratio: Optional[float]


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _build_roi_kwargs(template_path: Optional[str]) -> dict:
    if not template_path:
        # Safe fallback template
        return {
            'enabled': True,
            'strategy_order': ['ellipse', 'tube', 'wedge'],
            'wedge': {'fov_deg': 60.0, 'r_max_m': 8.0},
            'ellipse': {'safety_scale': 1.08},
            'tube': {'r0_m': 0.5, 'kappa_gain': 0.4, 'v_tau_m': 0.3},
            'guardrail': {'n_min': 10, 'n_max': 500, 'relax_step': 1.15, 'tighten_step': 0.9},
            'always_keep': {'near_radius_m': 1.5, 'goal_radius_m': 1.5},
        }
    try:
        cfg = _load_yaml(template_path)
        roi = cfg.get('roi', cfg)  # allow file to be either full or roi-rooted
        roi = dict(roi)
        roi['enabled'] = True
        # enforce n_min=10 if provided
        gr = roi.get('guardrail', {})
        if 'n_min' in gr:
            gr['n_min'] = int(gr['n_min'])
        else:
            gr['n_min'] = 10
        roi['guardrail'] = gr
        return roi
    except Exception:
        return _build_roi_kwargs(None)


def simulate_once(example: str,
                  kin: str,
                  config_id: str,
                  ckpt: str,
                  roi_template: Optional[str],
                  max_steps: int,
                  no_display: bool,
                  quiet: bool,
                  results_dir: Optional[Path] = None,
                  run_idx: Optional[int] = None,
                  save_last_frame: bool = True) -> RunMetrics:

    env_file = f"example/{example}/{kin}/env.yaml"
    planner_file = f"example/{example}/{kin}/planner.yaml"

    # Enable frame saving when requested (similar to example/run_exp.py)
    env = irsim.make(env_file, save_ani=bool(save_last_frame), full=False, display=not no_display)

    # Train/front params
    front_cfg = CONFIG_TO_FRONT[config_id]
    train_kwargs = dict(
        projection='hard',        # keep DUNE hard projection
        monitor_dual_norm=True,
        direct_train=True,        # do not trigger training
        front=front_cfg['front'],
        front_learned=front_cfg['front_learned'],
    )

    pan_kwargs = dict(dune_checkpoint=ckpt)

    # ROI kwargs
    roi_kwargs = None
    if front_cfg.get('roi', False):
        roi_kwargs = _build_roi_kwargs(roi_template)

    planner = neupan.init_from_yaml(
        planner_file,
        pan=pan_kwargs,
        train=train_kwargs,
        roi=roi_kwargs,
    )

    # Step loop with metrics
    stuck_threshold = 0.01
    stuck_count = 0
    stuck_count_thresh = 5

    step_times = []
    v_list = []
    path_length = 0.0
    roi_nin = []
    roi_nroi = []
    roi_ratios = []
    roi_counts: Dict[str, int] = {}

    prev_pos = None
    run_min_dist = float('inf')

    for step in range(max_steps):
        t0 = time.perf_counter()

        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()
        # Prefer velocity-aware scan; fall back gracefully if unavailable
        points, point_velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
        if points is None:
            points = planner.scan_to_point(robot_state, lidar_scan)
            point_velocities = None
        action, info = planner(robot_state, points, point_velocities)

        # Metrics: ROI
        rinfo = info.get('roi')
        if rinfo:
            strat = rinfo.get('strategy', 'none')
            roi_counts[strat] = roi_counts.get(strat, 0) + 1
            n_in = rinfo.get('n_in')
            n_roi = rinfo.get('n_roi')
            if isinstance(n_in, (int, float)) and isinstance(n_roi, (int, float)) and n_in:
                roi_nin.append(float(n_in))
                roi_nroi.append(float(n_roi))
                roi_ratios.append(float(n_in) / max(1.0, float(n_roi)))

        # Min distance across run (take per-step current, keep global min)
        try:
            cur_min = float(planner.min_distance)
            if cur_min < run_min_dist:
                run_min_dist = cur_min
        except Exception:
            pass

        # Timings
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        step_times.append(step_time_ms)

        # Velocity metrics (assume plain v in [0]-th row)
        try:
            v = float(action[0, 0])
            v_list.append(v)
        except Exception:
            pass

        # Path length increment
        cur_pos = env.get_robot_state()[0:2]
        if prev_pos is not None:
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos

        if not no_display:
            env.draw_points(planner.dune_points, s=25, c='g', refresh=True)
            env.draw_points(planner.nrmp_points, s=13, c='r', refresh=True)
            # draw optimized and reference trajectories
            env.draw_trajectory(planner.opt_trajectory, 'r', refresh=True)
            env.draw_trajectory(planner.ref_trajectory, 'b', refresh=True)
            # draw initial path once (mimic run_exp behavior)
            if step == 0:
                try:
                    env.draw_trajectory(planner.initial_path, traj_type='-k', show_direction=False)
                except Exception:
                    try:
                        env.draw_trajectory(planner.initial_path, '-k', refresh=True)
                    except Exception:
                        pass
            env.render()

        # Stuck detection
        pre_pos = robot_state[0:2]
        env.step(action)
        cur_pos2 = env.get_robot_state()[0:2]
        if np.linalg.norm(cur_pos2 - pre_pos) < stuck_threshold:
            stuck_count += 1
        else:
            stuck_count = 0
        if stuck_count > stuck_count_thresh:
            if not quiet:
                print(f"stuck: True, diff_distance < {stuck_threshold}")
            break

        if info.get('arrive') or info.get('stop') or env.done():
            break

    # Optional: overlay ROI region before final save
    try:
        if CONFIG_TO_FRONT.get(config_id, {}).get('roi', False):
            planner.visualize_roi_region(env)
            env.render()
    except Exception:
        pass

    if save_last_frame and results_dir is not None:
        try:
            frames_dir = Path(results_dir) / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            ani_name = str(frames_dir / f"{example}_{kin}_{config_id}_run{run_idx}_{ts}")
            # Use non-zero end delay to ensure renderer flushes and saves frame
            env.end(1, ani_name=ani_name)
        except Exception:
            # fallback: just close without saving
            env.end(0)
    else:
        env.end(0)

    steps = len(step_times)
    avg_ms = float(np.mean(step_times)) if step_times else 0.0
    max_v = float(np.max(v_list)) if v_list else 0.0
    avg_v = float(np.mean(v_list)) if v_list else 0.0
    roi_avg_in = float(np.mean(roi_nin)) if roi_nin else None
    roi_avg_roi = float(np.mean(roi_nroi)) if roi_nroi else None
    roi_avg_ratio = float(np.mean(roi_ratios)) if roi_ratios else None

    metrics = RunMetrics(
        steps=steps,
        arrive=bool(info.get('arrive', False)),
        stop=bool(info.get('stop', False)),
        min_distance=run_min_dist,
        path_length=path_length,
        avg_step_time_ms=avg_ms,
        max_v=max_v,
        avg_v=avg_v,
        roi_strategy_counts=roi_counts,
        roi_avg_n_in=roi_avg_in,
        roi_avg_n_roi=roi_avg_roi,
        roi_avg_reduction_ratio=roi_avg_ratio,
    )

    # Save per-run detailed result
    try:
        if results_dir is not None:
            runs_dir = Path(results_dir) / 'runs'
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_tag = f"run{run_idx}" if run_idx is not None else datetime.now().strftime('%H%M%S')
            out_file = runs_dir / f"{example}_{kin}_{config_id}_{run_tag}.json"
            cfg_front = CONFIG_TO_FRONT.get(config_id, {})
            planner_dt = getattr(planner, 'dt', None)
            planner_T = getattr(planner, 'T', None)
            robot_name = getattr(getattr(planner, 'robot', None), 'name', kin)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': {
                        'example': example,
                        'kinematics': kin,
                        'config_id': config_id,
                        'env_file': env_file,
                        'planner_file': planner_file,
                        'checkpoint': ckpt,
                        'front_type': cfg_front.get('front'),
                        'front_learned': cfg_front.get('front_learned'),
                        'roi_enabled': cfg_front.get('roi', False),
                        'roi_template': roi_template,
                        'receding': planner_T,
                        'step_time': planner_dt,
                        'robot': robot_name,
                        'max_steps': max_steps,
                    },
                    'metrics': {
                        'steps': steps,
                        'arrive': metrics.arrive,
                        'stop': metrics.stop,
                        'min_distance': metrics.min_distance,
                        'path_length': metrics.path_length,
                        'avg_step_time_ms': metrics.avg_step_time_ms,
                        'max_velocity': metrics.max_v,
                        'avg_velocity': metrics.avg_v,
                        'roi_strategy_counts': metrics.roi_strategy_counts,
                        'roi_avg_n_in': metrics.roi_avg_n_in,
                        'roi_avg_n_roi': metrics.roi_avg_n_roi,
                        'roi_reduction_ratio': metrics.roi_avg_reduction_ratio,
                        'total_compute_time_ms': float(np.sum(step_times)) if step_times else 0.0,
                    }
                }, f, indent=2)
    except Exception:
        # do not fail the run if saving has any issues
        pass

    return metrics


def aggregate_runs(runs: List[RunMetrics]) -> Dict[str, Any]:
    if not runs:
        return {}

    def arr(fn):
        return np.array([fn(r) for r in runs], dtype=float)

    agg: Dict[str, Any] = {
        'runs': len(runs),
        'steps_mean': float(arr(lambda r: r.steps).mean()),
        'path_length_mean': float(arr(lambda r: r.path_length).mean()),
        'min_distance_mean': float(arr(lambda r: r.min_distance).mean()),
        'avg_step_time_ms_mean': float(arr(lambda r: r.avg_step_time_ms).mean()),
        'max_v_mean': float(arr(lambda r: r.max_v).mean()),
        'avg_v_mean': float(arr(lambda r: r.avg_v).mean()),
        'success_rate': float(np.mean([1.0 if r.arrive else 0.0 for r in runs])),
        'stop_rate': float(np.mean([1.0 if r.stop else 0.0 for r in runs])),
    }

    # ROI aggregation
    roi_in = [r.roi_avg_n_in for r in runs if r.roi_avg_n_in is not None]
    roi_roi = [r.roi_avg_n_roi for r in runs if r.roi_avg_n_roi is not None]
    roi_rr = [r.roi_avg_reduction_ratio for r in runs if r.roi_avg_reduction_ratio is not None]
    if roi_in:
        agg['roi_avg_n_in_mean'] = float(np.mean(roi_in))
    if roi_roi:
        agg['roi_avg_n_roi_mean'] = float(np.mean(roi_roi))
    if roi_rr:
        agg['roi_reduction_ratio_mean'] = float(np.mean(roi_rr))

    # Merge strategy counts
    strat_counts: Dict[str, int] = {}
    for r in runs:
        for k, v in r.roi_strategy_counts.items():
            strat_counts[k] = strat_counts.get(k, 0) + int(v)
    if strat_counts:
        agg['roi_strategy_counts'] = strat_counts

    return agg


def save_summary(batch: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = out_dir / f'summary_{ts}.md'
    csv_path = out_dir / f'summary_{ts}.csv'
    json_path = out_dir / f'summary_{ts}.json'

    lines = [
        '# Core Modules Evaluation Summary',
        '',
        '| Example | Kin | Config | Runs | Success | Stop | Steps | PathLen | MinDist | AvgStep(ms) | MaxV | AvgV | ROI n_in | ROI n_roi | ROI ratio |',
        '|---------|-----|--------|------|---------|------|-------|---------|---------|-------------|------|------|---------|-----------|-----------|',
    ]
    csv_lines = [
        'example,kin,config,runs,success,stop,steps,path_length,min_distance,avg_step_ms,max_v,avg_v,roi_n_in,roi_n_roi,roi_ratio'
    ]

    for ex, mp in batch.items():
        for kin, kp in mp.items():
            for cfg, aggr in kp.items():
                lines.append(
                    f"| {ex} | {kin} | {cfg} | {aggr.get('runs','')} | "
                    f"{aggr.get('success_rate',''):.2f} | {aggr.get('stop_rate',''):.2f} | "
                    f"{aggr.get('steps_mean',''):.1f} | {aggr.get('path_length_mean',''):.2f} | "
                    f"{aggr.get('min_distance_mean',''):.2f} | {aggr.get('avg_step_time_ms_mean',''):.2f} | "
                    f"{aggr.get('max_v_mean',''):.2f} | {aggr.get('avg_v_mean',''):.2f} | "
                    f"{aggr.get('roi_avg_n_in_mean','') if 'roi_avg_n_in_mean' in aggr else 'NA'} | "
                    f"{aggr.get('roi_avg_n_roi_mean','') if 'roi_avg_n_roi_mean' in aggr else 'NA'} | "
                    f"{aggr.get('roi_reduction_ratio_mean','') if 'roi_reduction_ratio_mean' in aggr else 'NA'} |"
                )
                csv_lines.append(
                    f"{ex},{kin},{cfg},{aggr.get('runs','')},{aggr.get('success_rate','')},{aggr.get('stop_rate','')},"
                    f"{aggr.get('steps_mean','')},{aggr.get('path_length_mean','')},{aggr.get('min_distance_mean','')},"
                    f"{aggr.get('avg_step_time_ms_mean','')},{aggr.get('max_v_mean','')},{aggr.get('avg_v_mean','')},"
                    f"{aggr.get('roi_avg_n_in_mean','') if 'roi_avg_n_in_mean' in aggr else ''},"
                    f"{aggr.get('roi_avg_n_roi_mean','') if 'roi_avg_n_roi_mean' in aggr else ''},"
                    f"{aggr.get('roi_reduction_ratio_mean','') if 'roi_reduction_ratio_mean' in aggr else ''}"
                )

    md_path.write_text("\n".join(lines), encoding='utf-8')
    csv_path.write_text("\n".join(csv_lines), encoding='utf-8')
    json_path.write_text(json.dumps(batch, indent=2), encoding='utf-8')
    return md_path


def main():
    parser = argparse.ArgumentParser(description='Core modules batch evaluation')

    # Config selection or file
    parser.add_argument('-c', '--config', dest='config', type=str, default='baseline',
                        help='baseline | flex_no_learned | flex_learned | flex_roi')
    parser.add_argument('--config-file', dest='config_file', type=str, default='',
                        help='YAML file defining configurations (overrides --config)')

    # Matrix selection
    parser.add_argument('-e', '--examples', dest='examples', type=str, default='corridor,pf_obs,non_obs,convex_obs',
                        help="comma-separated example names or 'all'")
    parser.add_argument('-k', '--kinematics', dest='kinematics', type=str, default='diff,acker',
                        help='comma-separated robot types')
    parser.add_argument('-r', '--runs', dest='runs', type=int, default=10, help='runs per combo')
    parser.add_argument('-ms', '--max-steps', dest='max_steps', type=int, default=800, help='max steps per run')

    # Behavior
    parser.add_argument('-nd', '--no-display', dest='no_display', action='store_true', help='disable rendering')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='suppress prints')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default='', help='results folder')
    parser.add_argument('--save_results', dest='save_results', action='store_true',
                        help='save per-run JSON details and last-frame images')

    # ROI template
    parser.add_argument('--roi-template', dest='roi_template', type=str, default='test/configs/roi_config_template.yaml',
                        help='ROI config template path (used when ROI enabled)')

    args = parser.parse_args()

    # Build configuration map
    if args.config_file:
        file_cfg = _load_yaml(args.config_file)
        if 'configurations' in file_cfg:
            cfg_map = file_cfg['configurations']
        else:
            # accept flat YAML with one config
            cfg_map = {args.config: file_cfg}
    else:
        # construct from defaults
        cfg_map = {
            'baseline': {
                'front_type': 'obs_point',
                'front_learned': False,
                'roi_enabled': False,
                'ckpt_diff': DEFAULT_CKPT['baseline']['diff'],
                'ckpt_acker': DEFAULT_CKPT['baseline']['acker'],
            },
            'flex_no_learned': {
                'front_type': 'flex_pdhg',
                'front_learned': False,
                'roi_enabled': False,
                'ckpt_diff': DEFAULT_CKPT['flex_no_learned']['diff'],
                'ckpt_acker': DEFAULT_CKPT['flex_no_learned']['acker'],
            },
            'flex_learned': {
                'front_type': 'flex_pdhg',
                'front_learned': True,
                'roi_enabled': False,
                'ckpt_diff': DEFAULT_CKPT['flex_learned']['diff'],
                'ckpt_acker': DEFAULT_CKPT['flex_learned']['acker'],
            },
            'flex_roi': {
                'front_type': 'flex_pdhg',
                'front_learned': True,
                'roi_enabled': True,
                'ckpt_diff': DEFAULT_CKPT['flex_roi']['diff'],
                'ckpt_acker': DEFAULT_CKPT['flex_roi']['acker'],
            },
        }

    # Determine which config IDs to run
    cfg_ids = [args.config] if not args.config_file else list(cfg_map.keys())

    # Examples and kinematics
    all_examples = discover_examples('example', ('diff', 'acker'))
    req_examples = [s.strip() for s in args.examples.split(',') if s.strip()]
    examples = all_examples if args.examples == 'all' else [e for e in all_examples if e in set(req_examples)]
    if not examples:
        print('No valid examples found.')
        return

    kins = [s.strip() for s in args.kinematics.split(',') if s.strip()]
    kins = [k for k in kins if k in ('diff', 'acker')]
    if not kins:
        kins = ['diff', 'acker']

    # Results dir
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) if args.output_dir else Path('test/results') / f'core_modules_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Examples: {examples}")
    print(f"Kinematics: {kins}")
    print(f"Configs: {cfg_ids}")

    batch: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ex in examples:
        batch.setdefault(ex, {})
        for kin in kins:
            batch[ex].setdefault(kin, {})
            for cfg_id in cfg_ids:
                cfg = cfg_map[cfg_id]
                front_type = cfg.get('front_type', 'obs_point')
                front_learned = bool(cfg.get('front_learned', False))
                roi_enabled = bool(cfg.get('roi_enabled', False))
                ckpt = cfg.get('ckpt_diff' if kin == 'diff' else 'ckpt_acker')

                # Sanity check ckpt existence
                if ckpt and not os.path.exists(ckpt):
                    print(f"[WARN] ckpt not found: {ckpt}")

                # Align with CONFIG_TO_FRONT keys
                tmp_cfg_id = 'baseline' if front_type == 'obs_point' else ('flex_roi' if roi_enabled else ('flex_learned' if front_learned else 'flex_no_learned'))

                runs: List[RunMetrics] = []
                for i in range(args.runs):
                    if not args.quiet:
                        print(f"Run {i+1}/{args.runs} | {ex} | {kin} | {cfg_id}")
                    m = simulate_once(
                        example=ex,
                        kin=kin,
                        config_id=tmp_cfg_id,
                        ckpt=ckpt,
                        roi_template=args.roi_template if roi_enabled else None,
                        max_steps=args.max_steps,
                        no_display=args.no_display,
                        quiet=args.quiet,
                        save_last_frame=bool(args.save_results),
                        results_dir=(out_dir if args.save_results else None),
                        run_idx=i+1,
                    )
                    runs.append(m)

                aggr = aggregate_runs(runs)
                # Save per-config JSON
                cfg_json = out_dir / f"{ex}_{kin}_{cfg_id}_summary.json"
                with open(cfg_json, 'w', encoding='utf-8') as f:
                    json.dump(aggr, f, indent=2)
                batch[ex][kin][cfg_id] = aggr

    md_path = save_summary(batch, out_dir)
    print(f"\nSummary saved to: {md_path}")


if __name__ == '__main__':
    main()
