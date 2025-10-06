"""
Shared utilities for batch evaluation scripts.
Provides reusable components for projection/unroll evaluations.

Extracted from batch_projection_evaluation.py to enable code reuse
across multiple evaluation scripts (projection, unroll, combined).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch
import irsim
from neupan import neupan


class MetricsEvaluator:
    """Collect per-step dual-feasibility metrics and aggregate a summary."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: dict with evaluation configuration (projection, unroll_J, etc.)
        """
        self.config = config
        self.step_records: List[Dict[str, Any]] = []
        self.timing_records: List[Dict[str, float]] = []  # 时延记录
        self.profile_timing = config.get('profile_timing', False)
        self.timing_warmup = config.get('timing_warmup', 3)

    def record_step(self, step: int, dune_layer, lam_list, timing: Dict[str, float] = None):
        """
        Record metrics from a single planning step.

        Args:
            step: int, current step number
            dune_layer: DUNE instance (or None if no_obs)
            lam_list: list of lambda tensors from PAN
            timing: dict with timing info (optional)
        """
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

        # 记录时延（跳过预热步）
        if self.profile_timing and timing and step >= self.timing_warmup:
            self.timing_records.append(timing)

    def summary(self) -> Dict[str, Any]:
        """Aggregate metrics across all recorded steps."""
        if not self.step_records:
            return {}

        pre_viol = [r['pre_violation_rate'] for r in self.step_records if r['pre_violation_rate'] is not None]
        pre_p95s = [r['pre_p95'] for r in self.step_records if r['pre_p95'] is not None]
        pre_excs = [r['pre_excess'] for r in self.step_records if r['pre_excess'] is not None]
        post_excs = [r['post_excess'] for r in self.step_records if r['post_excess'] is not None]
        post_maxs = [r['post_max_via_lam'] for r in self.step_records if not np.isnan(r['post_max_via_lam'])]

        result = {
            'config': self.config,
            'steps_executed': len(self.step_records),
            'avg_pre_violation_rate': float(np.mean(pre_viol)) if pre_viol else 0.0,
            'avg_pre_p95': float(np.mean(pre_p95s)) if pre_p95s else 0.0,
            'avg_pre_excess': float(np.mean(pre_excs)) if pre_excs else 0.0,
            'avg_post_excess': float(np.mean(post_excs)) if post_excs else 0.0,
            'avg_post_max': float(np.mean(post_maxs)) if post_maxs else 0.0,
            'max_post_max': float(np.max(post_maxs)) if post_maxs else 0.0,
        }

        # 添加时延统计
        if self.timing_records:
            for key in ['total', 'dune', 'pdhg', 'nrmp', 'mpc']:
                times = [t.get(key, 0) for t in self.timing_records if key in t]
                if times:
                    result[f'avg_{key}_time_ms'] = float(np.mean(times)) * 1000  # 转换为毫秒
                    result[f'std_{key}_time_ms'] = float(np.std(times)) * 1000

        return result


def simulate(
    example_name: str,
    kinematics: str,
    planner_config: Dict[str, Any],
    max_steps: int,
    no_display: bool,
    save_results: bool,
    results_dir: Optional[Path] = None,
    save_animation: bool = False,
    full: bool = False,
    point_vel: bool = False,
    reverse: bool = False,
) -> Dict[str, Any]:
    """
    Generic simulation runner (supports projection/unroll/combined configs).
    
    Args:
        example_name: str, example directory name (e.g., 'corridor')
        kinematics: str, 'diff' or 'acker'
        planner_config: dict with keys:
            - 'projection': str ('hard'|'learned'|'none')
            - 'unroll_J': int (0|1|2|3)
            - 'pdhg_tau': float (optional)
            - 'pdhg_sigma': float (optional)
            - 'pdhg_learnable': bool (optional)
            - 'ckpt_override': str (optional checkpoint path)
        max_steps: int, maximum simulation steps
        no_display: bool, disable visualization
        save_results: bool, save per-run JSON
        results_dir: Optional[Path], output directory
        save_animation: bool, save animation
        full: bool, full rendering mode
        point_vel: bool, use point velocities
        reverse: bool, reverse mode (for 'reverse' example)
        
    Returns:
        dict with summary metrics
    """
    env_file = f"example/{example_name}/{kinematics}/env.yaml"
    planner_file = f"example/{example_name}/{kinematics}/planner.yaml"

    env = irsim.make(env_file, save_ani=save_animation, full=full, display=not no_display)

    # Build pan_kwargs from planner_config
    pan_kwargs = {}
    if planner_config.get('ckpt_override'):
        pan_kwargs['dune_checkpoint'] = planner_config['ckpt_override']

    # Build train_kwargs from planner_config
    train_kwargs = {
        'projection': planner_config.get('projection', 'hard'),
        'monitor_dual_norm': True,
        'direct_train': True,
        'unroll_J': planner_config.get('unroll_J', 0),
    }
    if 'pdhg_tau' in planner_config:
        train_kwargs['pdhg_tau'] = planner_config['pdhg_tau']
    if 'pdhg_sigma' in planner_config:
        train_kwargs['pdhg_sigma'] = planner_config['pdhg_sigma']
    if 'pdhg_learnable' in planner_config:
        train_kwargs['pdhg_learnable'] = planner_config['pdhg_learnable']

    planner = neupan.init_from_yaml(
        planner_file,
        pan=pan_kwargs if pan_kwargs else None,
        train=train_kwargs,
    )

    evaluator = MetricsEvaluator(planner_config)
    profile_timing = planner_config.get('profile_timing', False)

    for step in range(max_steps):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()

        if point_vel:
            points, point_velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
        else:
            points = planner.scan_to_point(robot_state, lidar_scan)
            point_velocities = None

        # 时延分析
        timing = {}
        if profile_timing:
            import time
            t_start = time.perf_counter()

        action, info = planner(robot_state, points, point_velocities)

        if profile_timing:
            timing['total'] = time.perf_counter() - t_start
            # 尝试获取子模块时延（如果 planner 记录了）
            if hasattr(planner, 'last_timing'):
                timing.update(planner.last_timing)

        dune_layer = getattr(planner.pan, 'dune_layer', None)
        lam_list = planner.pan.current_nom_values[3] if planner.pan.current_nom_values else []
        evaluator.record_step(step, dune_layer, lam_list, timing if profile_timing else None)

        if not no_display:
            env.draw_points(planner.dune_points, s=25, c='g', refresh=True)
            env.draw_points(planner.nrmp_points, s=13, c='r', refresh=True)
            env.draw_trajectory(planner.opt_trajectory, 'r', refresh=True)
            env.draw_trajectory(planner.ref_trajectory, 'b', refresh=True)
            env.render()

        env.step(action)
        if info.get('arrive') or info.get('stop') or env.done():
            break

    env.end(0)
    report = evaluator.summary()

    if save_results:
        out_dir = results_dir if results_dir is not None else Path("test/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        proj = planner_config.get('projection', 'none')
        unroll_j = planner_config.get('unroll_J', 0)
        result_file = out_dir / f"{example_name}_{kinematics}_{proj}_J{unroll_j}_{ts}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': planner_config,
                'summary': report,
            }, f, indent=2)

    return report


def safe_simulate(
    example: str,
    kin: str,
    planner_config: Dict[str, Any],
    max_steps: int,
    no_display: bool,
    save_results: bool,
    quiet: bool,
    results_dir: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Wrapper that catches exceptions and returns (success, report, err) tuple.
    
    Args:
        example: example name
        kin: kinematics ('diff' or 'acker')
        planner_config: planner configuration dict
        max_steps: max simulation steps
        no_display: disable display
        save_results: save JSON results
        quiet: suppress stdout
        results_dir: output directory
        
    Returns:
        (success: bool, report: dict, error_msg: str)
    """
    try:
        if quiet:
            f = StringIO()
            with redirect_stdout(f):
                rep = simulate(
                    example_name=example,
                    kinematics=kin,
                    planner_config=planner_config,
                    max_steps=max_steps,
                    no_display=no_display,
                    save_results=save_results,
                    results_dir=results_dir,
                    save_animation=False,
                    full=False,
                    point_vel=False,
                    reverse=(example == "reverse" and kin == "diff"),
                )
        else:
            rep = simulate(
                example_name=example,
                kinematics=kin,
                planner_config=planner_config,
                max_steps=max_steps,
                no_display=no_display,
                save_results=save_results,
                results_dir=results_dir,
                save_animation=False,
                full=False,
                point_vel=False,
                reverse=(example == "reverse" and kin == "diff"),
            )
        return True, rep, ""
    except Exception as e:
        return False, {}, str(e)


def aggregate_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple run reports into mean/std statistics.
    
    Args:
        reports: list of summary dicts from simulate()
        
    Returns:
        dict with aggregated metrics (mean, std, rates)
    """
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
    
    out['num_runs'] = len(reports)
    return out

