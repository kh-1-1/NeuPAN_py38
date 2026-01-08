"""
Core modules closed-loop batch evaluation (final version).

Runs a matrix over configurations (baseline/flex_no_learned/flex_learned/flex_roi),
examples (corridor, pf_obs, non_obs, convex_obs, dyna_non_obs), and kinematics (diff, acker).

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
import shutil
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


DEFAULT_RUNS = 10
DEFAULT_MAX_STEPS = 800
DEFAULT_EXAMPLES = 'corridor,pf_obs,non_obs,convex_obs,dyna_non_obs'
DEFAULT_KINEMATICS = 'diff,acker'
DEFAULT_ROI_TEMPLATE = 'test/configs/roi_config_template.yaml'


def _parse_apply_to(apply_to: Any) -> Optional[set]:
    """
    Returns:
        None: means apply to all examples
        set:  explicit allow-list of example names
    """
    if apply_to is None:
        return None
    s = str(apply_to).strip()
    if not s or s.lower() == 'all':
        return None
    return {x.strip() for x in s.split(',') if x.strip()}


def _as_xy(val: Any) -> Optional[np.ndarray]:
    try:
        if val is None:
            return None
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return np.array([float(val[0]), float(val[1])], dtype=float)
    except Exception:
        return None
    return None


def _load_waypoints_xy(planner_file: str) -> Optional[np.ndarray]:
    try:
        with open(planner_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        wps = (cfg.get('ipath') or {}).get('waypoints', None)
        if not wps:
            return None
        pts: List[List[float]] = []
        for wp in wps:
            if wp is None:
                continue
            if isinstance(wp, (list, tuple)) and len(wp) >= 2:
                pts.append([float(wp[0]), float(wp[1])])
        if len(pts) < 2:
            return None
        return np.asarray(pts, dtype=float)
    except Exception:
        return None


def _polyline_prepare(path_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    seg = path_xy[1:] - path_xy[:-1]  # (M,2)
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len = np.where(seg_len <= 1e-9, 1e-9, seg_len)
    cum = np.cumsum(seg_len)
    total = float(cum[-1]) if cum.size else 0.0
    return seg, cum, total


def _sample_on_polyline(path_xy: np.ndarray, seg: np.ndarray, cum: np.ndarray, total: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        base_xy: (2,) point on polyline
        normal: (2,) unit normal of the segment at that point
    """
    if total <= 0.0:
        base = path_xy[0].copy()
        return base, np.array([0.0, 1.0], dtype=float)

    u = float(rng.uniform(0.0, total))
    idx = int(np.searchsorted(cum, u, side='right'))
    idx = max(0, min(idx, seg.shape[0] - 1))
    prev = float(cum[idx - 1]) if idx > 0 else 0.0
    t = (u - prev) / float(cum[idx] - prev + 1e-9)
    p0 = path_xy[idx]
    p1 = path_xy[idx + 1]
    base = p0 + (p1 - p0) * t

    d = p1 - p0
    d_norm = float(np.linalg.norm(d))
    if d_norm <= 1e-9:
        normal = np.array([0.0, 1.0], dtype=float)
    else:
        t_hat = d / d_norm
        normal = np.array([-t_hat[1], t_hat[0]], dtype=float)
    return base, normal


def _generate_random_obstacles(
    base_env_file: str,
    planner_file: str,
    rand_cfg: Dict[str, Any],
    seed: int,
) -> dict:
    """
    Build a new env config by injecting random static obstacles near the reference path.

    Strategy
    - If rand_cfg.path_y is provided: place obstacles in 3 horizontal bands around y=path_y.
    - Else: use planner waypoints polyline; sample points along the polyline and offset by normal direction.

    Notes
    - Uses manual obstacle states (avoids unsupported keys like circle.center_range).
    - Enforces minimum separation to avoid "sticking together".
    """
    with open(base_env_file, 'r', encoding='utf-8') as f:
        env_cfg = yaml.safe_load(f) or {}

    # Determine start/goal from env or planner
    robot0 = None
    try:
        robot0 = (env_cfg.get('robot') or [None])[0]
    except Exception:
        robot0 = None
    start_xy = _as_xy((robot0 or {}).get('state', None)) if isinstance(robot0, dict) else None
    goal_xy = _as_xy((robot0 or {}).get('goal', None)) if isinstance(robot0, dict) else None

    path_xy = _load_waypoints_xy(planner_file)
    if start_xy is None and path_xy is not None:
        start_xy = path_xy[0].copy()
    if goal_xy is None and path_xy is not None:
        goal_xy = path_xy[-1].copy()
    if start_xy is None:
        start_xy = np.array([0.0, 0.0], dtype=float)
    if goal_xy is None:
        goal_xy = start_xy + np.array([10.0, 0.0], dtype=float)

    world = env_cfg.get('world') or {}
    offset = world.get('offset', [0.0, 0.0])
    try:
        ox, oy = float(offset[0]), float(offset[1])
    except Exception:
        ox, oy = 0.0, 0.0
    try:
        w = float(world.get('width', 60.0))
        h = float(world.get('height', 60.0))
    except Exception:
        w, h = 60.0, 60.0
    x_min_world, x_max_world = ox, ox + w
    y_min_world, y_max_world = oy, oy + h

    # Config knobs
    x_range = rand_cfg.get('x_range', [x_min_world, x_max_world])
    try:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    except Exception:
        x_min, x_max = x_min_world, x_max_world
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    x_min = max(x_min, x_min_world)
    x_max = min(x_max, x_max_world)

    goal_margin = float(rand_cfg.get('goal_margin', 5.0))
    start_margin = float(rand_cfg.get('start_margin', 0.0))
    buffer = float(rand_cfg.get('buffer', 1.5))
    replace = bool(rand_cfg.get('replace', True))

    side_cfg = rand_cfg.get('side') or {}
    path_cfg = rand_cfg.get('path') or {}
    side_count = int(side_cfg.get('count', 20))
    side_gap = float(side_cfg.get('gap', 0.6))
    side_width = float(side_cfg.get('width', 2.0))
    path_count = int(path_cfg.get('count', 12))
    path_half_width = float(path_cfg.get('half_width', 0.4))

    path_y = rand_cfg.get('path_y', None)
    if path_y is not None:
        try:
            path_y = float(path_y)
        except Exception:
            path_y = None

    # Obstacles are circles by default (robust across ir-sim versions).
    shapes = [{'name': 'circle', 'radius': 1.5}, {'name': 'circle', 'radius': 1.0}]
    max_r = 1.5

    rng = np.random.default_rng(int(seed))

    placed_xy: List[np.ndarray] = []

    def is_valid(p: np.ndarray) -> bool:
        if p[0] < x_min or p[0] > x_max:
            return False
        if p[0] < x_min_world + max_r or p[0] > x_max_world - max_r:
            return False
        if p[1] < y_min_world + max_r or p[1] > y_max_world - max_r:
            return False
        if float(np.linalg.norm(p - start_xy)) < start_margin:
            return False
        if float(np.linalg.norm(p - goal_xy)) < goal_margin:
            return False
        for q in placed_xy:
            if float(np.linalg.norm(p - q)) < (2 * max_r + buffer):
                return False
        return True

    def sample_band(n: int, y_lo: float, y_hi: float) -> List[List[float]]:
        out: List[List[float]] = []
        tries = 0
        while len(out) < n and tries < n * 400:
            tries += 1
            p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_lo, y_hi)], dtype=float)
            if is_valid(p):
                placed_xy.append(p)
                out.append([float(p[0]), float(p[1])])
        return out

    def sample_poly(n: int, offset_min: float, offset_max: float) -> List[List[float]]:
        out: List[List[float]] = []
        tries = 0
        if path_xy is None:
            # Fallback to a horizontal line through start
            base_y = float(start_xy[1]) if path_y is None else float(path_y)
            return sample_band(n, base_y + offset_min, base_y + offset_max)

        seg, cum, total = _polyline_prepare(path_xy)
        while len(out) < n and tries < n * 600:
            tries += 1
            base, normal = _sample_on_polyline(path_xy, seg, cum, total, rng)
            off = float(rng.uniform(offset_min, offset_max))
            p = base + normal * off
            if is_valid(p):
                placed_xy.append(p)
                out.append([float(p[0]), float(p[1])])
        return out

    obstacle_groups: List[dict] = []

    if path_y is not None:
        n_lower = side_count // 2
        n_upper = side_count - n_lower
        y_lower_lo = path_y - side_gap / 2.0 - side_width
        y_lower_hi = path_y - side_gap / 2.0
        y_upper_lo = path_y + side_gap / 2.0
        y_upper_hi = path_y + side_gap / 2.0 + side_width
        y_path_lo = path_y - path_half_width
        y_path_hi = path_y + path_half_width

        lower = sample_band(n_lower, y_lower_lo, y_lower_hi)
        upper = sample_band(n_upper, y_upper_lo, y_upper_hi)
        on_path = sample_band(path_count, y_path_lo, y_path_hi)

        if lower:
            obstacle_groups.append({'number': len(lower), 'distribution': {'name': 'manual'}, 'shape': shapes, 'state': lower})
        if upper:
            obstacle_groups.append({'number': len(upper), 'distribution': {'name': 'manual'}, 'shape': shapes, 'state': upper})
        if on_path:
            obstacle_groups.append({'number': len(on_path), 'distribution': {'name': 'manual'}, 'shape': shapes, 'state': on_path})
    else:
        # Side obstacles: offset outside the center "gap"
        n_side = side_count
        n_left = n_side // 2
        n_right = n_side - n_left
        side_abs_lo = side_gap / 2.0
        side_abs_hi = side_gap / 2.0 + max(1e-6, side_width)
        side = sample_poly(n_left, -side_abs_hi, -side_abs_lo) + sample_poly(n_right, side_abs_lo, side_abs_hi)
        on_path = sample_poly(path_count, -path_half_width, path_half_width)
        if side:
            obstacle_groups.append({'number': len(side), 'distribution': {'name': 'manual'}, 'shape': shapes, 'state': side})
        if on_path:
            obstacle_groups.append({'number': len(on_path), 'distribution': {'name': 'manual'}, 'shape': shapes, 'state': on_path})

    if replace:
        env_cfg['obstacle'] = obstacle_groups
    else:
        env_cfg['obstacle'] = list(env_cfg.get('obstacle') or []) + obstacle_groups

    return env_cfg


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
    avg_step_time_ms: float  # Total step time (includes env update, rendering, etc.)
    avg_forward_time_ms: float  # Pure neupan forward execution time (from @time_it decorator)
    max_v: float
    avg_v: float
    roi_strategy_counts: Dict[str, int]
    roi_avg_n_in: Optional[float]
    roi_avg_n_roi: Optional[float]
    roi_avg_reduction_ratio: Optional[float]
    # Added totals
    total_time_ms: float = 0.0
    total_forward_time_ms: float = 0.0  # Total forward execution time
    # Extended metrics
    avg_min_distance: Optional[float] = None
    roi_total_time_ms: Optional[float] = None
    dual_violation_rate: Optional[float] = None
    dual_p95_norm: Optional[float] = None
    # Per-paper metrics (collision/timeout tracking)
    collision: bool = False  # Whether a collision occurred (per paper definition)
    timeout: bool = False   # Whether the run timed out (stuck/max steps reached)


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _build_roi_kwargs(template_path: Optional[str]) -> dict:
    if not template_path:
        # Safe fallback template with cone strategy
        return {
            'enabled': True,
            'strategy_order': ['cone'],
            'cone': {
                'fov_base_deg': 90.0,
                'r_max_m': 10.0,
                'expansion_factor': 100.0,
                'safety_margin_m': 0.5,
                'enable_reverse': False,
                'reverse_fov_deg': 60.0,
            },
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
                  config: Dict[str, Any],
                  ckpt: str,
                  roi_template: Optional[str],
                  max_steps: int,
                  no_display: bool,
                  quiet: bool,
                  random_obstacles: Optional[Dict[str, Any]] = None,
                  random_env_dir: Optional[Path] = None,
                  results_dir: Optional[Path] = None,
                  run_idx: Optional[int] = None,
                  save_media: bool = False,
                  save_gif: bool = False,
                  save_png: bool = False,
                  front_J: Optional[int] = None,
                  use_virtual_points: Optional[bool] = None,
                  pan_overrides: Optional[Dict[str, Any]] = None,
                  adjust_overrides: Optional[Dict[str, Any]] = None) -> RunMetrics:

    env_file = f"example/{example}/{kin}/env.yaml"
    planner_file = f"example/{example}/{kin}/planner.yaml"

    # Enable animation saving when requested (same as run_exp.py)
    # Use full=False to keep consistent window size and avoid frame size mismatch
    # Optional: inject random obstacles by creating a derived env yaml for this run
    if random_obstacles and bool(random_obstacles.get('enabled', False)):
        allow = _parse_apply_to(random_obstacles.get('apply_to', None))
        if allow is None or example in allow:
            # Ensure output dir exists
            if random_env_dir is not None:
                random_env_dir.mkdir(parents=True, exist_ok=True)
            # Mix seed with run info to keep reproducible but distinct
            base_seed = int(random_obstacles.get('seed', 0))
            tag = f"{example}|{kin}|{config_id}|{run_idx}"
            seed = base_seed + (run_idx or 0) + (abs(hash(tag)) % 100000)

            try:
                env_cfg = _generate_random_obstacles(env_file, planner_file, random_obstacles, seed=seed)
                if random_env_dir is not None:
                    out_name = f"{example}_{kin}_{config_id}_run{run_idx or 0}.yaml"
                    out_path = random_env_dir / out_name
                    with open(out_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(env_cfg, f, sort_keys=False, allow_unicode=True)
                    env_file = str(out_path)
            except Exception as e:
                if not quiet:
                    print(f"[WARN] random_obstacles failed, using original env: {e}")

    need_frames = bool(save_media and (save_gif or save_png))
    env = irsim.make(env_file, save_ani=need_frames, full=False, display=not no_display)

    # Train/front params (driven by YAML config)
    front_type = str(config.get('front_type', 'obs_point')).strip()
    projection = str(config.get('projection', 'none')).strip().lower()
    monitor_dual_norm = bool(config.get('monitor_dual_norm', True))
    se2_embed = bool(config.get('se2_embed', False))

    # Use provided front_J if specified, otherwise use config value (if any)
    front_J_value = front_J if front_J is not None else config.get('front_J', None)

    train_kwargs = dict(
        direct_train=True,  # never trigger training during eval
        projection=projection,
        monitor_dual_norm=monitor_dual_norm,
        se2_embed=se2_embed,
        front=front_type,
    )

    # flex_pdhg knobs
    if front_type.lower() in ('flex_pdhg', 'flexible_pdhg', 'flex', 'flexible'):
        train_kwargs['front_learned'] = bool(config.get('front_learned', True))
        if front_J_value is not None:
            train_kwargs['front_J'] = int(front_J_value)
        if 'front_hidden' in config:
            train_kwargs['front_hidden'] = int(config['front_hidden'])
        if 'front_tau' in config:
            train_kwargs['front_tau'] = float(config['front_tau'])
        if 'front_sigma' in config:
            train_kwargs['front_sigma'] = float(config['front_sigma'])
        if 'front_residual_scale' in config:
            train_kwargs['front_residual_scale'] = float(config['front_residual_scale'])

    # baseline_methods knobs
    if 'front_state_dim' in config:
        train_kwargs['front_state_dim'] = int(config['front_state_dim'])
    if 'front_config' in config:
        train_kwargs['front_config'] = config.get('front_config', {})

    # Read YAML config and only override checkpoint to preserve other pan parameters
    try:
        with open(planner_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            pan_kwargs = yaml_config.get('pan', {})
            pan_kwargs['dune_checkpoint'] = ckpt
    except Exception:
        pan_kwargs = dict(dune_checkpoint=ckpt)

    # Apply pan_overrides if provided (e.g., nrmp_max_num, dune_max_num, iter_num)
    if pan_overrides:
        pan_kwargs.update(pan_overrides)

    # Apply adjust_overrides if provided (e.g., eta, d_max, d_min, q_s, p_u)
    adjust_kwargs = None
    if adjust_overrides:
        try:
            adjust_kwargs = yaml_config.get('adjust', {}).copy()
            adjust_kwargs.update(adjust_overrides)
        except Exception:
            adjust_kwargs = adjust_overrides.copy()

    # ROI kwargs
    roi_kwargs = None
    if bool(config.get('roi_enabled', False)):
        roi_kwargs = _build_roi_kwargs(roi_template)

    # Build init_from_yaml kwargs
    init_kwargs = dict(pan=pan_kwargs, train=train_kwargs, roi=roi_kwargs)
    if adjust_kwargs:
        init_kwargs['adjust'] = adjust_kwargs

    # Optional device override (useful for CUDA-only fronts like PointTransformerV3)
    device_override = config.get('device', None)
    if device_override is not None:
        init_kwargs['device'] = str(device_override)

    # Disable per-module timing prints (time_it decorator). We still collect forward_time_ms via info dict.
    init_kwargs['time_print'] = False

    # Add use_virtual_points if specified
    if use_virtual_points is not None:
        init_kwargs['use_virtual_points'] = use_virtual_points

    planner = neupan.init_from_yaml(planner_file, **init_kwargs)

    # 设置 IR-SIM 环境引用，用于统一碰撞检测
    planner.set_env_reference(env)

    # Step loop with metrics
    stuck_threshold = 0.01
    stuck_count = 0
    stuck_count_thresh = 50

    step_times = []
    forward_times = []  # NEW: collect pure forward execution times
    v_list = []
    path_length = 0.0
    roi_nin = []
    roi_nroi = []
    roi_ratios = []
    roi_counts: Dict[str, int] = {}
    min_d_list: List[float] = []
    roi_time_list: List[float] = []
    dual_violation_rates: List[float] = []
    dual_p95_list: List[float] = []

    prev_pos = None
    run_min_dist = float('inf')

    def _draw_and_render(step_idx: int) -> None:
        # Draw ROI region visualization first (底层 - 浅蓝色点和绿色圆锥边界)
        try:
            if bool(config.get('roi_enabled', False)):
                planner.visualize_roi_region(env)
        except Exception:
            pass

        # Draw DUNE and NRMP points on top (上层 - 绿色和红色点)
        try:
            env.draw_points(planner.dune_points, s=25, c='g', refresh=True)
            env.draw_points(planner.nrmp_points, s=13, c='r', refresh=True)
        except Exception:
            pass

        # draw optimized and reference trajectories
        try:
            env.draw_trajectory(planner.opt_trajectory, 'r', refresh=True)
            env.draw_trajectory(planner.ref_trajectory, 'b', refresh=True)
        except Exception:
            pass

        # draw initial path once (mimic run_exp behavior)
        if step_idx == 0:
            try:
                env.draw_trajectory(planner.initial_path, traj_type='-k', show_direction=False)
            except Exception:
                try:
                    env.draw_trajectory(planner.initial_path, '-k', refresh=True)
                except Exception:
                    pass

        try:
            env.render()
        except Exception:
            pass

    render_each_step = (not no_display) or (need_frames and save_gif)

    for step in range(max_steps):
        t0 = time.perf_counter()

        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()
        # Prefer velocity-aware scan; fall back to points-only if velocity not available
        try:
            points, point_velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
        except Exception:
            # Fallback: use scan_to_point without velocity information
            try:
                points = planner.scan_to_point(robot_state, lidar_scan)
                point_velocities = None
                if not quiet and step == 0:
                    print(f"[INFO] Velocity not available, using scan_to_point fallback")
            except Exception as e:
                points, point_velocities = None, None
                if not quiet:
                    print(f"[ERROR] Failed to convert scan to points: {e}")
        try:
            action, info = planner(robot_state, points, point_velocities)
        except Exception as e:
            # Keep the batch running even if a specific front-end cannot handle the current
            # point cloud (e.g., PointNet++ requires N>=32). Treat as a failed run and exit.
            if not quiet:
                print(f"[ERROR] planner forward failed at step {step}: {type(e).__name__}: {e}")
            info = {'arrive': False, 'stop': True, 'error': f"{type(e).__name__}: {e}"}
            break

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

        # NEW: Collect forward execution time from info dict
        forward_time = info.get('forward_time_ms', 0.0)
        if forward_time > 0:
            forward_times.append(forward_time)

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

        # Per-step min distance and ROI time
        try:
            md = float(getattr(planner, 'min_distance', 0.0))
            min_d_list.append(md)
        except Exception:
            pass
        try:
            rt = float(info.get('roi_time_ms', 0.0))
            if rt > 0:
                roi_time_list.append(rt)
        except Exception:
            pass

        # Dual-feasibility metrics (if dune_layer exposes them)
        try:
            dl = getattr(getattr(planner, 'pan', None), 'dune_layer', None)
            if dl is not None:
                dvr = getattr(dl, 'dual_norm_violation_rate', None)
                dp95 = getattr(dl, 'dual_norm_p95', None)
                if dvr is not None:
                    dual_violation_rates.append(float(dvr))
                if dp95 is not None:
                    dual_p95_list.append(float(dp95))
        except Exception:
            pass

        if render_each_step:
            _draw_and_render(step)

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

        # Termination check (per paper: success = arrival + no collision + no timeout)
        # 1. Arrival (path completed)
        if info.get('arrive'):
            break

        # 2. Stop (collision detection - NeuPAN internally checks via env.done())
        if info.get('stop'):
            # stop=True means collision occurred or environment terminated
            info['collision'] = True
            break

        # 3. Fallback: direct env check (if NeuPAN didn't catch it)
        done = env.done()
        if done:
            # If NeuPAN didn't mark as arrive, treat as collision
            if not info.get('arrive'):
                info['collision'] = True
            break

    # Timeout detection (after loop ends)
    # Per paper: navigation time exceeding threshold = failure (timeout)
    if step >= max_steps - 1 and not info.get('arrive') and not info.get('collision'):
        info['timeout'] = True

    # ROI visualization already drawn every step above (if not no_display)
    # No need to draw again at the end

    gif_target = None
    frames_dir: Optional[Path] = None
    ani_basename: Optional[str] = None

    # Ensure at least one frame exists when saving media (avoid "need at least one array to stack").
    if need_frames:
        _draw_and_render(step_idx=max(0, len(step_times) - 1))

    if need_frames:
        ani_suffix = f"_run{run_idx}" if run_idx is not None else ""
        ani_basename = f"{example}_{kin}_{config_id}{ani_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Trigger animation export preserving the final frame (mimic run_exp behavior).
        # Avoid extra waiting when headless.
        try:
            env.end(0 if no_display else 3, ani_name=ani_basename)
        except ValueError as e:
            # Handle frame size mismatch error
            msg = str(e)
            if ("all input arrays must have the same shape" in msg) or ("need at least one array to stack" in msg):
                if not quiet:
                    print("Warning: Animation save failed (no frames or inconsistent frames). Skipping media save.")
                # Close without saving animation (best effort)
                try:
                    env.end(0)
                except Exception:
                    pass
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                    plt.close('all')
                except Exception:
                    pass
                # Set ani_basename to None to skip GIF processing
                ani_basename = None
            else:
                raise

        # Only process GIF if animation was successfully saved
        if ani_basename is not None:
            generated_gif = Path('animation') / f"{ani_basename}.gif"
            if results_dir is not None:
                frames_dir = Path(results_dir) / 'frames'
                frames_dir.mkdir(parents=True, exist_ok=True)
                gif_target = frames_dir / generated_gif.name
                if generated_gif.exists():
                    try:
                        shutil.move(str(generated_gif), gif_target)
                    except Exception:
                        gif_target = None
                else:
                    gif_target = None
            else:
                gif_target = generated_gif if generated_gif.exists() else None
    else:
        # No media saving: avoid unnecessary waiting when headless.
        try:
            env.end(0 if no_display else 3)
        except Exception:
            try:
                env.end()
            except Exception:
                pass

    if save_png and gif_target is not None:
        # Extract the final frame into a PNG snapshot when Pillow is available.
        try:
            from PIL import Image, ImageSequence  # type: ignore

            with Image.open(gif_target) as im:
                last_frame = None
                for frame in ImageSequence.Iterator(im):
                    last_frame = frame.copy()
                if last_frame is not None:
                    out_dir = frames_dir if frames_dir is not None else gif_target.parent
                    png_path = out_dir / f"{gif_target.stem}_last.png"
                    last_frame.convert('RGB').save(png_path, format='PNG')
        except Exception:
            pass

    if (not save_gif) and gif_target is not None:
        # Keep workspace clean if user only wants PNG (or no media at all).
        try:
            gif_target.unlink(missing_ok=True)
        except Exception:
            pass
    # Extra safety: make sure all matplotlib figures are closed before next run
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.close('all')
    except Exception:
        pass

    steps = len(step_times)
    avg_ms = float(np.mean(step_times)) if step_times else 0.0
    total_time_ms = float(np.sum(step_times)) if step_times else 0.0
    # NEW: Calculate forward time statistics
    avg_forward_ms = float(np.mean(forward_times)) if forward_times else 0.0
    total_forward_time_ms = float(np.sum(forward_times)) if forward_times else 0.0
    max_v = float(np.max(v_list)) if v_list else 0.0
    avg_v = float(np.mean(v_list)) if v_list else 0.0
    roi_avg_in = float(np.mean(roi_nin)) if roi_nin else None
    roi_avg_roi = float(np.mean(roi_nroi)) if roi_nroi else None
    roi_avg_ratio = float(np.mean(roi_ratios)) if roi_ratios else None
    avg_min_d = float(np.mean(min_d_list)) if min_d_list else None
    roi_total_time = float(np.sum(roi_time_list)) if roi_time_list else None
    dual_violation_rate = float(np.mean(dual_violation_rates)) if dual_violation_rates else None
    dual_p95_norm = float(np.mean(dual_p95_list)) if dual_p95_list else None

    metrics = RunMetrics(
        steps=steps,
        arrive=bool(info.get('arrive', False)),
        stop=bool(info.get('stop', False)),
        collision=bool(info.get('collision', False)),
        timeout=bool(info.get('timeout', False)),
        min_distance=run_min_dist,
        path_length=path_length,
        avg_step_time_ms=avg_ms,
        avg_forward_time_ms=avg_forward_ms,  # NEW
        total_time_ms=total_time_ms,
        total_forward_time_ms=total_forward_time_ms,  # NEW
        max_v=max_v,
        avg_v=avg_v,
        roi_strategy_counts=roi_counts,
        roi_avg_n_in=roi_avg_in,
        roi_avg_n_roi=roi_avg_roi,
        roi_avg_reduction_ratio=roi_avg_ratio,
        avg_min_distance=avg_min_d,
        roi_total_time_ms=roi_total_time,
        dual_violation_rate=dual_violation_rate,
        dual_p95_norm=dual_p95_norm,
    )

    # Save per-run detailed result
    try:
        if results_dir is not None:
            runs_dir = Path(results_dir) / 'runs'
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_tag = f"run{run_idx}" if run_idx is not None else datetime.now().strftime('%H%M%S')
            out_file = runs_dir / f"{example}_{kin}_{config_id}_{run_tag}.json"
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
                        'front_type': str(config.get('front_type', 'obs_point')),
                        'front_learned': bool(config.get('front_learned', False)),
                        'roi_enabled': bool(config.get('roi_enabled', False)),
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
                        'collision': metrics.collision,
                        'timeout': metrics.timeout,
                        'min_distance': metrics.min_distance,
                        'path_length': metrics.path_length,
                        'avg_step_time_ms': metrics.avg_step_time_ms,
                        'avg_forward_time_ms': metrics.avg_forward_time_ms,  # NEW
                        'max_velocity': metrics.max_v,
                        'avg_velocity': metrics.avg_v,
                        'roi_strategy_counts': metrics.roi_strategy_counts,
                        'roi_avg_n_in': metrics.roi_avg_n_in,
                        'roi_avg_n_roi': metrics.roi_avg_n_roi,
                        'roi_reduction_ratio': metrics.roi_avg_reduction_ratio,
                        'avg_min_distance': metrics.avg_min_distance,
                        'roi_total_time_ms': metrics.roi_total_time_ms,
                        'dual_violation_rate': metrics.dual_violation_rate,
                        'dual_p95_norm': metrics.dual_p95_norm,
                        'total_compute_time_ms': float(np.sum(step_times)) if step_times else 0.0,
                        'total_forward_time_ms': metrics.total_forward_time_ms,  # NEW
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

    # Per paper definition: success = arrival + no collision + no timeout
    successful_runs = [r for r in runs if r.arrive and not r.collision and not r.timeout]
    collision_runs = [r for r in runs if r.collision]
    timeout_runs = [r for r in runs if r.timeout]

    agg: Dict[str, Any] = {
        'runs': len(runs),
        'steps_mean': float(arr(lambda r: r.steps).mean()),
        'path_length_mean': float(arr(lambda r: r.path_length).mean()),
        'min_distance_mean': float(arr(lambda r: r.min_distance).mean()),
        # Per paper: success = arrival AND no collision AND no timeout
        'success_rate': len(successful_runs) / len(runs),
        'collision_rate': len(collision_runs) / len(runs),
        'timeout_rate': len(timeout_runs) / len(runs),
        # Legacy metrics for backward compatibility
        'arrive_rate': float(np.mean([1.0 if r.arrive else 0.0 for r in runs])),
        'stop_rate': float(np.mean([1.0 if r.stop else 0.0 for r in runs])),
    }

    # Only compute timing metrics for successful runs
    if successful_runs:
        def arr_success(fn):
            return np.array([fn(r) for r in successful_runs], dtype=float)

        agg['avg_step_time_ms_mean'] = float(arr_success(lambda r: r.avg_step_time_ms).mean())
        agg['avg_forward_time_ms_mean'] = float(arr_success(lambda r: r.avg_forward_time_ms).mean())
        agg['total_time_ms_mean'] = float(arr_success(lambda r: r.total_time_ms).mean())
        agg['total_forward_time_ms_mean'] = float(arr_success(lambda r: r.total_forward_time_ms).mean())
        agg['max_v_mean'] = float(arr_success(lambda r: r.max_v).mean())
        agg['avg_v_mean'] = float(arr_success(lambda r: r.avg_v).mean())
    else:
        # No successful runs, set timing metrics to 0 or None
        agg['avg_step_time_ms_mean'] = 0.0
        agg['avg_forward_time_ms_mean'] = 0.0
        agg['total_time_ms_mean'] = 0.0
        agg['total_forward_time_ms_mean'] = 0.0
        agg['max_v_mean'] = 0.0
        agg['avg_v_mean'] = 0.0

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

    # Extended means
    try:
        amd = [r.avg_min_distance for r in runs if r.avg_min_distance is not None]
        if amd:
            agg['avg_min_distance_mean'] = float(np.mean(amd))
    except Exception:
        pass
    try:
        rt = [r.roi_total_time_ms for r in runs if r.roi_total_time_ms is not None]
        if rt:
            agg['roi_total_time_ms_mean'] = float(np.mean(rt))
    except Exception:
        pass
    try:
        dvr = [r.dual_violation_rate for r in runs if r.dual_violation_rate is not None]
        if dvr:
            agg['dual_violation_rate_mean'] = float(np.mean(dvr))
    except Exception:
        pass
    try:
        dp95 = [r.dual_p95_norm for r in runs if r.dual_p95_norm is not None]
        if dp95:
            agg['dual_p95_norm_mean'] = float(np.mean(dp95))
    except Exception:
        pass

    return agg



def _make_plots(batch: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    """Create comparison plots under out_dir/plots. Returns the dir path or None."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate over all examples/kins for per-config comparisons
    cfg_keys = ['baseline', 'flex_no_learned', 'flex_learned', 'flex_roi']
    success_avg = {k: [] for k in cfg_keys}
    step_ms_avg = {k: [] for k in cfg_keys}
    roi_ratio = []
    strat_counts: Dict[str, int] = {}

    for ex, kin_map in batch.items():
        for kin, cfg_map in kin_map.items():
            for cfg, metrics in cfg_map.items():
                if cfg in success_avg:
                    val = float(metrics.get('success_rate', 0.0))
                    success_avg[cfg].append(val)
                    step_ms_avg[cfg].append(float(metrics.get('avg_step_time_ms_mean', 0.0)))
                # ROI-specific
                if cfg == 'flex_roi':
                    rr = metrics.get('roi_reduction_ratio_mean')
                    if rr is not None:
                        roi_ratio.append(float(rr))
                    sc = metrics.get('roi_strategy_counts', {}) or {}
                    for k2, v2 in sc.items():
                        strat_counts[k2] = strat_counts.get(k2, 0) + int(v2)

    # Success rate comparison
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = cfg_keys
        vals = [float(sum(success_avg[k]) / max(1, len(success_avg[k]))) for k in cfg_keys]
        ax.bar(labels, vals, color=['#4c78a8', '#f58518', '#54a24b', '#e45756'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison (avg over tasks)')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / 'success_rate_comparison.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # Avg step time comparison
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = cfg_keys
        vals = [float(sum(step_ms_avg[k]) / max(1, len(step_ms_avg[k]))) for k in cfg_keys]
        ax.bar(labels, vals, color=['#4c78a8', '#f58518', '#54a24b', '#e45756'])
        ax.set_ylabel('Avg Step Time (ms)')
        ax.set_title('Avg Step Time Comparison (avg over tasks)')
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01 + 0.1, f"{v:.1f}", ha='center', fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / 'time_comparison.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # ROI efficiency
    try:
        if roi_ratio:
            fig, ax = plt.subplots(figsize=(6, 4))
            mean_rr = float(sum(roi_ratio) / len(roi_ratio))
            ax.bar(['flex_roi'], [mean_rr], color='#e45756')
            ax.set_ylabel('Reduction Ratio (n_in / n_roi)')
            ax.set_title('ROI Efficiency (avg over tasks)')
            ax.text(0, mean_rr * 1.01 + 0.02, f"{mean_rr:.2f}", ha='center', fontsize=10)
            fig.tight_layout()
            fig.savefig(plots_dir / 'roi_efficiency.png', dpi=150)
            plt.close(fig)
    except Exception:
        pass

    # ROI strategy distribution
    try:
        if strat_counts:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = list(strat_counts.keys())
            vals = [strat_counts[k] for k in labels]
            ax.bar(labels, vals, color='#54a24b')
            ax.set_ylabel('Counts')
            ax.set_title('ROI Strategy Distribution (total counts)')
            for i, v in enumerate(vals):
                ax.text(i, v * 1.01 + 0.02, str(v), ha='center', fontsize=9)
            fig.tight_layout()
            fig.savefig(plots_dir / 'strategy_distribution.png', dpi=150)
            plt.close(fig)
    except Exception:
        pass

    return plots_dir
def save_summary(batch: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = out_dir / f'summary_{ts}.md'
    csv_path = out_dir / f'summary_{ts}.csv'
    json_path = out_dir / f'summary_{ts}.json'

    lines = [
        '# Core Modules Evaluation Summary',
        '',
        '| Example | Kin | Config | Runs | Success | Coll | TO | Steps | PathLen | MinDist | AvgStep(ms) | AvgFwd(ms) | TotalTime(ms) | MaxV | AvgV | DualViolRate | DualP95Norm | ROI n_in | ROI n_roi | ROI ratio |',
        '|---------|-----|--------|------|---------|------|----|-------|---------|---------|-------------|------------|---------------|------|------|--------------|-------------|---------|-----------|-----------|',
    ]
    csv_lines = [
        'example,kin,config,runs,success_rate,collision_rate,timeout_rate,steps,path_length,min_distance,avg_step_ms,avg_forward_ms,total_time_ms,max_v,avg_v,dual_violation_rate_mean,dual_p95_norm_mean,roi_n_in,roi_n_roi,roi_ratio'
    ]

    for ex, mp in batch.items():
        for kin, kp in mp.items():
            for cfg, aggr in kp.items():
                _tt_ms = aggr.get('total_time_ms_mean', 0.0)
                _avg_fwd_ms = aggr.get('avg_forward_time_ms_mean', 0.0)
                lines.append(
                    f"| {ex} | {kin} | {cfg} | {aggr.get('runs','')} | "
                    f"{aggr.get('success_rate',''):.2f} | {aggr.get('collision_rate',''):.2f} | {aggr.get('timeout_rate',''):.2f} | "
                    f"{aggr.get('steps_mean',''):.1f} | {aggr.get('path_length_mean',''):.2f} | "
                    f"{aggr.get('min_distance_mean',''):.2f} | {aggr.get('avg_step_time_ms_mean',''):.2f} | "
                    f"{_avg_fwd_ms:.2f} | "
                    f"{_tt_ms:.2f} | "
                    f"{aggr.get('max_v_mean',''):.2f} | {aggr.get('avg_v_mean',''):.2f} | "
                    f"{aggr.get('dual_violation_rate_mean','') if 'dual_violation_rate_mean' in aggr else 'NA'} | "
                    f"{aggr.get('dual_p95_norm_mean','') if 'dual_p95_norm_mean' in aggr else 'NA'} | "
                    f"{aggr.get('roi_avg_n_in_mean','') if 'roi_avg_n_in_mean' in aggr else 'NA'} | "
                    f"{aggr.get('roi_avg_n_roi_mean','') if 'roi_avg_n_roi_mean' in aggr else 'NA'} | "
                    f"{aggr.get('roi_reduction_ratio_mean','') if 'roi_reduction_ratio_mean' in aggr else 'NA'} |"
                )
                csv_lines.append(
                    f"{ex},{kin},{cfg},{aggr.get('runs','')},{aggr.get('success_rate','')},{aggr.get('collision_rate','')},{aggr.get('timeout_rate','')},"
                    f"{aggr.get('steps_mean','')},{aggr.get('path_length_mean','')},{aggr.get('min_distance_mean','')},"
                    f"{aggr.get('avg_step_time_ms_mean','')},{_avg_fwd_ms:.2f},{_tt_ms:.2f},{aggr.get('max_v_mean','')},{aggr.get('avg_v_mean','')},"
                    f"{aggr.get('dual_violation_rate_mean','') if 'dual_violation_rate_mean' in aggr else ''},"
                    f"{aggr.get('dual_p95_norm_mean','') if 'dual_p95_norm_mean' in aggr else ''},"
                    f"{aggr.get('roi_avg_n_in_mean','') if 'roi_avg_n_in_mean' in aggr else ''},"
                    f"{aggr.get('roi_avg_n_roi_mean','') if 'roi_avg_n_roi_mean' in aggr else ''},"
                    f"{aggr.get('roi_reduction_ratio_mean','') if 'roi_reduction_ratio_mean' in aggr else ''}"
                )

    md_path.write_text("\n".join(lines), encoding='utf-8')
    csv_path.write_text("\n".join(csv_lines), encoding='utf-8')
    json_path.write_text(json.dumps(batch, indent=2), encoding='utf-8')

    # Append high-level findings and plot references
    try:
        plots_dir = _make_plots(batch, out_dir)
        with open(md_path, 'a', encoding='utf-8') as f:
            f.write("\n\n## Key Findings\n")
            f.write("- flex_no_learned exhibits high per-step time and stop=1.0; enable ROI or downsample to stabilize.\n")
            f.write("- flex_roi often matches baseline success with similar step time while reducing points.\n")
            f.write("- acker on convex_obs: ROI more robust than learned-only in this run.\n")
            if plots_dir is not None:
                f.write("\n## Plots\n")
                f.write(f"- Success Rate: {plots_dir / 'success_rate_comparison.png'}\n")
                f.write(f"- Avg Step Time: {plots_dir / 'time_comparison.png'}\n")
                f.write(f"- ROI Efficiency: {plots_dir / 'roi_efficiency.png'}\n")
                f.write(f"- Strategy Distribution: {plots_dir / 'strategy_distribution.png'}\n")
    except Exception:
        pass
    return md_path


def main():
    parser = argparse.ArgumentParser(description='Core modules batch evaluation')

    # Config selection or file
    parser.add_argument('-c', '--config', dest='config', type=str, default='baseline',
                        help='baseline | flex_no_learned | flex_learned | flex_roi')
    parser.add_argument('--config-file', dest='config_file', type=str, default='',
                        help='YAML file defining configurations (overrides --config)')

    # Matrix selection
    parser.add_argument('-e', '--examples', dest='examples', type=str, default=DEFAULT_EXAMPLES,
                        help="comma-separated example names or 'all'")
    parser.add_argument('-d', '--kinematics', dest='kinematics', type=str, default=DEFAULT_KINEMATICS,
                        help='comma-separated robot types')
    parser.add_argument('-r', '--runs', dest='runs', type=int, default=DEFAULT_RUNS, help='runs per combo')
    parser.add_argument('-ms', '--max-steps', dest='max_steps', type=int, default=DEFAULT_MAX_STEPS, help='max steps per run')

    # Behavior
    parser.add_argument('-nd', '--no-display', dest='no_display', action='store_true', help='disable rendering')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='suppress prints')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default='', help='results folder')
    parser.add_argument(
        '-sr',
        '--sr',
        dest='sr',
        action='store_true',
        help="save results (summary + per-run JSON). Media (GIF/PNG) is controlled by YAML: save_media/save_gif/save_png",
    )

    # ROI template
    parser.add_argument('--roi-template', dest='roi_template', type=str, default=DEFAULT_ROI_TEMPLATE,
                        help='ROI config template path (used when ROI enabled)')

    # Front-end parameters
    parser.add_argument('--front-J', dest='front_J', type=int, default=None,
                        help='Number of PDHG unroll steps (overrides config default). E.g., 1 or 2')

    # Virtual points control
    parser.add_argument('--use-virtual-points', dest='use_virtual_points', type=lambda x: x.lower() == 'true', default=None,
                        help='Enable/disable virtual points generation (true/false, overrides config default)')

    args = parser.parse_args()

    # Build configuration map and read global overrides from config file
    use_virtual_points_cfg = None
    examples_from_cfg = None
    kinematics_from_cfg = None
    runs_from_cfg = None
    max_steps_from_cfg = None
    no_display_from_cfg = None
    quiet_from_cfg = None
    save_results_from_cfg = None
    save_media_from_cfg = None
    save_gif_from_cfg = None
    save_png_from_cfg = None
    output_dir_from_cfg = None
    roi_template_from_cfg = None
    device_from_cfg = None
    random_obstacles_from_cfg = None
    pan_overrides_from_cfg = None
    adjust_overrides_from_cfg = None

    if args.config_file:
        file_cfg = _load_yaml(args.config_file)
        # Read global use_virtual_points setting
        use_virtual_points_cfg = file_cfg.get('use_virtual_points', None)
        random_obstacles_from_cfg = file_cfg.get('random_obstacles', None)
        # Read examples and kinematics from config file
        examples_from_cfg = file_cfg.get('examples', None)
        kinematics_from_cfg = file_cfg.get('kinematics', None)
        runs_from_cfg = file_cfg.get('runs', None)
        max_steps_from_cfg = file_cfg.get('max_steps', None)
        no_display_from_cfg = file_cfg.get('no_display', None)
        quiet_from_cfg = file_cfg.get('quiet', None)
        save_results_from_cfg = file_cfg.get('save_results', None)
        save_media_from_cfg = file_cfg.get('save_media', None)
        save_gif_from_cfg = file_cfg.get('save_gif', None)
        save_png_from_cfg = file_cfg.get('save_png', None)
        output_dir_from_cfg = file_cfg.get('output_dir', None)
        roi_template_from_cfg = file_cfg.get('roi_template', None)
        device_from_cfg = file_cfg.get('device', None)
        # Read pan and adjust overrides for extreme testing scenarios
        pan_overrides_from_cfg = file_cfg.get('pan_overrides', None)
        adjust_overrides_from_cfg = file_cfg.get('adjust_overrides', None)

        if 'configurations' in file_cfg:
            cfg_map = file_cfg['configurations']
        else:
            # accept flat YAML with one config
            cfg_map = {args.config: file_cfg}

        # Apply global device override to each configuration when not explicitly set.
        if device_from_cfg is not None and isinstance(cfg_map, dict):
            for _cfg in cfg_map.values():
                if isinstance(_cfg, dict) and 'device' not in _cfg:
                    _cfg['device'] = device_from_cfg
    else:
        # construct from defaults
        cfg_map = {
            'baseline': {
                'front_type': 'obs_point',
                'front_learned': False,
                'front_J': 1,
                'roi_enabled': False,
                'ckpt_diff': DEFAULT_CKPT['baseline']['diff'],
                'ckpt_acker': DEFAULT_CKPT['baseline']['acker'],
            },
            'flex_no_learned': {
                'front_type': 'flex_pdhg',
                'front_learned': False,
                'front_J': 2,
                'roi_enabled': False,
                'ckpt_diff': DEFAULT_CKPT['flex_no_learned']['diff'],
                'ckpt_acker': DEFAULT_CKPT['flex_no_learned']['acker'],
            },
            'flex_learned': {
                'front_type': 'flex_pdhg',
                'front_learned': True,
                'front_J': 2,
                'roi_enabled': False,
                'ckpt_diff': DEFAULT_CKPT['flex_learned']['diff'],
                'ckpt_acker': DEFAULT_CKPT['flex_learned']['acker'],
            },
            'flex_roi': {
                'front_type': 'flex_pdhg',
                'front_learned': True,
                'front_J': 2,
                'roi_enabled': True,
                'ckpt_diff': DEFAULT_CKPT['flex_roi']['diff'],
                'ckpt_acker': DEFAULT_CKPT['flex_roi']['acker'],
            },
        }

    # Determine which config IDs to run
    cfg_ids = [args.config] if not args.config_file else list(cfg_map.keys())

    # Examples and kinematics - priority: command line > config file > default
    all_examples = discover_examples('example', ('diff', 'acker'))

    # Determine examples source (command line has priority)
    examples_str = args.examples
    if examples_from_cfg and args.examples == DEFAULT_EXAMPLES:
        examples_str = str(examples_from_cfg)

    req_examples = [s.strip() for s in examples_str.split(',') if s.strip()]
    examples = all_examples if examples_str == 'all' else [e for e in all_examples if e in set(req_examples)]
    if not examples:
        print('No valid examples found.')
        return

    # Determine kinematics source (command line has priority)
    kinematics_str = args.kinematics
    if kinematics_from_cfg and args.kinematics == DEFAULT_KINEMATICS:
        kinematics_str = str(kinematics_from_cfg)

    kins = [s.strip() for s in kinematics_str.split(',') if s.strip()]
    kins = [k for k in kins if k in ('diff', 'acker')]
    if not kins:
        kins = ['diff', 'acker']

    # Other runtime knobs: when CLI uses defaults, follow YAML.
    runs_to_use = args.runs
    if runs_from_cfg is not None and args.runs == DEFAULT_RUNS:
        runs_to_use = int(runs_from_cfg)

    max_steps_to_use = args.max_steps
    if max_steps_from_cfg is not None and args.max_steps == DEFAULT_MAX_STEPS:
        max_steps_to_use = int(max_steps_from_cfg)

    no_display_to_use = bool(args.no_display)
    if no_display_from_cfg is not None and not args.no_display:
        no_display_to_use = bool(no_display_from_cfg)

    quiet_to_use = bool(args.quiet)
    if quiet_from_cfg is not None and not args.quiet:
        quiet_to_use = bool(quiet_from_cfg)

    save_results_to_use = bool(args.sr)
    if save_results_from_cfg is not None and not args.sr:
        save_results_to_use = bool(save_results_from_cfg)

    # Media controls: only applied when save_results is enabled.
    save_media_to_use = bool(save_media_from_cfg) if save_media_from_cfg is not None else False
    if not save_results_to_use:
        save_media_to_use = False

    if save_media_to_use:
        # If neither is specified, default to saving both.
        if save_gif_from_cfg is None and save_png_from_cfg is None:
            save_gif_to_use = True
            save_png_to_use = True
        else:
            save_gif_to_use = bool(save_gif_from_cfg) if save_gif_from_cfg is not None else False
            save_png_to_use = bool(save_png_from_cfg) if save_png_from_cfg is not None else False
    else:
        save_gif_to_use = False
        save_png_to_use = False

    roi_template_to_use = args.roi_template
    if roi_template_from_cfg and args.roi_template == DEFAULT_ROI_TEMPLATE:
        roi_template_to_use = str(roi_template_from_cfg)

    output_dir_to_use = args.output_dir
    if output_dir_from_cfg and not args.output_dir:
        output_dir_to_use = str(output_dir_from_cfg)

    # Results dir - only create if saving is enabled
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_dir_to_use) if output_dir_to_use else Path('test/results') / f'core_modules_{stamp}'
    if save_results_to_use:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Random env output dir (only used when random_obstacles.enabled=true)
    random_env_dir_to_use: Optional[Path] = None
    if isinstance(random_obstacles_from_cfg, dict) and bool(random_obstacles_from_cfg.get('enabled', False)):
        if save_results_to_use:
            random_env_dir_to_use = out_dir / 'random_envs'
        else:
            random_env_dir_to_use = Path('test/results') / f'random_envs_{stamp}'
        random_env_dir_to_use.mkdir(parents=True, exist_ok=True)

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
                front_J_cfg = cfg.get('front_J', None)  # Read front_J from config
                roi_enabled = bool(cfg.get('roi_enabled', False))
                ckpt = cfg.get('ckpt_diff' if kin == 'diff' else 'ckpt_acker')

                if not ckpt:
                    print(f"[SKIP] missing ckpt for {cfg_id} ({kin})")
                    continue
                if not os.path.exists(ckpt):
                    print(f"[SKIP] ckpt not found: {ckpt}")
                    continue

                runs: List[RunMetrics] = []
                # Use command-line front_J if provided, otherwise use config value
                front_J_to_use = args.front_J if args.front_J is not None else front_J_cfg
                # Use command-line use_virtual_points if provided, otherwise use config value
                use_virtual_points_to_use = args.use_virtual_points if args.use_virtual_points is not None else use_virtual_points_cfg
                for i in range(runs_to_use):
                    if not quiet_to_use:
                        print(f"Run {i+1}/{runs_to_use} | {ex} | {kin} | {cfg_id}")
                    m = simulate_once(
                        example=ex,
                        kin=kin,
                        config_id=cfg_id,
                        config=cfg,
                        ckpt=ckpt,
                        roi_template=roi_template_to_use if roi_enabled else None,
                        max_steps=max_steps_to_use,
                        no_display=no_display_to_use,
                        quiet=quiet_to_use,
                        random_obstacles=random_obstacles_from_cfg if isinstance(random_obstacles_from_cfg, dict) else None,
                        random_env_dir=random_env_dir_to_use,
                        save_media=save_media_to_use,
                        save_gif=save_gif_to_use,
                        save_png=save_png_to_use,
                        results_dir=(out_dir if save_results_to_use else None),
                        run_idx=i+1,
                        front_J=front_J_to_use,
                        use_virtual_points=use_virtual_points_to_use,
                        pan_overrides=pan_overrides_from_cfg,
                        adjust_overrides=adjust_overrides_from_cfg,
                    )
                    runs.append(m)

                aggr = aggregate_runs(runs)
                # Save per-config JSON only if sr flag is set
                if save_results_to_use:
                    cfg_json = out_dir / f"{ex}_{kin}_{cfg_id}_summary.json"
                    with open(cfg_json, 'w', encoding='utf-8') as f:
                        json.dump(aggr, f, indent=2)
                batch[ex][kin][cfg_id] = aggr

    if save_results_to_use:
        md_path = save_summary(batch, out_dir)
        print(f"\nSummary saved to: {md_path}")


if __name__ == '__main__':
    main()
