"""
NeuPAN vs RDA Comparison Script

Compares point-level NeuPAN against object-level RDA planner.
Supports multiple scenarios and command-line configuration.

Usage:
    # Single scenario
    python test/compare_neupan_vs_rda.py -e corridor -k acker -r 20

    # Multiple scenarios (batch)
    python test/compare_neupan_vs_rda.py -e corridor,convex_obs,dyna_obs -k acker -r 20

    # With config file
    python test/compare_neupan_vs_rda.py --config test/configs/rda_comparison_eval.yaml
"""

import argparse
import csv
import sys
import os
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml
import irsim
from collections import namedtuple

# Add RDA_planner to path
sys.path.append(os.path.join(os.getcwd(), 'RDA-planner'))

from neupan import neupan

try:
    from RDA_planner.mpc import MPC
    RDA_AVAILABLE = True
except ImportError:
    print("Warning: Could not import RDA_planner. RDA evaluation will be skipped.")
    print("Make sure RDA-planner submodule is initialized: git submodule update --init")
    MPC = None
    RDA_AVAILABLE = False

# Car tuple for RDA
Car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce dynamics')


def _draw_and_render(env, planner, step_idx: int, config: Dict[str, Any]) -> None:
    """Draw and render the visualization (from batch_core_modules_evaluation.py)"""
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


def run_neupan(
    env,
    planner_file: str,
    runs: int,
    max_steps: int,
    pan_overrides: Optional[Dict] = None,
    adjust_overrides: Optional[Dict] = None,
    neupan_cfg: Optional[Dict] = None,
    kinematics: str = 'acker',
    quiet: bool = False,
    no_display: bool = False,
    save_ani: bool = False,
    save_gif: bool = False,
    save_png: bool = False,
    results_dir: Optional[Path] = None,
    run_idx: Optional[int] = None,
) -> Tuple[int, int, float, float, float]:
    """Run NeuPAN evaluation with full visualization support."""
    if not quiet:
        print(f"\n>>> Running NeuPAN (Point-level) for {runs} runs...")

    # Build init kwargs with overrides
    init_kwargs = {'time_print': False}
    if pan_overrides:
        init_kwargs['pan'] = pan_overrides
    if adjust_overrides:
        init_kwargs['adjust'] = adjust_overrides

    # Apply neupan front-end config (checkpoint, front_type, etc.)
    if neupan_cfg:
        # Select checkpoint based on kinematics
        ckpt_key = f'ckpt_{kinematics}'
        if ckpt_key in neupan_cfg and neupan_cfg[ckpt_key]:
            if 'pan' not in init_kwargs:
                init_kwargs['pan'] = {}
            init_kwargs['pan']['dune_checkpoint'] = neupan_cfg[ckpt_key]

        # Apply front_type and other settings
        if 'front_type' in neupan_cfg:
            if 'pan' not in init_kwargs:
                init_kwargs['pan'] = {}
            init_kwargs['pan']['front'] = neupan_cfg['front_type']
        if 'front_J' in neupan_cfg:
            if 'pan' not in init_kwargs:
                init_kwargs['pan'] = {}
            init_kwargs['pan']['front_J'] = neupan_cfg['front_J']
        if 'front_learned' in neupan_cfg:
            if 'pan' not in init_kwargs:
                init_kwargs['pan'] = {}
            init_kwargs['pan']['front_learned'] = neupan_cfg['front_learned']
        if 'projection' in neupan_cfg:
            if 'pan' not in init_kwargs:
                init_kwargs['pan'] = {}
            init_kwargs['pan']['projection'] = neupan_cfg['projection']

    planner = neupan.init_from_yaml(planner_file, **init_kwargs)
    if not quiet:
        print(f"NeuPAN Config: nrmp_max_num={planner.pan.nrmp_layer.max_num}, "
              f"dune_max_num={planner.pan.dune_layer.max_num}, "
              f"iter_num={planner.pan.iter_num}")
    planner.set_env_reference(env)

    success_count = 0
    collision_count = 0
    timeout_count = 0
    all_times = []
    path_lengths = []

    need_frames = bool(save_ani and (save_gif or save_png))
    render_each_step = (not no_display) or (need_frames and save_gif)

    for i in range(runs):
        env.reset()
        planner.reset()

        # Re-init to ensure clean state
        planner = neupan.init_from_yaml(planner_file, **init_kwargs)
        planner.set_env_reference(env)

        run_times = []
        path_length = 0.0
        prev_pos = None

        for step in range(max_steps):
            start_t = time.perf_counter()

            robot_state = env.get_robot_state()
            lidar_scan = env.get_lidar_scan()

            try:
                points, velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
            except:
                points = planner.scan_to_point(robot_state, lidar_scan)
                velocities = None

            action, info = planner(robot_state, points, velocities)

            end_t = time.perf_counter()
            run_times.append((end_t - start_t) * 1000)

            # Track path length
            cur_pos = robot_state[:2]
            if prev_pos is not None:
                path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos

            # Render visualization
            if render_each_step:
                _draw_and_render(env, planner, step, neupan_cfg or {})

            env.step(action)

            if info.get('arrive', False):
                success_count += 1
                break

            if info.get('collision', False) or info.get('stop', False) or env.done():
                if not info.get('arrive', False):
                    collision_count += 1
                else:
                    success_count += 1
                break
        else:
            # Max steps reached without termination
            timeout_count += 1

        # Final render to ensure at least one frame exists
        if need_frames:
            try:
                _draw_and_render(env, planner, max(0, len(run_times) - 1), neupan_cfg or {})
            except Exception:
                pass

        # Handle animation saving
        gif_target = None
        ani_basename = None

        if need_frames:
            ani_suffix = f"_run{run_idx}" if run_idx is not None else ""
            ani_basename = f"neupan_{ani_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            try:
                env.end(0 if no_display else 3, ani_name=ani_basename)
            except ValueError as e:
                msg = str(e)
                if ("all input arrays must have the same shape" in msg) or ("need at least one array to stack" in msg):
                    if not quiet:
                        print("Warning: Animation save failed (no frames or inconsistent frames). Skipping media save.")
                    try:
                        env.end(0)
                    except Exception:
                        pass
                    try:
                        import matplotlib.pyplot as plt
                        plt.close('all')
                    except Exception:
                        pass
                    ani_basename = None
                else:
                    raise

            # Process GIF if animation was saved
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
            # No media saving: avoid unnecessary waiting
            try:
                env.end(0 if no_display else 3)
            except Exception:
                try:
                    env.end()
                except Exception:
                    pass

        # Extract PNG if requested
        if save_png and gif_target is not None:
            try:
                from PIL import Image, ImageSequence

                with Image.open(gif_target) as im:
                    last_frame = None
                    for frame in ImageSequence.Iterator(im):
                        last_frame = frame.copy()
                    if last_frame is not None:
                        out_dir = Path(results_dir) / 'frames' if results_dir else gif_target.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        png_path = out_dir / f"{gif_target.stem}_last.png"
                        last_frame.convert('RGB').save(png_path, format='PNG')
            except Exception:
                pass

        # Remove GIF if only PNG wanted
        if (not save_gif) and gif_target is not None:
            try:
                gif_target.unlink(missing_ok=True)
            except Exception:
                pass

        # Clean up matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass

        all_times.extend(run_times)
        path_lengths.append(path_length)

        if not quiet:
            sys.stdout.write('.')
            sys.stdout.flush()

    avg_time = np.mean(all_times) if all_times else 0.0
    std_time = np.std(all_times) if all_times else 0.0
    avg_path = np.mean(path_lengths) if path_lengths else 0.0

    return success_count, collision_count, avg_time, std_time, avg_path


def run_rda(
    env,
    planner_file: str,
    runs: int,
    max_steps: int,
    kinematics: str = 'diff',
    quiet: bool = False,
    no_display: bool = False,
) -> Tuple[int, int, float, float, float]:
    """Run RDA evaluation with visualization support."""
    if not quiet:
        print(f"\n>>> Running RDA (Object-level) for {runs} runs...")

    if not RDA_AVAILABLE:
        print("RDA_planner module not available. Skipping.")
        return 0, 0, 0.0, 0.0, 0.0

    # Load waypoints from planner config
    with open(planner_file, 'r') as f:
        cfg = yaml.safe_load(f)
    waypoints = np.array(cfg['ipath']['waypoints'])

    # Generate dense reference path for RDA
    ref_path_list = []
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = np.linalg.norm(p1[:2] - p2[:2])
        num = max(1, int(dist / 0.1))
        for j in range(num):
            alpha = j / num
            p = p1 * (1 - alpha) + p2 * alpha
            ref_path_list.append(p.reshape(3, 1))
    ref_path_list.append(waypoints[-1].reshape(3, 1))

    # Get robot info from environment
    robot_info = env.get_robot_info()
    dynamics = 'acker' if kinematics == 'acker' else 'diff'
    car_tuple = Car(
        robot_info.G, robot_info.h, robot_info.cone_type,
        robot_info.wheelbase, [8, 3], [8, 3], dynamics
    )

    # RDA MPC configuration
    mpc_opt = MPC(
        car_tuple, ref_path_list,
        receding=10,
        sample_time=env.step_time,
        process_num=1,
        iter_num=2,
        max_edge_num=4,
        max_obs_num=30,
        min_sd=0.1,
        wu=0.2,
        ws=1.0,
        obstacle_order=True
    )

    success_count = 0
    collision_count = 0
    timeout_count = 0
    all_times = []
    path_lengths = []

    render_each_step = not no_display

    for i in range(runs):
        env.reset()
        mpc_opt.reset()

        run_times = []
        path_length = 0.0
        prev_pos = None

        for step in range(max_steps):
            start_t = time.perf_counter()

            # RDA uses ground truth obstacle list
            obs_list = env.get_obstacle_info_list()

            try:
                state = env.robot.state
                if state.ndim == 1:
                    state = state.reshape(3, 1)
                opt_vel, info = mpc_opt.control(state, 4.0, obs_list)
                action = opt_vel
            except Exception as e:
                action = np.zeros((2, 1))

            end_t = time.perf_counter()
            run_times.append((end_t - start_t) * 1000)

            # Track path length
            cur_pos = env.robot.state[:2].flatten()
            if prev_pos is not None:
                path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos

            # Render visualization
            if render_each_step:
                try:
                    env.render()
                except Exception:
                    pass

            env.step(action)

            if env.done():
                if env.robot.arrive_flag:
                    success_count += 1
                elif env.robot.collision_flag:
                    collision_count += 1
                break

            if info.get('arrive', False):
                success_count += 1
                break
        else:
            timeout_count += 1

        # Clean up environment
        try:
            env.end(0 if no_display else 3)
        except Exception:
            try:
                env.end()
            except Exception:
                pass

        all_times.extend(run_times)
        path_lengths.append(path_length)

        if not quiet:
            sys.stdout.write('.')
            sys.stdout.flush()

    avg_time = np.mean(all_times) if all_times else 0.0
    std_time = np.std(all_times) if all_times else 0.0
    avg_path = np.mean(path_lengths) if path_lengths else 0.0

    return success_count, collision_count, avg_time, std_time, avg_path


def evaluate_scenario(
    example: str,
    kinematics: str,
    runs: int,
    max_steps: int,
    pan_overrides: Optional[Dict] = None,
    adjust_overrides: Optional[Dict] = None,
    neupan_cfg: Optional[Dict] = None,
    display: bool = True,
    quiet: bool = False,
    save_media: bool = False,
    save_gif: bool = False,
    save_png: bool = False,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate both NeuPAN and RDA on a single scenario."""
    planner_file = f"example/{example}/{kinematics}/planner.yaml"
    env_file = f"example/{example}/{kinematics}/env.yaml"

    if not os.path.exists(planner_file) or not os.path.exists(env_file):
        print(f"Skipping {example}/{kinematics}: files not found")
        return {}

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Scenario: {example}/{kinematics}")
        print(f"Runs: {runs}, Max Steps: {max_steps}")
        print(f"{'='*60}")

    no_display = not display

    # Run NeuPAN (create new environment)
    need_frames = bool(save_media and (save_gif or save_png))
    env_neupan = irsim.make(env_file, save_ani=need_frames, full=False, display=display)
    n_succ, n_coll, n_time_avg, n_time_std, n_path = run_neupan(
        env_neupan, planner_file, runs, max_steps, pan_overrides, adjust_overrides,
        neupan_cfg, kinematics, quiet, no_display,
        save_ani=save_media, save_gif=save_gif, save_png=save_png,
        results_dir=results_dir, run_idx=1
    )

    # Close NeuPAN environment
    try:
        env_neupan.end(0)
    except:
        pass

    # Clean up matplotlib before creating new environment
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    # Run RDA (create fresh environment)
    env_rda = irsim.make(env_file, save_ani=False, full=False, display=display)
    r_succ, r_coll, r_time_avg, r_time_std, r_path = run_rda(
        env_rda, planner_file, runs, max_steps, kinematics, quiet, no_display
    )

    # Close RDA environment
    try:
        env_rda.end(0)
    except:
        pass

    # Final matplotlib cleanup
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    result = {
        'example': example,
        'kinematics': kinematics,
        'runs': runs,
        'neupan_success_rate': n_succ / runs if runs > 0 else 0,
        'neupan_collision_rate': n_coll / runs if runs > 0 else 0,
        'neupan_avg_time_ms': n_time_avg,
        'neupan_std_time_ms': n_time_std,
        'neupan_avg_path_length': n_path,
        'rda_success_rate': r_succ / runs if runs > 0 else 0,
        'rda_collision_rate': r_coll / runs if runs > 0 else 0,
        'rda_avg_time_ms': r_time_avg,
        'rda_std_time_ms': r_time_std,
        'rda_avg_path_length': r_path,
    }

    if not quiet:
        print(f"\n{'-'*60}")
        print(f"{'Metric':<20} | {'NeuPAN (Ours)':<20} | {'RDA (Baseline)':<20}")
        print(f"{'-'*60}")
        print(f"{'Success Rate':<20} | {n_succ/runs*100:>19.1f}% | {r_succ/runs*100:>19.1f}%")
        print(f"{'Collision Rate':<20} | {n_coll/runs*100:>19.1f}% | {r_coll/runs*100:>19.1f}%")
        print(f"{'Avg Time (ms)':<20} | {n_time_avg:>19.2f}  | {r_time_avg:>19.2f}")
        print(f"{'Std Time (ms)':<20} | {n_time_std:>19.2f}  | {r_time_std:>19.2f}")
        print(f"{'Avg Path Length':<20} | {n_path:>19.2f}  | {r_path:>19.2f}")

    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description='NeuPAN vs RDA Comparison')
    parser.add_argument('--config', type=str, default='',
                        help='YAML config file path (e.g., test/configs/rda_comparison_eval.yaml)')
    parser.add_argument('-e', '--examples', type=str, default='',
                        help='Comma-separated scenario names (overrides config)')
    parser.add_argument('-k', '--kinematics', type=str, default='',
                        help='Robot kinematics: diff or acker (overrides config)')
    parser.add_argument('-r', '--runs', type=int, default=None,
                        help='Number of runs per scenario (overrides config)')
    parser.add_argument('-ms', '--max-steps', type=int, default=None,
                        help='Maximum steps per run (overrides config)')
    parser.add_argument('--iter-num', type=int, default=None,
                        help='Override iter_num in pan config')
    parser.add_argument('--d-min', type=float, default=None,
                        help='Override d_min in adjust config')
    parser.add_argument('--nrmp-max-num', type=int, default=None,
                        help='Override nrmp_max_num in pan config')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('-nd', '--no-display', action='store_true',
                        help='Disable visualization display')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='Output CSV path (overrides config)')
    parser.add_argument('--save-media', action='store_true',
                        help='Save animation media (GIF/PNG)')
    parser.add_argument('--save-gif', action='store_true',
                        help='Save GIF animation')
    parser.add_argument('--save-png', action='store_true',
                        help='Save PNG snapshot')
    args = parser.parse_args()

    # Load config file if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    # Determine parameters: CLI > config > defaults
    examples_str = args.examples or cfg.get('examples', 'corridor,convex_obs,dyna_obs')
    examples = [e.strip() for e in examples_str.split(',') if e.strip()]

    kinematics = args.kinematics or cfg.get('kinematics', 'acker')
    runs = args.runs if args.runs is not None else cfg.get('runs', 20)
    max_steps = args.max_steps if args.max_steps is not None else cfg.get('max_steps', 600)
    quiet = args.quiet or cfg.get('quiet', False)
    display = not args.no_display if args.no_display else cfg.get('display', True)
    output = args.output or cfg.get('output_dir', '')

    # Media saving options
    save_media = args.save_media or cfg.get('save_media', False)
    save_gif = args.save_gif or cfg.get('save_gif', False)
    save_png = args.save_png or cfg.get('save_png', False)

    # Build overrides from config + CLI
    pan_overrides = dict(cfg.get('pan_overrides', {}) or {})
    if args.iter_num is not None:
        pan_overrides['iter_num'] = args.iter_num
    if args.nrmp_max_num is not None:
        pan_overrides['nrmp_max_num'] = args.nrmp_max_num

    adjust_overrides = dict(cfg.get('adjust_overrides', {}) or {})
    if args.d_min is not None:
        adjust_overrides['d_min'] = args.d_min

    # Load neupan front-end config
    neupan_cfg = cfg.get('neupan', {}) or {}

    # Setup results directory
    results_dir = None
    if save_media or output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output:
            if os.path.isdir(output) or output.endswith('/') or output.endswith('\\'):
                results_dir = Path(output)
                results_dir.mkdir(parents=True, exist_ok=True)
            elif os.path.basename(output):
                results_dir = Path(output).parent
        if results_dir is None:
            results_dir = Path('test/results/rda_comparison')
        results_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("=" * 60)
        print("NeuPAN vs RDA Comparison")
        print("=" * 60)
        if args.config:
            print(f"Config: {args.config}")
        print(f"Examples: {examples}")
        print(f"Kinematics: {kinematics}")
        print(f"Runs: {runs}, Max Steps: {max_steps}")
        print(f"Display: {display}")
        if save_media:
            print(f"Save Media: GIF={save_gif}, PNG={save_png}")
        if pan_overrides:
            print(f"PAN Overrides: {pan_overrides}")
        if adjust_overrides:
            print(f"Adjust Overrides: {adjust_overrides}")
        if neupan_cfg:
            print(f"NeuPAN Front: {neupan_cfg.get('front_type', 'obs_point')}, "
                  f"J={neupan_cfg.get('front_J', 1)}, "
                  f"ckpt={neupan_cfg.get(f'ckpt_{kinematics}', 'default')}")

    results = []
    for example in examples:
        result = evaluate_scenario(
            example=example,
            kinematics=kinematics,
            runs=runs,
            max_steps=max_steps,
            pan_overrides=pan_overrides if pan_overrides else None,
            adjust_overrides=adjust_overrides if adjust_overrides else None,
            neupan_cfg=neupan_cfg if neupan_cfg else None,
            display=display,
            quiet=quiet,
            save_media=save_media,
            save_gif=save_gif,
            save_png=save_png,
            results_dir=results_dir,
        )
        if result:
            results.append(result)

    # Save results
    if results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output and not os.path.isdir(output) and not output.endswith('/') and not output.endswith('\\'):
            output_path = output
        else:
            if results_dir:
                output_path = results_dir / f'neupan_vs_rda_{timestamp}.csv'
            else:
                output_path = f'test/results/neupan_vs_rda_{timestamp}.csv'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")

        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Scenario':<20} | {'NeuPAN Success':<15} | {'RDA Success':<15} | {'NeuPAN Time':<12} | {'RDA Time':<12}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['example']:<20} | {r['neupan_success_rate']*100:>13.1f}% | {r['rda_success_rate']*100:>13.1f}% | {r['neupan_avg_time_ms']:>10.2f}ms | {r['rda_avg_time_ms']:>10.2f}ms")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
