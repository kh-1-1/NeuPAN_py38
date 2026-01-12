import numpy as np
import yaml
import torch
import json
import os
import sys

# Ensure project root is in python path
sys.path.append(os.getcwd())

from neupan.neupan import neupan
import irsim

def run_simulation(env_file, planner_file, config_overrides, label):
    print(f"Running simulation for: {label}")

    # Initialize environment
    env = irsim.make(env_file, save_ani=False, full=False, display=False)

    # Initialize planner
    # We need to handle overrides carefully.
    # neupan.init_from_yaml takes kwargs that override config

    train_kwargs = config_overrides.get('train', {})
    pan_kwargs = config_overrides.get('pan', {})

    # Load default planner config to get checkpoints if not overridden
    with open(planner_file, 'r') as f:
        default_planner_cfg = yaml.safe_load(f)

    # Ensure checkpoints exist
    if 'dune_checkpoint' not in pan_kwargs:
        # Default fallback logic
        if 'flex' in str(train_kwargs.get('front', '')):
             pan_kwargs['dune_checkpoint'] = 'example/model/acker_flex_pdhg_robot/model_5000.pth'
        else:
             pan_kwargs['dune_checkpoint'] = 'example/model/acker_CLARABEL_robot/model_5000.pth'

    planner = neupan.init_from_yaml(
        planner_file,
        train_kwargs=train_kwargs,
        pan_kwargs=pan_kwargs,
        time_print=False
    )
    planner.set_env_reference(env)

    trajectory = []

    max_steps = 200
    for i in range(max_steps):
        state = env.get_robot_state()
        trajectory.append(state.flatten().tolist()) # [x, y, theta, v]

        scan = env.get_lidar_scan()
        points = planner.scan_to_point(state, scan)

        action, info = planner(state, points)

        env.step(action)

        if info['arrive'] or info['stop'] or env.done():
            print(f"  Ended at step {i}")
            break

    return trajectory

def main():
    env_file = "example/convex_obs/acker/env.yaml"
    planner_file = "example/convex_obs/acker/planner.yaml"

    # Define configurations
    configs = [
        {
            "label": "Baseline (PointNet++)",
            "overrides": {
                "train": {"front": "obs_point", "front_learned": False}
            }
        },
        {
            "label": "PDPL-Net (J=1)",
            "overrides": {
                "train": {"front": "flex_pdhg", "front_learned": True, "front_J": 1}
            }
        },
        {
            "label": "PDPL-Net (J=10, Proxy GT)",
            "overrides": {
                "train": {"front": "flex_pdhg", "front_learned": True, "front_J": 10}
            }
        }
    ]

    results = {}

    # Extract Obstacles for plotting
    env = irsim.make(env_file, display=False)
    obstacles = []
    if hasattr(env, 'env_plot') and hasattr(env.env_plot, 'components'):
         # This is tricky with irsim's structure, let's try to parse yaml directly or use env properties
         pass

    # Parse env yaml for obstacles (simple approximation)
    with open(env_file, 'r') as f:
        env_cfg = yaml.safe_load(f)
    results['env_config'] = env_cfg

    for cfg in configs:
        traj = run_simulation(env_file, planner_file, cfg['overrides'], cfg['label'])
        results[cfg['label']] = traj

    output_path = "paper/figures/generated/trajectory_data.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Trajectory data saved to {output_path}")

if __name__ == "__main__":
    main()
