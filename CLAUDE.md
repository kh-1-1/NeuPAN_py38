# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NeuPAN** (Neural Proximal Alternating-minimization Network) is an end-to-end, real-time, map-free robot motion planner. It directly maps obstacle points to control actions by solving an optimization problem with point-level collision avoidance constraints. The algorithm integrates learning-based (DUNE) and optimization-based (NRMP) techniques using a proximal alternating-minimization framework.

Published in IEEE Transactions on Robotics (T-RO 2025).

## Installation

```bash
pip install -e .                    # Install in development mode
pip install -e ".[irsim]"           # Include IR-SIM for simulation examples
```

Python 3.8+ required. Key dependencies: `cvxpylayers`, `torch>=2.0.1`, `ecos`, `clarabel`, `scipy<=1.13.0`.

## Running Examples

Examples use IR-SIM simulation and are run from the project root:

```bash
# Basic usage: python example/run_exp.py -e <scenario> -d <kinematics>
python example/run_exp.py -e non_obs -d acker
python example/run_exp.py -e corridor -d diff
python example/run_exp.py -e dyna_obs -d diff -v    # -v enables point velocities
python example/run_exp.py -e polygon_robot -d diff -vr  # -vr visualizes ROI region
```

Scenarios: `non_obs`, `convex_obs`, `corridor`, `dyna_obs`, `dyna_non_obs`, `pf`, `pf_obs`, `polygon_robot`, `reverse`.

Kinematics: `diff` (differential drive), `acker` (Ackermann steering)

## High-Level Architecture

### Two-Layer Optimization

NeuPAN alternates between two layers in each MPC step:

1. **DUNE (Distance Uncertainty Network Estimation)**: Neural network that predicts signed distance values (mu) and Lagrange multipliers (lambda) for obstacle points. Trained offline for specific robot geometries. The implementation supports two front-ends:
   - **Original DUNE**: PointNet-based architecture with hard projection for dual feasibility
   - **Flexible PDHG**: Unrolled PDHG-style network with learned residuals (see `flexible_pdhg.py`)
     - PDHG-J: Controls number of unrolled iterations (J=1, J=2)
     - With/without learned residuals for ablation studies
     - Optional KKT condition enforcement

2. **NRMP (Neural Risk-aware Motion Planning)**: Optimization layer using cvxpy with ECOS/CLARABEL solvers. Solves the MPC problem with collision avoidance constraints derived from DUNE outputs.

### Key Files and Their Roles

- **`neupan/neupan.py`**: Main entry point class with user-friendly interface (`NeuPAN.init_from_yaml()`)
- **`neupan/blocks/pan.py`**: Core PAN algorithm implementing the alternating minimization loop
- **`neupan/blocks/dune.py`**: DUNE layer - neural network for distance prediction with PointNet backbone
- **`neupan/blocks/flexible_pdhg.py`**: Alternative PDHG-style front-end for mu prediction with learned residuals
- **`neupan/blocks/nrmp.py`**: NRMP layer - optimization solver using cvxpylayers
- **`neupan/blocks/roi_selector.py`**: Region of Interest selector for efficient point filtering (reachability cone)
- **`neupan/blocks/initial_path.py`**: Initial path generation (Dubins/Reeds-Shepp/line)
- **`neupan/blocks/learned_prox.py`**: Learned proximal operators for DUNE
- **`neupan/blocks/obs_point_net.py`**: PointNet for obstacle feature extraction
- **`neupan/robot/robot.py`**: Robot kinematics and geometry definitions

### Data Flow

```
Lidar Scan → scan_to_point() → obs_points
                                  ↓
waypoints → initial_path → ref_trajectory → PAN.forward()
                                                  ↓
                                  iterations of:
                                    DUNE: points → mu, lambda
                                    NRMP: solve MPC with constraints
                                                  ↓
                                  opt_state, opt_u → control action
```

## Configuration

All parameters are configured via YAML files (see `example/*/planner.yaml`):

- **`robot`**: kinematics (`diff`/`acker`), vertices (geometry), max_speed, max_acce, wheelbase
- **`receding`**: MPC horizon steps (default: 10)
- **`step_time`**: MPC time step (default: 0.1s)
- **`ref_speed`**: Reference speed (default: 4.0 m/s)
- **`pan.iter_num`**: PAN iterations (default: 2)
- **`pan.dune_max_num`**: Max points for DUNE layer (default: 100)
- **`pan.nrmp_max_num`**: Max points for NRMP layer (default: 10)
- **`adjust`**: Cost weights (q_s, p_u, eta, d_max, d_min, ro_obs, bk, solver)
- **`train`**: DUNE training parameters (see Training section below)
- **`roi_selector`**: ROI filtering configuration (enabled, strategy, cone_fov, cone_r_max)
- **`collision_threshold`**: Distance threshold for collision detection (default: 0.1m)

Important: Set `device: "cpu"` - cvxpy solver is CPU-only.

## DUNE Model Training

Train a new DUNE model when changing robot geometry (size/shape):

```bash
# Training scripts are in example/dune_train/
python example/dune_train/<training_script>.py <config.yaml>
```

Key training parameters in YAML `train` section:
- `data_size`: training data size (default: 100000)
- `data_range`: [x_min, y_min, x_max, y_max] - must cover max obstacle range
- `epoch`: training epochs (default: 5000)
- `batch_size`: training batch size (default: 256)
- `lr`: learning rate (default: 5e-5)

Models are saved to `example/model/<robot_name>/` and only need to be trained once per robot geometry.

## Testing and Evaluation

Tests are executable scripts (not pytest). Run via `python -m test.<module>` or directly.

### Projection Tests (no IR-SIM required)
```bash
python -m test.test_projection_ab_synthetic   # Synthetic A/B test for hard projection
```

### Baseline Evaluation
```bash
python test/evaluate_dual_baselines.py --config test/configs/dual_baselines_eval.yaml
python test/evaluate_point_level_variants.py --config test/configs/point_level_variants_eval.yaml
```

### Performance Analysis
```bash
python test/benchmark_front_j.py     # Benchmark different J values for PDHG front
python test/analyze_performance.py   # Analyze evaluation results
```

### Paper Results Reproduction
```bash
python test/reproduce_table3.py      # Compare NeuPAN vs RDA_planner (IR-SIM)
python test/reproduce_table4.py      # Ablation studies and component analysis
```

Test outputs go to `test/results/`. Do not commit generated results.

## LON (Learning Optimization Network)

LON is an online learning framework that automatically tunes MPC parameters (`p_u`, `eta`, `d_max`) during runtime through gradient-based optimization. This is useful when default weights don't work well for a specific scenario.

```bash
# Run LON training to auto-tune MPC parameters
python example/LON/run_lon.py --config example/LON/lon_config.yaml

# Or specify components directly
python example/LON/run_lon.py -e example/corridor/acker/env.yaml -p example/corridor/acker/planner.yaml --epochs 50 --lr 0.005
```

LON optimizes these parameters via gradient descent on a loss function combining:
- State tracking error (MSE vs reference trajectory)
- Speed tracking error
- Distance penalty (collision avoidance)

The tuned parameters are printed at the end and can be copied to `planner.yaml`.

## Model Variants

Trained DUNE models are organized in `example/model/<variant>/` with descriptive names:

| Variant | Description |
|---------|-------------|
| `acker_robot_default` | Original DUNE (PointNet) for Ackermann |
| `diff_robot_default` | Original DUNE (PointNet) for differential drive |
| `acker_flex_pdhg-1_robot` | PDHG with J=1 iteration + learned residuals |
| `acker_flex_pdhg-2_robot` | PDHG with J=2 iterations + learned residuals |
| `acker_flex_pdhg-nolearned_robot` | PDHG without learned residuals (baseline) |
| `*_CLARABEL_robot` | Models using CLARABEL solver instead of ECOS |
| `*_kkt_robot` | Models with KKT condition enforcement |

Use the appropriate model for your robot kinematics and desired front-end.

## Important Implementation Notes

1. **CPU-based optimization**: cvxpylayers only runs on CPU. Use powerful CPU (Intel i7+) for real-time performance (>10 Hz).

2. **Robot geometry specificity**: DUNE models are tied to robot shape defined in `robot.vertices`. Must retrain when changing geometry.

3. **Dual feasibility**: Both DUNE front-ends enforce dual constraints (mu >= 0, ||G^T mu||_2 <= 1) via hard projection in `dune.py` or `flexible_pdhg.py`.

4. **Point velocities**: Optional support for dynamic obstacles via `point_velocities` parameter. Enable with `-v` flag in examples.

5. **ROI selector**: For efficiency, ROI selector filters points before DUNE processing using reachability cones. Enable via `roi_selector.enabled` in config. Uses reachability cone strategy with O(N) complexity. Configurable parameters include FOV, max distance, and reverse motion support.

6. **Initial path updates**: Use `set_initial_path()` or `update_initial_path_from_waypoints()` for dynamic path updates from external planners (A*, etc.).

7. **Reverse motion**: For Ackermann reverse, modify initial path direction (see `run_exp.py` lines 77-81`).

8. **Virtual points**: Can generate virtual points at max sensor range (10m) for sparse environments. Enable via `use_virtual_points` parameter in config.

9. **SolverError handling**: Optimization may become infeasible with extreme parameter values (e.g., `d_max` too large). LON handles this by resetting the environment when SolverError occurs.

10. **ROI visualization**: Use `-vr` flag in examples to visualize the ROI region for debugging cone filtering behavior.

## Code Architecture Tips

- When adding new robot kinematics: modify constraints in `nrmp.py` and kinematics in `robot.py`
- When tuning performance: adjust `receding`, `nrmp_max_num`, `dune_max_num`, `iter_num`
- For real-world deployment: integrate via ROS wrapper (see [neupan_ros](https://github.com/hanruihua/neupan_ros))
- Simulation environment reference can be set via `set_env_reference()` for unified collision detection

## Coding Style

- Python: 4-space indentation; match surrounding style (no enforced formatter)
- Names: `snake_case` for functions/files and YAML keys; avoid renaming public APIs
- Configuration-first: prefer adding/tuning options via YAML with backwards-compatible defaults
- Large artifacts (`*.pth`, `*.pt`, `*.pkl`) are tracked via Git LFS
- Treat `baseline_methods/` as vendored/third-party code
