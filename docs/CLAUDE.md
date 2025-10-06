# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuPAN (Neural Proximal Alternating-minimization Network) is an end-to-end, real-time, map-free robot motion planner that integrates learning-based and optimization-based techniques. It directly maps obstacle points data to control actions by solving an end-to-end mathematical model with point-level collision avoidance constraints.

## Key Architecture Components

### Core Modules
- **`neupan/neupan.py`**: Main class that wraps the PAN algorithm with user-friendly interface
- **`neupan/blocks/`**: Contains the core algorithmic blocks:
  - `pan.py`: Proximal Alternating-minimization Network main algorithm
  - `dune.py`: Distance Uncertainty Network Estimation layer
  - `nrmp.py`: Neural Risk-aware Motion Planning layer
  - `initial_path.py`: Initial path generation from waypoints
- **`neupan/robot/robot.py`**: Robot kinematics and geometry definitions
- **`neupan/configuration/`**: Configuration management and utilities

### Algorithm Flow
1. **Initial Path Generation**: Creates path from waypoints using Dubins/Reeds-Shepp/straight line
2. **PAN Algorithm**: Iterative optimization with DUNE and NRMP layers
3. **Collision Avoidance**: Point-level constraints with neural network distance estimation
4. **Control Output**: Generates control actions for differential or Ackermann robots

## Common Development Tasks

### Installation and Setup
```bash
pip install -e .  # Install in development mode
pip install ir-sim  # For simulation examples
```

### Running Examples
```bash
# Run specific examples from the example/ directory
python example/run_exp.py -e non_obs -d acker
python example/run_exp.py -e corridor -d diff
```

### Configuration Management
- Use YAML files for planner configuration (see `example/*/planner.yaml`)
- Key parameters: `receding`, `step_time`, `ref_speed`, robot kinematics, PAN iterations
- DUNE model checkpoints stored in `example/model/`

### DUNE Model Training
- Train custom DUNE models for new robot geometries using `example/dune_train/`
- Training parameters configured in YAML `train` section
- Models are robot-specific and don't need retraining for different environments

## Important Notes

- **CPU-based optimization**: cvxpy solver runs on CPU, so powerful CPU recommended for real-time performance
- **Robot geometry**: DUNE models are specific to robot shape/size defined in `robot` section
- **Initial path**: Can be updated dynamically from external planners via `set_initial_path()`
- **Supported kinematics**: Differential drive (`diff`) and Ackermann (`acker`)

## Testing and Validation

- Examples serve as integration tests in various scenarios
- Use IR-SIM for simulation-based validation
- Monitor forward time cost for real-time performance assessment
- Collision detection with configurable threshold

## Development Workflow

1. Modify robot kinematics in `neupan/robot/robot.py` if adding new robot types
2. Adjust PAN parameters in YAML configuration for performance tuning
3. Train DUNE models for new robot geometries using training examples
4. Test with IR-SIM simulation environments
5. Validate real-time performance on target hardware