# Repository Guidelines

## Project Structure & Module Organization
- `neupan/`: core library (`neupan.py` entrypoint; `blocks/` PAN/DUNE/NRMP; `robot/` kinematics; `configuration/` helpers).
- `example/`: runnable IR-SIM demos (`run_exp.py`) and DUNE training (`dune_train/`); scenario folders contain `planner.yaml`.
- `test/`: evaluation/reproduction scripts + configs (`test/configs/`); writes outputs to `test/results/`.
- `baseline_methods/`: third-party baselines (treat as vendored code).
- `docs/`, `img/`, `animation/`: documentation and media assets.

## Setup, Run, and Development Commands
Install in editable mode (repo root):
```bash
python -m pip install -e .
python -m pip install -e ".[irsim]"   # required for simulation examples
```

Run an example (from repo root):
```bash
python example/run_exp.py -e corridor -d acker
python example/run_exp.py -e dyna_obs -d diff -v    # enable point velocities
```

Common evaluations:
```bash
python test/evaluate_dual_baselines.py --config test/configs/dual_baselines_eval.yaml
python -m test.test_projection_ab_synthetic
```

Note: planning uses `cvxpylayers` and runs on CPU; set `device: "cpu"` in YAML configs (GPU is mainly for training).

## Coding Style & Naming Conventions
- Python: 4-space indentation; keep imports explicit and match surrounding style (no enforced formatter).
- Names: `snake_case` for functions/files and YAML keys; avoid renaming public APIs (e.g., `neupan.neupan`).
- Configuration-first: prefer adding/tuning options via YAML with backwards-compatible defaults.

## Testing Guidelines
- Tests are mostly executable scripts (not a single `pytest` suite). New checks should live in `test/` and be runnable via `python -m test.<module>`.
- For simulation-dependent changes, document the exact scenario/flags and keep runs short.

## Commit & Pull Request Guidelines
- History uses mixed languages; keep messages short and descriptive (optional prefixes: `feat:`, `fix:`, `docs:`, `chore:`).
- Large artifacts: `*.pth`, `*.pt`, `*.pkl` are tracked via Git LFS (`.gitattributes`). Don’t commit generated results; use `test/results/` and similar output folders.
- PRs: include purpose, how to reproduce (command + YAML path), and a screenshot/GIF when planner behavior changes.
