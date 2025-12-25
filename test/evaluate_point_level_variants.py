import argparse
import csv
import datetime as dt
import os

import numpy as np
import torch

from . import evaluate_dual_baselines as eval_mod


def _parse_variants(cfg, cfg_dir):
    items = cfg.get("variants", [])
    if not isinstance(items, list) or not items:
        raise ValueError("variants must be a non-empty list")
    out = []
    for entry in items:
        if not isinstance(entry, dict):
            raise ValueError("variant entry must be a mapping")
        vtype = entry.get("type") or entry.get("name") or entry.get("baseline")
        if not vtype:
            raise ValueError("variant entry missing type/name")
        vid = entry.get("id") or entry.get("alias") or entry.get("label") or vtype
        cfg_item = entry.get("config") or {}
        if not isinstance(cfg_item, dict):
            raise ValueError("variant config must be a mapping")
        w = entry.get("weights") or entry.get("weight")
        if w:
            w = eval_mod._resolve_path(cfg_dir, str(w))
        out.append(
            {
                "id": str(vid),
                "type": str(vtype).strip().lower(),
                "config": cfg_item,
                "weights": w,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point-level eval for multiple variants (alias support)"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--device", default="", help="Override device")
    parser.add_argument("--progress-interval", type=int, default=None)
    args = parser.parse_args()

    cfg = eval_mod._load_yaml_config(args.config)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    variants = _parse_variants(cfg, cfg_dir)

    device_name = args.device or str(cfg.get("device", "cuda"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    seed = int(cfg.get("seed", 42))
    eval_mod._seed_all(seed)

    samples = int(cfg.get("samples", 200))
    min_points = int(cfg.get("min_points", 50))
    max_points = int(cfg.get("max_points", 200))
    data_range = list(cfg.get("data_range", [-25, -25, 25, 25]))
    edge_dim = int(cfg.get("edge_dim", 4))
    state_dim = int(cfg.get("state_dim", 3))
    gt_solver_name = str(cfg.get("gt_solver", "CLARABEL"))
    kkt_rho = float(cfg.get("kkt_rho", 0.5))
    tol = float(cfg.get("tol", 1e-6))
    progress_interval = (
        int(args.progress_interval)
        if args.progress_interval is not None
        else int(cfg.get("progress_interval", 0))
    )

    robot_cfg_raw = cfg.get("robot", {})
    if robot_cfg_raw and not isinstance(robot_cfg_raw, dict):
        raise ValueError("robot config must be a mapping")
    robot_cfg = {
        "length": float(robot_cfg_raw.get("length", 4.6)),
        "width": float(robot_cfg_raw.get("width", 1.6)),
        "wheelbase": float(robot_cfg_raw.get("wheelbase", 3.0)),
        "kinematics": str(robot_cfg_raw.get("kinematics", "acker")),
        "name": str(robot_cfg_raw.get("name", "eval_robot")),
        "step_time": float(robot_cfg_raw.get("step_time", 0.1)),
    }

    G_2d, G_full, h_np = eval_mod._build_geometry(
        robot_cfg["length"],
        robot_cfg["width"],
        robot_cfg["wheelbase"],
        state_dim,
    )
    if edge_dim != G_2d.shape[0]:
        raise ValueError("edge_dim does not match generated G rows.")

    rng = np.random.default_rng(seed)
    point_clouds = eval_mod._generate_point_clouds(
        rng, samples, min_points, max_points, data_range
    )

    gt_solver = eval_mod.CVXPYSolver(
        edge_dim=edge_dim,
        state_dim=state_dim,
        G=G_full,
        h=h_np,
        solver=gt_solver_name,
    )

    gt_mu_list = []
    gt_lam_list = []
    for pc in point_clouds:
        mu_gt, lam_gt = gt_solver(pc)
        gt_mu_list.append(mu_gt)
        gt_lam_list.append(lam_gt)

    results = []
    for var in variants:
        vid = var["id"]
        base = var["type"]
        weights_path = var.get("weights")
        if base in eval_mod.REQUIRES_WEIGHTS and not weights_path:
            raise ValueError(f"Missing weights for variant '{vid}'.")
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights not found for '{vid}': {weights_path}")

        model = eval_mod._instantiate_baseline(
            base,
            edge_dim,
            state_dim,
            G_full,
            h_np,
            gt_solver_name,
            device,
            baseline_cfg=var.get("config") or {},
            G_2d=G_2d,
            h_2d=h_np,
            weights_path=weights_path,
            robot_cfg=robot_cfg,
            max_points=max_points,
        )

        model_device = torch.device("cpu") if base in eval_mod.CPU_ONLY else device
        if isinstance(model, torch.nn.Module):
            model.eval()
            model.to(model_device)
            if base != "dune" and weights_path:
                eval_mod._load_weights(model, weights_path)

        result = eval_mod.evaluate(
            vid,
            model,
            point_clouds,
            gt_mu_list,
            gt_lam_list,
            torch.from_numpy(G_full),
            torch.from_numpy(h_np),
            model_device,
            kkt_rho,
            tol,
            progress_interval,
        )
        result["type"] = base
        result["device"] = str(model_device)
        results.append(result)
        print(f"{vid}: mse_mu={result['mse_mu']:.6e} mse_lam={result['mse_lam']:.6e}")

    if not results:
        raise RuntimeError("No variants evaluated.")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = str(cfg.get("output", ""))
    if not output:
        output = os.path.join("test", "results", f"point_eval_{timestamp}.csv")
    else:
        output = eval_mod._resolve_output_path(output, timestamp)
    output = eval_mod._ensure_unique_path(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fieldnames = [
        "baseline",
        "type",
        "samples",
        "total_points",
        "mse_mu",
        "mse_lam",
        "kkt_primal_mse",
        "kkt_rel_mean",
        "constraint_rate",
        "norm_violation_mean",
        "mu_neg_rate",
        "objective_gap_mean",
        "avg_ms_per_sample",
        "avg_ms_per_point",
        "device",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved results: {output}")


if __name__ == "__main__":
    main()
