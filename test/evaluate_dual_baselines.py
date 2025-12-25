"""
Unified dual-variable baseline evaluation script.

Evaluates point-level mu/lambda prediction against CVXPY ground truth.
Outputs aggregated metrics to CSV.
"""

import argparse
import csv
import datetime as dt
import inspect
import os
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from baseline_methods.implementations import (
    ADMMUnrolling,
    CenterDistanceMPC,
    CvxpyLayersSolver,
    CVXPYSolver,
    DeepInverseUnrolling,
    ESDFMPCSolver,
    ISTAUnrolling,
    MLPBaseline,
    PointNetPlusPlus,
    PointTransformerV3,
)
from neupan import configuration as neu_cfg
from neupan.blocks.dune import DUNE
from neupan.blocks.flexible_pdhg import FlexiblePDHGFront
from neupan.robot.robot import robot as Robot
from neupan.util import gen_inequal_from_vertex


BASELINE_REGISTRY = {
    "cvxpy": CVXPYSolver,
    "cvxpylayers": CvxpyLayersSolver,
    "center_distance": CenterDistanceMPC,
    "esdf_mpc": ESDFMPCSolver,
    "mlp": MLPBaseline,
    "ista": ISTAUnrolling,
    "admm": ADMMUnrolling,
    "deepinverse": DeepInverseUnrolling,
    "point_transformer_v3": PointTransformerV3,
    "pointnet_plusplus": PointNetPlusPlus,
    "pdpl_net": None,
    "dune": None,
}

CPU_ONLY = {"cvxpy", "cvxpylayers"}
REQUIRES_WEIGHTS = {
    "mlp",
    "ista",
    "admm",
    "deepinverse",
    "point_transformer_v3",
    "pointnet_plusplus",
    "pdpl_net",
    "dune",
}

DEFAULT_PDPL_CFG = {
    "front_J": 4,
    "front_hidden": 32,
    "front_learned": True,
    "se2_embed": False,
    "front_tau": 0.5,
    "front_sigma": 0.5,
    "front_residual_scale": 0.5,
}

DEFAULT_DUNE_CFG = {
    "receding": 0,
    "dune_max_num": None,
    "use_directional_sampling": False,
    "key_directions": [],
    "nearest_num": 2,
}

DEFAULT_DUNE_TRAIN_KWARGS = {
    "projection": "none",
    "monitor_dual_norm": True,
    "se2_embed": False,
    "front": "obs_point",
    "front_J": 1,
    "front_hidden": 32,
    "front_learned": True,
    "front_tau": 0.5,
    "front_sigma": 0.5,
    "front_residual_scale": 0.5,
}

DUNE_INIT_KEYS = {
    "receding",
    "dune_max_num",
    "use_directional_sampling",
    "key_directions",
    "nearest_num",
    "train_kwargs",
}

_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_output_path(output: str, timestamp: str) -> str:
    """
    Resolve output path with timestamp support.

    - Supports placeholders: {timestamp}, {date}, {time}
    - If output is a directory, writes dual_eval_{timestamp}.csv inside it.
    - If output is a file path without placeholders, appends _{timestamp} before extension.
    """
    date, time_part = timestamp.split("_", 1)
    expanded = (
        str(output)
        .replace("{timestamp}", timestamp)
        .replace("{date}", date)
        .replace("{time}", time_part)
    )

    is_dir_like = expanded.endswith(("\\", "/")) or (os.path.exists(expanded) and os.path.isdir(expanded))
    if is_dir_like:
        return os.path.join(expanded, f"dual_eval_{timestamp}.csv")

    base, ext = os.path.splitext(expanded)
    has_placeholder = any(token in str(output) for token in ("{timestamp}", "{date}", "{time}"))
    if has_placeholder:
        return expanded
    if _TIMESTAMP_RE.search(base):
        return expanded
    if ext:
        return f"{base}_{timestamp}{ext}"
    return f"{expanded}_{timestamp}.csv"


def _ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{base}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --config. Install pyyaml.") from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    return cfg


def _resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _parse_config_baselines(
    cfg: Dict[str, Any],
    cfg_dir: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
    baseline_cfg_map: Dict[str, Dict[str, Any]] = {}
    weights_map: Dict[str, str] = {}

    weights_section = cfg.get("weights", {})
    if isinstance(weights_section, dict):
        for name, path in weights_section.items():
            if path is None:
                continue
            weights_map[str(name).strip().lower()] = _resolve_path(cfg_dir, str(path))

    baselines_cfg = cfg.get("baselines", [])
    if not baselines_cfg:
        return [], baseline_cfg_map, weights_map

    if isinstance(baselines_cfg, str):
        baselines_cfg = [baselines_cfg]
    if not isinstance(baselines_cfg, list):
        raise ValueError("baselines must be a list")

    baseline_names: List[str] = []
    for entry in baselines_cfg:
        cfg_item: Dict[str, Any] = {}
        if isinstance(entry, str):
            name = entry.strip().lower()
        elif isinstance(entry, dict):
            raw_name = entry.get("name") or entry.get("baseline")
            if not raw_name:
                raise ValueError("Baseline entry missing name")
            name = str(raw_name).strip().lower()
            cfg_item = entry.get("config") or {}
            if cfg_item is None:
                cfg_item = {}
            if not isinstance(cfg_item, dict):
                raise ValueError(f"Baseline config for '{name}' must be a mapping")
            weight_path = entry.get("weights") or entry.get("weight")
            if weight_path:
                weights_map[name] = _resolve_path(cfg_dir, str(weight_path))
        else:
            raise ValueError("Baseline entry must be string or mapping")

        baseline_names.append(name)
        if cfg_item:
            baseline_cfg_map[name] = cfg_item

    return baseline_names, baseline_cfg_map, weights_map


def _set_neupan_device(device: torch.device) -> None:
    neu_cfg.device = device
    neu_cfg.time_print = False


def _rect_vertices(length: float, width: float, wheelbase: float = 0.0) -> np.ndarray:
    start_x = -(length - wheelbase) / 2.0
    start_y = -width / 2.0

    point0 = np.array([[start_x], [start_y]])
    point1 = np.array([[start_x + length], [start_y]])
    point2 = np.array([[start_x + length], [start_y + width]])
    point3 = np.array([[start_x], [start_y + width]])
    return np.hstack((point0, point1, point2, point3))


def _build_geometry(
    length: float,
    width: float,
    wheelbase: float,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = _rect_vertices(length, width, wheelbase)
    G_2d, h = gen_inequal_from_vertex(vertices)
    if G_2d is None or h is None:
        raise ValueError("Failed to generate G/h from vertices.")

    if state_dim < 2:
        raise ValueError("state_dim must be >= 2.")

    if G_2d.shape[1] < state_dim:
        pad = np.zeros((G_2d.shape[0], state_dim - G_2d.shape[1]), dtype=G_2d.dtype)
        G_full = np.hstack((G_2d, pad))
    else:
        G_full = G_2d[:, :state_dim]

    h = h.reshape(-1)
    return G_2d.astype(np.float32), G_full.astype(np.float32), h.astype(np.float32)


def _augment_points(points: torch.Tensor, state_dim: int) -> torch.Tensor:
    if state_dim == 2:
        return points
    if state_dim < 2:
        raise ValueError("state_dim must be >= 2.")
    aug = torch.zeros(points.shape[0], state_dim, device=points.device, dtype=points.dtype)
    aug[:, :2] = points
    return aug


def _match_point_indices(sorted_points: torch.Tensor, original_points: torch.Tensor) -> List[int]:
    if sorted_points.shape != original_points.shape:
        raise ValueError("Point shapes do not match for reordering.")

    sorted_np = sorted_points.detach().cpu().numpy()
    orig_np = original_points.detach().cpu().numpy()

    buckets: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for idx, p in enumerate(orig_np):
        buckets[(float(p[0]), float(p[1]))].append(idx)

    perm = []
    for p in sorted_np:
        key = (float(p[0]), float(p[1]))
        if key not in buckets or not buckets[key]:
            raise ValueError("Failed to match sorted points back to original order.")
        perm.append(buckets[key].pop(0))
    return perm


class PDPLNetWrapper(torch.nn.Module):
    def __init__(
        self,
        G_2d: np.ndarray,
        h: np.ndarray,
        state_dim: int,
        front_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.edge_dim = int(G_2d.shape[0])

        cfg = _merge_dict(DEFAULT_PDPL_CFG, front_cfg)
        h_col = h.reshape(-1, 1).astype(np.float32)
        self.front = FlexiblePDHGFront(
            input_dim=2,
            E=self.edge_dim,
            G=torch.from_numpy(G_2d),
            h=torch.from_numpy(h_col),
            hidden=int(cfg["front_hidden"]),
            J=int(cfg["front_J"]),
            se2_embed=bool(cfg["se2_embed"]),
            use_learned_prox=bool(cfg["front_learned"]),
            residual_scale=float(cfg["front_residual_scale"]),
            tau=float(cfg["front_tau"]),
            sigma=float(cfg["front_sigma"]),
        )

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if point_cloud.dim() != 2 or point_cloud.shape[1] != 2:
            raise ValueError(f"Expected point_cloud shape [N,2], got {tuple(point_cloud.shape)}")
        mu_row = self.front(point_cloud)
        mu = mu_row.t()
        lam_2d = -self.front.G.t().mm(mu)
        if self.state_dim > 2:
            lam = torch.zeros(self.state_dim, mu.shape[1], device=mu.device, dtype=mu.dtype)
            lam[:2] = lam_2d
        else:
            lam = lam_2d
        return mu, lam


class DUNEBaselineAdapter(torch.nn.Module):
    def __init__(self, dune: DUNE, state_dim: int) -> None:
        super().__init__()
        self.dune = dune
        self.state_dim = int(state_dim)

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if point_cloud.dim() != 2 or point_cloud.shape[1] != 2:
            raise ValueError(f"Expected point_cloud shape [N,2], got {tuple(point_cloud.shape)}")
        point_flow = [point_cloud.t()]
        R_list = [torch.eye(2, device=point_cloud.device, dtype=point_cloud.dtype)]
        obs_points_list = [point_cloud.t()]

        mu_list, lam_list, sort_point_list = self.dune(point_flow, R_list, obs_points_list)
        mu_sorted = mu_list[0]
        lam_sorted = lam_list[0]
        sorted_points = sort_point_list[0].t()

        if mu_sorted.shape[1] != point_cloud.shape[0]:
            raise ValueError("DUNE returned a subset of points; increase dune_max_num.")

        perm = _match_point_indices(sorted_points, point_cloud)
        mu = torch.zeros(
            mu_sorted.shape[0],
            point_cloud.shape[0],
            device=point_cloud.device,
            dtype=mu_sorted.dtype,
        )
        lam = torch.zeros(
            self.state_dim,
            point_cloud.shape[0],
            device=point_cloud.device,
            dtype=lam_sorted.dtype,
        )
        mu[:, perm] = mu_sorted
        lam[:2, perm] = lam_sorted
        return mu, lam


def _parse_weights(items: Iterable[str]) -> Dict[str, str]:
    weights = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid weight spec '{item}', expected name=path.")
        name, path = item.split("=", 1)
        weights[name.strip().lower()] = path.strip()
    return weights


def _load_weights(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and hasattr(model, "front"):
        if not any(key.startswith("front.") for key in state.keys()):
            state = {f"front.{key}": value for key, value in state.items()}
        front_h_key = "front.h"
        if front_h_key in state and hasattr(model.front, "h"):
            target = model.front.h
            src = state[front_h_key]
            if isinstance(src, torch.Tensor) and src.numel() == target.numel():
                state[front_h_key] = src.reshape_as(target)
    model.load_state_dict(state, strict=True)


def _instantiate_baseline(
    name: str,
    edge_dim: int,
    state_dim: int,
    G_full: np.ndarray,
    h_full: np.ndarray,
    solver_name: str,
    device: torch.device,
    baseline_cfg: Optional[Dict[str, Any]] = None,
    G_2d: Optional[np.ndarray] = None,
    h_2d: Optional[np.ndarray] = None,
    weights_path: Optional[str] = None,
    robot_cfg: Optional[Dict[str, Any]] = None,
    max_points: Optional[int] = None,
) -> object:
    if name == "pdpl_net":
        if G_2d is None or h_2d is None:
            raise ValueError("PDPL-Net requires 2D G/h geometry.")
        return PDPLNetWrapper(G_2d, h_2d, state_dim, baseline_cfg or {})

    if name == "dune":
        if robot_cfg is None:
            raise ValueError("DUNE requires robot configuration.")
        if not weights_path:
            raise ValueError("DUNE requires a checkpoint path.")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DUNE checkpoint not found: {weights_path}")

        dune_cfg = _merge_dict(DEFAULT_DUNE_CFG, baseline_cfg)
        train_kwargs = _merge_dict(DEFAULT_DUNE_TRAIN_KWARGS, baseline_cfg)
        extra_train_kwargs = dune_cfg.get("train_kwargs")
        if isinstance(extra_train_kwargs, dict):
            train_kwargs.update(extra_train_kwargs)
        for key in list(train_kwargs.keys()):
            if key in DUNE_INIT_KEYS:
                train_kwargs.pop(key, None)

        receding = int(dune_cfg.get("receding", DEFAULT_DUNE_CFG["receding"]))
        dune_max_num = dune_cfg.get("dune_max_num", DEFAULT_DUNE_CFG["dune_max_num"])
        if dune_max_num is None:
            dune_max_num = max_points
        if dune_max_num is None:
            raise ValueError("dune_max_num must be set when max_points is unknown.")
        dune_max_num = int(dune_max_num)

        use_directional_sampling = bool(dune_cfg.get("use_directional_sampling", False))
        key_directions = dune_cfg.get("key_directions", []) or []
        nearest_num = int(dune_cfg.get("nearest_num", DEFAULT_DUNE_CFG["nearest_num"]))

        _set_neupan_device(device)
        robot = Robot(
            receding=receding,
            step_time=float(robot_cfg.get("step_time", 0.1)),
            kinematics=str(robot_cfg["kinematics"]),
            length=float(robot_cfg["length"]),
            width=float(robot_cfg["width"]),
            wheelbase=float(robot_cfg["wheelbase"]),
            name=str(robot_cfg.get("name", "eval_robot")),
        )
        dune = DUNE(
            receding=receding,
            checkpoint=weights_path,
            robot=robot,
            dune_max_num=dune_max_num,
            train_kwargs=train_kwargs,
            use_directional_sampling=use_directional_sampling,
            key_directions=key_directions,
            nearest_num=nearest_num,
        )
        dune.eval()
        if isinstance(dune, torch.nn.Module):
            dune.to(device)
        return DUNEBaselineAdapter(dune, state_dim)

    cls = BASELINE_REGISTRY[name]
    if cls is None:
        raise ImportError(f"Baseline '{name}' is unavailable (import failed).")
    kwargs = {"edge_dim": edge_dim, "state_dim": state_dim}
    sig = inspect.signature(cls.__init__)
    if "G" in sig.parameters:
        kwargs["G"] = G_full
    if "h" in sig.parameters:
        kwargs["h"] = h_full
    if name == "cvxpy":
        kwargs["solver"] = solver_name
    return cls(**kwargs)


def _generate_point_clouds(
    rng: np.random.Generator,
    num_samples: int,
    min_points: int,
    max_points: int,
    data_range: List[float],
) -> List[torch.Tensor]:
    clouds = []
    low = np.array(data_range[:2], dtype=np.float32)
    high = np.array(data_range[2:], dtype=np.float32)
    for _ in range(num_samples):
        n = int(rng.integers(min_points, max_points + 1))
        points = rng.uniform(low=low, high=high, size=(n, 2)).astype(np.float32)
        clouds.append(torch.from_numpy(points))
    return clouds


def _compute_metrics(
    mu_pred: torch.Tensor,
    lam_pred: torch.Tensor,
    mu_gt: torch.Tensor,
    lam_gt: torch.Tensor,
    points: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    kkt_rho: float,
    tol: float,
) -> Dict[str, float]:
    device = mu_pred.device
    mu_gt = mu_gt.to(device=device, dtype=mu_pred.dtype)
    lam_gt = lam_gt.to(device=device, dtype=lam_pred.dtype)
    G = G.to(device=device, dtype=mu_pred.dtype)
    h = h.to(device=device, dtype=mu_pred.dtype).reshape(-1, 1)

    E, N = mu_pred.shape
    sd = lam_pred.shape[0]

    mu_err = (mu_pred - mu_gt).pow(2).sum()
    lam_err = (lam_pred - lam_gt).pow(2).sum()

    Gt_mu = G.t().mm(mu_pred)
    lam_from_mu = -Gt_mu
    primal_res = (lam_from_mu - lam_pred).pow(2).sum(dim=0)

    norm = torch.norm(Gt_mu, dim=0)
    mu_nonneg = (mu_pred >= 0.0).all(dim=0)
    feasible = mu_nonneg & (norm <= (1.0 + tol))

    norm_violation = torch.clamp(norm - 1.0, min=0.0)
    mu_neg_rate = (mu_pred < 0.0).float().mean()

    points_aug = _augment_points(points, G.shape[1])
    a = G.mm(points_aug.t()) - h
    Gy = G.mm(Gt_mu)
    s = torch.relu(-mu_pred)
    r = -a + kkt_rho * Gy - s
    r_norm = torch.norm(r, dim=0)
    a_norm = torch.norm(a, dim=0)
    kkt_rel = (r_norm / (a_norm + 1e-6)).pow(2)

    obj_pred = (mu_pred * a).sum(dim=0)
    obj_gt = (mu_gt * a).sum(dim=0)
    obj_gap = (obj_pred - obj_gt).abs()

    return {
        "mu_sq_err": mu_err.item(),
        "lam_sq_err": lam_err.item(),
        "primal_sq_err": primal_res.sum().item(),
        "feasible_count": feasible.float().sum().item(),
        "norm_violation_sum": norm_violation.sum().item(),
        "mu_neg_rate_sum": mu_neg_rate.item() * (E * N),
        "kkt_rel_sum": kkt_rel.sum().item(),
        "obj_gap_sum": obj_gap.sum().item(),
        "points": N,
        "mu_entries": E * N,
        "lam_entries": sd * N,
    }


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(
    name: str,
    model: object,
    point_clouds: List[torch.Tensor],
    gt_mu_list: List[torch.Tensor],
    gt_lam_list: List[torch.Tensor],
    G: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
    kkt_rho: float,
    tol: float,
    progress_interval: int,
) -> Dict[str, float]:
    total_points = 0
    mu_sq_err = 0.0
    lam_sq_err = 0.0
    primal_sq_err = 0.0
    feasible_count = 0.0
    norm_violation_sum = 0.0
    mu_neg_rate_sum = 0.0
    kkt_rel_sum = 0.0
    obj_gap_sum = 0.0
    total_mu_entries = 0
    total_lam_entries = 0
    total_time_ms = 0.0

    total_samples = len(point_clouds)
    for idx, (points, mu_gt, lam_gt) in enumerate(zip(point_clouds, gt_mu_list, gt_lam_list), start=1):
        points_dev = points.to(device)
        _sync_if_cuda(device)
        start = time.perf_counter()
        with torch.no_grad():
            mu_pred, lam_pred = model(points_dev)
        _sync_if_cuda(device)
        total_time_ms += (time.perf_counter() - start) * 1000.0

        if progress_interval > 0 and (idx % progress_interval == 0 or idx == total_samples):
            avg_ms = total_time_ms / max(idx, 1)
            mse_mu = mu_sq_err / max(total_mu_entries, 1)
            mse_lam = lam_sq_err / max(total_lam_entries, 1)
            constraint_rate = feasible_count / max(total_points, 1)
            print(
                f"{name}: {idx}/{total_samples} samples | "
                f"mse_mu={mse_mu:.3e} mse_lam={mse_lam:.3e} "
                f"constraint={constraint_rate:.3f} avg={avg_ms:.2f}ms"
            )

        metrics = _compute_metrics(
            mu_pred,
            lam_pred,
            mu_gt,
            lam_gt,
            points_dev,
            G,
            h,
            kkt_rho,
            tol,
        )
        total_points += metrics["points"]
        mu_sq_err += metrics["mu_sq_err"]
        lam_sq_err += metrics["lam_sq_err"]
        primal_sq_err += metrics["primal_sq_err"]
        feasible_count += metrics["feasible_count"]
        norm_violation_sum += metrics["norm_violation_sum"]
        mu_neg_rate_sum += metrics["mu_neg_rate_sum"]
        kkt_rel_sum += metrics["kkt_rel_sum"]
        obj_gap_sum += metrics["obj_gap_sum"]
        total_mu_entries += metrics["mu_entries"]
        total_lam_entries += metrics["lam_entries"]

    return {
        "baseline": name,
        "samples": len(point_clouds),
        "total_points": total_points,
        "mse_mu": mu_sq_err / max(total_mu_entries, 1),
        "mse_lam": lam_sq_err / max(total_lam_entries, 1),
        "kkt_primal_mse": primal_sq_err / max(total_points, 1),
        "kkt_rel_mean": kkt_rel_sum / max(total_points, 1),
        "constraint_rate": feasible_count / max(total_points, 1),
        "norm_violation_mean": norm_violation_sum / max(total_points, 1),
        "mu_neg_rate": mu_neg_rate_sum / max(total_mu_entries, 1),
        "objective_gap_mean": obj_gap_sum / max(total_points, 1),
        "avg_ms_per_sample": total_time_ms / max(len(point_clouds), 1),
        "avg_ms_per_point": total_time_ms / max(total_points, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-variable baseline evaluation")
    parser.add_argument("--config", default="", help="YAML config path (optional)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for NN baselines")
    parser.add_argument(
        "--baselines",
        default="cvxpy,cvxpylayers,center_distance,esdf_mpc",
        help="Comma-separated baseline names or 'all'",
    )
    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        help="Weights mapping, format name=path (repeatable)",
    )
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Fail if a trained baseline is missing weights",
    )
    parser.add_argument("--samples", type=int, default=200, help="Number of point clouds")
    parser.add_argument("--min-points", type=int, default=50, help="Min points per cloud")
    parser.add_argument("--max-points", type=int, default=200, help="Max points per cloud")
    parser.add_argument(
        "--data-range",
        type=float,
        nargs=4,
        default=[-25, -25, 25, 25],
        help="Point range: xmin ymin xmax ymax",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--edge-dim", type=int, default=4, help="Number of edges E")
    parser.add_argument("--state-dim", type=int, default=3, help="State dimension for lambda")
    parser.add_argument("--robot-length", type=float, default=1.6, help="Robot length")
    parser.add_argument("--robot-width", type=float, default=2.0, help="Robot width")
    parser.add_argument("--robot-wheelbase", type=float, default=0.5, help="Robot wheelbase")
    parser.add_argument("--robot-kinematics", default="acker", help="Robot kinematics for DUNE")
    parser.add_argument("--robot-name", default="eval_robot", help="Robot name for DUNE")
    parser.add_argument("--robot-step-time", type=float, default=0.1, help="Robot step time for DUNE")
    parser.add_argument("--gt-solver", default="CLARABEL", help="CVXPY solver name for ground truth")
    parser.add_argument("--kkt-rho", type=float, default=0.5, help="KKT residual rho")
    parser.add_argument("--tol", type=float, default=1e-6, help="Constraint tolerance")
    parser.add_argument("--progress-interval", type=int, default=0,
                        help="Print progress every N samples (0 to disable)")
    parser.add_argument("--output", default="", help="CSV output path")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    cfg_dir = ""
    baseline_cfg_map: Dict[str, Dict[str, Any]] = {}
    weights_map: Dict[str, str] = {}
    baseline_names_cfg: List[str] = []

    if args.config:
        cfg = _load_yaml_config(args.config)
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        baseline_names_cfg, baseline_cfg_map, cfg_weights = _parse_config_baselines(cfg, cfg_dir)
        weights_map.update(cfg_weights)

    weights_map.update(_parse_weights(args.weights))

    device_name = str(cfg.get("device", args.device))
    seed = int(cfg.get("seed", args.seed))
    samples = int(cfg.get("samples", args.samples))
    min_points = int(cfg.get("min_points", args.min_points))
    max_points = int(cfg.get("max_points", args.max_points))
    data_range = cfg.get("data_range", args.data_range)
    edge_dim = int(cfg.get("edge_dim", args.edge_dim))
    state_dim = int(cfg.get("state_dim", args.state_dim))
    gt_solver_name = str(cfg.get("gt_solver", args.gt_solver))
    kkt_rho = float(cfg.get("kkt_rho", args.kkt_rho))
    tol = float(cfg.get("tol", args.tol))
    output = cfg.get("output", args.output)
    require_weights = bool(cfg.get("require_weights", args.require_weights))
    progress_interval = int(cfg.get("progress_interval", args.progress_interval))

    if not isinstance(data_range, (list, tuple)) or len(data_range) != 4:
        raise ValueError("data_range must be a list of 4 floats")
    data_range = [float(x) for x in data_range]

    if baseline_names_cfg:
        baseline_names = baseline_names_cfg
    else:
        baselines_arg = args.baselines.strip().lower()
        if baselines_arg == "all":
            baseline_names = list(BASELINE_REGISTRY.keys())
        else:
            baseline_names = [b.strip() for b in baselines_arg.split(",") if b.strip()]

    if any(name == "all" for name in baseline_names):
        baseline_names = list(BASELINE_REGISTRY.keys())
    baseline_names = list(dict.fromkeys([b.lower() for b in baseline_names]))

    unknown = [b for b in baseline_names if b not in BASELINE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")

    robot_cfg_raw = cfg.get("robot", {})
    if robot_cfg_raw and not isinstance(robot_cfg_raw, dict):
        raise ValueError("robot config must be a mapping")
    robot_cfg = {
        "length": float(robot_cfg_raw.get("length", args.robot_length)),
        "width": float(robot_cfg_raw.get("width", args.robot_width)),
        "wheelbase": float(robot_cfg_raw.get("wheelbase", args.robot_wheelbase)),
        "kinematics": str(robot_cfg_raw.get("kinematics", args.robot_kinematics)),
        "name": str(robot_cfg_raw.get("name", args.robot_name)),
        "step_time": float(robot_cfg_raw.get("step_time", args.robot_step_time)),
    }
    if not robot_cfg["kinematics"]:
        raise ValueError("robot.kinematics is required")

    _seed_all(seed)

    G_2d, G_full, h_np = _build_geometry(
        robot_cfg["length"],
        robot_cfg["width"],
        robot_cfg["wheelbase"],
        state_dim,
    )
    if edge_dim != G_2d.shape[0]:
        raise ValueError("edge_dim does not match generated G rows.")

    rng = np.random.default_rng(seed)
    point_clouds = _generate_point_clouds(
        rng,
        samples,
        min_points,
        max_points,
        data_range,
    )

    gt_solver = CVXPYSolver(
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

    device = torch.device(device_name)
    results = []

    for name in baseline_names:
        weights_path = weights_map.get(name)
        if weights_path and not os.path.isfile(weights_path):
            if require_weights:
                raise FileNotFoundError(f"Missing weights for baseline '{name}': {weights_path}")
            print(f"Skip {name}: weights not found at {weights_path}.")
            continue
        if name in REQUIRES_WEIGHTS and not weights_path:
            if require_weights:
                raise ValueError(f"Missing weights for baseline '{name}'.")
            print(f"Skip {name}: missing weights.")
            continue

        baseline_cfg = baseline_cfg_map.get(name, {})
        model = _instantiate_baseline(
            name,
            edge_dim,
            state_dim,
            G_full,
            h_np,
            gt_solver_name,
            device,
            baseline_cfg=baseline_cfg,
            G_2d=G_2d,
            h_2d=h_np,
            weights_path=weights_path,
            robot_cfg=robot_cfg,
            max_points=max_points,
        )

        model_device = device
        if name in CPU_ONLY:
            model_device = torch.device("cpu")

        if isinstance(model, torch.nn.Module):
            model.eval()
            model.to(model_device)
            if name in weights_map and name != "dune":
                _load_weights(model, weights_map[name])

        G_t = torch.from_numpy(G_full)
        h_t = torch.from_numpy(h_np)

        result = evaluate(
            name,
            model,
            point_clouds,
            gt_mu_list,
            gt_lam_list,
            G_t,
            h_t,
            model_device,
            kkt_rho,
            tol,
            progress_interval,
        )
        result["device"] = str(model_device)
        results.append(result)
        print(f"{name}: mse_mu={result['mse_mu']:.6e} mse_lam={result['mse_lam']:.6e}")

    if not results:
        raise RuntimeError("No baselines evaluated.")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output:
        output = os.path.join("test", "results", f"dual_eval_{timestamp}.csv")
    else:
        output = _resolve_output_path(str(output), timestamp)
    output = _ensure_unique_path(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fieldnames = [
        "baseline",
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
