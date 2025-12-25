"""
Unified dual-variable baseline training script (point-level, DUNETrain style).

Trains weight-required baselines (e.g., MLP/ISTA/ADMM/DeepInverse) to predict
per-point dual variables using the same loss structure as DUNETrain:
  - mu MSE
  - distance MSE
  - fa/fb alignment losses
  - optional constraint loss
  - optional KKT residual loss

This script is point-level: the GT in this repo is solved per point independently,
so we flatten point clouds into a point dataset.
"""

import argparse
import datetime as dt
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baseline_methods.implementations import (
    ADMMUnrolling,
    CvxpyLayersSolver,
    CVXPYSolver,
    DeepInverseUnrolling,
    ISTAUnrolling,
    MLPBaseline,
    PointNetPlusPlus,
    PointTransformerV3,
)
from neupan.util import gen_inequal_from_vertex


TRAINABLE_REGISTRY = {
    "mlp": MLPBaseline,
    "ista": ISTAUnrolling,
    "admm": ADMMUnrolling,
    "deepinverse": DeepInverseUnrolling,
    "point_transformer_v3": PointTransformerV3,
    "pointnet_plusplus": PointNetPlusPlus,
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _resolve_with_timestamp(path: str, timestamp: str) -> str:
    date, time_part = timestamp.split("_", 1)
    return (
        str(path)
        .replace("{timestamp}", timestamp)
        .replace("{date}", date)
        .replace("{time}", time_part)
    )


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


def _generate_points(
    rng: np.random.Generator,
    num_points: int,
    data_range: List[float],
) -> torch.Tensor:
    low = np.array(data_range[:2], dtype=np.float32)
    high = np.array(data_range[2:], dtype=np.float32)
    pts = rng.uniform(low=low, high=high, size=(int(num_points), 2)).astype(np.float32)
    return torch.from_numpy(pts)


def _compute_gt(
    solver: object,
    points: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    state_dim: int,
    chunk: int,
    progress_interval: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.dim() != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points shape [N,2], got {tuple(points.shape)}")

    mu_chunks: List[torch.Tensor] = []
    dist_chunks: List[torch.Tensor] = []
    lam_chunks: List[torch.Tensor] = []
    total = points.shape[0]
    start = time.perf_counter()
    for i in range(0, total, chunk):
        pts = points[i : i + chunk]
        mu, _ = solver(pts)
        mu_b = mu.t().contiguous()  # (B, E)

        pts_aug = torch.zeros(pts.shape[0], state_dim, dtype=pts.dtype)
        pts_aug[:, :2] = pts
        temp = pts_aug @ G.t() - h.reshape(1, -1)
        dist = (mu_b * temp).sum(dim=1, keepdim=True)
        lam = -(mu_b @ G)

        mu_chunks.append(mu_b)
        dist_chunks.append(dist)
        lam_chunks.append(lam)

        done = min(i + chunk, total)
        if progress_interval > 0 and (done % progress_interval == 0 or done == total):
            dt_ms = (time.perf_counter() - start) * 1000.0
            per_point = dt_ms / max(done, 1)
            print(f"GT: {done}/{total} points ({done / total:.1%}) | {per_point:.3f} ms/pt")

    mu_all = torch.cat(mu_chunks, dim=0)
    dist_all = torch.cat(dist_chunks, dim=0)
    lam_all = torch.cat(lam_chunks, dim=0)
    return mu_all, dist_all, lam_all


class PointDualDataset(Dataset):
    def __init__(
        self,
        points: torch.Tensor,
        mu: torch.Tensor,
        dist: torch.Tensor,
        lam: torch.Tensor,
    ) -> None:
        if points.shape[0] != mu.shape[0] or points.shape[0] != dist.shape[0]:
            raise ValueError("points/mu/dist must have the same length.")
        if points.shape[0] != lam.shape[0]:
            raise ValueError("points/lam must have the same length.")
        self.points = points
        self.mu = mu
        self.dist = dist
        self.lam = lam

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.points[idx], self.mu[idx], self.dist[idx], self.lam[idx]


@dataclass
class TrainCfg:
    data_size: int
    data_range: List[float]
    batch_size: int
    epoch: int
    valid_freq: int
    save_freq: int
    lr: float
    lr_decay: float
    decay_freq: int
    use_lconstr: bool
    w_constr: float
    use_kkt: bool
    w_kkt: float
    kkt_rho: float
    projection: str
    lam_weight: float
    early_stop: bool
    early_stop_patience: int
    early_stop_min_delta: float
    early_stop_min_epoch: int
    early_stop_metric: str


def _train_one(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainCfg,
    G: torch.Tensor,
    h: torch.Tensor,
    state_dim: int,
    save_dir: str,
) -> Tuple[float, float, Dict[str, float]]:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    best_metric = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stats: Dict[str, float] = {}
    warned_no_lam = False

    G = G.to(device)
    h = h.to(device).reshape(-1, 1)

    def _model_mu(out: Any, batch: int) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            mu = out[0]
        else:
            mu = out
        if mu.dim() == 2:
            if mu.shape[0] == G.shape[0] and mu.shape[1] == batch:
                mu = mu.t()
            elif mu.shape[1] != G.shape[0]:
                raise ValueError("Unexpected mu shape.")
            mu = mu.unsqueeze(-1)
        elif mu.dim() == 3:
            if mu.shape[0] == G.shape[0] and mu.shape[1] == batch:
                mu = mu.permute(1, 0, 2)
            elif mu.shape[1] != G.shape[0]:
                raise ValueError("Unexpected mu shape.")
        else:
            raise ValueError("Unexpected mu dimensions.")
        return mu

    def _model_lam(out: Any, batch: int, state_dim: int) -> Optional[torch.Tensor]:
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            return None
        lam = out[1]
        if lam.dim() == 2:
            if lam.shape[0] == state_dim and lam.shape[1] == batch:
                lam = lam.t()
            elif lam.shape[1] != state_dim:
                raise ValueError("Unexpected lam shape.")
        elif lam.dim() == 3:
            if lam.shape[0] == state_dim and lam.shape[1] == batch:
                lam = lam.permute(1, 0, 2).squeeze(-1)
            elif lam.shape[1] == state_dim:
                lam = lam.squeeze(-1)
            else:
                raise ValueError("Unexpected lam shape.")
        else:
            raise ValueError("Unexpected lam dimensions.")
        return lam

    def _cal_distance(mu_be1: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        ip = pts.unsqueeze(-1)
        G_b = G.unsqueeze(0).expand(mu_be1.shape[0], -1, -1)
        h_b = h.unsqueeze(0).expand(mu_be1.shape[0], -1, -1)
        temp = torch.bmm(G_b, ip) - h_b
        mu_t = mu_be1.transpose(1, 2)
        dist = torch.bmm(mu_t, temp)
        return dist.squeeze(-1)

    def _cal_loss_fab(mu_pred: torch.Tensor, mu_label: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = float(np.random.uniform(0.0, 2.0 * np.pi))
        R = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=pts.dtype,
            device=pts.device,
        )
        batch = pts.shape[0]
        GT_b = G.t().unsqueeze(0).expand(batch, -1, -1)
        R_b = R.unsqueeze(0).expand(batch, -1, -1)

        lam_pred = -torch.bmm(R_b, torch.bmm(GT_b, mu_pred))
        lam_label = -torch.bmm(R_b, torch.bmm(GT_b, mu_label))

        fa_pred = lam_pred.transpose(1, 2)
        fa_label = lam_label.transpose(1, 2)

        ip = pts.unsqueeze(-1)
        h_b = h.unsqueeze(0).expand(batch, -1, -1)
        mu_t = mu_pred.transpose(1, 2)
        mu_label_t = mu_label.transpose(1, 2)
        fb_pred = torch.bmm(fa_pred, ip) + torch.bmm(mu_t, h_b)
        fb_label = torch.bmm(fa_label, ip) + torch.bmm(mu_label_t, h_b)

        mse_fa = F.mse_loss(fa_pred, fa_label)
        mse_fb = F.mse_loss(fb_pred, fb_label)
        return mse_fa, mse_fb

    def _cal_constraints(mu_reg: torch.Tensor) -> torch.Tensor:
        GT_b = G.t().unsqueeze(0).expand(mu_reg.shape[0], -1, -1)
        v = torch.bmm(GT_b, mu_reg).squeeze(-1)
        nrm = torch.norm(v, dim=-1)
        viol = torch.clamp(nrm - 1.0, min=0.0)
        l_nonneg = (mu_reg.clamp_max(0.0) ** 2).mean()
        return (viol ** 2).mean() + l_nonneg

    def _cal_kkt(mu_reg: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        ip = pts.unsqueeze(-1)
        G_b = G.unsqueeze(0).expand(mu_reg.shape[0], -1, -1)
        h_b = h.unsqueeze(0).expand(mu_reg.shape[0], -1, -1)
        a = torch.bmm(G_b, ip) - h_b
        GT_b = G.t().unsqueeze(0).expand(mu_reg.shape[0], -1, -1)
        Gy = torch.bmm(G_b, torch.bmm(GT_b, mu_reg))
        s = torch.relu(-mu_reg)
        r = -a + cfg.kkt_rho * Gy - s
        r_flat = r.squeeze(-1)
        a_flat = a.squeeze(-1)
        eps = 1e-6
        num = torch.norm(r_flat, p=2, dim=-1)
        den = torch.norm(a_flat, p=2, dim=-1)
        return ((num / (den + eps)) ** 2).mean()

    for epoch in range(cfg.epoch + 1):
        t0 = time.perf_counter()
        model.train()
        if name == "pointnet_plusplus":
            # PointNet++ uses BatchNorm on a single cloud (batch=1); keep BN in eval to avoid errors.
            def _bn_eval(m: torch.nn.Module) -> None:
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
            model.apply(_bn_eval)
        train_mu = train_dist = train_fa = train_fb = 0.0
        train_lconstr = train_lkkt = train_lam = 0.0
        seen = 0

        for pts, mu_t, dist_t, lam_t in train_loader:
            pts = pts.to(device)
            mu_t = mu_t.to(device)
            dist_t = dist_t.to(device)
            lam_t = lam_t.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(pts)
            mu_p = _model_mu(out, pts.shape[0])
            lam_p = _model_lam(out, pts.shape[0], state_dim)

            mu_t_be1 = mu_t.unsqueeze(-1)
            loss_mu = F.mse_loss(mu_p, mu_t_be1)
            loss_dist = F.mse_loss(_cal_distance(mu_p, pts), dist_t)
            loss_fa, loss_fb = _cal_loss_fab(mu_p, mu_t_be1, pts)

            loss_lconstr = torch.zeros((), device=device)
            if cfg.use_lconstr:
                loss_lconstr = _cal_constraints(mu_p)

            loss_lkkt = torch.zeros((), device=device)
            if cfg.use_kkt:
                loss_lkkt = _cal_kkt(mu_p, pts)

            loss_lam = torch.zeros((), device=device)
            if cfg.lam_weight > 0.0:
                if lam_p is None:
                    if not warned_no_lam:
                        print(f"{name}: lam_weight > 0 but model returns no lam. Skipping lam loss.")
                        warned_no_lam = True
                else:
                    loss_lam = F.mse_loss(lam_p, lam_t)

            loss = loss_mu + loss_dist + loss_fa + loss_fb
            loss = loss + cfg.w_constr * loss_lconstr + cfg.w_kkt * loss_lkkt
            loss = loss + cfg.lam_weight * loss_lam
            loss.backward()
            opt.step()

            bs = int(pts.shape[0])
            seen += bs
            train_mu += loss_mu.item() * bs
            train_dist += loss_dist.item() * bs
            train_fa += loss_fa.item() * bs
            train_fb += loss_fb.item() * bs
            train_lconstr += loss_lconstr.item() * bs
            train_lkkt += loss_lkkt.item() * bs
            train_lam += loss_lam.item() * bs

        def _avg(val: float) -> float:
            return val / max(seen, 1)

        train_mu = _avg(train_mu)
        train_dist = _avg(train_dist)
        train_fa = _avg(train_fa)
        train_fb = _avg(train_fb)
        train_lconstr = _avg(train_lconstr)
        train_lkkt = _avg(train_lkkt)
        train_lam = _avg(train_lam)

        val_mu = val_dist = val_fa = val_fb = val_lconstr = val_lkkt = val_lam = 0.0
        vseen = 0
        if cfg.valid_freq > 0 and (epoch % cfg.valid_freq == 0):
            model.eval()
            with torch.no_grad():
                for pts, mu_t, dist_t, lam_t in val_loader:
                    pts = pts.to(device)
                    mu_t = mu_t.to(device)
                    dist_t = dist_t.to(device)
                    lam_t = lam_t.to(device)

                    out = model(pts)
                    mu_p = _model_mu(out, pts.shape[0])
                    lam_p = _model_lam(out, pts.shape[0], state_dim)

                    mu_t_be1 = mu_t.unsqueeze(-1)
                    loss_mu = F.mse_loss(mu_p, mu_t_be1)
                    loss_dist = F.mse_loss(_cal_distance(mu_p, pts), dist_t)
                    loss_fa, loss_fb = _cal_loss_fab(mu_p, mu_t_be1, pts)

                    loss_lconstr = torch.zeros((), device=device)
                    if cfg.use_lconstr:
                        loss_lconstr = _cal_constraints(mu_p)

                    loss_lkkt = torch.zeros((), device=device)
                    if cfg.use_kkt:
                        loss_lkkt = _cal_kkt(mu_p, pts)

                    loss_lam = torch.zeros((), device=device)
                    if cfg.lam_weight > 0.0 and lam_p is not None:
                        loss_lam = F.mse_loss(lam_p, lam_t)

                    bs = int(pts.shape[0])
                    vseen += bs
                    val_mu += loss_mu.item() * bs
                    val_dist += loss_dist.item() * bs
                    val_fa += loss_fa.item() * bs
                    val_fb += loss_fb.item() * bs
                    val_lconstr += loss_lconstr.item() * bs
                    val_lkkt += loss_lkkt.item() * bs
                    val_lam += loss_lam.item() * bs

            val_mu /= max(vseen, 1)
            val_dist /= max(vseen, 1)
            val_fa /= max(vseen, 1)
            val_fb /= max(vseen, 1)
            val_lconstr /= max(vseen, 1)
            val_lkkt /= max(vseen, 1)
            val_lam /= max(vseen, 1)

        elapsed = time.perf_counter() - t0
        print(
            f"{name} | epoch {epoch}/{cfg.epoch} "
            f"train_mu={train_mu:.2e} train_dist={train_dist:.2e} "
            f"train_fa={train_fa:.2e} train_fb={train_fb:.2e} "
            f"train_lconstr={train_lconstr:.2e} train_lkkt={train_lkkt:.2e} "
            f"train_lam={train_lam:.2e} time={elapsed:.1f}s"
        )

        if cfg.valid_freq > 0 and (epoch % cfg.valid_freq == 0):
            val_total = (
                val_mu + val_dist + val_fa + val_fb
                + cfg.w_constr * val_lconstr + cfg.w_kkt * val_lkkt
                + cfg.lam_weight * val_lam
            )
            print(
                f"{name} | val_mu={val_mu:.2e} val_dist={val_dist:.2e} "
                f"val_fa={val_fa:.2e} val_fb={val_fb:.2e} "
                f"val_lconstr={val_lconstr:.2e} val_lkkt={val_lkkt:.2e} "
                f"val_lam={val_lam:.2e} val_total={val_total:.2e}"
            )

            metric = val_total if cfg.early_stop_metric == "val_total" else val_mu
            improved = metric < (best_metric - cfg.early_stop_min_delta)
            if improved:
                best_metric = metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stats = {
                    "val_total": float(val_total),
                    "val_mu": float(val_mu),
                    "val_dist": float(val_dist),
                    "val_fa": float(val_fa),
                    "val_fb": float(val_fb),
                    "val_lconstr": float(val_lconstr),
                    "val_lkkt": float(val_lkkt),
                    "val_lam": float(val_lam),
                    "epoch": float(epoch),
                }
                best_path = os.path.join(save_dir, "model_best.pth")
                torch.save(model.state_dict(), best_path)

            if cfg.early_stop and best_epoch >= 0:
                if (epoch - best_epoch) >= cfg.early_stop_patience and epoch >= cfg.early_stop_min_epoch:
                    print(f"{name} | early stop at epoch {epoch}, best_epoch={best_epoch}")
                    break

        if cfg.save_freq > 0 and (epoch % cfg.save_freq == 0):
            save_path = os.path.join(save_dir, f"model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        if cfg.decay_freq > 0 and (epoch + 1) % cfg.decay_freq == 0:
            for group in opt.param_groups:
                group["lr"] = group["lr"] * cfg.lr_decay

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    last_path = os.path.join(save_dir, "model_last.pth")
    torch.save(model.state_dict(), last_path)
    return best_metric, float(stats.get("epoch", cfg.epoch)), stats


def _parse_baselines(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = cfg.get("baselines", [])
    if not items:
        raise ValueError("Config must include non-empty 'baselines' list.")
    parsed: List[Dict[str, Any]] = []
    for entry in items:
        if isinstance(entry, str):
            parsed.append({"name": entry.strip().lower(), "config": {}, "train": {}})
            continue
        if not isinstance(entry, dict):
            raise ValueError("Each baseline entry must be a string or mapping.")
        name = str(entry.get("name") or entry.get("baseline") or "").strip().lower()
        if not name:
            raise ValueError("Baseline entry missing name.")
        parsed.append(
            {
                "name": name,
                "config": entry.get("config") or {},
                "train": entry.get("train") or {},
            }
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-variable baselines (point-level, DUNETrain style)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--baselines", default="", help="Comma-separated subset to train (optional)")
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    device_name = str(cfg.get("device", "cuda"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    seed = int(cfg.get("seed", 42))
    _seed_all(seed)

    edge_dim = int(cfg.get("edge_dim", 4))
    state_dim = int(cfg.get("state_dim", 3))

    robot_cfg = cfg.get("robot", {}) or {}
    if not isinstance(robot_cfg, dict):
        raise ValueError("robot config must be a mapping")
    length = float(robot_cfg.get("length", 4.6))
    width = float(robot_cfg.get("width", 1.6))
    wheelbase = float(robot_cfg.get("wheelbase", 3.0))

    G_2d, G_full, h = _build_geometry(length, width, wheelbase, state_dim)
    if edge_dim != int(G_full.shape[0]):
        raise ValueError(f"edge_dim={edge_dim} does not match generated G rows={G_full.shape[0]}.")

    gt_backend = str(cfg.get("gt_backend", "cvxpy")).strip().lower()
    gt_solver_name = str(cfg.get("gt_solver", "CLARABEL"))
    if gt_backend not in ("cvxpy", "cvxpylayers"):
        raise ValueError("gt_backend must be 'cvxpy' or 'cvxpylayers'.")

    if gt_backend == "cvxpylayers":
        gt_solver = CvxpyLayersSolver(edge_dim=edge_dim, state_dim=state_dim, G=G_full, h=h)
    else:
        gt_solver = CVXPYSolver(edge_dim=edge_dim, state_dim=state_dim, G=G_full, h=h, solver=gt_solver_name)

    train_section = cfg.get("train", {})
    if not isinstance(train_section, dict):
        train_section = {}

    if train_section:
        data_size = int(train_section.get("data_size", 100000))
        data_range = list(train_section.get("data_range", [-25, -25, 25, 25]))
        if len(data_range) != 4:
            raise ValueError("data_range must be [xmin, ymin, xmax, ymax].")
        train_size = int(data_size * 0.8)
        val_size = data_size - train_size
    else:
        train_points = int(cfg.get("train_points", 200000))
        val_points = int(cfg.get("val_points", 20000))
        data_size = train_points + val_points
        data_range = list(cfg.get("data_range", [-25, -25, 25, 25]))
        if len(data_range) != 4:
            raise ValueError("data_range must be [xmin, ymin, xmax, ymax].")
        train_size = train_points
        val_size = val_points

    ds_cache_raw = cfg.get("dataset_cache", "")
    dataset_cache = _resolve_path(cfg_dir, str(ds_cache_raw)) if ds_cache_raw else ""
    if dataset_cache:
        dataset_cache = _resolve_with_timestamp(dataset_cache, timestamp)

    gt_chunk = int(cfg.get("gt_chunk", 256))
    gt_progress = int(cfg.get("gt_progress_interval", 10000))

    G_t = torch.from_numpy(G_full)
    h_t = torch.from_numpy(h)
    G2_t = torch.from_numpy(G_2d)

    if dataset_cache and os.path.isfile(dataset_cache):
        payload = torch.load(dataset_cache, map_location="cpu")
        points_all = payload["points"]
        mu_all = payload["mu"]
        dist_all = payload.get("dist")
        lam_all = payload.get("lam")
        if dist_all is None:
            pts_aug = torch.zeros(points_all.shape[0], state_dim, dtype=points_all.dtype)
            pts_aug[:, :2] = points_all
            temp = pts_aug @ G_t.t() - h_t.reshape(1, -1)
            dist_all = (mu_all * temp).sum(dim=1, keepdim=True)
        if lam_all is None:
            lam_all = -(mu_all @ G_t)
        print(f"Loaded dataset cache: {dataset_cache}")
    else:
        rng = np.random.default_rng(seed)
        points_all = _generate_points(rng, data_size, data_range)
        print(f"Generating GT with {gt_backend} ({gt_solver_name}) for {points_all.shape[0]} points...")
        mu_all, dist_all, lam_all = _compute_gt(
            gt_solver,
            points_all,
            G_t,
            h_t,
            state_dim,
            chunk=gt_chunk,
            progress_interval=gt_progress,
        )
        if dataset_cache:
            os.makedirs(os.path.dirname(dataset_cache), exist_ok=True)
            torch.save(
                {
                    "points": points_all,
                    "mu": mu_all,
                    "dist": dist_all,
                    "lam": lam_all,
                    "meta": {
                        "seed": seed,
                        "edge_dim": edge_dim,
                        "state_dim": state_dim,
                        "data_range": data_range,
                        "robot": {"length": length, "width": width, "wheelbase": wheelbase},
                        "gt_backend": gt_backend,
                        "gt_solver": gt_solver_name,
                        "timestamp": timestamp,
                    },
                },
                dataset_cache,
            )
            print(f"Saved dataset cache: {dataset_cache}")

    points_train = points_all[:train_size]
    mu_train = mu_all[:train_size]
    dist_train = dist_all[:train_size]
    lam_train = lam_all[:train_size]
    points_val = points_all[train_size : train_size + val_size]
    mu_val = mu_all[train_size : train_size + val_size]
    dist_val = dist_all[train_size : train_size + val_size]
    lam_val = lam_all[train_size : train_size + val_size]

    train_ds = PointDualDataset(points_train, mu_train, dist_train, lam_train)
    val_ds = PointDualDataset(points_val, mu_val, dist_val, lam_val)

    default_train_cfg = TrainCfg(
        data_size=data_size,
        data_range=data_range,
        batch_size=int(train_section.get("batch_size", 256)),
        epoch=int(train_section.get("epoch", 5000)),
        valid_freq=int(train_section.get("valid_freq", 1)),
        save_freq=int(train_section.get("save_freq", 500)),
        lr=float(train_section.get("lr", 5e-5)),
        lr_decay=float(train_section.get("lr_decay", 0.5)),
        decay_freq=int(train_section.get("decay_freq", 1500)),
        use_lconstr=bool(train_section.get("use_lconstr", False)),
        w_constr=float(train_section.get("w_constr", 0.10)),
        use_kkt=bool(train_section.get("use_kkt", False)),
        w_kkt=float(train_section.get("w_kkt", 1e-3)),
        kkt_rho=float(train_section.get("kkt_rho", 0.5)),
        projection=str(train_section.get("projection", "none")),
        lam_weight=float(train_section.get("lam_weight", 0.0)),
        early_stop=bool(train_section.get("early_stop", False)),
        early_stop_patience=int(train_section.get("early_stop_patience", 500)),
        early_stop_min_delta=float(train_section.get("early_stop_min_delta", 1.0e-7)),
        early_stop_min_epoch=int(train_section.get("early_stop_min_epoch", 0)),
        early_stop_metric=str(train_section.get("early_stop_metric", "val_total")),
    )

    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", device.type == "cuda"))
    train_loader = DataLoader(
        train_ds,
        batch_size=default_train_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=default_train_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    out_dir_raw = str(cfg.get("output_dir", "test/weights/dual_baselines/{timestamp}"))
    out_dir = _resolve_path(cfg_dir, out_dir_raw) or out_dir_raw
    out_dir = _resolve_with_timestamp(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    baselines = _parse_baselines(cfg)
    if args.baselines:
        allow = {b.strip().lower() for b in args.baselines.split(",") if b.strip()}
        baselines = [b for b in baselines if b["name"] in allow]

    if not baselines:
        raise ValueError("No baselines selected to train.")

    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "device": str(device),
        "dataset_cache": dataset_cache,
        "output_dir": out_dir,
        "trained": {},
    }

    for item in baselines:
        name = item["name"]
        if name not in TRAINABLE_REGISTRY:
            raise ValueError(f"Unknown/unsupported trainable baseline: {name}")
        if name == "point_transformer_v3" and device.type != "cuda":
            raise ValueError("point_transformer_v3 requires CUDA; set device: cuda.")
        model_cls = TRAINABLE_REGISTRY[name]
        model_cfg = item.get("config") or {}
        if not isinstance(model_cfg, dict):
            raise ValueError(f"config for {name} must be a mapping.")

        kwargs: Dict[str, Any] = {"edge_dim": edge_dim, "state_dim": state_dim}
        if name in {"ista", "admm"}:
            kwargs["G"] = G_full
            kwargs["h"] = h
        kwargs.update(model_cfg)

        model = model_cls(**kwargs)

        train_over = item.get("train") or {}
        if not isinstance(train_over, dict):
            raise ValueError(f"train for {name} must be a mapping.")

        if "data_size" in train_over or "data_range" in train_over:
            raise ValueError("Per-baseline data_size/data_range is not supported in unified training.")

        train_cfg = TrainCfg(
            data_size=default_train_cfg.data_size,
            data_range=default_train_cfg.data_range,
            batch_size=int(train_over.get("batch_size", default_train_cfg.batch_size)),
            epoch=int(train_over.get("epoch", default_train_cfg.epoch)),
            valid_freq=int(train_over.get("valid_freq", default_train_cfg.valid_freq)),
            save_freq=int(train_over.get("save_freq", default_train_cfg.save_freq)),
            lr=float(train_over.get("lr", default_train_cfg.lr)),
            lr_decay=float(train_over.get("lr_decay", default_train_cfg.lr_decay)),
            decay_freq=int(train_over.get("decay_freq", default_train_cfg.decay_freq)),
            use_lconstr=bool(train_over.get("use_lconstr", default_train_cfg.use_lconstr)),
            w_constr=float(train_over.get("w_constr", default_train_cfg.w_constr)),
            use_kkt=bool(train_over.get("use_kkt", default_train_cfg.use_kkt)),
            w_kkt=float(train_over.get("w_kkt", default_train_cfg.w_kkt)),
            kkt_rho=float(train_over.get("kkt_rho", default_train_cfg.kkt_rho)),
            projection=str(train_over.get("projection", default_train_cfg.projection)),
            lam_weight=float(train_over.get("lam_weight", default_train_cfg.lam_weight)),
            early_stop=bool(train_over.get("early_stop", default_train_cfg.early_stop)),
            early_stop_patience=int(train_over.get("early_stop_patience", default_train_cfg.early_stop_patience)),
            early_stop_min_delta=float(train_over.get("early_stop_min_delta", default_train_cfg.early_stop_min_delta)),
            early_stop_min_epoch=int(train_over.get("early_stop_min_epoch", default_train_cfg.early_stop_min_epoch)),
            early_stop_metric=str(train_over.get("early_stop_metric", default_train_cfg.early_stop_metric)),
        )
        if name == "pointnet_plusplus" and train_cfg.batch_size < 32:
            raise ValueError("pointnet_plusplus requires batch_size >= 32.")

        if train_cfg.batch_size != default_train_cfg.batch_size:
            train_loader = DataLoader(
                train_ds,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=train_cfg.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

        print(f"\n=== Train {name} on {device} ===")
        baseline_dir = os.path.join(out_dir, name)
        os.makedirs(baseline_dir, exist_ok=True)
        best_val, best_epoch, stats = _train_one(
            name,
            model,
            train_loader,
            val_loader,
            device,
            train_cfg,
            G2_t,
            h_t,
            state_dim,
            baseline_dir,
        )

        weights_path = os.path.join(baseline_dir, "model_last.pth")
        print(f"Saved weights: {weights_path}")

        summary["trained"][name] = {
            "weights": weights_path,
            "best_val": float(best_val),
            "best_epoch": float(best_epoch),
            "model_config": model_cfg,
            "train_config": train_over,
            "stats": stats,
        }

    try:
        import yaml

        with open(os.path.join(out_dir, "train_summary.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    except Exception:
        pass

    print("\nDone. Fill these into test/configs/dual_baselines_eval.yaml:")
    for name, info in summary["trained"].items():
        print(f"  - {name}: {info['weights']}")


if __name__ == "__main__":
    main()
