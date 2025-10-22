'''
ROI (Region of Interest) Selector for Neural Focusing in NeuPAN

Implements three ROI strategies inspired by Neural Informed RRT* (ICRA 2024):
- Ellipse: Informed elliptical corridor using c_best and c_min (for straight/gentle paths)
- Tube: Morphological dilation around reference trajectory (for curved paths)
- Wedge: Forward fan-shaped region (for cold start/fallback)



NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
'''

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ROIConfig:
    """Configuration for ROI selector"""
    enabled: bool = False
    strategy_order: list = field(default_factory=lambda: ['ellipse', 'tube', 'wedge'])
    
    # Wedge parameters (forward fan-shaped region)
    wedge_fov_deg: float = 60.0
    wedge_r_max_m: float = 8.0
    
    # Ellipse parameters (Informed RRT* style)
    ellipse_safety_scale: float = 1.08
    
    # Tube parameters (path corridor)
    tube_r0_m: float = 0.5
    tube_kappa_gain: float = 0.4
    tube_v_tau_m: float = 0.3
    
    # Guardrail (adaptive relaxation/tightening)
    guardrail_n_min: int = 30
    guardrail_n_max: int = 500
    guardrail_relax_step: float = 1.15
    guardrail_tighten_step: float = 0.9
    
    # Always keep (safety retention set)
    always_keep_near_radius_m: float = 1.5
    always_keep_goal_radius_m: float = 1.5


@dataclass
class ROIInputs:
    """Inputs for ROI selection"""
    pts: np.ndarray  # (2, N) global coordinates
    path_xy: Optional[np.ndarray] = None  # (2, T+1) reference path in global coords
    heading_rad: Optional[float] = None  # robot heading for wedge
    v_robot: Optional[float] = None  # robot velocity
    c_best_hint: Optional[float] = None  # current best path length
    goal_xy: Optional[np.ndarray] = None  # (2, 1) goal position


@dataclass
class ROIOutputs:
    """Outputs from ROI selection"""
    pts: np.ndarray  # (2, N_roi) filtered points in global coords
    strategy: str  # 'ellipse', 'tube', 'wedge', or 'none'
    n_in: int  # input point count
    n_roi: int  # output point count
    relax_count: int = 0
    tighten_count: int = 0
    # Indices (into input pts) of the kept points, shape (N_roi,)
    indices: Optional[np.ndarray] = None


class ROISelector:
    """
    ROI Selector for Neural Focusing
    
    Filters obstacle points to a relevant corridor before feeding to DUNE/Flexible PDHG.
    Supports three strategies with automatic switching and guardrail mechanism.
    """
    
    def __init__(self, cfg: ROIConfig):
        self.cfg = cfg
        self._relax_count = 0
        self._tighten_count = 0
        # Working copy of parameters (can be relaxed/tightened)
        self._work_cfg = {
            'ellipse_safety': cfg.ellipse_safety_scale,
            'tube_r0': cfg.tube_r0_m,
            'wedge_fov': cfg.wedge_fov_deg,
            'wedge_r_max': cfg.wedge_r_max_m,
        }
    
    def select(self, pts: np.ndarray, path_xy: Optional[np.ndarray] = None,
               heading_rad: Optional[float] = None, v_robot: Optional[float] = None,
               c_best_hint: Optional[float] = None, goal_xy: Optional[np.ndarray] = None) -> ROIOutputs:
        """
        Select ROI from obstacle points
        
        Args:
            pts: (2, N) obstacle points in global coordinates
            path_xy: (2, T+1) reference path in global coordinates
            heading_rad: robot heading in radians
            v_robot: robot velocity (m/s)
            c_best_hint: current best path length (m)
            goal_xy: (2, 1) goal position in global coordinates
            
        Returns:
            ROIOutputs with filtered points and metadata
        """
        if pts is None or pts.shape[1] == 0:
            return ROIOutputs(pts=pts if pts is not None else np.zeros((2, 0)),
                            strategy='none', n_in=0, n_roi=0)
        
        n_in = pts.shape[1]
        inputs = ROIInputs(pts=pts, path_xy=path_xy, heading_rad=heading_rad,
                          v_robot=v_robot, c_best_hint=c_best_hint, goal_xy=goal_xy)
        
        # Try strategies in order
        for strategy in self.cfg.strategy_order:
            if strategy == 'ellipse' and self._can_use_ellipse(inputs):
                mask = self._ellipse_mask(inputs)
            elif strategy == 'tube' and self._can_use_tube(inputs):
                mask = self._tube_mask(inputs)
            elif strategy == 'wedge' and self._can_use_wedge(inputs):
                mask = self._wedge_mask(inputs)
            else:
                continue
            
            # Apply always_keep safety retention set
            mask = self._apply_always_keep(inputs, mask)

            sel_idx = np.flatnonzero(mask)
            pts_roi = pts[:, sel_idx]
            n_roi = pts_roi.shape[1]

            # Apply guardrail
            if n_roi < self.cfg.guardrail_n_min:
                # Too few points, relax and retry next strategy
                self._relax_parameters()
                continue
            elif n_roi > self.cfg.guardrail_n_max:
                # Too many points, downsample deterministically (preserve indices)
                m = int(self.cfg.guardrail_n_max)
                ds_local = np.linspace(0, n_roi - 1, m, dtype=int)
                sel_idx = sel_idx[ds_local]
                pts_roi = pts[:, sel_idx]
                n_roi = pts_roi.shape[1]

            # Success
            return ROIOutputs(
                pts=pts_roi,
                strategy=strategy,
                n_in=n_in,
                n_roi=n_roi,
                relax_count=self._relax_count,
                tighten_count=self._tighten_count,
                indices=sel_idx,
            )
        
        # Fallback: return all points (with downsampling if needed)
        if n_in > self.cfg.guardrail_n_max:
            m = int(self.cfg.guardrail_n_max)
            sel_idx = np.linspace(0, n_in - 1, m, dtype=int)
            pts_roi = pts[:, sel_idx]
            n_roi = pts_roi.shape[1]
        else:
            pts_roi = pts
            n_roi = n_in
            sel_idx = np.arange(n_in, dtype=int)

        return ROIOutputs(
            pts=pts_roi,
            strategy='none',
            n_in=n_in,
            n_roi=n_roi,
            relax_count=self._relax_count,
            tighten_count=self._tighten_count,
            indices=sel_idx,
        )
    
    def _can_use_ellipse(self, inputs: ROIInputs) -> bool:
        """Check if ellipse strategy can be used"""
        return (inputs.path_xy is not None and inputs.c_best_hint is not None 
                and inputs.c_best_hint > 0)
    
    def _can_use_tube(self, inputs: ROIInputs) -> bool:
        """Check if tube strategy can be used"""
        return inputs.path_xy is not None and inputs.path_xy.shape[1] >= 2
    
    def _can_use_wedge(self, inputs: ROIInputs) -> bool:
        """Check if wedge strategy can be used"""
        return inputs.heading_rad is not None
    
    def _ellipse_mask(self, inputs: ROIInputs) -> np.ndarray:
        """
        Ellipse ROI: Informed RRT* style elliptical corridor
        
        Uses start (first point of path), goal (last point of path), and c_best
        to define an ellipse. Points inside the ellipse are kept.
        """
        path = inputs.path_xy
        s = path[:, 0:1]  # start (2, 1)
        g = path[:, -1:]  # goal (2, 1)
        c_best = inputs.c_best_hint
        
        # Compute ellipse parameters
        v = g - s
        c_min = np.linalg.norm(v) + 1e-6
        
        # Safety check
        if c_best <= c_min * 1.001:
            # c_best too close to c_min, ellipse would be degenerate
            return np.ones(inputs.pts.shape[1], dtype=bool)
        
        # Ellipse center and axes
        center = 0.5 * (s + g)  # (2, 1)
        a = 0.5 * c_best * self._work_cfg['ellipse_safety']
        b = 0.5 * np.sqrt(max(c_best**2 - c_min**2, 1e-6)) * self._work_cfg['ellipse_safety']
        
        # Rotation matrix (align major axis with s->g direction)
        e = v / c_min  # unit vector along major axis
        e_perp = np.array([[-e[1, 0]], [e[0, 0]]])  # perpendicular
        R = np.hstack([e, e_perp])  # (2, 2)
        
        # Transform points to ellipse frame
        X = R.T @ (inputs.pts - center)  # (2, N)
        
        # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        mask = (X[0, :]**2 / a**2 + X[1, :]**2 / b**2) <= 1.0
        
        return mask
    
    def _tube_mask(self, inputs: ROIInputs) -> np.ndarray:
        """
        Tube ROI: Morphological dilation around reference path
        
        Keeps points within adaptive radius of the path.
        Radius = r0 + v_tau * v + kappa_gain * |kappa|
        """
        path = inputs.path_xy  # (2, T+1)
        pts = inputs.pts  # (2, N)
        
        # Compute adaptive radius
        r0 = self._work_cfg['tube_r0']
        v = inputs.v_robot if inputs.v_robot is not None else 0.0
        r = r0 + self.cfg.tube_v_tau_m * abs(v)
        
        # Compute distance from each point to path
        # For each point, find minimum distance to any path segment
        min_dist = np.full(pts.shape[1], np.inf)
        
        for i in range(path.shape[1] - 1):
            p1 = path[:, i:i+1]  # (2, 1)
            p2 = path[:, i+1:i+2]  # (2, 1)
            
            # Distance from pts to line segment p1-p2
            seg_vec = p2 - p1  # (2, 1)
            seg_len_sq = np.sum(seg_vec**2) + 1e-9
            
            # Project pts onto line
            t = np.sum((pts - p1) * seg_vec, axis=0, keepdims=True) / seg_len_sq  # (1, N)
            t = np.clip(t, 0, 1)  # clamp to segment
            
            # Closest point on segment
            closest = p1 + seg_vec * t  # (2, N)
            dist = np.linalg.norm(pts - closest, axis=0)  # (N,)
            
            min_dist = np.minimum(min_dist, dist)
        
        # Estimate curvature and adjust radius (simple approximation)
        if path.shape[1] >= 3:
            # Use maximum turning angle as curvature proxy
            angles = []
            for i in range(path.shape[1] - 2):
                v1 = path[:, i+1] - path[:, i]
                v2 = path[:, i+2] - path[:, i+1]
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                angle = np.arctan2(np.sin(angle), np.cos(angle))  # wrap to [-pi, pi]
                angles.append(abs(angle))
            max_kappa_proxy = max(angles) if angles else 0.0
            r += self.cfg.tube_kappa_gain * max_kappa_proxy
        
        mask = min_dist <= r
        return mask
    
    def _wedge_mask(self, inputs: ROIInputs) -> np.ndarray:
        """
        Wedge ROI: Forward fan-shaped region
        
        Keeps points within FOV cone and max range from robot (assumed at origin in robot frame).
        """
        pts = inputs.pts  # (2, N) in global coords
        heading = inputs.heading_rad
        
        # Transform points to robot frame (robot at origin, heading along x-axis)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R_inv = np.array([[cos_h, sin_h], [-sin_h, cos_h]])  # rotation from global to robot frame
        
        # Assume robot is at first point of path (if available) or origin
        if inputs.path_xy is not None:
            robot_pos = inputs.path_xy[:, 0:1]  # (2, 1)
        else:
            robot_pos = np.zeros((2, 1))
        
        pts_robot = R_inv @ (pts - robot_pos)  # (2, N)
        
        # Wedge criteria: forward (x > 0), within FOV, within range
        fov_rad = np.deg2rad(self._work_cfg['wedge_fov'])
        r_max = self._work_cfg['wedge_r_max']
        
        x = pts_robot[0, :]
        y = pts_robot[1, :]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        mask = (x > 0) & (np.abs(theta) <= fov_rad / 2) & (r <= r_max)
        return mask
    
    def _apply_always_keep(self, inputs: ROIInputs, mask: np.ndarray) -> np.ndarray:
        """
        Apply safety retention set: always keep near-body and goal-vicinity points
        """
        pts = inputs.pts
        
        # Near-body circle (assume robot at first point of path or origin)
        if inputs.path_xy is not None:
            robot_pos = inputs.path_xy[:, 0:1]
        else:
            robot_pos = np.zeros((2, 1))
        
        dist_to_robot = np.linalg.norm(pts - robot_pos, axis=0)
        near_mask = dist_to_robot <= self.cfg.always_keep_near_radius_m
        
        # Goal vicinity (if goal available)
        goal_mask = np.zeros(pts.shape[1], dtype=bool)
        if inputs.goal_xy is not None:
            dist_to_goal = np.linalg.norm(pts - inputs.goal_xy, axis=0)
            goal_mask = dist_to_goal <= self.cfg.always_keep_goal_radius_m
        elif inputs.path_xy is not None:
            # Use last point of path as goal proxy
            goal_pos = inputs.path_xy[:, -1:]
            dist_to_goal = np.linalg.norm(pts - goal_pos, axis=0)
            goal_mask = dist_to_goal <= self.cfg.always_keep_goal_radius_m
        
        # Union of ROI mask and safety masks
        return mask | near_mask | goal_mask
    
    def _relax_parameters(self):
        """Relax ROI parameters to include more points"""
        self._work_cfg['ellipse_safety'] *= self.cfg.guardrail_relax_step
        self._work_cfg['tube_r0'] *= self.cfg.guardrail_relax_step
        self._work_cfg['wedge_fov'] *= self.cfg.guardrail_relax_step
        self._work_cfg['wedge_r_max'] *= self.cfg.guardrail_relax_step
        self._relax_count += 1
    
    def _tighten_parameters(self):
        """Tighten ROI parameters (currently unused, for future adaptive logic)"""
        self._work_cfg['ellipse_safety'] *= self.cfg.guardrail_tighten_step
        self._work_cfg['tube_r0'] *= self.cfg.guardrail_tighten_step
        self._work_cfg['wedge_fov'] *= self.cfg.guardrail_tighten_step
        self._work_cfg['wedge_r_max'] *= self.cfg.guardrail_tighten_step
        self._tighten_count += 1
    
    def _downsample_deterministic(self, pts: np.ndarray, m: int) -> np.ndarray:
        """
        Downsample points deterministically using uniform decimation
        
        Args:
            pts: (2, N) points
            m: target number of points
            
        Returns:
            (2, m) downsampled points
        """
        n = pts.shape[1]
        if n <= m:
            return pts
        
        # Uniform decimation
        indices = np.linspace(0, n - 1, m, dtype=int)
        return pts[:, indices]

