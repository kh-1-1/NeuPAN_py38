'''
ROI (Region of Interest) Selector for Neural Focusing in NeuPAN

Implements Reachability Cone strategy optimized for MPC prediction horizon:
- Cone: Reachability-based cone filtering using dynamic FOV and max reachable distance
  * O(N) computational complexity (vs O(N×T) for path-based methods)
  * Adapts FOV based on path curvature (straight: 90°, curved: up to 150°)
  * Considers robot dynamics and prediction horizon
  * Optional reverse motion detection

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
    strategy_order: list = field(default_factory=lambda: ['cone'])

    # Reachability Cone parameters (optimized for MPC prediction horizon)
    cone_fov_base_deg: float = 90.0          # Base field of view (degrees)
    cone_r_max_m: float = 8.0                # Maximum reachable distance (meters)
    cone_expansion_factor: float = 100.0     # FOV expansion factor for curved paths
    cone_safety_margin_m: float = 0.5        # Safety margin added to R_max (meters)
    cone_enable_reverse: bool = False        # Enable reverse cone for backward motion
    cone_reverse_fov_deg: float = 60.0       # FOV for reverse cone (degrees)

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
    heading_rad: Optional[float] = None  # robot heading for cone/wedge
    v_robot: Optional[float] = None  # robot velocity (for cone radius and reverse detection)
    c_best_hint: Optional[float] = None  # current best path length
    goal_xy: Optional[np.ndarray] = None  # (2, 1) goal position
    robot_xy: Optional[np.ndarray] = None  # (2, 1) current robot position (for cone/always-keep)


@dataclass
class ROIOutputs:
    """Outputs from ROI selection"""
    pts: np.ndarray  # (2, N_roi) filtered points in global coords
    strategy: str  # 'cone' or 'none'
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
    Uses Reachability Cone strategy optimized for MPC prediction horizon with automatic
    parameter adaptation via guardrail mechanism.
    """
    
    def __init__(self, cfg: ROIConfig):
        self.cfg = cfg
        self._relax_count = 0
        self._tighten_count = 0
        # Working copy of parameters (can be relaxed/tightened)
        self._work_cfg = {
            'cone_fov': cfg.cone_fov_base_deg,
            'cone_r_max': cfg.cone_r_max_m,
        }
    
    def select(self, pts: np.ndarray, path_xy: Optional[np.ndarray] = None,
               heading_rad: Optional[float] = None, v_robot: Optional[float] = None,
               c_best_hint: Optional[float] = None, goal_xy: Optional[np.ndarray] = None,
               robot_xy: Optional[np.ndarray] = None) -> ROIOutputs:
        """
        Select ROI from obstacle points

        Args:
            pts: (2, N) obstacle points in global coordinates
            path_xy: (2, T+1) reference path in global coordinates
            heading_rad: robot heading in radians
            v_robot: robot velocity (m/s)
            c_best_hint: current best path length (m)
            goal_xy: (2, 1) goal position in global coordinates
            robot_xy: (2, 1) current robot position in global coordinates

        Returns:
            ROIOutputs with filtered points and metadata
        """
        if pts is None or pts.shape[1] == 0:
            return ROIOutputs(pts=pts if pts is not None else np.zeros((2, 0)),
                            strategy='none', n_in=0, n_roi=0)

        # Reset parameters and counters for each frame
        # This ensures parameters don't accumulate relaxation across frames
        self._work_cfg = {
            'cone_fov': self.cfg.cone_fov_base_deg,
            'cone_r_max': self.cfg.cone_r_max_m,
        }
        self._relax_count = 0
        self._tighten_count = 0

        n_in = pts.shape[1]
        inputs = ROIInputs(pts=pts, path_xy=path_xy, heading_rad=heading_rad,
                          v_robot=v_robot, c_best_hint=c_best_hint, goal_xy=goal_xy,
                          robot_xy=robot_xy)

        # Direct cone strategy (simplified - no strategy chain)
        if self._can_use_cone(inputs):
            # Apply reachability cone mask
            mask = self._reachability_cone_mask(inputs)

            # Apply always_keep safety retention set
            mask = self._apply_always_keep(inputs, mask)

            sel_idx = np.flatnonzero(mask)
            pts_roi = pts[:, sel_idx]
            n_roi = pts_roi.shape[1]

            # Apply guardrail
            if n_roi < self.cfg.guardrail_n_min:
                # Too few points, relax parameters and retry
                self._relax_parameters()
                # Retry with relaxed parameters
                mask = self._reachability_cone_mask(inputs)
                mask = self._apply_always_keep(inputs, mask)
                sel_idx = np.flatnonzero(mask)
                pts_roi = pts[:, sel_idx]
                n_roi = pts_roi.shape[1]

                # If still too few after relaxation, fall back to returning all points
                if n_roi < self.cfg.guardrail_n_min:
                    # Use fallback logic below
                    pass
                else:
                    # Relaxation succeeded, check if too many
                    if n_roi > self.cfg.guardrail_n_max:
                        m = int(self.cfg.guardrail_n_max)
                        ds_local = np.linspace(0, n_roi - 1, m, dtype=int)
                        sel_idx = sel_idx[ds_local]
                        pts_roi = pts[:, sel_idx]
                        n_roi = pts_roi.shape[1]

                    # Success after relaxation
                    return ROIOutputs(
                        pts=pts_roi,
                        strategy='cone',
                        n_in=n_in,
                        n_roi=n_roi,
                        relax_count=self._relax_count,
                        tighten_count=self._tighten_count,
                        indices=sel_idx,
                    )

            elif n_roi > self.cfg.guardrail_n_max:
                # Too many points, downsample deterministically
                m = int(self.cfg.guardrail_n_max)
                ds_local = np.linspace(0, n_roi - 1, m, dtype=int)
                sel_idx = sel_idx[ds_local]
                pts_roi = pts[:, sel_idx]
                n_roi = pts_roi.shape[1]

                # Success
                return ROIOutputs(
                    pts=pts_roi,
                    strategy='cone',
                    n_in=n_in,
                    n_roi=n_roi,
                    relax_count=self._relax_count,
                    tighten_count=self._tighten_count,
                    indices=sel_idx,
                )
            else:
                # Point count is within range, success
                return ROIOutputs(
                    pts=pts_roi,
                    strategy='cone',
                    n_in=n_in,
                    n_roi=n_roi,
                    relax_count=self._relax_count,
                    tighten_count=self._tighten_count,
                    indices=sel_idx,
                )

        # Fallback: cone requirements not met, return all points (with downsampling if needed)
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
    
    def _can_use_cone(self, inputs: ROIInputs) -> bool:
        """Check if reachability cone strategy can be used

        Requires:
        - robot_xy: current robot position (for cone center)
        - heading_rad: robot heading (for cone axis)
        - path_xy: optional, used for curvature calculation (for dynamic FOV)
        """
        return (inputs.robot_xy is not None and inputs.heading_rad is not None)

    def _reachability_cone_mask(self, inputs: ROIInputs) -> np.ndarray:
        """
        Reachability Cone ROI: Dynamic cone based on prediction horizon reachability

        Core principle: Only keep obstacle points that are within the robot's reachable
        region during the prediction horizon, considering:
        1. Maximum reachable distance (based on max velocity and time horizon)
        2. Dynamic field of view (expands for curved paths)
        3. Robot shape (safety margin)
        4. Optional reverse motion detection

        This strategy is optimized for MPC with short prediction horizons, where:
        - Near-term predictions are accurate (narrow cone)
        - Far-term predictions may change (wider cone via FOV expansion)
        - Computational efficiency is critical (O(N) complexity)

        Args:
            inputs: ROIInputs containing robot state and predicted path

        Returns:
            Boolean mask (N,) indicating which points to keep
        """
        pts = inputs.pts  # (2, N) in global coords
        robot_pos = inputs.robot_xy  # (2, 1)
        heading = inputs.heading_rad  # scalar
        path = inputs.path_xy  # (2, T+1)
        v_robot = inputs.v_robot if inputs.v_robot is not None else 0.0

        # Step 1: Calculate maximum reachable distance R_max
        # R_max = v_max * T_horizon + r_vehicle + safety_margin

        # Estimate v_max from path (if available) or use default
        if path is not None and path.shape[1] >= 2:
            # Estimate from path segment lengths and time
            path_dists = np.linalg.norm(np.diff(path, axis=1), axis=0)
            v_max_estimate = np.max(path_dists) / 0.1  # Assume 0.1s time step
            v_max = max(v_max_estimate, 8.0)  # At least 8 m/s
        else:
            v_max = 8.0  # Default max velocity

        # Time horizon (assume 1 second for typical MPC)
        T_horizon = 1.0

        # Vehicle radius (approximate from typical robot size)
        # For a robot with length=1.6m, width=2.0m: r_vehicle ≈ sqrt(0.8^2 + 1.0^2) ≈ 1.28m
        r_vehicle = 1.3  # Conservative estimate

        # Total maximum reachable distance
        R_max = v_max * T_horizon + r_vehicle + self.cfg.cone_safety_margin_m

        # Override with config if explicitly set
        if self._work_cfg['cone_r_max'] > 0:
            R_max = self._work_cfg['cone_r_max']

        # Step 2: Calculate path curvature to determine dynamic FOV
        # Higher curvature → wider FOV to cover turning region

        kappa_max = 0.0
        if path is not None and path.shape[1] >= 3:
            # Calculate maximum angle change between consecutive segments
            for i in range(path.shape[1] - 2):
                v1 = path[:, i+1] - path[:, i]
                v2 = path[:, i+2] - path[:, i+1]

                # Avoid division by zero
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                # Angle between vectors
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                # Normalize to [-pi, pi]
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                kappa_max = max(kappa_max, abs(angle))

        # Step 3: Calculate dynamic FOV
        FOV_base = self._work_cfg['cone_fov']  # degrees

        if kappa_max < 0.1:  # Straight path (< 0.1 rad ≈ 5.7 degrees)
            FOV = FOV_base
        else:
            # Expand FOV for curved paths
            FOV_expansion = kappa_max * self.cfg.cone_expansion_factor
            FOV = min(FOV_base + FOV_expansion, 150.0)  # Cap at 150 degrees

        FOV_rad = np.deg2rad(FOV)

        # Step 4: Handle reverse motion (optional)
        cone_axis = heading  # Default: forward direction

        if self.cfg.cone_enable_reverse and v_robot < -0.1:  # Moving backward
            # Reverse the cone axis by 180 degrees
            cone_axis = heading + np.pi
            # Normalize to [-pi, pi]
            cone_axis = np.arctan2(np.sin(cone_axis), np.cos(cone_axis))
            # Use narrower FOV for reverse
            FOV_rad = np.deg2rad(self.cfg.cone_reverse_fov_deg)

        # Step 5: Filter obstacle points
        # Calculate relative position of each point
        v = pts - robot_pos  # (2, N)

        # Distance from robot
        dist = np.linalg.norm(v, axis=0)  # (N,)

        # Angle of each point relative to global x-axis
        angle_pts = np.arctan2(v[1, :], v[0, :])  # (N,)

        # Angle difference from cone axis
        angle_diff = angle_pts - cone_axis
        # Normalize to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Keep points within cone
        mask = (dist <= R_max) & (np.abs(angle_diff) <= FOV_rad / 2)

        return mask


    def _apply_always_keep(self, inputs: ROIInputs, mask: np.ndarray) -> np.ndarray:
        """
        Apply safety retention set: always keep near-body and goal-vicinity points
        """
        pts = inputs.pts

        # Near-body circle: use robot_xy if available, otherwise fallback to path start
        if inputs.robot_xy is not None:
            robot_pos = inputs.robot_xy
        elif inputs.path_xy is not None:
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
        self._work_cfg['cone_fov'] *= self.cfg.guardrail_relax_step
        self._work_cfg['cone_r_max'] *= self.cfg.guardrail_relax_step
        self._relax_count += 1

    def _tighten_parameters(self):
        """Tighten ROI parameters (currently unused, for future adaptive logic)"""
        self._work_cfg['cone_fov'] *= self.cfg.guardrail_tighten_step
        self._work_cfg['cone_r_max'] *= self.cfg.guardrail_tighten_step
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

