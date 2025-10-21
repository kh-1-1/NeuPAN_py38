'''
neupan file is the main class for the NeuPan algorithm. It wraps the PAN class and provides a more user-friendly interface.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
'''

import yaml
import torch
from neupan.robot import robot
from neupan.blocks import InitialPath, PAN
from neupan import configuration
from neupan.util import time_it, file_check, get_transform
import numpy as np
from neupan.configuration import np_to_tensor, tensor_to_np
from math import cos, sin
from neupan.blocks.roi_selector import ROISelector, ROIConfig

class neupan(torch.nn.Module):

    """
    Args:
        receding: int, the number of steps in the receding horizon.
        step_time: float, the time step in the MPC framework.
        ref_speed: float, the reference speed of the robot.
        device: str, the device to run the algorithm on. 'cpu' or 'cuda'.
        robot_kwargs: dict, the keyword arguments for the robot class.
        ipath_kwargs: dict, the keyword arguments for the initial path class.
        pan_kwargs: dict, the keyword arguments for the PAN class.
        adjust_kwargs: dict, the keyword arguments for the adjust class
        train_kwargs: dict, the keyword arguments for the train class
        time_print: bool, whether to print the forward time of the algorithm.
        collision_threshold: float, the threshold for the collision detection. If collision, the algorithm will stop.
    """

    def __init__(
        self,
        receding: int = 10,
        step_time: float = 0.1,
        ref_speed: float = 4.0,
        device: str = "cpu",
        robot_kwargs: dict = None,
        ipath_kwargs: dict = None,
        pan_kwargs: dict = None,
        adjust_kwargs: dict = None,
        train_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super(neupan, self).__init__()

        # mpc parameters
        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed

        configuration.device = torch.device(device)
        configuration.time_print = kwargs.get("time_print", False)
        self.collision_threshold = kwargs.get("collision_threshold", 0.1)

        # initialization
        self.cur_vel_array = np.zeros((2, self.T))
        self.robot = robot(receding, step_time, **robot_kwargs)

        self.ipath = InitialPath(
            receding, step_time, ref_speed, self.robot, **ipath_kwargs
        )

        pan_kwargs["adjust_kwargs"] = adjust_kwargs
        pan_kwargs["train_kwargs"] = train_kwargs
        self.dune_train_kwargs = train_kwargs

        self.pan = PAN(receding, step_time, self.robot, **pan_kwargs)

        self.info = {"stop": False, "arrive": False, "collision": False}
        # Cache last scan-derived obstacle point velocities (auto-used when caller doesn't pass them)
        self._last_point_velocities = None

        # ROI/Neural Focusing configuration
        roi_kwargs = kwargs.get("roi_kwargs", {})
        self.roi_enabled = bool(roi_kwargs.get("enabled", False))
        self.roi_selector = None
        self._roi_state = {"last_c_best": None}

        if self.roi_enabled:
            # Build ROIConfig from roi_kwargs
            roi_cfg = ROIConfig(
                enabled=True,
                strategy_order=roi_kwargs.get("strategy_order", ['ellipse', 'tube', 'wedge']),
                wedge_fov_deg=roi_kwargs.get("wedge", {}).get("fov_deg", 60.0),
                wedge_r_max_m=roi_kwargs.get("wedge", {}).get("r_max_m", 8.0),
                ellipse_safety_scale=roi_kwargs.get("ellipse", {}).get("safety_scale", 1.08),
                tube_r0_m=roi_kwargs.get("tube", {}).get("r0_m", 0.5),
                tube_kappa_gain=roi_kwargs.get("tube", {}).get("kappa_gain", 0.4),
                tube_v_tau_m=roi_kwargs.get("tube", {}).get("v_tau_m", 0.3),
                guardrail_n_min=roi_kwargs.get("guardrail", {}).get("n_min", 30),
                guardrail_n_max=roi_kwargs.get("guardrail", {}).get("n_max", 500),
                guardrail_relax_step=roi_kwargs.get("guardrail", {}).get("relax_step", 1.15),
                guardrail_tighten_step=roi_kwargs.get("guardrail", {}).get("tighten_step", 0.9),
                always_keep_near_radius_m=roi_kwargs.get("always_keep", {}).get("near_radius_m", 1.5),
                always_keep_goal_radius_m=roi_kwargs.get("always_keep", {}).get("goal_radius_m", 1.5),
            )
            self.roi_selector = ROISelector(roi_cfg)

    @classmethod
    def init_from_yaml(cls, yaml_file, **kwargs):
        abs_path = file_check(yaml_file)

        with open(abs_path, "r") as f:
            config = yaml.safe_load(f)
            config.update(kwargs)

        # Robustly coerce optional sections to dicts (handle None from overrides)
        def _as_dict(section: str) -> dict:
            val = config.pop(section, dict())
            return val if isinstance(val, dict) else dict()

        config["robot_kwargs"] = _as_dict("robot")
        config["ipath_kwargs"] = _as_dict("ipath")
        config["pan_kwargs"] = _as_dict("pan")
        config["adjust_kwargs"] = _as_dict("adjust")
        config["train_kwargs"] = _as_dict("train")
        config["roi_kwargs"] = _as_dict("roi")

        # Auto-inject more permissive arrival settings for diff-drive robots
        try:
            kin = str(config["robot_kwargs"].get("kinematics", "")).lower()
            if kin == "diff":
                ipath_defaults = {
                    "arrive_threshold": 0.4,          # meters (was 0.1)
                    "arrive_index_threshold": 3,       # was 1
                    "ind_range": 20,                   # search window for nearest path index
                }
                for k, v in ipath_defaults.items():
                    config["ipath_kwargs"].setdefault(k, v)
        except Exception:
            pass

        return cls(**config)

    @time_it("neupan forward")
    def forward(self, state, points, velocities=None):
        """
        state: current state of the robot, matrix (3, 1), x, y, theta
        points: current input obstacle point positions, matrix (2, N), N is the number of obstacle points.
        velocities: current velocity of each obstacle point, matrix (2, N), N is the number of obstacle points. vx, vy
        """

        assert state.shape[0] >= 3

        if self.ipath.check_arrive(state):
            self.info["arrive"] = True
            return np.zeros((2, 1)), self.info

        nom_input_np = self.ipath.generate_nom_ref_state(
            state, self.cur_vel_array, self.ref_speed
        )

        # convert to tensor
        nom_input_tensor = [np_to_tensor(n) for n in nom_input_np]

        # Apply ROI filtering if enabled
        if self.roi_enabled and points is not None:
            points = self._apply_roi(points, nom_input_np, state)

        obstacle_points_tensor = np_to_tensor(points) if points is not None else None
        # Auto-enable obstacle point velocities: if caller doesn't provide, but last scan carried it, use cached
        if velocities is None and getattr(self, "_last_point_velocities", None) is not None:
            velocities = self._last_point_velocities
        point_velocities_tensor = (
            np_to_tensor(velocities) if velocities is not None else None
        )

        opt_state_tensor, opt_vel_tensor, opt_distance_tensor = self.pan(
            *nom_input_tensor, obstacle_points_tensor, point_velocities_tensor
        )

        opt_state_np, opt_vel_np = tensor_to_np(opt_state_tensor), tensor_to_np(
            opt_vel_tensor
        )

        self.cur_vel_array = opt_vel_np

        self.info["state_tensor"] = opt_state_tensor
        self.info["vel_tensor"] = opt_vel_tensor
        self.info["distance_tensor"] = opt_distance_tensor
        self.info['ref_state_tensor'] = nom_input_tensor[2]
        self.info['ref_speed_tensor'] = nom_input_tensor[3]

        # propagate PDHG profiling info to info dict if available
        try:
            dune_layer = self.pan.dune_layer
            self.info['pdhg_profile'] = getattr(dune_layer, 'pdhg_profile', None) if dune_layer is not None else None
        except Exception:
            pass

        self.info["ref_state_list"] = [
            state[:, np.newaxis] for state in nom_input_np[2].T
        ]
        self.info["opt_state_list"] = [state[:, np.newaxis] for state in opt_state_np.T]

        if self.check_stop():
            self.info["stop"] = True
            return np.zeros((2, 1)), self.info
        else:
            self.info["stop"] = False

        action = opt_vel_np[:, 0:1]

        return action, self.info

    def check_stop(self):
        return self.min_distance < self.collision_threshold

    def _apply_roi(self, points: np.ndarray, nom_input_np: list, state: np.ndarray) -> np.ndarray:
        """
        Apply ROI filtering to obstacle points

        Args:
            points: (2, N) obstacle points in global coordinates
            nom_input_np: list of nominal inputs [nom_s, nom_u, ref_s, ref_us]
            state: (3, 1) current robot state [x, y, theta]

        Returns:
            (2, N_roi) filtered obstacle points in global coordinates
        """
        # Select reference path (prioritize previous optimized trajectory)
        path_xy = self._select_path(nom_input_np)

        # Estimate c_best from path
        c_best = self._estimate_cbest(path_xy)

        # Extract robot state
        heading_rad = float(state[2, 0])
        v_robot = float(self.cur_vel_array[0, 0]) if self.cur_vel_array.shape[1] > 0 else 0.0

        # Apply ROI selection
        roi_output = self.roi_selector.select(
            pts=points,
            path_xy=path_xy,
            heading_rad=heading_rad,
            v_robot=v_robot,
            c_best_hint=c_best,
            goal_xy=None  # Could extract from ipath if needed
        )

        # Log ROI statistics
        self.info["roi"] = {
            "strategy": roi_output.strategy,
            "n_in": roi_output.n_in,
            "n_roi": roi_output.n_roi,
            "relax_count": roi_output.relax_count,
            "tighten_count": roi_output.tighten_count,
        }

        # Update state for visualization
        self._roi_state["last_c_best"] = c_best
        self._roi_state["path_xy"] = path_xy
        self._roi_state["robot_xy"] = state[:2, :].copy()
        self._roi_state["heading_rad"] = heading_rad
        self._roi_state["roi_params_snapshot"] = getattr(self.roi_selector, "_work_cfg", {}).copy() if self.roi_selector else {}
        self._roi_state["pts_roi"] = roi_output.pts

        return roi_output.pts

    def _select_path(self, nom_input_np: list) -> np.ndarray:
        """
        Select reference path for ROI

        Priority:
        1. Previous cycle optimized trajectory (if available)
        2. Current cycle reference trajectory

        Args:
            nom_input_np: list of nominal inputs [nom_s, nom_u, ref_s, ref_us]

        Returns:
            (2, T+1) reference path in global coordinates
        """
        # Try previous optimized trajectory first
        if "opt_state_list" in self.info and self.info["opt_state_list"]:
            opt_states = self.info["opt_state_list"]
            # Extract x, y from state list (each state is (3, 1))
            path_xy = np.hstack([s[:2, :] for s in opt_states])  # (2, T+1)
            return path_xy

        # Fallback to current reference trajectory
        ref_s = nom_input_np[2]  # (3, T+1)
        path_xy = ref_s[:2, :]  # (2, T+1)
        return path_xy

    def _estimate_cbest(self, path_xy: np.ndarray) -> float:
        """
        Estimate c_best (current best path length) from reference path

        Uses geometric arc length of the path with safety margin.

        Args:
            path_xy: (2, T+1) reference path in global coordinates

        Returns:
            c_best: estimated path length (meters)
        """
        if path_xy is None or path_xy.shape[1] < 2:
            return None

        # Compute arc length
        diffs = np.diff(path_xy, axis=1)  # (2, T)
        seg_lengths = np.linalg.norm(diffs, axis=0)  # (T,)
        arc_length = np.sum(seg_lengths)

        # Apply safety margin and lower bound
        c_min = np.linalg.norm(path_xy[:, -1] - path_xy[:, 0])  # straight-line distance
        c_best = max(arc_length, c_min * 1.001)  # ensure c_best > c_min

        return float(c_best)

    def visualize_roi_region(self, env):
        """
        Visualize current ROI region (ellipse/tube/wedge) in the simulation environment

        Args:
            env: irsim environment with draw_points/draw_trajectory methods
        """
        if not self.roi_enabled or "roi" not in self.info:
            return

        strategy = self.info["roi"].get("strategy", "none")
        if strategy == "none":
            return

        # Draw filtered points (ROI output) in orange for comparison
        pts_roi = self._roi_state.get("pts_roi")
        if pts_roi is not None and pts_roi.shape[1] > 0:
            env.draw_points(pts_roi, s=18, c="orange", alpha=0.6, refresh=True)

        # Draw ROI region boundary based on strategy
        if strategy == "ellipse":
            self._visualize_ellipse_roi(env)
        elif strategy == "tube":
            self._visualize_tube_roi(env)
        elif strategy == "wedge":
            self._visualize_wedge_roi(env)

    def _visualize_ellipse_roi(self, env):
        """Visualize ellipse ROI region boundary"""
        path_xy = self._roi_state.get("path_xy")
        c_best = self._roi_state.get("last_c_best")
        params = self._roi_state.get("roi_params_snapshot", {})

        if path_xy is None or c_best is None or path_xy.shape[1] < 2:
            return

        # Ellipse parameters
        start = path_xy[:, 0:1]  # (2, 1)
        goal = path_xy[:, -1:]   # (2, 1)
        c_min = np.linalg.norm(goal - start) + 1e-6

        if c_best <= c_min * 1.001:
            return  # Degenerate ellipse

        safety_scale = params.get("ellipse_safety", 1.08)
        center = 0.5 * (start + goal)
        a = 0.5 * c_best * safety_scale
        b = 0.5 * np.sqrt(max(c_best**2 - c_min**2, 1e-6)) * safety_scale

        # Rotation matrix (align major axis with start->goal)
        v = goal - start
        e = v / c_min  # unit vector along major axis
        e_perp = np.array([[-e[1, 0]], [e[0, 0]]])
        R = np.hstack([e, e_perp])  # (2, 2)

        # Sample ellipse boundary
        angles = np.linspace(0, 2*np.pi, 100)
        ellipse_local = np.vstack([a * np.cos(angles), b * np.sin(angles)])  # (2, 100)
        ellipse_global = R @ ellipse_local + center  # (2, 100)

        # Draw as points (yellow ellipse boundary)
        # Convert (2, N) to list of [x, y] to avoid irsim bug
        ellipse_list = [[ellipse_global[0, i], ellipse_global[1, i]] for i in range(ellipse_global.shape[1])]
        env.draw_points(ellipse_list, s=8, c="yellow", alpha=0.7, refresh=True)

    def _visualize_tube_roi(self, env):
        """Visualize tube ROI region boundary"""
        path_xy = self._roi_state.get("path_xy")
        params = self._roi_state.get("roi_params_snapshot", {})

        if path_xy is None or path_xy.shape[1] < 2:
            return

        # Tube radius
        tube_r0 = params.get("tube_r0", 0.5)
        v_robot = float(self.cur_vel_array[0, 0]) if self.cur_vel_array.shape[1] > 0 else 0.0
        r = tube_r0 + self.roi_selector.cfg.tube_v_tau_m * abs(v_robot)

        # Estimate curvature adjustment (simplified)
        if path_xy.shape[1] >= 3:
            angles = []
            for i in range(path_xy.shape[1] - 2):
                v1 = path_xy[:, i+1] - path_xy[:, i]
                v2 = path_xy[:, i+2] - path_xy[:, i+1]
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                angles.append(abs(angle))
            max_kappa_proxy = max(angles) if angles else 0.0
            r += self.roi_selector.cfg.tube_kappa_gain * max_kappa_proxy

        # Generate boundary points (left and right offset)
        left_boundary = []
        right_boundary = []

        for i in range(path_xy.shape[1] - 1):
            p1 = path_xy[:, i]
            p2 = path_xy[:, i+1]
            tangent = p2 - p1
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-6:
                continue
            tangent = tangent / tangent_norm
            normal = np.array([-tangent[1], tangent[0]])  # perpendicular

            left_boundary.append(p1 + r * normal)
            right_boundary.append(p1 - r * normal)

        # Add last point
        if path_xy.shape[1] >= 2:
            p_last = path_xy[:, -1]
            tangent = path_xy[:, -1] - path_xy[:, -2]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm >= 1e-6:
                tangent = tangent / tangent_norm
                normal = np.array([-tangent[1], tangent[0]])
                left_boundary.append(p_last + r * normal)
                right_boundary.append(p_last - r * normal)

        if left_boundary:
            left_pts = np.column_stack(left_boundary)  # (2, N)
            right_pts = np.column_stack(right_boundary)
            # Convert to list format to avoid irsim bug
            left_list = [[left_pts[0, i], left_pts[1, i]] for i in range(left_pts.shape[1])]
            right_list = [[right_pts[0, i], right_pts[1, i]] for i in range(right_pts.shape[1])]
            env.draw_points(left_list, s=8, c="cyan", alpha=0.7, refresh=True)
            env.draw_points(right_list, s=8, c="cyan", alpha=0.7, refresh=True)

    def _visualize_wedge_roi(self, env):
        """Visualize wedge ROI region boundary"""
        robot_xy = self._roi_state.get("robot_xy")
        heading_rad = self._roi_state.get("heading_rad")
        params = self._roi_state.get("roi_params_snapshot", {})

        if robot_xy is None or heading_rad is None:
            return

        # Wedge parameters
        fov_deg = params.get("wedge_fov", 60.0)
        r_max = params.get("wedge_r_max", 8.0)
        fov_rad = np.deg2rad(fov_deg)

        # Sample wedge boundary (arc + two rays)
        # Arc
        angles = np.linspace(heading_rad - fov_rad/2, heading_rad + fov_rad/2, 50)
        arc_x = robot_xy[0, 0] + r_max * np.cos(angles)
        arc_y = robot_xy[1, 0] + r_max * np.sin(angles)
        arc_pts = np.vstack([arc_x, arc_y])  # (2, 50)

        # Left ray
        left_angle = heading_rad - fov_rad/2
        left_ray_x = np.linspace(robot_xy[0, 0], robot_xy[0, 0] + r_max * np.cos(left_angle), 20)
        left_ray_y = np.linspace(robot_xy[1, 0], robot_xy[1, 0] + r_max * np.sin(left_angle), 20)
        left_ray_pts = np.vstack([left_ray_x, left_ray_y])

        # Right ray
        right_angle = heading_rad + fov_rad/2
        right_ray_x = np.linspace(robot_xy[0, 0], robot_xy[0, 0] + r_max * np.cos(right_angle), 20)
        right_ray_y = np.linspace(robot_xy[1, 0], robot_xy[1, 0] + r_max * np.sin(right_angle), 20)
        right_ray_pts = np.vstack([right_ray_x, right_ray_y])

        # Draw all boundary elements (convert to list format to avoid irsim bug)
        arc_list = [[arc_pts[0, i], arc_pts[1, i]] for i in range(arc_pts.shape[1])]
        left_ray_list = [[left_ray_pts[0, i], left_ray_pts[1, i]] for i in range(left_ray_pts.shape[1])]
        right_ray_list = [[right_ray_pts[0, i], right_ray_pts[1, i]] for i in range(right_ray_pts.shape[1])]
        env.draw_points(arc_list, s=8, c="red", alpha=0.7, refresh=True)
        env.draw_points(left_ray_list, s=8, c="red", alpha=0.7, refresh=True)
        env.draw_points(right_ray_list, s=8, c="red", alpha=0.7, refresh=True)

    def scan_to_point(
        self,
        state: np.ndarray,
        scan: dict,
        scan_offset = [0, 0, 0],
        angle_range = [-np.pi, np.pi],
        down_sample: int = 1,
    ):

        """
        input:
            state: [x, y, theta]
            scan: {}
                ranges: list[float], the range of the scan
                angle_min: float, the minimum angle of the scan
                angle_max: float, the maximum angle of the scan
                range_max: float, the maximum range of the scan
                range_min: float, the minimum range of the scan

            scan_offset: [x, y, theta], the relative position of the sensor to the robot state coordinate

        return point cloud: (2, n)
        """
        point_cloud = []
        velocity_points = []

        ranges = np.array(scan["ranges"])
        angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range > scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)
                    # If per-ray velocity is available in scan, align and cache it
                    if "velocity" in scan:
                        v = np.asarray(scan["velocity"])  # expect shape (2, N)
                        if v.ndim == 2 and v.shape[1] == len(ranges):
                            velocity_points.append(v[:, i : i + 1])

        if len(point_cloud) == 0:
            # Clear cache when no points
            self._last_point_velocities = None
            return None

        point_array = np.hstack(point_cloud)
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R @ point_array + s_trans

        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        # Cache aligned velocities if provided in scan (assumed already in global frame; keep consistent with scan_to_point_velocity)
        if velocity_points:
            try:
                velocity = np.hstack(velocity_points)[:, ::down_sample]
                self._last_point_velocities = velocity
            except Exception:
                self._last_point_velocities = None
        else:
            self._last_point_velocities = None

        return points

    def scan_to_point_velocity(
        self,
        state,
        scan,
        scan_offset=[0, 0, 0],
        angle_range=[-np.pi, np.pi],
        down_sample=1,
    ):
        """
        input:
            state: [x, y, theta]
            scan: {}
                ranges: list[float], the ranges of the scan
                angle_min: float, the minimum angle of the scan
                angle_max: float, the maximum angle of the scan
                range_max: float, the maximum range of the scan
                range_min: float, the minimum range of the scan
                velocity: list[float], the velocity of the scan

            scan_offset: [x, y, theta], the relative position of the sensor to the robot state coordinate

        return point cloud: (2, n)
        """
        point_cloud = []
        velocity_points = []

        ranges = np.array(scan["ranges"])
        angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))
        scan_velocity = scan.get("velocity", np.zeros((2, len(ranges))))

        # lidar_state = self.lidar_state_transform(state, np.c_[self.lidar_offset])
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range >= scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)
                    velocity_points.append(scan_velocity[:, i : i + 1])

        if len(point_cloud) == 0:
            return None, None

        point_array = np.hstack(point_cloud)
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R.T @ (
            point_array - s_trans
        )  # points in the robot state coordinate

        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        velocity = np.hstack(velocity_points)[:, ::down_sample]

        return points, velocity


    def train_dune(self):
        self.pan.dune_layer.train_dune(self.dune_train_kwargs)


    def reset(self):
        self.ipath.point_index = 0
        self.ipath.curve_index = 0
        self.ipath.arrive_flag = False
        self.info["stop"] = False
        self.info["arrive"] = False
        self.cur_vel_array = np.zeros_like(self.cur_vel_array)

    def set_initial_path(self, path):

        '''
        set the initial path from the given path
        path: list of [x, y, theta, gear] 4x1 vector, gear is -1 (back gear) or 1 (forward gear)
        '''

        self.ipath.set_initial_path(path)

    def set_initial_path_from_state(self, state):
        """
        Args:
            states: [x, y, theta] or 3x1 vector

        """
        self.ipath.init_check(state)

    def set_reference_speed(self, speed: float):

        """
        Args:
            speed: float, the reference speed of the robot
        """

        self.ipath.ref_speed = speed
        self.ref_speed = speed

    def update_initial_path_from_goal(self, start, goal):

        """
        Args:
            start: [x, y, theta] or 3x1 vector
            goal: [x, y, theta] or 3x1 vector
        """

        self.ipath.update_initial_path_from_goal(start, goal)


    def update_initial_path_from_waypoints(self, waypoints):

        """
        Args:
            waypoints: list of [x, y, theta] or 3x1 vector
        """

        self.ipath.set_ipath_with_waypoints(waypoints)


    def update_adjust_parameters(self, **kwargs):

        """
        update the adjust parameters value: q_s, p_u, eta, d_max, d_min

        Args:
            q_s: float, the weight of the state cost
            p_u: float, the weight of the speed cost
            eta: float, the weight of the collision avoidance cost
            d_max: float, the maximum distance to the obstacle
            d_min: float, the minimum distance to the obstacle
        """

        self.pan.nrmp_layer.update_adjust_parameters_value(**kwargs)

    @property
    def min_distance(self):
        return self.pan.min_distance

    @property
    def dune_points(self):
        return self.pan.dune_points

    @property
    def nrmp_points(self):
        return self.pan.nrmp_points

    @property
    def initial_path(self):
        return self.ipath.initial_path

    @property
    def adjust_parameters(self):
        return self.pan.nrmp_layer.adjust_parameters

    @property
    def waypoints(self):

        '''
        Waypoints for generating the initial path
        '''

        return self.ipath.waypoints

    @property
    def opt_trajectory(self):

        '''
        MPC receding horizon trajectory under the velocity sequence
        return a list of state sequence, each state is a 3x1 vector
        '''

        return self.info["opt_state_list"]

    @property
    def ref_trajectory(self):

        '''
        Reference trajectory on the initial path
        return a list of state sequence, each state is a 3x1 vector
        '''

        return self.info["ref_state_list"]





