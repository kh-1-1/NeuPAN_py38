import argparse
import torch

import irsim
from neupan import neupan


def simulate(
    example_name: str,
    kinematics: str,
    projection: str,
    save_animation: bool,
    ani_name: str,
    full: bool,
    no_display: bool,
    point_vel: bool,
    max_steps: int,
    reverse: bool,
):
    env_file = f"example/{example_name}/{kinematics}/env.yaml"
    planner_file = f"example/{example_name}/{kinematics}/planner.yaml"

    env = irsim.make(env_file, save_ani=save_animation, full=full, display=no_display)

    train_overrides = dict(
        projection=projection,
        monitor_dual_norm=True,
        unroll_J=0,
        se2_embed=False,
    )

    planner = neupan.init_from_yaml(planner_file, train=train_overrides)

    print(f"projection={projection}")

    pre_excess_records = []
    post_excess_records = []

    for step in range(max_steps):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()

        if point_vel:
            points, point_velocities = planner.scan_to_point_velocity(robot_state, lidar_scan)
        else:
            points = planner.scan_to_point(robot_state, lidar_scan)
            point_velocities = None

        action, info = planner(robot_state, points, point_velocities)

        dune_layer = getattr(planner.pan, "dune_layer", None)
        if dune_layer is not None:
            pre_v = getattr(dune_layer, "dual_norm_violation_rate", None)
            pre_p95 = getattr(dune_layer, "dual_norm_p95", None)
            pre_exc = getattr(dune_layer, "dual_norm_max_excess_pre", None)
            post_exc = getattr(dune_layer, "dual_norm_max_excess_post", None)

            current_values = getattr(planner.pan, "current_nom_values", None)
            lam_list = current_values[3] if current_values else []
            post_max = float("nan")
            if lam_list:
                vals = []
                for lam in lam_list:
                    if lam is None or lam.numel() == 0:
                        continue
                    vals.append(torch.norm(lam, dim=0).max().item())
                if vals:
                    post_max = max(vals)

            if pre_exc is not None:
                pre_excess_records.append(pre_exc)
            if post_exc is not None:
                post_excess_records.append(post_exc)

            print(
                f"  step {step:03d} | pre-viol {pre_v} | pre-p95 {pre_p95} "
                f"| post-max {post_max} | pre-excess {pre_exc} | post-excess {post_exc}"
            )

        if info["stop"]:
            print("NeuPAN stops because of minimum distance")

        if info["arrive"]:
            print("NeuPAN arrives at the target")
            break

        env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(planner.opt_trajectory, "r", refresh=True)
        env.draw_trajectory(planner.ref_trajectory, "b", refresh=True)
        env.render()

        env.step(action)

        if env.done():
            break

        if step == 0:
            if reverse:
                for j in range(len(planner.initial_path)):
                    planner.initial_path[j][-1, 0] = -1
                    planner.initial_path[j][-2, 0] = planner.initial_path[j][-2, 0] + 3.14

                env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=True)
            else:
                env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=False)

            env.render()

    env.end(3, ani_name=ani_name)

    if pre_excess_records:
        avg_pre_exc = sum(pre_excess_records) / len(pre_excess_records)
    else:
        avg_pre_exc = 0.0

    if post_excess_records:
        avg_post_exc = sum(post_excess_records) / len(post_excess_records)
    else:
        avg_post_exc = 0.0

    print(f"  avg pre-proj excess: {avg_pre_exc}")
    print(f"  avg post-proj excess: {avg_post_exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", type=str, default="polygon_robot", help="pf, pf_obs, corridor, dyna_obs, dyna_non_obs, convex_obs, non_obs, polygon_robot, reverse")
    parser.add_argument("-d", "--kinematics", type=str, default="diff", help="acker, diff")
    parser.add_argument("-a", "--save_animation", action="store_true", help="save animation")
    parser.add_argument("-f", "--full", action="store_true", help="full screen")
    parser.add_argument("-n", "--no_display", action="store_false", help="no display")
    parser.add_argument("-v", "--point_vel", action='store_true', help="point vel")
    parser.add_argument("-m", "--max_steps", type=int, default=1000, help="max steps")
    parser.add_argument("--projection", type=str, default="hard", choices=["hard", "none", "learned"], help="projection mode passed to DUNE")
    parser.add_argument("--compare", action="store_true", help="run hard and none sequentially")
    args = parser.parse_args()

    projections = [args.projection]
    if args.compare:
        projections = ["hard", "none"]

    ani_base = f"{args.example}_{args.kinematics}"

    for proj in projections:
        simulate(
            args.example,
            args.kinematics,
            proj,
            args.save_animation,
            f"{ani_base}_{proj}",
            args.full,
            args.no_display,
            args.point_vel,
            args.max_steps,
            reverse=(args.example == "reverse" and args.kinematics == "diff"),
        )


if __name__ == "__main__":
    main()
