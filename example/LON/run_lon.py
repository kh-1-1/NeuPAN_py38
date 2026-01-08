import irsim
from neupan import neupan
import torch
import numpy as np
import argparse
import os
import sys
import yaml
from diffcp.cone_program import SolverError

# 增加项目根目录到路径，防止导入错误
sys.path.append(os.getcwd())

def cal_distance_loss(distance, min_distance, collision_threshold, stuck):
    """
    计算基于距离的惩罚损失
    """
    # 如果距离过近（碰撞风险），惩罚急剧增加
    if min_distance <= collision_threshold:
        distance_loss = 50 - torch.sum(distance)
    # 如果卡住了，也给予惩罚
    elif stuck:
        distance_loss = 50 + torch.sum(distance)
    else:
        # 正常行驶时，我们也希望保持一定距离，但允许 requires_grad
        distance_loss = torch.tensor(0.0, requires_grad=True)

    return distance_loss

def cal_loss(neupan_planner, stuck):
    """
    计算总损失：状态跟踪误差 + 速度误差 + 距离惩罚
    """
    states = neupan_planner.info['state_tensor']
    vel = neupan_planner.info["vel_tensor"]
    distance = neupan_planner.info["distance_tensor"]
    
    ref_state_tensor = neupan_planner.info['ref_state_tensor']
    ref_speed_tensor = neupan_planner.info['ref_speed_tensor']

    # 1. 轨迹跟踪损失
    state_loss = torch.nn.MSELoss()(states, ref_state_tensor)
    # 2. 速度跟踪损失
    speed_loss = torch.nn.MSELoss()(vel[0, :], ref_speed_tensor)
    # 3. 距离/安全损失
    # Fix: Access min_distance via pan sub-module
    distance_loss = cal_distance_loss(distance, neupan_planner.pan.min_distance, neupan_planner.collision_threshold, stuck)

    return state_loss, speed_loss, distance_loss

def train_one_epoch(env, planner, opt, max_steps=500, render=True):
    
    loss_of_each_step = []
    
    # [Alignment] Clear gradients ONCE at the start of the epoch
    # Gradients will accumulate throughout the steps, matching the original LON behavior.
    opt.zero_grad() 

    stuck_threshold = 0.01
    stack_number_threshold = 5
    stack_number = 0
    arrive_flag = False
    
    pre_position = env.get_robot_state()[0:2]

    for i in range(max_steps):

        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()
        
        try:
            points = planner.scan_to_point(robot_state, lidar_scan)
        except AttributeError:
             points = None

        # Always call planner. NeuPAN handles None points gracefully (pure tracking mode).
        # This ensures info['state_tensor'] is always populated for loss calculation.
        try:
            action, info = planner(robot_state, points)
        except SolverError:
            print(f"  [Solver Error] Optimization infeasible at step {i}. Resetting environment.")
            # Infeasible optimization means we pushed parameters too far (e.g. d_max too large, eta too high)
            # We treat this as a crash/stuck scenario to reset.
            # Ideally we'd penalize, but gradients are broken. 
            # We just break the loop and let the accumulated gradients so far (if any) update the params.
            env.reset()
            planner.reset()
            break

        if render:
            env.render()
        
        env.step(action)

        cur_position = env.get_robot_state()[0:2]
        diff_distance = np.linalg.norm(cur_position - pre_position)
        pre_position = cur_position

        # 卡死检测逻辑
        if diff_distance < stuck_threshold:
            stack_number += 1
        else:
            stack_number = 0
        
        stuck = (stack_number > stack_number_threshold)

        if stuck and i % 20 == 0:
            print(f'  [Stuck Detected] diff_distance: {diff_distance:.4f}')

        # --- 核心：计算损失 ---
        _, _, distance_loss = cal_loss(planner, stuck)
        
        loss = 10 * distance_loss

        if info.get('arrive', False):
            arrive_flag = True
            print("  [Arrived] Target reached!")
        
        loss_of_each_step.append(loss.item())

        # --- 核心：反向传播与参数更新 ---
        loss.backward()
        opt.step()
        
        # [Alignment] Do NOT zero_grad here to match original cumulative behavior.
        # opt.zero_grad() 

        # 绘制轨迹可视化
        if render:
            planner.visualize_roi_region(env)
            env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
            env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
            env.draw_trajectory(planner.opt_trajectory, "r", refresh=True)
            if i == 0:
                env.draw_trajectory(planner.initial_path, traj_type="-k", show_direction=False)

        if info.get('arrive', False) or info.get('stop', False) or stuck or env.done():
            break

    env.reset()
    planner.reset()
    
    avg_loss = sum(loss_of_each_step) / max(len(loss_of_each_step), 1)
    return avg_loss, arrive_flag

def main():
    parser = argparse.ArgumentParser(description="LON: Learning Optimization Network - Auto-tuning MPC parameters")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    
    parser.add_argument("-e", "--env", type=str, default=None, help="Path to env config")
    parser.add_argument("-p", "--planner", type=str, default=None, help="Path to planner config")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None, help="Optional: Override DUNE model checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for parameters")
    parser.add_argument("--no-render", action="store_true", default=None, help="Disable visualization")
    
    args = parser.parse_args()

    # --- 配置加载逻辑 ---
    config = {
        "env_file": "example/corridor/acker/env.yaml",
        "planner_file": "example/corridor/acker/planner.yaml",
        "checkpoint": None,
        "epochs": 50,
        "learning_rate": 5e-3,
        "no_render": False
    }

    if args.config:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            # Update config with non-null values from yaml
            for k, v in yaml_config.items():
                if k in config:  # Only update known keys
                    config[k] = v

    # 2. 如果命令行指定了参数，覆盖配置文件/默认值
    if args.env: config["env_file"] = args.env
    if args.planner: config["planner_file"] = args.planner
    if args.checkpoint: config["checkpoint"] = args.checkpoint
    if args.epochs is not None: config["epochs"] = args.epochs
    if args.lr is not None: config["learning_rate"] = args.lr
    if args.no_render is not None: config["no_render"] = args.no_render

    env_path = config["env_file"]
    planner_path = config["planner_file"]
    ckpt_path = config["checkpoint"]
    
    # Safety check: Ensure ckpt_path is truly None if it's empty/null
    if not ckpt_path or str(ckpt_path).lower() == 'null':
        ckpt_path = None

    epochs = config["epochs"]
    lr = config["learning_rate"]
    no_render = config["no_render"]

    # 1. 初始化环境
    print(f"Loading environment: {env_path}")
    try:
        env = irsim.make(env_path, display=not no_render)
    except Exception as e:
        print(f"Error loading env: {e}")
        return

    # 2. 初始化规划器
    print(f"Loading planner: {planner_path}")
    init_kwargs = {}
    if ckpt_path:
        print(f"Overriding checkpoint: {ckpt_path}")
        init_kwargs['pan'] = {'dune_checkpoint': ckpt_path}
        
    neupan_planner = neupan.init_from_yaml(planner_path, **init_kwargs)
    neupan_planner.set_env_reference(env)

    # [Auto-Align] Force update initial path to match env's robot state and goal
    # This prevents back-up behavior caused by mismatch between planner waypoints and actual robot pos.
    try:
        robot_state = env.get_robot_state()
        start_wp = robot_state[:3].reshape(3, 1) # x, y, theta
        
        # Get goal from env config if available, otherwise assume far right
        # We need to read env yaml again to get 'goal' reliably as ir-sim object structure varies
        with open(env_path, 'r', encoding='utf-8') as f:
            env_cfg = yaml.safe_load(f)
            goal_list = env_cfg['robot'][0].get('goal', [60, 20, 0])
            goal_wp = np.array(goal_list[:3]).reshape(3, 1)
            
        print(f"Auto-aligning path: Start={start_wp.flatten()}, Goal={goal_wp.flatten()}")
        neupan_planner.update_initial_path_from_waypoints([start_wp, goal_wp])
    except Exception as e:
        print(f"Warning: Failed to auto-align path: {e}")

    # Fix: Manually inject collision_threshold from config since neupan class doesn't store it
    with open(planner_path, 'r', encoding='utf-8') as f:
        p_cfg = yaml.safe_load(f)
        neupan_planner.collision_threshold = p_cfg.get('collision_threshold', 0.1)

    # 3. 设置优化器
    params = neupan_planner.adjust_parameters
    if len(params) >= 4:
        p_u_tune = params[1]
        eta_tune = params[2]
        d_max_tune = params[3]
        
        print(f"Optimizing parameters: p_u (Speed), eta (Obstacle), d_max (Safety Dist)")
        print(f"Initial values -> p_u: {p_u_tune.item():.3f}, eta: {eta_tune.item():.3f}, d_max: {d_max_tune.item():.3f}")
        
        opt = torch.optim.Adam([p_u_tune, eta_tune, d_max_tune], lr=lr)
    else:
        print("Error: Planner does not expose enough adjustable parameters.")
        return

    # 4. 训练循环
    print("-" * 60)
    print(f"Starting LON Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        render_step = not no_render
        
        total_loss, arrive = train_one_epoch(env, neupan_planner, opt, max_steps=400, render=render_step) 

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss:.4f} | Arrive: {arrive} | "
              f"Params -> p_u: {p_u_tune.item():.3f}, eta: {eta_tune.item():.3f}, d_max: {d_max_tune.item():.3f}")

        if total_loss < 0.05 and arrive and epoch > 10:
            print("Converged! Stopping early.")
            break
            
    print("-" * 60)
    print("Final Optimized Parameters:")
    print(f"p_u:   {p_u_tune.item():.4f}")
    print(f"eta:   {eta_tune.item():.4f}")
    print(f"d_max: {d_max_tune.item():.4f}")
    
    print("\nYou can update your planner.yaml with these values to improve performance.")

    env.end()

if __name__ == "__main__":
    main()
