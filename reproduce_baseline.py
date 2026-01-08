import irsim
import numpy as np
import yaml
import time
import os
import argparse
import matplotlib.pyplot as plt
from neupan import neupan

# 全局配置
TRIALS = 100
MAX_STEPS = 1000
VISUALIZE = False
EXPERIMENT = "default"

# 实验路径映射
EXP_CONFIGS = {
    "default": {
        "env": "example/non_obs/acker/env.yaml",
        "planner": "example/non_obs/acker/planner.yaml",
        "random_obs": True  # 需要在运行时随机化
    },
    "narrow_gap": {
        "env": "example/narrow_gap/acker/env.yaml",
        "planner": "example/narrow_gap/acker/planner.yaml",
        "random_obs": False # 固定场景
    },
    "dynamic": {
        "env": "example/dynamic/env.yaml",
        "planner": "example/dynamic/planner.yaml",
        "random_obs": False # 固定场景(但物体会动)
    },
    "complex": {
        "env": "example/complex/env.yaml",
        "planner": "example/complex/planner.yaml",
        "random_obs": True # 随机生成大量障碍
    }
}

def setup_env_config(trial_id):
    """根据实验类型准备环境配置文件"""
    
    # 1. 确定配置文件路径
    if EXPERIMENT in EXP_CONFIGS:
        # 使用预定义实验
        base_env_path = EXP_CONFIGS[EXPERIMENT]["env"]
        base_planner_path = EXP_CONFIGS[EXPERIMENT]["planner"]
        is_random = EXP_CONFIGS[EXPERIMENT]["random_obs"]
    else:
        # 尝试加载现有标准场景 (Standard Benchmarks)
        # 默认使用 acker 运动学
        base_env_path = f"example/{EXPERIMENT}/acker/env.yaml"
        base_planner_path = f"example/{EXPERIMENT}/acker/planner.yaml"
        is_random = False # 默认不额外随机化(相信原配置)

        # 检查文件是否存在
        if not os.path.exists(base_env_path):
            raise FileNotFoundError(f"Experiment '{EXPERIMENT}' not found. Checked: {base_env_path}")
            
    # 更新全局 planner 路径以便 main 函数使用
    # 注意：这里有点 hack，理想情况应该返回 planner_path，但为了少改代码
    # 我们把 planner_path 存入一个临时属性或直接在 run_single_trial 里重新获取
    # 这里我们选择在 run_single_trial 里重新解析逻辑，或者让 setup 返回 planner_path
    
    with open(base_env_path, 'r', encoding='utf-8') as f:
        env_config = yaml.safe_load(f)
    
    # 针对 default 和 complex 实验进行随机化
    if is_random:
        if EXPERIMENT == "default":
            # 论文基线复现逻辑
            if 'distribution' in env_config['obstacle'][0]:
                 env_config['obstacle'][0]['distribution'] = {
                    'name': 'random', 
                    'mode': 'uniform', 
                    'range': [5, 45, 15, 35]
                }
            if 'state' in env_config['obstacle'][0]:
                del env_config['obstacle'][0]['state']

    # 保存临时文件
    temp_env_file = f"temp_env_{EXPERIMENT}_{trial_id}.yaml"
    with open(temp_env_file, 'w') as f:
        yaml.dump(env_config, f)
        
    return temp_env_file, env_config, base_planner_path

def run_single_trial(trial_id):
    try:
        temp_env_file, env_data, planner_path = setup_env_config(trial_id)
    except FileNotFoundError as e:
        print(e)
        return False, 0, 0
    except Exception as e:
        print(f"Config Error: {e}")
        return False, 0, 0

    # 初始化环境
    try:
        env = irsim.make(temp_env_file, display=VISUALIZE, save_ani=False)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return False, 0, 0


    # 初始化规划器
    # Disable per-module timing prints (time_it decorator). Timing is still available via info['forward_time_ms'].
    neupan_planner = neupan.init_from_yaml(planner_path, time_print=False)
    neupan_planner.set_env_reference(env)

    # --- 自动路线对齐 (Auto Path Alignment) ---
    # 从 env 配置中提取起点和终点，覆盖 planner 的初始路径
    try:
        robot_conf = env_data['robot'][0]
        start_state = np.array(robot_conf['state'])
        
        if 'goal' in robot_conf:
            goal_state = np.array(robot_conf['goal'])
        else:
            # 如果没有显式goal，假设终点在很远的地方(仅针对 default)
            goal_state = np.array([50, 25, 0]) 

        # 构造 waypoints: [Start, Goal]
        # 注意：waypoint 格式通常是 [x, y, theta]
        start_wp = start_state[:3].reshape(3, 1)
        goal_wp = goal_state[:3].reshape(3, 1) if len(goal_state) >=3 else np.pad(goal_state, (0, 3-len(goal_state))).reshape(3,1)
        
        # 强制更新初始路径
        neupan_planner.update_initial_path_from_waypoints([start_wp, goal_wp])
        if VISUALIZE:
            print(f"  [Auto-Align] Path updated: Start={start_wp.flatten()}, Goal={goal_wp.flatten()}")

    except Exception as e:
        print(f"  [Warning] Auto path alignment failed: {e}")

    # 运行仿真
    success = False
    collision = False
    navigation_time = 0
    step_speeds = []
    
    for i in range(MAX_STEPS):
        robot_state = env.get_robot_state()
        lidar_scan = env.get_lidar_scan()

        points = neupan_planner.scan_to_point(robot_state, lidar_scan)
        
        # 处理动态障碍物速度 (Dynamic Experiment)
        point_velocities = None
        if EXPERIMENT == "dynamic":
             points, point_velocities = neupan_planner.scan_to_point_velocity(robot_state, lidar_scan)

        action, info = neupan_planner(robot_state, points, point_velocities)
        
        env.step(action)

        # 可视化绘图
        if VISUALIZE:
            env.draw_points(neupan_planner.dune_points, s=25, c="g", refresh=True)
            env.draw_points(neupan_planner.nrmp_points, s=13, c="r", refresh=True)
            env.draw_trajectory(neupan_planner.opt_trajectory, "r", refresh=True)
            # 画一下全局路径确认对齐
            # env.draw_trajectory(neupan_planner.initial_path, "k", refresh=False) 
            env.render()
        
        # 记录速度
        if isinstance(action, np.ndarray):
            v = action[0, 0] if action.ndim > 1 else action[0]
        else:
            v = action[0]
        step_speeds.append(v)

        if env.done():
            if info["arrive"]:
                success = True
            if env.robot.collision: 
                collision = True
            break
            
        if info["stop"] and not info["arrive"]:
             # 有时候 stop 是因为避障减速，不一定是卡死，但如果 max_steps 到了还没到就算卡死
             pass

    # 统计
    navigation_time = env.step_time * (i + 1)
    avg_speed = np.mean(step_speeds) if len(step_speeds) > 0 else 0
    
    is_success = success and not collision
    
    # 清理
    env.end()
    if VISUALIZE:
        time.sleep(0.5)
        try:
            plt.close('all')
        except:
            pass

    try:
        os.remove(temp_env_file)
    except:
        pass
        
    return is_success, navigation_time, avg_speed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("-n", "--num_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("-e", "--experiment", type=str, default="default", 
                        help="Choose experiment scenario (e.g., default, corridor, dyna_obs, non_obs)")
    
    args = parser.parse_args()
    
    global VISUALIZE, TRIALS, EXPERIMENT
    VISUALIZE = args.visualize
    TRIALS = args.num_trials
    EXPERIMENT = args.experiment
    
    if VISUALIZE:
        print(f"Visualization Mode: Enabled. Running {TRIALS} trials.")
    
    print(f"Starting Experiment: [{EXPERIMENT}]")
    print("-" * 65)
    print(f"{'Trial':<5} | {'Result':<10} | {'Time (s)':<10} | {'Avg Speed':<10}")
    print("-" * 65)

    success_count = 0
    total_time = []
    total_speed = []

    for i in range(TRIALS):
        is_success, nav_time, speed = run_single_trial(i)
        
        status = "SUCCESS" if is_success else "FAIL"
        # 红色显示失败，绿色显示成功 (在支持的终端)
        print(f"{i+1:<5} | {status:<10} | {nav_time:<10.2f} | {speed:<10.2f}")
        
        if is_success:
            success_count += 1
            total_time.append(nav_time)
            total_speed.append(speed)

    print("-" * 65)
    print(f"FINAL RESULTS for [{EXPERIMENT}]:")
    print(f"Success Rate: {success_count}/{TRIALS} = {success_count/TRIALS*100:.2f}%")
    if len(total_time) > 0:
        print(f"Avg Nav Time: {np.mean(total_time):.2f} s")
        print(f"Avg Speed:    {np.mean(total_speed):.2f} m/s")
    else:
        print("No successful trials.")

if __name__ == "__main__":
    main()
