"""
诊断虚拟点是否被正确禁用
"""
import numpy as np
import irsim
from neupan import neupan

def diagnose_virtual_points():
    """诊断虚拟点生成情况"""
    
    print("=" * 80)
    print("虚拟点诊断工具")
    print("=" * 80)
    print()
    
    # 测试场景
    env_file = "example/corridor/acker/env.yaml"
    planner_file = "example/corridor/acker/planner.yaml"
    
    print("测试 1: use_virtual_points=False (禁用虚拟点)")
    print("-" * 80)
    
    # 创建环境
    env = irsim.make(env_file, save_ani=False, full=False, display=False)
    
    # 创建规划器（禁用虚拟点）
    planner = neupan.init_from_yaml(planner_file, use_virtual_points=False)
    planner.set_env_reference(env)
    
    print(f"✓ 规划器初始化成功")
    print(f"  use_virtual_points = {planner.use_virtual_points}")
    print()
    
    # 运行几步，收集点云数据
    point_counts = []
    point_distances = []
    
    for step in range(10):
        robot_state = env.get_robot_state()[0:3]
        lidar_scan = env.get_lidar_scan()
        
        # 转换为点云
        points = planner.scan_to_point(robot_state, lidar_scan)
        
        if points is not None:
            num_points = points.shape[1]
            point_counts.append(num_points)
            
            # 计算点的距离（相对于机器人）
            robot_pos = robot_state[0:2].reshape(2, 1)
            distances = np.linalg.norm(points - robot_pos, axis=0)
            point_distances.extend(distances.tolist())
            
            if step == 0:
                print(f"步骤 {step}:")
                print(f"  点数: {num_points}")
                print(f"  距离范围: [{distances.min():.2f}, {distances.max():.2f}]")
                print(f"  平均距离: {distances.mean():.2f}")
                
                # 检查是否有10m附近的点（虚拟点的特征）
                near_10m = np.sum((distances > 9.5) & (distances < 10.5))
                if near_10m > 0:
                    print(f"  ⚠️ 警告: 发现 {near_10m} 个约10m处的点（可能是虚拟点）")
                else:
                    print(f"  ✓ 没有发现10m附近的点（虚拟点已禁用）")
        else:
            point_counts.append(0)
            if step == 0:
                print(f"步骤 {step}:")
                print(f"  点数: 0 (没有生成点云)")
        
        # 执行一步
        action, info = planner(robot_state, points)
        env.step(action)
        
        if env.done():
            break
    
    print()
    print(f"总结 (前10步):")
    print(f"  平均点数: {np.mean(point_counts):.1f}")
    print(f"  点数范围: [{min(point_counts)}, {max(point_counts)}]")
    
    if point_distances:
        all_distances = np.array(point_distances)
        near_10m_total = np.sum((all_distances > 9.5) & (all_distances < 10.5))
        print(f"  总点数: {len(all_distances)}")
        print(f"  约10m处的点: {near_10m_total} ({near_10m_total/len(all_distances)*100:.1f}%)")
        
        if near_10m_total > len(all_distances) * 0.1:  # 超过10%
            print(f"  ❌ 虚拟点可能未被正确禁用！")
        else:
            print(f"  ✓ 虚拟点已成功禁用")
    
    env.end(0)
    print()
    
    # 测试 2: use_virtual_points=True (启用虚拟点)
    print("=" * 80)
    print("测试 2: use_virtual_points=True (启用虚拟点)")
    print("-" * 80)
    
    env = irsim.make(env_file, save_ani=False, full=False, display=False)
    planner = neupan.init_from_yaml(planner_file, use_virtual_points=True)
    planner.set_env_reference(env)
    
    print(f"✓ 规划器初始化成功")
    print(f"  use_virtual_points = {planner.use_virtual_points}")
    print()
    
    point_counts = []
    point_distances = []
    
    for step in range(10):
        robot_state = env.get_robot_state()[0:3]
        lidar_scan = env.get_lidar_scan()
        
        points = planner.scan_to_point(robot_state, lidar_scan)
        
        if points is not None:
            num_points = points.shape[1]
            point_counts.append(num_points)
            
            robot_pos = robot_state[0:2].reshape(2, 1)
            distances = np.linalg.norm(points - robot_pos, axis=0)
            point_distances.extend(distances.tolist())
            
            if step == 0:
                print(f"步骤 {step}:")
                print(f"  点数: {num_points}")
                print(f"  距离范围: [{distances.min():.2f}, {distances.max():.2f}]")
                print(f"  平均距离: {distances.mean():.2f}")
                
                near_10m = np.sum((distances > 9.5) & (distances < 10.5))
                if near_10m > 0:
                    print(f"  ✓ 发现 {near_10m} 个约10m处的点（虚拟点已启用）")
                else:
                    print(f"  ⚠️ 没有发现10m附近的点")
        else:
            point_counts.append(0)
        
        action, info = planner(robot_state, points)
        env.step(action)
        
        if env.done():
            break
    
    print()
    print(f"总结 (前10步):")
    print(f"  平均点数: {np.mean(point_counts):.1f}")
    print(f"  点数范围: [{min(point_counts)}, {max(point_counts)}]")
    
    if point_distances:
        all_distances = np.array(point_distances)
        near_10m_total = np.sum((all_distances > 9.5) & (all_distances < 10.5))
        print(f"  总点数: {len(all_distances)}")
        print(f"  约10m处的点: {near_10m_total} ({near_10m_total/len(all_distances)*100:.1f}%)")
        
        if near_10m_total > len(all_distances) * 0.1:
            print(f"  ✓ 虚拟点已成功启用")
        else:
            print(f"  ⚠️ 虚拟点可能未生效")
    
    env.end(0)
    print()
    
    print("=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == '__main__':
    diagnose_virtual_points()

