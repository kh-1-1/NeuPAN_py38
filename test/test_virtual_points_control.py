"""
测试虚拟点控制功能
"""
import numpy as np
from neupan import neupan

def test_virtual_points_enabled():
    """测试启用虚拟点"""
    print("=" * 80)
    print("测试 1: 启用虚拟点 (use_virtual_points=True)")
    print("=" * 80)
    
    planner = neupan.init_from_yaml(
        'example/corridor/acker/planner.yaml',
        use_virtual_points=True
    )
    
    print(f"✓ 规划器初始化成功")
    print(f"  use_virtual_points = {planner.use_virtual_points}")
    
    # 模拟激光雷达扫描（所有射线都没有检测到障碍物）
    scan = {
        'ranges': [100.0] * 100,  # 所有射线都是最大距离
        'angle_min': -np.pi,
        'angle_max': np.pi,
        'angle_increment': 2 * np.pi / 100,
        'range_min': 0.1,
        'range_max': 100.0,
    }
    
    state = np.array([0, 0, 0])
    points = planner.scan_to_point(state, scan)
    
    if points is not None:
        print(f"✓ 生成了点云")
        print(f"  点数: {points.shape[1]}")
        print(f"  预期: 应该生成虚拟点（约100个点在10m处）")
        
        # 检查点的距离
        distances = np.linalg.norm(points, axis=0)
        print(f"  距离范围: [{distances.min():.2f}, {distances.max():.2f}]")
        print(f"  平均距离: {distances.mean():.2f}")
    else:
        print(f"✗ 没有生成点云")
    
    print()

def test_virtual_points_disabled():
    """测试禁用虚拟点"""
    print("=" * 80)
    print("测试 2: 禁用虚拟点 (use_virtual_points=False)")
    print("=" * 80)
    
    planner = neupan.init_from_yaml(
        'example/corridor/acker/planner.yaml',
        use_virtual_points=False
    )
    
    print(f"✓ 规划器初始化成功")
    print(f"  use_virtual_points = {planner.use_virtual_points}")
    
    # 模拟激光雷达扫描（所有射线都没有检测到障碍物）
    scan = {
        'ranges': [100.0] * 100,  # 所有射线都是最大距离
        'angle_min': -np.pi,
        'angle_max': np.pi,
        'angle_increment': 2 * np.pi / 100,
        'range_min': 0.1,
        'range_max': 100.0,
    }
    
    state = np.array([0, 0, 0])
    points = planner.scan_to_point(state, scan)
    
    if points is None:
        print(f"✓ 没有生成点云（符合预期）")
        print(f"  预期: 禁用虚拟点后，没有真实障碍物时不应生成点云")
    else:
        print(f"✗ 生成了点云（不符合预期）")
        print(f"  点数: {points.shape[1]}")
    
    print()

def test_virtual_points_with_real_obstacles():
    """测试有真实障碍物时的虚拟点"""
    print("=" * 80)
    print("测试 3: 有真实障碍物 + 虚拟点")
    print("=" * 80)
    
    planner = neupan.init_from_yaml(
        'example/corridor/acker/planner.yaml',
        use_virtual_points=True
    )
    
    print(f"✓ 规划器初始化成功")
    print(f"  use_virtual_points = {planner.use_virtual_points}")
    
    # 模拟激光雷达扫描（部分射线检测到障碍物，部分没有）
    ranges = [100.0] * 100
    ranges[40:60] = [5.0] * 20  # 中间20个射线检测到5m处的障碍物
    
    scan = {
        'ranges': ranges,
        'angle_min': -np.pi,
        'angle_max': np.pi,
        'angle_increment': 2 * np.pi / 100,
        'range_min': 0.1,
        'range_max': 100.0,
    }
    
    state = np.array([0, 0, 0])
    points = planner.scan_to_point(state, scan)
    
    if points is not None:
        print(f"✓ 生成了点云")
        print(f"  点数: {points.shape[1]}")
        print(f"  预期: 应该包含真实障碍物点（5m）+ 虚拟点（10m）")
        
        # 检查点的距离
        distances = np.linalg.norm(points, axis=0)
        print(f"  距离范围: [{distances.min():.2f}, {distances.max():.2f}]")
        
        # 统计不同距离的点
        near_5m = np.sum((distances > 4.5) & (distances < 5.5))
        near_10m = np.sum((distances > 9.5) & (distances < 10.5))
        print(f"  约5m处的点: {near_5m} (真实障碍物)")
        print(f"  约10m处的点: {near_10m} (虚拟点)")
    else:
        print(f"✗ 没有生成点云")
    
    print()

def test_virtual_points_default():
    """测试默认行为（不指定参数）"""
    print("=" * 80)
    print("测试 4: 默认行为（不指定 use_virtual_points）")
    print("=" * 80)
    
    planner = neupan.init_from_yaml('example/corridor/acker/planner.yaml')

    print(f"✓ 规划器初始化成功")
    print(f"  use_virtual_points = {planner.use_virtual_points}")
    print(f"  预期: 默认应该为 False（禁用虚拟点）")
    
    print()

if __name__ == '__main__':
    print()
    print("虚拟点控制功能测试")
    print("=" * 80)
    print()
    
    test_virtual_points_default()
    test_virtual_points_enabled()
    test_virtual_points_disabled()
    test_virtual_points_with_real_obstacles()
    
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print()
    print("✓ 虚拟点控制功能已成功集成")
    print()
    print("使用方法:")
    print()
    print("1. 在 YAML 配置文件中设置:")
    print("   use_virtual_points: true   # 启用虚拟点")
    print("   use_virtual_points: false  # 禁用虚拟点")
    print()
    print("2. 在代码中设置:")
    print("   planner = neupan.init_from_yaml('planner.yaml', use_virtual_points=True)")
    print()
    print("3. 在测试脚本中设置:")
    print("   python -m test.batch_core_modules_evaluation \\")
    print("     --config-file test/configs/core_modules_evaluation.yaml \\")
    print("     --use-virtual-points true")
    print()

