"""
对比基线、J=1 和 J=2 的推理时间
"""
import torch
import numpy as np
import time
from pathlib import Path
from neupan import neupan

def benchmark_inference(planner, num_runs=100, num_points_list=None):
    """
    测试推理时间
    
    Args:
        planner: NeuPAN 规划器
        num_runs: 每个点数运行次数
        num_points_list: 要测试的点数列表
    
    Returns:
        dict: 包含时间统计的字典
    """
    if num_points_list is None:
        num_points_list = [10, 20, 50, 100]
    
    results = {}
    
    for num_points in num_points_list:
        times = []
        
        for _ in range(num_runs):
            # 创建随机输入
            state = np.array([[0.0], [0.0], [0.0]])  # x, y, theta
            points = np.random.randn(2, num_points) * 10  # 随机障碍点
            
            # 测量推理时间
            start = time.perf_counter()
            planner.forward(state, points)
            elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
            
            times.append(elapsed)
        
        times = np.array(times)
        results[num_points] = {
            'mean': times.mean(),
            'std': times.std(),
            'min': times.min(),
            'max': times.max(),
            'p95': np.percentile(times, 95),
        }
    
    return results

def main():
    print("=" * 100)
    print("推理时间对比测试：基线 vs J=1 vs J=2")
    print("=" * 100)
    print()
    
    # 配置
    planner_file = 'example/pf_obs/acker/planner.yaml'
    
    configs = {
        'baseline': {
            'front': 'obs_point',
            'front_learned': False,
            'front_J': 1,
            'ckpt': 'example/model/acker_CLARABEL_robot/model_5000.pth',
        },
        'flex_J=1': {
            'front': 'flex_pdhg',
            'front_learned': True,
            'front_J': 1,
            'ckpt': 'example/model/acker_flex_pdhg_robot/model_5000.pth',
        },
        'flex_J=2': {
            'front': 'flex_pdhg',
            'front_learned': True,
            'front_J': 2,
            'ckpt': 'example/model/acker_flex_pdhg_robot/model_5000.pth',
        },
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'=' * 100}")
        print(f"配置: {config_name}")
        print(f"  front: {config['front']}")
        print(f"  front_learned: {config['front_learned']}")
        print(f"  front_J: {config['front_J']}")
        print(f"  checkpoint: {config['ckpt']}")
        print(f"{'=' * 100}")
        
        # 初始化规划器
        try:
            train_kwargs = {
                'front': config['front'],
                'front_learned': config['front_learned'],
                'front_J': config['front_J'],
                'projection': 'hard',
                'monitor_dual_norm': True,
                'direct_train': True,
            }
            
            planner = neupan.init_from_yaml(
                planner_file,
                pan={'dune_checkpoint': config['ckpt']},
                train=train_kwargs,
            )
            
            print(f"✓ 规划器初始化成功")
            print(f"  DUNE front_type: {planner.pan.dune_layer.front_type}")
            print(f"  DUNE model.J: {planner.pan.dune_layer.model.J if hasattr(planner.pan.dune_layer.model, 'J') else 'N/A'}")
            print()
            
            # 运行基准测试
            print(f"运行基准测试 (100 次运行/点数)...")
            results = benchmark_inference(planner, num_runs=100)
            all_results[config_name] = results
            
            # 打印结果
            print()
            print(f"{'点数':<10} {'平均(ms)':<12} {'标准差':<12} {'最小(ms)':<12} {'最大(ms)':<12} {'P95(ms)':<12}")
            print("-" * 70)
            for num_points in sorted(results.keys()):
                r = results[num_points]
                print(f"{num_points:<10} {r['mean']:<12.4f} {r['std']:<12.4f} {r['min']:<12.4f} {r['max']:<12.4f} {r['p95']:<12.4f}")
            
        except Exception as e:
            print(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比分析
    print()
    print("=" * 100)
    print("对比分析")
    print("=" * 100)
    print()
    
    if len(all_results) >= 2:
        baseline_results = all_results.get('baseline')
        flex_j1_results = all_results.get('flex_J=1')
        flex_j2_results = all_results.get('flex_J=2')
        
        if baseline_results and flex_j1_results:
            print("基线 vs J=1 对比:")
            print(f"{'点数':<10} {'基线(ms)':<15} {'J=1(ms)':<15} {'增长(%)':<15}")
            print("-" * 55)
            for num_points in sorted(baseline_results.keys()):
                baseline_mean = baseline_results[num_points]['mean']
                j1_mean = flex_j1_results[num_points]['mean']
                increase = ((j1_mean - baseline_mean) / baseline_mean) * 100
                print(f"{num_points:<10} {baseline_mean:<15.4f} {j1_mean:<15.4f} {increase:<15.2f}%")
            print()
        
        if flex_j1_results and flex_j2_results:
            print("J=1 vs J=2 对比:")
            print(f"{'点数':<10} {'J=1(ms)':<15} {'J=2(ms)':<15} {'增长(%)':<15}")
            print("-" * 55)
            for num_points in sorted(flex_j1_results.keys()):
                j1_mean = flex_j1_results[num_points]['mean']
                j2_mean = flex_j2_results[num_points]['mean']
                increase = ((j2_mean - j1_mean) / j1_mean) * 100
                print(f"{num_points:<10} {j1_mean:<15.4f} {j2_mean:<15.4f} {increase:<15.2f}%")
            print()
        
        if baseline_results and flex_j2_results:
            print("基线 vs J=2 对比:")
            print(f"{'点数':<10} {'基线(ms)':<15} {'J=2(ms)':<15} {'增长(%)':<15}")
            print("-" * 55)
            for num_points in sorted(baseline_results.keys()):
                baseline_mean = baseline_results[num_points]['mean']
                j2_mean = flex_j2_results[num_points]['mean']
                increase = ((j2_mean - baseline_mean) / baseline_mean) * 100
                print(f"{num_points:<10} {baseline_mean:<15.4f} {j2_mean:<15.4f} {increase:<15.2f}%")
            print()
    
    print("=" * 100)
    print("总结")
    print("=" * 100)
    print()
    print("✓ 测试完成")
    print()
    print("关键发现:")
    print("  1. 基线使用 ObsPointNet（不需要迭代）")
    print("  2. J=1 执行 1 步 PDHG 迭代")
    print("  3. J=2 执行 2 步 PDHG 迭代")
    print()
    print("预期结果:")
    print("  - J=2 应该比 J=1 慢约 50-100%（因为多执行一步）")
    print("  - J=1 可能比基线快或慢，取决于算法复杂度")
    print()

if __name__ == '__main__':
    main()

