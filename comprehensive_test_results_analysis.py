#!/usr/bin/env python3
"""
综合分析 test/results 目录下的批量消融实验结果
提取关键数据并生成深度分析报告
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import statistics

# 实验目录
TEST_RESULTS_DIR = Path("test/results")

# 关键实验目录
KEY_EXPERIMENTS = [
    "batch_unroll_modes-none-hard_J-0-1-2-3_20251005_001140",
    "batch_unroll_modes-none-hard_J-0-1-2-3_20251005_164242",
    "batch_unroll_modes-hard_J-0-1-2-3_20251006_000627"
]

def load_experiment_data(exp_dir: str) -> Dict:
    """加载实验数据"""
    exp_path = TEST_RESULTS_DIR / exp_dir
    
    # 查找 batch_summary JSON 文件
    json_files = list(exp_path.glob("batch_summary_*.json"))
    if not json_files:
        return None
    
    with open(json_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_key_metrics(data: Dict) -> Dict:
    """提取关键指标"""
    results = {}
    
    for key, value in data.items():
        if 'aggregate' not in value:
            continue
        
        # 解析 key: "example::robot::projection::J#"
        parts = key.split("::")
        if len(parts) < 4:
            continue
        
        example, robot, projection, j_str = parts[0], parts[1], parts[2], parts[3]
        
        agg = value['aggregate']
        
        results[key] = {
            'example': example,
            'robot': robot,
            'projection': projection,
            'J': int(j_str[1:]),  # 从 "J0" 提取 0
            'avg_pre_violation_rate': agg['avg_pre_violation_rate_mean'],
            'avg_pre_p95': agg['avg_pre_p95_mean'],
            'avg_pre_excess': agg['avg_pre_excess_mean'],
            'avg_post_max': agg['avg_post_max_mean'],
            'avg_post_excess': agg['avg_post_excess_mean'],
            'steps_executed': agg['steps_executed_mean'],
            'num_runs': agg['num_runs']
        }
    
    return results

def analyze_pdhg_contribution(results: Dict) -> Dict:
    """分析 PDHG 的独立贡献"""
    analysis = {
        'projection_none': {},
        'projection_hard': {}
    }
    
    for key, metrics in results.items():
        proj = metrics['projection']
        j = metrics['J']
        robot = metrics['robot']
        example = metrics['example']
        
        config_key = f"{example}::{robot}"
        
        if proj == 'none':
            if config_key not in analysis['projection_none']:
                analysis['projection_none'][config_key] = {}
            analysis['projection_none'][config_key][j] = metrics['avg_pre_violation_rate']
        
        elif proj == 'hard':
            if config_key not in analysis['projection_hard']:
                analysis['projection_hard'][config_key] = {}
            analysis['projection_hard'][config_key][j] = metrics['avg_pre_violation_rate']
    
    return analysis

def main():
    print("="*80)
    print("test/results 批量消融实验综合分析")
    print("="*80)
    
    all_results = {}
    
    # 加载所有实验数据
    for exp_dir in KEY_EXPERIMENTS:
        print(f"\n加载实验: {exp_dir}")
        data = load_experiment_data(exp_dir)
        if data:
            metrics = extract_key_metrics(data)
            all_results[exp_dir] = metrics
            print(f"  - 提取了 {len(metrics)} 个配置的数据")
    
    # 使用最新的实验数据进行分析
    latest_exp = "batch_unroll_modes-hard_J-0-1-2-3_20251006_000627"
    if latest_exp in all_results:
        results = all_results[latest_exp]
        
        print("\n" + "="*80)
        print("关键发现 1: PDHG-Unroll 的独立贡献 (projection='hard')")
        print("="*80)
        
        # 筛选 corridor + diff + hard 的数据
        corridor_diff_hard = {k: v for k, v in results.items() 
                              if v['example'] == 'corridor' and v['robot'] == 'diff' and v['projection'] == 'hard'}
        
        if corridor_diff_hard:
            print("\n场景: corridor, 机器人: diff, 投影: hard")
            print(f"{'J值':<5} {'违反率':<15} {'改进幅度':<15} {'执行步数':<10}")
            print("-" * 50)
            
            baseline_viol = None
            for j in [0, 1, 2, 3]:
                key = f"corridor::diff::hard::J{j}"
                if key in corridor_diff_hard:
                    viol = corridor_diff_hard[key]['avg_pre_violation_rate']
                    steps = corridor_diff_hard[key]['steps_executed']
                    
                    if j == 0:
                        baseline_viol = viol
                        improvement = "Baseline"
                    else:
                        improvement = f"{(1 - viol/baseline_viol)*100:.1f}%" if baseline_viol else "N/A"
                    
                    print(f"J={j:<3} {viol:<15.4f} {improvement:<15} {steps:<10.0f}")
        
        print("\n" + "="*80)
        print("关键发现 2: Hard Projection 的独立贡献")
        print("="*80)
        
        # 对比 projection='none' vs 'hard' (J=0)
        print("\n对比: projection='none' vs 'hard' (J=0, 无 PDHG)")
        print(f"{'场景':<15} {'机器人':<8} {'None违反率':<15} {'Hard违反率':<15} {'Hard改进':<10}")
        print("-" * 70)
        
        # 从第一个实验获取 projection='none' 的数据
        none_exp = "batch_unroll_modes-none-hard_J-0-1-2-3_20251005_001140"
        if none_exp in all_results:
            none_results = all_results[none_exp]
            
            for example in ['convex_obs', 'corridor']:
                for robot in ['diff', 'acker']:
                    none_key = f"{example}::{robot}::none::J0"
                    hard_key = f"{example}::{robot}::hard::J0"
                    
                    if none_key in none_results and hard_key in none_results:
                        none_viol = none_results[none_key]['avg_pre_violation_rate']
                        hard_viol = none_results[hard_key]['avg_pre_violation_rate']
                        improvement = (1 - hard_viol/none_viol) * 100 if none_viol > 0 else 0
                        
                        print(f"{example:<15} {robot:<8} {none_viol:<15.4f} {hard_viol:<15.4f} {improvement:<10.1f}%")
        
        print("\n" + "="*80)
        print("关键发现 3: 双重投影的必要性分析")
        print("="*80)
        
        print("\n对比: projection='none', J=1 vs projection='hard', J=1")
        print(f"{'场景':<15} {'机器人':<8} {'None+J1违反率':<18} {'Hard+J1违反率':<18} {'Hard额外改进':<15}")
        print("-" * 80)
        
        if none_exp in all_results:
            none_results = all_results[none_exp]
            
            for example in ['convex_obs', 'corridor']:
                for robot in ['diff', 'acker']:
                    none_j1_key = f"{example}::{robot}::none::J1"
                    hard_j1_key = f"{example}::{robot}::hard::J1"
                    
                    if none_j1_key in none_results and hard_j1_key in none_results:
                        none_j1_viol = none_results[none_j1_key]['avg_pre_violation_rate']
                        hard_j1_viol = none_results[hard_j1_key]['avg_pre_violation_rate']
                        extra_improvement = (1 - hard_j1_viol/none_j1_viol) * 100 if none_j1_viol > 0 else 0
                        
                        print(f"{example:<15} {robot:<8} {none_j1_viol:<18.6f} {hard_j1_viol:<18.6f} {extra_improvement:<15.1f}%")
        
        print("\n" + "="*80)
        print("关键发现 4: 最优 J 值分析")
        print("="*80)
        
        print("\n在 projection='hard' 模式下,不同 J 值的性能对比:")
        print(f"{'场景':<15} {'机器人':<8} {'J=0':<12} {'J=1':<12} {'J=2':<12} {'J=3':<12} {'最优J':<8}")
        print("-" * 85)
        
        for example in ['corridor', 'convex_obs']:
            for robot in ['diff', 'acker']:
                j_viols = {}
                for j in [0, 1, 2, 3]:
                    key = f"{example}::{robot}::hard::J{j}"
                    if key in results:
                        j_viols[j] = results[key]['avg_pre_violation_rate']
                
                if len(j_viols) == 4:
                    best_j = min(j_viols, key=j_viols.get)
                    print(f"{example:<15} {robot:<8} {j_viols[0]:<12.6f} {j_viols[1]:<12.6f} {j_viols[2]:<12.6f} {j_viols[3]:<12.6f} J={best_j}")
        
        print("\n" + "="*80)
        print("关键发现 5: 数值稳定性分析")
        print("="*80)
        
        print("\nPDHG 后的 dual norm 最大值 (avg_post_max):")
        print(f"{'场景':<15} {'机器人':<8} {'J值':<5} {'Post Max':<15} {'是否稳定':<10}")
        print("-" * 60)
        
        for example in ['corridor']:
            for robot in ['diff']:
                for j in [0, 1, 2, 3]:
                    key = f"{example}::{robot}::hard::J{j}"
                    if key in results:
                        post_max = results[key]['avg_post_max']
                        stable = "✅ 是" if post_max <= 1.0001 else "⚠️ 否"
                        print(f"{example:<15} {robot:<8} J={j:<3} {post_max:<15.10f} {stable:<10}")
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)

if __name__ == "__main__":
    main()

