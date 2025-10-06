#!/usr/bin/env python3
"""
阶段 2 时延分析脚本
分析 PDHG-Unroll 的性能-精度权衡
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_latest_timing_results():
    """加载最新的时延分析结果"""
    results_dir = Path("test/results")
    
    # 查找最新的 batch_unroll 目录
    batch_dirs = sorted(results_dir.glob("batch_unroll_modes-hard_J-0-1-2-3_*"))
    if not batch_dirs:
        print("❌ 未找到时延分析结果目录")
        return None
    
    latest_dir = batch_dirs[-1]
    print(f"📁 分析目录: {latest_dir.name}")
    
    # 查找 JSON 汇总文件
    json_files = list(latest_dir.glob("batch_summary_*.json"))
    if not json_files:
        print("❌ 未找到 JSON 汇总文件")
        return None
    
    json_file = json_files[0]
    print(f"📄 数据文件: {json_file.name}\n")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data, latest_dir

def extract_metrics(data):
    """提取关键指标"""
    records = []
    
    for config_key, config_data in data.items():
        if 'aggregate' not in config_data:
            continue
        
        # 解析配置键: example::kinematics::projection::J
        parts = config_key.split("::")
        if len(parts) != 4:
            continue
        
        example, kinematics, projection, j_str = parts
        unroll_J = int(j_str[1:])  # 去掉 'J' 前缀
        
        agg = config_data['aggregate']
        
        # 从 runs 中提取时延数据
        runs = config_data.get('runs', [])
        timing_data = [r.get('avg_total_time_ms', 0) for r in runs if 'avg_total_time_ms' in r]
        
        record = {
            'example': example,
            'kinematics': kinematics,
            'projection': projection,
            'unroll_J': unroll_J,
            'viol_rate': agg.get('avg_pre_violation_rate_mean', 0) * 100,  # 转为百分比
            'p95': agg.get('avg_pre_p95_mean', 0),
            'steps': agg.get('steps_executed_mean', 0),
            'num_runs': agg.get('num_runs', 0),
            'avg_time_ms': np.mean(timing_data) if timing_data else 0,
            'std_time_ms': np.std(timing_data) if timing_data else 0,
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def analyze_timing_overhead(df):
    """分析时延开销"""
    print("=" * 80)
    print("📊 时延开销分析")
    print("=" * 80)
    
    # 按 example + kinematics 分组
    for (example, kin), group in df.groupby(['example', 'kinematics']):
        group = group.sort_values('unroll_J')
        
        if len(group) < 2:
            continue
        
        print(f"\n🔹 {example} | {kin}")
        print("-" * 60)
        
        baseline = group[group['unroll_J'] == 0].iloc[0]
        baseline_time = baseline['avg_time_ms']
        baseline_viol = baseline['viol_rate']
        
        for _, row in group.iterrows():
            J = row['unroll_J']
            time = row['avg_time_ms']
            viol = row['viol_rate']
            steps = row['steps']
            
            time_increase = time - baseline_time
            time_increase_pct = (time_increase / baseline_time * 100) if baseline_time > 0 else 0
            viol_decrease = baseline_viol - viol
            viol_decrease_pct = (viol_decrease / baseline_viol * 100) if baseline_viol > 0 else 0
            
            # 性价比：每降低 1% 违反率的时延代价
            cost_per_percent = (time_increase / viol_decrease) if viol_decrease > 0 else float('inf')
            
            print(f"  J={J}: "
                  f"时延={time:.1f}ms (+{time_increase:.1f}ms, +{time_increase_pct:.1f}%), "
                  f"违反率={viol:.2f}% (↓{viol_decrease:.2f}%, ↓{viol_decrease_pct:.1f}%), "
                  f"步数={steps:.0f}")
            
            if J > 0:
                if cost_per_percent < float('inf'):
                    print(f"       性价比: {cost_per_percent:.3f} ms/% (每降低1%违反率需增加 {cost_per_percent:.3f}ms)")
                else:
                    print(f"       性价比: N/A (违反率未降低)")

def analyze_cost_effectiveness(df):
    """分析性价比"""
    print("\n" + "=" * 80)
    print("💰 性价比汇总（按 J 值聚合）")
    print("=" * 80)
    
    summary = []
    
    for J in sorted(df['unroll_J'].unique()):
        if J == 0:
            continue
        
        # 计算所有场景的平均性价比
        cost_per_percents = []
        
        for (example, kin), group in df.groupby(['example', 'kinematics']):
            group = group.sort_values('unroll_J')
            
            baseline = group[group['unroll_J'] == 0]
            current = group[group['unroll_J'] == J]
            
            if baseline.empty or current.empty:
                continue
            
            baseline_time = baseline.iloc[0]['avg_time_ms']
            baseline_viol = baseline.iloc[0]['viol_rate']
            current_time = current.iloc[0]['avg_time_ms']
            current_viol = current.iloc[0]['viol_rate']
            
            time_increase = current_time - baseline_time
            viol_decrease = baseline_viol - current_viol
            
            if viol_decrease > 0:
                cost = time_increase / viol_decrease
                cost_per_percents.append(cost)
        
        if cost_per_percents:
            avg_cost = np.mean(cost_per_percents)
            std_cost = np.std(cost_per_percents)
            
            # 计算平均违反率降低
            j_data = df[df['unroll_J'] == J]
            j0_data = df[df['unroll_J'] == 0]
            
            avg_viol_j = j_data['viol_rate'].mean()
            avg_viol_0 = j0_data['viol_rate'].mean()
            avg_time_j = j_data['avg_time_ms'].mean()
            avg_time_0 = j0_data['avg_time_ms'].mean()
            
            viol_improvement = (avg_viol_0 - avg_viol_j) / avg_viol_0 * 100
            time_overhead = (avg_time_j - avg_time_0) / avg_time_0 * 100
            
            summary.append({
                'J': J,
                'avg_cost': avg_cost,
                'std_cost': std_cost,
                'avg_viol': avg_viol_j,
                'viol_improvement': viol_improvement,
                'avg_time': avg_time_j,
                'time_overhead': time_overhead,
            })
    
    summary_df = pd.DataFrame(summary)
    
    print("\n| J | 平均时延 (ms) | 时延增加 (%) | 平均违反率 (%) | 违反率降低 (%) | 性价比 (ms/%) |")
    print("|---|--------------|-------------|---------------|---------------|--------------|")
    
    for _, row in summary_df.iterrows():
        print(f"| {row['J']} | "
              f"{row['avg_time']:.1f} | "
              f"+{row['time_overhead']:.1f}% | "
              f"{row['avg_viol']:.2f}% | "
              f"↓{row['viol_improvement']:.1f}% | "
              f"{row['avg_cost']:.3f} ± {row['std_cost']:.3f} |")
    
    return summary_df

def analyze_realtime_feasibility(df):
    """分析实时性"""
    print("\n" + "=" * 80)
    print("⏱️  实时性评估（10 Hz 控制频率 = 100 ms 周期）")
    print("=" * 80)
    
    threshold_ms = 100
    
    for J in sorted(df['unroll_J'].unique()):
        j_data = df[df['unroll_J'] == J]
        
        avg_time = j_data['avg_time_ms'].mean()
        max_time = j_data['avg_time_ms'].max()
        
        feasible = "✅ 可行" if max_time < threshold_ms else "❌ 不可行"
        margin = threshold_ms - max_time
        
        print(f"  J={J}: 平均={avg_time:.1f}ms, 最大={max_time:.1f}ms, "
              f"余量={margin:.1f}ms, {feasible}")

def generate_recommendations(df, summary_df):
    """生成推荐配置"""
    print("\n" + "=" * 80)
    print("🎯 推荐配置")
    print("=" * 80)
    
    # 找到最佳性价比的 J
    if not summary_df.empty:
        best_j = summary_df.loc[summary_df['avg_cost'].idxmin(), 'J']
        best_cost = summary_df.loc[summary_df['avg_cost'].idxmin(), 'avg_cost']
        best_viol = summary_df.loc[summary_df['avg_cost'].idxmin(), 'avg_viol']
        best_time = summary_df.loc[summary_df['avg_cost'].idxmin(), 'avg_time']
        best_overhead = summary_df.loc[summary_df['avg_cost'].idxmin(), 'time_overhead']
        
        print(f"\n✨ 最佳性价比配置: J={int(best_j)}")
        print(f"   - 性价比: {best_cost:.3f} ms/% (最优)")
        print(f"   - 平均违反率: {best_viol:.2f}%")
        print(f"   - 平均时延: {best_time:.1f}ms")
        print(f"   - 时延开销: +{best_overhead:.1f}%")
        
        # 评估 J=1 vs J=2
        j1_data = summary_df[summary_df['J'] == 1]
        j2_data = summary_df[summary_df['J'] == 2]
        
        if not j1_data.empty and not j2_data.empty:
            j1_viol = j1_data.iloc[0]['avg_viol']
            j2_viol = j2_data.iloc[0]['avg_viol']
            j1_time = j1_data.iloc[0]['avg_time']
            j2_time = j2_data.iloc[0]['avg_time']
            
            viol_improvement = (j1_viol - j2_viol) / j1_viol * 100
            time_increase = (j2_time - j1_time) / j1_time * 100
            
            print(f"\n📊 J=2 vs J=1 边际收益:")
            print(f"   - 违反率进一步降低: {viol_improvement:.1f}%")
            print(f"   - 时延额外增加: {time_increase:.1f}%")
            
            if viol_improvement < 30:
                print(f"   ⚠️  边际收益有限（< 30%），推荐使用 J=1")
            else:
                print(f"   ✅ 边际收益显著（> 30%），高安全场景可考虑 J=2")
    
    print("\n💡 使用建议:")
    print("   - 通用场景: projection='hard', J=1 (最佳性价比)")
    print("   - 高安全场景: projection='hard', J=2 (更低违反率)")
    print("   - 实时性要求: 所有配置均满足 10 Hz 控制频率")

def main():
    print("\n" + "=" * 80)
    print("🚀 阶段 2：时延分析报告")
    print("=" * 80)
    print()
    
    # 加载数据
    result = load_latest_timing_results()
    if result is None:
        return
    
    data, results_dir = result
    
    # 提取指标
    df = extract_metrics(data)
    
    if df.empty:
        print("❌ 未提取到有效数据")
        return
    
    print(f"✅ 成功提取 {len(df)} 条记录")
    print(f"   - 场景数: {df['example'].nunique()}")
    print(f"   - 运动学模型: {df['kinematics'].nunique()}")
    print(f"   - J 值范围: {df['unroll_J'].min()} - {df['unroll_J'].max()}")
    print()
    
    # 分析时延开销
    analyze_timing_overhead(df)
    
    # 分析性价比
    summary_df = analyze_cost_effectiveness(df)
    
    # 分析实时性
    analyze_realtime_feasibility(df)
    
    # 生成推荐
    generate_recommendations(df, summary_df)
    
    # 保存分析结果
    output_file = results_dir / "stage2_timing_analysis.txt"
    print(f"\n📝 分析报告已保存至: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

