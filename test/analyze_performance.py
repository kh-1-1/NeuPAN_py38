"""
性能分析脚本：深入分析实验结果中的时间性能问题

分析维度：
1. 数据层面：时间指标对比、瓶颈识别
2. 配置层面：参数设置分析
3. 算法层面：flex_pdhg性能分析
4. 代码层面：潜在性能问题定位
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_data(csv_path):
    """加载CSV数据"""
    df = pd.read_csv(csv_path)
    return df


def analyze_time_metrics(df):
    """1. 数据分析：时间指标对比"""
    print("=" * 80)
    print("1. 时间性能数据分析")
    print("=" * 80)
    
    # 按配置分组统计
    configs = ['baseline', 'flex_learned', 'flex_roi']
    
    print("\n【总体时间对比】")
    print("-" * 80)
    for config in configs:
        config_data = df[df['config'] == config]
        if len(config_data) == 0:
            continue
        
        avg_step = config_data['avg_step_ms'].mean()
        avg_forward = config_data['avg_forward_ms'].mean()
        total_time = config_data['total_time_s'].mean()
        
        print(f"\n{config.upper()}:")
        print(f"  平均单步时间: {avg_step:.2f} ms")
        print(f"  平均forward时间: {avg_forward:.2f} ms")
        print(f"  平均总时间: {total_time:.2f} s")
        print(f"  forward占比: {(avg_forward/avg_step*100):.1f}%")
    
    # 计算相对于baseline的时间增长
    print("\n【相对baseline的时间增长】")
    print("-" * 80)
    baseline_data = df[df['config'] == 'baseline']
    baseline_avg_forward = baseline_data['avg_forward_ms'].mean()
    baseline_total = baseline_data['total_time_s'].mean()
    
    for config in ['flex_learned', 'flex_roi']:
        config_data = df[df['config'] == config]
        if len(config_data) == 0:
            continue
        
        avg_forward = config_data['avg_forward_ms'].mean()
        total_time = config_data['total_time_s'].mean()
        
        forward_increase = ((avg_forward - baseline_avg_forward) / baseline_avg_forward) * 100
        total_increase = ((total_time - baseline_total) / baseline_total) * 100
        
        print(f"\n{config.upper()}:")
        print(f"  forward时间增长: {forward_increase:+.1f}%")
        print(f"  总时间增长: {total_increase:+.1f}%")
    
    # 识别时间开销最大的测试用例
    print("\n【时间开销最大的测试用例 TOP 5】")
    print("-" * 80)
    top5 = df.nlargest(5, 'avg_forward_ms')[['example', 'kin', 'config', 'avg_forward_ms', 'total_time_s']]
    print(top5.to_string(index=False))
    
    # 分析不同配置下的时间分布
    print("\n【不同配置的时间统计】")
    print("-" * 80)
    for config in configs:
        config_data = df[df['config'] == config]
        if len(config_data) == 0:
            continue
        
        print(f"\n{config.upper()}:")
        print(f"  样本数: {len(config_data)}")
        print(f"  avg_forward_ms - 均值: {config_data['avg_forward_ms'].mean():.2f}, "
              f"标准差: {config_data['avg_forward_ms'].std():.2f}, "
              f"最小: {config_data['avg_forward_ms'].min():.2f}, "
              f"最大: {config_data['avg_forward_ms'].max():.2f}")
    
    return df


def analyze_by_scenario(df):
    """按场景和运动学模型分析"""
    print("\n" + "=" * 80)
    print("2. 按场景和运动学模型的时间分析")
    print("=" * 80)
    
    examples = df['example'].unique()
    kins = df['kin'].unique()
    
    for example in examples:
        print(f"\n【场景: {example}】")
        print("-" * 80)
        
        for kin in kins:
            subset = df[(df['example'] == example) & (df['kin'] == kin)]
            if len(subset) == 0:
                continue
            
            print(f"\n  运动学模型: {kin}")
            
            baseline = subset[subset['config'] == 'baseline']
            flex_learned = subset[subset['config'] == 'flex_learned']
            flex_roi = subset[subset['config'] == 'flex_roi']
            
            if len(baseline) > 0:
                baseline_time = baseline['avg_forward_ms'].values[0]
                print(f"    baseline: {baseline_time:.2f} ms")
                
                if len(flex_learned) > 0:
                    flex_time = flex_learned['avg_forward_ms'].values[0]
                    increase = ((flex_time - baseline_time) / baseline_time) * 100
                    print(f"    flex_learned: {flex_time:.2f} ms ({increase:+.1f}%)")
                
                if len(flex_roi) > 0:
                    roi_time = flex_roi['avg_forward_ms'].values[0]
                    increase = ((roi_time - baseline_time) / baseline_time) * 100
                    roi_overhead = flex_roi['roi_n_roi'].values[0] if 'roi_n_roi' in flex_roi.columns else 0
                    print(f"    flex_roi: {roi_time:.2f} ms ({increase:+.1f}%), ROI点数: {roi_overhead:.1f}")


def analyze_roi_overhead(df):
    """分析ROI的时间开销"""
    print("\n" + "=" * 80)
    print("3. ROI时间开销分析")
    print("=" * 80)
    
    roi_data = df[df['config'] == 'flex_roi']
    
    if len(roi_data) == 0:
        print("没有ROI数据")
        return
    
    print(f"\n【ROI统计信息】")
    print(f"  平均输入点数: {roi_data['roi_n_in'].mean():.1f}")
    print(f"  平均ROI点数: {roi_data['roi_n_roi'].mean():.1f}")
    print(f"  平均压缩比: {roi_data['roi_ratio'].mean():.2f}x")
    
    # 对比flex_learned和flex_roi的时间差异
    print("\n【flex_roi vs flex_learned 时间对比】")
    print("-" * 80)
    
    for idx, row in roi_data.iterrows():
        example = row['example']
        kin = row['kin']
        
        flex_learned = df[(df['example'] == example) & 
                          (df['kin'] == kin) & 
                          (df['config'] == 'flex_learned')]
        
        if len(flex_learned) > 0:
            roi_time = row['avg_forward_ms']
            learned_time = flex_learned['avg_forward_ms'].values[0]
            diff = roi_time - learned_time
            diff_pct = (diff / learned_time) * 100
            
            print(f"{example}_{kin}: ROI={roi_time:.2f}ms, Learned={learned_time:.2f}ms, "
                  f"差异={diff:+.2f}ms ({diff_pct:+.1f}%)")


def analyze_flex_pdhg_overhead(df):
    """分析flex_pdhg相对于baseline的额外开销"""
    print("\n" + "=" * 80)
    print("4. FlexPDHG算法开销分析")
    print("=" * 80)
    
    print("\n【baseline vs flex_learned 时间对比】")
    print("(flex_learned使用FlexPDHG前端，baseline使用ObsPointNet)")
    print("-" * 80)
    
    baseline_data = df[df['config'] == 'baseline']
    flex_learned_data = df[df['config'] == 'flex_learned']
    
    total_overhead = 0
    count = 0
    
    for idx, baseline_row in baseline_data.iterrows():
        example = baseline_row['example']
        kin = baseline_row['kin']
        
        flex_row = flex_learned_data[(flex_learned_data['example'] == example) & 
                                      (flex_learned_data['kin'] == kin)]
        
        if len(flex_row) > 0:
            baseline_time = baseline_row['avg_forward_ms']
            flex_time = flex_row['avg_forward_ms'].values[0]
            overhead = flex_time - baseline_time
            overhead_pct = (overhead / baseline_time) * 100
            
            total_overhead += overhead
            count += 1
            
            print(f"{example}_{kin}: baseline={baseline_time:.2f}ms, "
                  f"flex={flex_time:.2f}ms, 开销={overhead:+.2f}ms ({overhead_pct:+.1f}%)")
    
    if count > 0:
        avg_overhead = total_overhead / count
        print(f"\n平均FlexPDHG开销: {avg_overhead:+.2f} ms")


def identify_performance_issues(df):
    """识别性能问题"""
    print("\n" + "=" * 80)
    print("5. 性能问题识别")
    print("=" * 80)
    
    # 问题1: flex_learned比baseline慢的情况
    print("\n【问题1: flex_learned比baseline慢】")
    print("-" * 80)
    
    baseline_data = df[df['config'] == 'baseline']
    flex_learned_data = df[df['config'] == 'flex_learned']
    
    slow_cases = []
    for idx, baseline_row in baseline_data.iterrows():
        example = baseline_row['example']
        kin = baseline_row['kin']
        
        flex_row = flex_learned_data[(flex_learned_data['example'] == example) & 
                                      (flex_learned_data['kin'] == kin)]
        
        if len(flex_row) > 0:
            baseline_time = baseline_row['avg_forward_ms']
            flex_time = flex_row['avg_forward_ms'].values[0]
            
            if flex_time > baseline_time:
                overhead_pct = ((flex_time - baseline_time) / baseline_time) * 100
                slow_cases.append({
                    'case': f"{example}_{kin}",
                    'baseline': baseline_time,
                    'flex': flex_time,
                    'overhead_pct': overhead_pct
                })
    
    slow_cases_df = pd.DataFrame(slow_cases).sort_values('overhead_pct', ascending=False)
    print(slow_cases_df.to_string(index=False))
    
    # 问题2: 异常慢的测试用例
    print("\n【问题2: 异常慢的测试用例 (>100ms)】")
    print("-" * 80)
    
    slow_tests = df[df['avg_forward_ms'] > 100][['example', 'kin', 'config', 'avg_forward_ms', 'steps']]
    if len(slow_tests) > 0:
        print(slow_tests.to_string(index=False))
    else:
        print("没有发现异常慢的测试用例")


def generate_visualizations(df, output_dir):
    """生成可视化图表"""
    print("\n" + "=" * 80)
    print("6. 生成可视化图表")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 图1: 不同配置的平均forward时间对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = ['baseline', 'flex_learned', 'flex_roi']
    config_times = []
    
    for config in configs:
        config_data = df[df['config'] == config]
        if len(config_data) > 0:
            config_times.append(config_data['avg_forward_ms'].mean())
        else:
            config_times.append(0)
    
    bars = ax.bar(configs, config_times, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('平均Forward时间 (ms)', fontsize=12)
    ax.set_title('不同配置的平均Forward时间对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'config_comparison.png', dpi=150)
    print(f"  已保存: {output_dir / 'config_comparison.png'}")
    plt.close()
    
    # 图2: 各场景的时间对比
    fig, ax = plt.subplots(figsize=(14, 8))
    
    examples = sorted(df['example'].unique())
    x = np.arange(len(examples))
    width = 0.25
    
    baseline_times = []
    flex_times = []
    roi_times = []
    
    for example in examples:
        baseline = df[(df['example'] == example) & (df['config'] == 'baseline')]['avg_forward_ms'].mean()
        flex = df[(df['example'] == example) & (df['config'] == 'flex_learned')]['avg_forward_ms'].mean()
        roi = df[(df['example'] == example) & (df['config'] == 'flex_roi')]['avg_forward_ms'].mean()
        
        baseline_times.append(baseline if not np.isnan(baseline) else 0)
        flex_times.append(flex if not np.isnan(flex) else 0)
        roi_times.append(roi if not np.isnan(roi) else 0)
    
    ax.bar(x - width, baseline_times, width, label='baseline', color='#3498db')
    ax.bar(x, flex_times, width, label='flex_learned', color='#e74c3c')
    ax.bar(x + width, roi_times, width, label='flex_roi', color='#2ecc71')
    
    ax.set_ylabel('平均Forward时间 (ms)', fontsize=12)
    ax.set_title('各场景的Forward时间对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(examples, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_comparison.png', dpi=150)
    print(f"  已保存: {output_dir / 'scenario_comparison.png'}")
    plt.close()


def main():
    # 数据文件路径
    csv_path = "test/results/core_modules_20251026_092238_一次成功的实验，但是时间偏高/summary_20251026_134844.csv"
    output_dir = "test/results/core_modules_20251026_092238_一次成功的实验，但是时间偏高/performance_analysis"
    
    print("性能分析报告")
    print("=" * 80)
    print(f"数据源: {csv_path}")
    print("=" * 80)
    
    # 加载数据
    df = load_data(csv_path)
    
    # 执行各项分析
    analyze_time_metrics(df)
    analyze_by_scenario(df)
    analyze_roi_overhead(df)
    analyze_flex_pdhg_overhead(df)
    identify_performance_issues(df)
    generate_visualizations(df, output_dir)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

