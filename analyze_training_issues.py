#!/usr/bin/env python3
"""
深度根因分析脚本: NeuPAN 模块训练问题诊断
分析 PDHG、SE(2)、Learned-Prox 的训练失败原因
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# 模型目录
NEW_MODEL_DIR = Path("example/dune_train/model")
CONFIG_DIR = Path("example/dune_train")

def parse_results_txt(results_path: Path) -> Dict:
    """解析 results.txt 文件,提取训练曲线"""
    if not results_path.exists():
        return None
    
    with open(results_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有 epoch 的指标
    epochs = re.findall(
        r'Epoch (\d+)/\d+ learning rate ([\d.e-]+)\s+-+\s+Losses:\s+'
        r'Mu Loss:\s+([\d.e-]+)\s+\|\s+Validate Mu Loss:\s+([\d.e-]+)\s+'
        r'Distance Loss:\s+([\d.e-]+)\s+\|\s+Validate Distance Loss:\s+([\d.e-]+)\s+'
        r'Fa Loss:\s+([\d.e-]+)\s+\|\s+Validate Fa Loss:\s+([\d.e-]+)\s+'
        r'Fb Loss:\s+([\d.e-]+)\s+\|\s+Validate Fb Loss:\s+([\d.e-]+)',
        content
    )
    
    if not epochs:
        return None
    
    # 构建训练曲线
    curve = {
        'epochs': [],
        'lr': [],
        'train_mu': [],
        'val_mu': [],
        'train_dist': [],
        'val_dist': [],
        'train_fa': [],
        'val_fa': [],
        'train_fb': [],
        'val_fb': []
    }
    
    for epoch_data in epochs:
        curve['epochs'].append(int(epoch_data[0]))
        curve['lr'].append(float(epoch_data[1]))
        curve['train_mu'].append(float(epoch_data[2]))
        curve['val_mu'].append(float(epoch_data[3]))
        curve['train_dist'].append(float(epoch_data[4]))
        curve['val_dist'].append(float(epoch_data[5]))
        curve['train_fa'].append(float(epoch_data[6]))
        curve['val_fa'].append(float(epoch_data[7]))
        curve['train_fb'].append(float(epoch_data[8]))
        curve['val_fb'].append(float(epoch_data[9]))
    
    return curve

def parse_results_reg_txt(results_reg_path: Path) -> Dict:
    """解析 results_reg.txt 文件,提取 KKT 损失曲线"""
    if not results_reg_path.exists():
        return None
    
    with open(results_reg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有 epoch 的 KKT 指标
    reg_epochs = re.findall(
        r'Epoch (\d+)/\d+ learning rate ([\d.e-]+)\s+-+\s+Reg Losses:\s+'
        r'L_constr:\s+([\d.e-]+)\s+\|\s+Validate L_constr:\s+([\d.e-]+)\s+'
        r'L_kkt:\s+([\d.e-]+)\s+\|\s+Validate L_kkt:\s+([\d.e-]+)',
        content
    )
    
    if not reg_epochs:
        return None
    
    curve = {
        'epochs': [],
        'train_l_constr': [],
        'val_l_constr': [],
        'train_l_kkt': [],
        'val_l_kkt': []
    }
    
    for epoch_data in reg_epochs:
        curve['epochs'].append(int(epoch_data[0]))
        curve['train_l_constr'].append(float(epoch_data[2]))
        curve['val_l_constr'].append(float(epoch_data[3]))
        curve['train_l_kkt'].append(float(epoch_data[4]))
        curve['val_l_kkt'].append(float(epoch_data[5]))
    
    return curve

def analyze_convergence(curve: Dict) -> Dict:
    """分析训练收敛性"""
    if not curve or len(curve['epochs']) < 10:
        return {'converged': False, 'reason': '数据不足'}
    
    # 检查最后 10% 的 epoch
    n = len(curve['epochs'])
    last_10pct = max(10, n // 10)
    
    val_mu_last = curve['val_mu'][-last_10pct:]
    val_dist_last = curve['val_dist'][-last_10pct:]
    
    # 计算最后阶段的平均值和标准差
    import statistics
    mu_mean = statistics.mean(val_mu_last)
    mu_std = statistics.stdev(val_mu_last) if len(val_mu_last) > 1 else 0
    dist_mean = statistics.mean(val_dist_last)
    dist_std = statistics.stdev(val_dist_last) if len(val_dist_last) > 1 else 0
    
    # 判断是否收敛
    converged = True
    issues = []
    
    # 检查损失值是否异常高
    if mu_mean > 1e-3:
        converged = False
        issues.append(f'Mu Loss 异常高 ({mu_mean:.2e}, 正常应为 1e-6 量级)')
    
    if dist_mean > 1e-2:
        converged = False
        issues.append(f'Distance Loss 异常高 ({dist_mean:.2e}, 正常应为 1e-5 量级)')
    
    # 检查是否震荡
    if mu_std / (mu_mean + 1e-10) > 0.5:
        issues.append(f'Mu Loss 震荡严重 (std/mean = {mu_std/mu_mean:.2f})')
    
    if dist_std / (dist_mean + 1e-10) > 0.5:
        issues.append(f'Distance Loss 震荡严重 (std/mean = {dist_std/dist_mean:.2f})')
    
    # 检查是否过拟合
    train_mu_last = curve['train_mu'][-last_10pct:]
    train_mu_mean = statistics.mean(train_mu_last)
    if mu_mean / (train_mu_mean + 1e-10) > 2.0:
        issues.append(f'可能过拟合 (val/train = {mu_mean/train_mu_mean:.2f})')
    
    return {
        'converged': converged,
        'val_mu_mean': mu_mean,
        'val_mu_std': mu_std,
        'val_dist_mean': dist_mean,
        'val_dist_std': dist_std,
        'issues': issues
    }

def load_yaml_config(yaml_path: Path) -> Dict:
    """加载 YAML 配置文件"""
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_model(model_name: str) -> Dict:
    """分析单个模型的训练问题"""
    model_dir = NEW_MODEL_DIR / model_name
    results_path = model_dir / "results.txt"
    results_reg_path = model_dir / "results_reg.txt"
    
    # 解析训练曲线
    curve = parse_results_txt(results_path)
    reg_curve = parse_results_reg_txt(results_reg_path)
    
    # 分析收敛性
    convergence = analyze_convergence(curve) if curve else {'converged': False, 'reason': '无训练数据'}
    
    # 提取配置信息
    config = {}
    if 'acker' in model_name:
        config['robot_type'] = 'acker'
    elif 'diff' in model_name:
        config['robot_type'] = 'diff'
    
    config['has_learned_prox'] = 'learned' in model_name
    config['has_kkt'] = 'kkt' in model_name
    config['has_pdhg'] = 'pdhg' in model_name
    config['has_se2'] = 'se2' in model_name
    
    return {
        'model_name': model_name,
        'config': config,
        'curve': curve,
        'reg_curve': reg_curve,
        'convergence': convergence
    }

def main():
    print("=" * 80)
    print("NeuPAN 模块训练问题深度分析")
    print("=" * 80)
    
    # 扫描所有模型
    problem_models = []
    
    for model_dir in NEW_MODEL_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        analysis = analyze_model(model_name)
        
        # 只关注有问题的模型
        if not analysis['convergence']['converged']:
            problem_models.append(analysis)
    
    print(f"\n发现 {len(problem_models)} 个训练未收敛的模型:\n")
    
    # 按问题类型分组
    pdhg_issues = []
    se2_issues = []
    learned_prox_issues = []
    
    for model in problem_models:
        config = model['config']
        convergence = model['convergence']
        
        print(f"模型: {model['model_name']}")
        print(f"  配置: Robot={config.get('robot_type')}, Learned-Prox={config.get('has_learned_prox')}, "
              f"KKT={config.get('has_kkt')}, PDHG={config.get('has_pdhg')}, SE(2)={config.get('has_se2')}")
        print(f"  收敛状态: {'✅ 收敛' if convergence['converged'] else '❌ 未收敛'}")
        if 'val_mu_mean' in convergence:
            print(f"  Val Mu Loss: {convergence['val_mu_mean']:.2e} ± {convergence['val_mu_std']:.2e}")
            print(f"  Val Dist Loss: {convergence['val_dist_mean']:.2e} ± {convergence['val_dist_std']:.2e}")
        if convergence.get('issues'):
            print(f"  问题:")
            for issue in convergence['issues']:
                print(f"    - {issue}")
        print()
        
        # 分类
        if config.get('has_pdhg'):
            pdhg_issues.append(model)
        if config.get('has_se2'):
            se2_issues.append(model)
        if config.get('has_learned_prox') and not config.get('has_kkt'):
            learned_prox_issues.append(model)
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("问题总结")
    print("=" * 80)
    
    print(f"\n1. PDHG 相关问题 ({len(pdhg_issues)} 个模型):")
    for model in pdhg_issues:
        print(f"   - {model['model_name']}: {', '.join(model['convergence'].get('issues', []))}")
    
    print(f"\n2. SE(2) 相关问题 ({len(se2_issues)} 个模型):")
    for model in se2_issues:
        print(f"   - {model['model_name']}: {', '.join(model['convergence'].get('issues', []))}")
    
    print(f"\n3. Learned-Prox 单独使用问题 ({len(learned_prox_issues)} 个模型):")
    for model in learned_prox_issues:
        print(f"   - {model['model_name']}: {', '.join(model['convergence'].get('issues', []))}")
    
    # 保存详细分析结果
    output_path = Path("docs/训练问题分析结果.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'problem_models': [
                {
                    'model_name': m['model_name'],
                    'config': m['config'],
                    'convergence': m['convergence']
                }
                for m in problem_models
            ],
            'summary': {
                'total_problems': len(problem_models),
                'pdhg_issues': len(pdhg_issues),
                'se2_issues': len(se2_issues),
                'learned_prox_issues': len(learned_prox_issues)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细分析结果已保存到: {output_path}")

if __name__ == "__main__":
    main()

