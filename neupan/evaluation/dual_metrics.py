"""
对偶可行性评估指标

提供完整的对偶约束违反率计算和分析工具
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class DualFeasibilityMetrics:
    """
    对偶可行性评估指标计算器
    
    用于评估对偶变量 μ 是否满足约束:
        μ ≥ 0
        ||G^T μ||_2 ≤ 1
    """
    
    def __init__(self, G: torch.Tensor, tolerance: float = 1e-6):
        """
        Args:
            G: 机器人几何矩阵 [E, 2]
            tolerance: 浮点数容差,用于软阈值判断
        """
        self.G = G
        self.tolerance = tolerance
    
    def compute_all_metrics(self, mu: torch.Tensor) -> Dict[str, float]:
        """
        计算所有违反率指标
        
        Args:
            mu: 对偶权重 [E, N]
        
        Returns:
            dict: 包含所有指标的字典
        """
        v = self.G.T @ mu  # [2, N]
        norms = torch.norm(v, dim=0)  # [N]
        
        metrics = {}
        
        # 1. 硬阈值违反率 (严格 >1.0)
        metrics['violation_rate_hard'] = (norms > 1.0).float().mean().item()
        
        # 2. 软违反率 (考虑浮点数精度)
        metrics['violation_rate_soft'] = (
            norms > 1.0 + self.tolerance
        ).float().mean().item()
        
        # 3. 加权违反率 (违反程度的平均)
        metrics['weighted_violation'] = (
            torch.clamp(norms - 1.0, min=0.0).mean().item()
        )
        
        # 4. 相对违反率
        metrics['relative_violation'] = (
            torch.clamp((norms - 1.0) / 1.0, min=0.0).mean().item()
        )
        
        # 5. 最大违反
        metrics['max_violation'] = (norms - 1.0).max().item()
        
        # 6. 百分位数
        if norms.numel() > 0:
            try:
                metrics['p50_norm'] = torch.quantile(norms, 0.50).item()
                metrics['p95_norm'] = torch.quantile(norms, 0.95).item()
                metrics['p99_norm'] = torch.quantile(norms, 0.99).item()
            except Exception:
                # fallback for older PyTorch versions
                sorted_norms = torch.sort(norms)[0]
                n = len(sorted_norms)
                metrics['p50_norm'] = sorted_norms[int(0.50 * n)].item()
                metrics['p95_norm'] = sorted_norms[int(0.95 * n)].item()
                metrics['p99_norm'] = sorted_norms[int(0.99 * n)].item()
        
        # 7. 分层违反率
        metrics['mild_violation_rate'] = (
            (norms > 1.0) & (norms <= 1.01)
        ).float().mean().item()
        metrics['moderate_violation_rate'] = (
            (norms > 1.01) & (norms <= 1.05)
        ).float().mean().item()
        metrics['severe_violation_rate'] = (
            norms > 1.05
        ).float().mean().item()
        
        # 8. 统计信息
        metrics['mean_norm'] = norms.mean().item()
        metrics['std_norm'] = norms.std().item()
        metrics['min_norm'] = norms.min().item()
        metrics['max_norm'] = norms.max().item()
        
        # 9. 非负约束违反
        metrics['nonneg_violation_rate'] = (mu < 0).float().mean().item()
        
        return metrics
    
    def compute_basic_metrics(self, mu: torch.Tensor) -> Tuple[float, float, float]:
        """
        计算基本指标 (快速版本)
        
        Args:
            mu: 对偶权重 [E, N]
        
        Returns:
            violation_rate: 软阈值违反率
            p95_norm: P95范数
            max_violation: 最大违反幅度
        """
        v = self.G.T @ mu
        norms = torch.norm(v, dim=0)
        
        violation_rate = (norms > 1.0 + self.tolerance).float().mean().item()
        
        try:
            p95_norm = torch.quantile(norms, 0.95).item()
        except Exception:
            sorted_norms = torch.sort(norms)[0]
            p95_norm = sorted_norms[int(0.95 * len(sorted_norms))].item()
        
        max_violation = (norms - 1.0).max().item()
        
        return violation_rate, p95_norm, max_violation
    
    def print_report(self, mu: torch.Tensor, name: str = '') -> Dict[str, float]:
        """
        打印详细报告
        
        Args:
            mu: 对偶权重 [E, N]
            name: 报告名称
        
        Returns:
            dict: 所有指标
        """
        metrics = self.compute_all_metrics(mu)
        
        print(f"\n{'='*70}")
        print(f"Dual Feasibility Report {name}")
        print(f"{'='*70}")
        
        print(f"\n基本统计:")
        print(f"  点数: {mu.shape[1]}")
        print(f"  平均范数: {metrics['mean_norm']:.6f}")
        print(f"  标准差: {metrics['std_norm']:.6f}")
        print(f"  范围: [{metrics['min_norm']:.6f}, {metrics['max_norm']:.6f}]")
        
        print(f"\n违反率:")
        print(f"  硬阈值 (>1.0): {metrics['violation_rate_hard']:.2%}")
        print(f"  软阈值 (>1.0+{self.tolerance}): {metrics['violation_rate_soft']:.2%}")
        print(f"  加权违反: {metrics['weighted_violation']:.6f}")
        print(f"  相对违反: {metrics['relative_violation']:.2%}")
        
        print(f"\n百分位数:")
        print(f"  P50: {metrics['p50_norm']:.6f}")
        print(f"  P95: {metrics['p95_norm']:.6f}")
        print(f"  P99: {metrics['p99_norm']:.6f}")
        
        print(f"\n分层违反:")
        print(f"  轻微 (1.0-1.01): {metrics['mild_violation_rate']:.2%}")
        print(f"  中等 (1.01-1.05): {metrics['moderate_violation_rate']:.2%}")
        print(f"  严重 (>1.05): {metrics['severe_violation_rate']:.2%}")
        
        print(f"\n最大违反:")
        print(f"  幅度: {metrics['max_violation']:.6f}")
        if metrics['max_violation'] > 0:
            print(f"  相对: {metrics['max_violation'] / 1.0 * 100:.2f}%")
        
        print(f"\n非负约束:")
        print(f"  违反率: {metrics['nonneg_violation_rate']:.2%}")
        
        print(f"\n{'='*70}\n")
        
        return metrics
    
    def compare_methods(self, mu_dict: Dict[str, torch.Tensor]) -> str:
        """
        对比多个方法的违反率
        
        Args:
            mu_dict: {method_name: mu_tensor}
        
        Returns:
            str: 格式化的对比表格
        """
        results = []
        for name, mu in mu_dict.items():
            metrics = self.compute_all_metrics(mu)
            results.append({
                'Method': name,
                'Violation Rate': f"{metrics['violation_rate_soft']:.2%}",
                'P95 Norm': f"{metrics['p95_norm']:.6f}",
                'Max Violation': f"{metrics['max_violation']:.2e}",
                'Weighted': f"{metrics['weighted_violation']:.6f}",
            })
        
        # 格式化输出
        header = f"{'Method':<20} {'Violation Rate':<15} {'P95 Norm':<12} {'Max Violation':<15} {'Weighted':<10}"
        separator = "=" * len(header)
        
        lines = [separator, header, separator]
        for r in results:
            line = f"{r['Method']:<20} {r['Violation Rate']:<15} {r['P95 Norm']:<12} {r['Max Violation']:<15} {r['Weighted']:<10}"
            lines.append(line)
        lines.append(separator)
        
        return '\n'.join(lines)
    
    def plot_distribution(self, mu: torch.Tensor, title: str = 'Dual Norm Distribution'):
        """
        绘制对偶范数分布图
        
        Args:
            mu: 对偶权重 [E, N]
            title: 图表标题
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return
        
        v = self.G.T @ mu
        norms = torch.norm(v, dim=0).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(norms, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Threshold (1.0)')
        plt.axvline(1.0 + self.tolerance, color='orange', linestyle=':', linewidth=2, 
                   label=f'Soft Threshold (1.0+{self.tolerance})')
        
        plt.xlabel('||G^T μ||_2', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def quick_violation_check(mu: torch.Tensor, G: torch.Tensor, 
                          tolerance: float = 1e-6) -> Tuple[float, float]:
    """
    快速检查违反率 (无需创建类实例)
    
    Args:
        mu: 对偶权重 [E, N]
        G: 几何矩阵 [E, 2]
        tolerance: 容差
    
    Returns:
        violation_rate: 违反率
        max_norm: 最大范数
    """
    v = G.T @ mu
    norms = torch.norm(v, dim=0)
    violation_rate = (norms > 1.0 + tolerance).float().mean().item()
    max_norm = norms.max().item()
    return violation_rate, max_norm


# 使用示例
if __name__ == '__main__':
    # 模拟数据
    torch.manual_seed(42)
    E, N = 4, 1000
    G = torch.randn(E, 2)
    mu = torch.randn(E, N) * 2.0  # 故意让一些违反约束
    
    # 创建评估器
    metrics = DualFeasibilityMetrics(G, tolerance=1e-6)
    
    # 打印详细报告
    metrics.print_report(mu, name='(Before Projection)')
    
    # 快速检查
    vr, mn = quick_violation_check(mu, G)
    print(f"Quick check: violation_rate={vr:.2%}, max_norm={mn:.6f}")

