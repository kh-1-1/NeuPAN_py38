"""
Visualize PDHG-Unroll evaluation results.
Generates comparison charts for J=0 vs J=1.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(json_path: str) -> pd.DataFrame:
    """Load batch results JSON and convert to DataFrame."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for cfg_key, cfg_data in data.items():
        if not isinstance(cfg_data, dict) or 'aggregate' not in cfg_data:
            continue
        
        parts = cfg_key.split("::")
        if len(parts) != 4:
            continue
        
        example, kin, proj, j_str = parts
        J = int(j_str.replace("J", ""))
        
        aggr = cfg_data['aggregate']
        rows.append({
            'example': example,
            'kinematics': kin,
            'projection': proj,
            'unroll_J': J,
            'avg_pre_violation_rate': aggr.get('avg_pre_violation_rate_mean', 0),
            'avg_pre_p95': aggr.get('avg_pre_p95_mean', 0),
            'avg_pre_excess': aggr.get('avg_pre_excess_mean', 0),
            'avg_post_max': aggr.get('avg_post_max_mean', 0),
            'max_post_max': aggr.get('max_post_max_mean', 0),
            'avg_post_excess': aggr.get('avg_post_excess_mean', 0),
            'steps_executed': aggr.get('steps_executed_mean', 0),
        })
    
    return pd.DataFrame(rows)


def plot_violation_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot pre-violation rate comparison (J=0 vs J=1)."""
    df_pivot = df.pivot_table(
        index=['example', 'kinematics'],
        columns='unroll_J',
        values='avg_pre_violation_rate'
    ).reset_index()
    
    df_pivot = df_pivot.sort_values(by=0, ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(df_pivot))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df_pivot[0], width, 
                   label='J=0 (Baseline)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], df_pivot[1], width,
                   label='J=1 (PDHG-Unroll)', color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('Example (sorted by J=0 violation rate)', fontsize=12)
    ax.set_ylabel('Avg Pre-Violation Rate', fontsize=12)
    ax.set_title('PDHG-Unroll Impact on Dual Feasibility\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['example']}\n({row['kinematics']})" 
                        for _, row in df_pivot.iterrows()], 
                       rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add improvement percentage labels
    for i, (_, row) in enumerate(df_pivot.iterrows()):
        if row[0] > 0:
            improvement = (row[0] - row[1]) / row[0] * 100
            ax.text(i, max(row[0], row[1]) + 0.02, f'-{improvement:.1f}%',
                   ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'violation_rate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'violation_rate_comparison.png'}")
    plt.close()


def plot_p95_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot P95 dual norm comparison."""
    df_pivot = df.pivot_table(
        index=['example', 'kinematics'],
        columns='unroll_J',
        values='avg_pre_p95'
    ).reset_index()
    
    # Filter out zero entries (pf example)
    df_pivot = df_pivot[df_pivot[0] > 0].sort_values(by=0, ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(df_pivot))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df_pivot[0], width,
                   label='J=0 (Baseline)', color='#3498db', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], df_pivot[1], width,
                   label='J=1 (PDHG-Unroll)', color='#9b59b6', alpha=0.8)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Feasibility Threshold (1.0)')
    
    ax.set_xlabel('Example (sorted by J=0 P95)', fontsize=12)
    ax.set_ylabel('Avg Pre-P95 (||G^T μ||)', fontsize=12)
    ax.set_title('PDHG-Unroll Impact on Dual Norm P95\n(Target: ≤ 1.0)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['example']}\n({row['kinematics']})" 
                        for _, row in df_pivot.iterrows()],
                       rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p95_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'p95_comparison.png'}")
    plt.close()


def plot_steps_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot execution steps comparison."""
    df_pivot = df.pivot_table(
        index=['example', 'kinematics'],
        columns='unroll_J',
        values='steps_executed'
    ).reset_index()
    
    df_pivot = df_pivot[df_pivot[0] > 0].sort_values(by=0, ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(df_pivot))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df_pivot[0], width,
                   label='J=0 (Baseline)', color='#f39c12', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], df_pivot[1], width,
                   label='J=1 (PDHG-Unroll)', color='#1abc9c', alpha=0.8)
    
    ax.set_xlabel('Example', fontsize=12)
    ax.set_ylabel('Steps Executed', fontsize=12)
    ax.set_title('PDHG-Unroll Impact on Execution Steps\n(Fewer Steps = Faster Convergence)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['example']}\n({row['kinematics']})" 
                        for _, row in df_pivot.iterrows()],
                       rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add step reduction labels
    for i, (_, row) in enumerate(df_pivot.iterrows()):
        if row[0] > 0:
            reduction = row[0] - row[1]
            if abs(reduction) > 1:
                ax.text(i, max(row[0], row[1]) + 2, f'{reduction:+.0f}',
                       ha='center', va='bottom', fontsize=8, 
                       color='green' if reduction > 0 else 'red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'steps_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'steps_comparison.png'}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table."""
    summary = []
    
    for J in [0, 1]:
        df_j = df[df['unroll_J'] == J]
        summary.append({
            'unroll_J': J,
            'avg_violation_rate': df_j['avg_pre_violation_rate'].mean(),
            'max_violation_rate': df_j['avg_pre_violation_rate'].max(),
            'avg_p95': df_j[df_j['avg_pre_p95'] > 0]['avg_pre_p95'].mean(),
            'max_p95': df_j['avg_pre_p95'].max(),
            'avg_steps': df_j[df_j['steps_executed'] > 0]['steps_executed'].mean(),
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Calculate improvements
    improvement = {
        'Metric': ['Avg Violation Rate', 'Max Violation Rate', 'Avg P95', 'Max P95', 'Avg Steps'],
        'J=0': [
            f"{summary_df.loc[0, 'avg_violation_rate']:.4f}",
            f"{summary_df.loc[0, 'max_violation_rate']:.4f}",
            f"{summary_df.loc[0, 'avg_p95']:.4f}",
            f"{summary_df.loc[0, 'max_p95']:.4f}",
            f"{summary_df.loc[0, 'avg_steps']:.1f}",
        ],
        'J=1': [
            f"{summary_df.loc[1, 'avg_violation_rate']:.4f}",
            f"{summary_df.loc[1, 'max_violation_rate']:.4f}",
            f"{summary_df.loc[1, 'avg_p95']:.4f}",
            f"{summary_df.loc[1, 'max_p95']:.4f}",
            f"{summary_df.loc[1, 'avg_steps']:.1f}",
        ],
        'Improvement': [
            f"{(summary_df.loc[0, 'avg_violation_rate'] - summary_df.loc[1, 'avg_violation_rate']) / summary_df.loc[0, 'avg_violation_rate'] * 100:.1f}%",
            f"{(summary_df.loc[0, 'max_violation_rate'] - summary_df.loc[1, 'max_violation_rate']) / summary_df.loc[0, 'max_violation_rate'] * 100:.1f}%",
            f"{(summary_df.loc[0, 'avg_p95'] - summary_df.loc[1, 'avg_p95']) / summary_df.loc[0, 'avg_p95'] * 100:.1f}%",
            f"{(summary_df.loc[0, 'max_p95'] - summary_df.loc[1, 'max_p95']) / summary_df.loc[0, 'max_p95'] * 100:.1f}%",
            f"{(summary_df.loc[0, 'avg_steps'] - summary_df.loc[1, 'avg_steps']) / summary_df.loc[0, 'avg_steps'] * 100:.1f}%",
        ]
    }
    
    improvement_df = pd.DataFrame(improvement)
    
    # Save as markdown
    md_path = output_dir / 'summary_statistics.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Summary Statistics (J=0 vs J=1)\n\n")
        f.write(improvement_df.to_markdown(index=False))
        f.write("\n\n**Key Findings**:\n")
        f.write(f"- Violation rate reduced by **{improvement['Improvement'][0]}**\n")
        f.write(f"- P95 dual norm reduced by **{improvement['Improvement'][2]}**\n")
        f.write(f"- Execution steps changed by **{improvement['Improvement'][4]}**\n")
    
    print(f"✓ Saved: {md_path}")
    print("\n" + improvement_df.to_string(index=False))


def main():
    # Find latest batch results (by modification time)
    results_dir = Path("test/results")
    # Prefer Stage 2 timing runs: modes-hard only
    batch_dirs = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("batch_unroll_modes-hard_")
    ] or [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("batch_unroll")]
    if not batch_dirs:
        print("No batch results found in test/results/")
        return

    latest_batch = max(batch_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Analyzing: {latest_batch.name}\n")

    # Find most recent JSON file
    json_files = list(latest_batch.glob("*.json"))
    if not json_files:
        print(f"No JSON file found in {latest_batch}")
        return
    
    json_path = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading: {json_path.name}\n")
    
    # Load and process data
    df = load_results(str(json_path))
    print(f"Loaded {len(df)} configurations\n")
    
    # Create output directory
    output_dir = latest_batch / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating visualizations...")
    plot_violation_rate_comparison(df, output_dir)
    plot_p95_comparison(df, output_dir)
    plot_steps_comparison(df, output_dir)
    
    # Generate summary
    print("\nGenerating summary statistics...")
    generate_summary_table(df, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
