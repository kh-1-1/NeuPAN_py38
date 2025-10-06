"""
Analyze timing metrics from PDHG-Unroll Stage 2 runs.
- Computes avg_total_time_ms, avg_pdhg_time_ms, and pdhg_time_share per config
- Computes cost_per_percent for J=1/2/3 relative to J=0
Usage:
  python test/analyze_timing_results.py [path_to_batch_json]
If no path is provided, it finds the latest batch_unroll_* directory and its JSON.
"""

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Any, Tuple, List


def find_latest_batch_json() -> Path:
    results_dir = Path("test/results")
    batch_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("batch_unroll_")]
    if not batch_dirs:
        raise FileNotFoundError("No batch_unroll_* directory found under test/results/")

    # Pick most recently modified directory
    latest_dir = max(batch_dirs, key=lambda d: d.stat().st_mtime)
    json_files = list(latest_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.json in {latest_dir}")

    # Pick most recently modified JSON
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    return latest_json


def parse_key(k: str) -> Tuple[str, str, str, int]:
    parts = k.split("::")
    if len(parts) != 4:
        raise ValueError(f"Bad key: {k}")
    ex, kin, proj, j = parts
    J = int(j.replace("J", ""))
    return ex, kin, proj, J


def avg_time_from_runs(runs: List[Dict[str, Any]], field: str) -> float:
    vals = [r.get(field) for r in runs if r.get(field) is not None]
    return float(mean(vals)) if vals else float('nan')


def main():
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = find_latest_batch_json()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect per-config timing and violation
    rows: List[Dict[str, Any]] = []
    by_ex_kin: Dict[Tuple[str, str], Dict[int, Dict[str, float]]] = {}

    for cfg_key, cfg in data.items():
        if not isinstance(cfg, dict) or 'aggregate' not in cfg:
            continue
        ex, kin, proj, J = parse_key(cfg_key)
        aggr = cfg['aggregate']
        runs = cfg.get('runs', [])

        mean_total = avg_time_from_runs(runs, 'avg_total_time_ms')
        mean_pdhg = avg_time_from_runs(runs, 'avg_pdhg_time_ms')
        share = (mean_pdhg / mean_total) if (mean_total and mean_total == mean_total and mean_total > 0) and (mean_pdhg == mean_pdhg) else float('nan')

        viol = aggr.get('avg_pre_violation_rate_mean')

        rows.append({
            'example': ex, 'kin': kin, 'proj': proj, 'J': J,
            'avg_total_time_ms': mean_total,
            'avg_pdhg_time_ms': mean_pdhg,
            'pdhg_time_share': share,
            'viol_mean': viol,
            'steps_mean': aggr.get('steps_executed_mean'),
        })

        by_ex_kin.setdefault((ex, kin), {})[J] = {
            'time': mean_total,
            'viol': viol,
        }

    # Print summary table
    print("\nPer-config timing summary (ms):")
    print("example,kin,proj,J,avg_total_time_ms,avg_pdhg_time_ms,pdhg_time_share,viol_mean,steps_mean")
    for r in sorted(rows, key=lambda x: (x['example'], x['kin'], x['proj'], x['J'])):
        print(f"{r['example']},{r['kin']},{r['proj']},{r['J']},{r['avg_total_time_ms']:.3f},{r['avg_pdhg_time_ms']:.3f},{r['pdhg_time_share']:.3f},{r['viol_mean']},{r['steps_mean']}")

    # Compute cost_per_percent per example/kin and overall
    def cpp(tJ, t0, v0, vJ):
        if v0 is None or vJ is None or v0 <= vJ:
            return float('nan')
        return (tJ - t0) / ((v0 - vJ) * 100.0)

    cpp_rows = []
    for (ex, kin), mp in sorted(by_ex_kin.items()):
        base = mp.get(0)
        if not base:
            continue
        for J in [1, 2, 3]:
            cur = mp.get(J)
            if not cur:
                continue
            cpp_val = cpp(cur['time'], base['time'], base['viol'], cur['viol'])
            cpp_rows.append({'example': ex, 'kin': kin, 'J': J, 'cost_per_percent_ms_per_pct': cpp_val})

    print("\nCost-per-percent (ms per 1% viol reduction) vs J=0:")
    print("example,kin,J,cost_per_percent_ms_per_pct")
    for r in cpp_rows:
        val = r['cost_per_percent_ms_per_pct']
        print(f"{r['example']},{r['kin']},{r['J']},{val if val==val else 'NA'}")

    # Overall averages
    shares = [r['pdhg_time_share'] for r in rows if r['pdhg_time_share'] == r['pdhg_time_share']]
    overall_share = mean(shares) if shares else float('nan')
    print("\nOverall PDHG share (avg across configs):", f"{overall_share*100:.2f}%" if overall_share==overall_share else "NA")

    # Realtime check
    viol_summary = {}
    for r in rows:
        key = (r['example'], r['kin'], r['J'])
        viol_summary[key] = r['avg_total_time_ms']
    bad = [(k, t) for k, t in viol_summary.items() if t == t and t > 100.0]
    if bad:
        print("\nRealtime check failures (avg_total_time_ms > 100 ms):")
        for (ex, kin, J), t in bad:
            print(f"  {ex}|{kin}|J{J}: {t:.2f} ms")
    else:
        print("\nRealtime check passed for all configs (<=100 ms).")

    print(f"\nAnalyzed: {json_path}")


if __name__ == '__main__':
    main()
