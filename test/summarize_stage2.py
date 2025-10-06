"""
Summarize Stage 2 timing analysis for PDHG-Unroll.

Outputs a concise markdown highlighting:
- Avg total time per config
- PDHG share if available (planner must expose last_timing['pdhg'])
- Cost-per-percent for J=1/2/3 vs J=0 per (example,kin)
- Best J per (example,kin) by lowest positive cost-per-percent
- Realtime pass/fail (avg_total_time_ms <= 100)

Usage:
  python test/summarize_stage2.py [path_to_batch_json]
If no path given, finds the most recent 'batch_unroll_modes-hard_*' JSON under test/results.
"""

from pathlib import Path
from statistics import mean
import json
import sys
from typing import Dict, Any, Tuple, List


def find_stage2_json() -> Path:
    root = Path("test/results")
    batch_dirs = [
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("batch_unroll_modes-hard_")
    ]
    if not batch_dirs:
        raise FileNotFoundError("No Stage 2 batch dir starting with 'batch_unroll_modes-hard_' found")
    latest = max(batch_dirs, key=lambda d: d.stat().st_mtime)
    json_files = list(latest.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON in {latest}")
    return max(json_files, key=lambda f: f.stat().st_mtime)


def parse_key(k: str) -> Tuple[str, str, str, int]:
    parts = k.split("::")
    if len(parts) != 4:
        raise ValueError(f"Bad key: {k}")
    ex, kin, proj, j = parts
    return ex, kin, proj, int(j.replace("J", ""))


def avg_from_runs(runs: List[Dict[str, Any]], field: str) -> float:
    vals = [r.get(field) for r in runs if r.get(field) is not None]
    return float(mean(vals)) if vals else float('nan')


def main():
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = find_stage2_json()

    data = json.loads(Path(json_path).read_text(encoding='utf-8'))

    # Collect
    per_cfg: List[Dict[str, Any]] = []
    by_ex_kin: Dict[Tuple[str, str], Dict[int, Dict[str, float]]] = {}

    for k, v in data.items():
        if not isinstance(v, dict) or 'aggregate' not in v:
            continue
        ex, kin, proj, J = parse_key(k)
        if proj != 'hard':
            continue
        aggr = v['aggregate']
        runs = v.get('runs', [])
        total = avg_from_runs(runs, 'avg_total_time_ms')
        pdhg = avg_from_runs(runs, 'avg_pdhg_time_ms')
        share = (pdhg / total) if (pdhg == pdhg) and (total == total) and total > 0 else float('nan')
        viol = aggr.get('avg_pre_violation_rate_mean')
        per_cfg.append({'example': ex, 'kin': kin, 'J': J, 'total': total, 'pdhg': pdhg, 'share': share, 'viol': viol})
        by_ex_kin.setdefault((ex, kin), {})[J] = {'time': total, 'viol': viol}

    # Compute cost-per-percent and best J per (ex,kin)
    def cpp(tJ, t0, v0, vJ):
        if v0 is None or vJ is None or v0 <= vJ:
            return float('nan')
        return (tJ - t0) / ((v0 - vJ) * 100.0)

    cpp_table: List[Tuple[str, str, int, float]] = []
    bestJ: Dict[Tuple[str, str], Tuple[int, float]] = {}

    for (ex, kin), mp in sorted(by_ex_kin.items()):
        base = mp.get(0)
        if not base:
            continue
        best = (None, float('inf'))
        for J in (1, 2, 3):
            cur = mp.get(J)
            if not cur:
                continue
            val = cpp(cur['time'], base['time'], base['viol'], cur['viol'])
            cpp_table.append((ex, kin, J, val))
            if val == val and val > 0 and val < best[1]:
                best = (J, val)
        bestJ[(ex, kin)] = best

    # Global summaries
    shares = [r['share'] for r in per_cfg if r['share'] == r['share'] and r['J'] == 1]
    share_pass = [s for s in shares if s < 0.05]
    overall_share = mean(shares) if shares else float('nan')

    realtime_bad = [(r['example'], r['kin'], r['J'], r['total']) for r in per_cfg if r['total'] == r['total'] and r['total'] > 100.0]

    # Write markdown
    out_dir = Path(json_path).parent
    md = out_dir / 'stage2_highlights.md'
    lines = []
    lines.append('# Stage 2 Timing Analysis — Highlights\n')
    lines.append(f'Source JSON: `{json_path}`\n')
    lines.append('## Key Findings\n')
    if shares:
        lines.append(f'- J=1 PDHG share avg: {overall_share*100:.2f}% ({len(share_pass)}/{len(shares)} configs < 5%)')
    else:
        lines.append('- PDHG share unavailable (planner did not report `last_timing[\'pdhg\']`)')
    lines.append('- Cost-per-percent (ms per 1% viol reduction) — best J per scenario:')
    for (ex, kin), (bj, val) in bestJ.items():
        if bj is None or val == float('inf') or val != val:
            lines.append(f'  - {ex}|{kin}: NA (insufficient reduction or data)')
        else:
            lines.append(f'  - {ex}|{kin}: J={bj} (cpp={val:.3f} ms/%)')
    if realtime_bad:
        lines.append('\n## Realtime Check (>100 ms avg total FAIL)')
        for ex, kin, J, t in sorted(realtime_bad):
            lines.append(f'- {ex}|{kin}|J{J}: {t:.2f} ms')
    else:
        lines.append('\n## Realtime Check\n- All configs pass (avg_total_time_ms <= 100 ms)')

    (md).write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote: {md}")

    # Also write CSV of cpp values
    import csv
    csv_path = out_dir / 'stage2_cost_per_percent.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['example', 'kin', 'J', 'cost_per_percent_ms_per_pct'])
        for ex, kin, J, val in cpp_table:
            w.writerow([ex, kin, J, 'NA' if val != val else f'{val:.6f}'])
    print(f"Wrote: {csv_path}")


if __name__ == '__main__':
    main()

