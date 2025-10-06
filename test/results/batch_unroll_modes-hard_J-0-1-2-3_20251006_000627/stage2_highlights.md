# Stage 2 Timing Analysis — Highlights

Source JSON: `test\results\batch_unroll_modes-hard_J-0-1-2-3_20251006_000627\batch_summary_modes-hard_J-0-1-2-3_20251006_072004.json`

## Key Findings

- PDHG share unavailable (planner did not report `last_timing['pdhg']`)
- Cost-per-percent (ms per 1% viol reduction) — best J per scenario:
  - corridor|acker: NA (insufficient reduction or data)
  - corridor|diff: J=2 (cpp=0.407 ms/%)
  - dyna_obs|acker: J=2 (cpp=0.080 ms/%)
  - dyna_obs|diff: NA (insufficient reduction or data)
  - polygon_robot|diff: J=1 (cpp=0.068 ms/%)

## Realtime Check (>100 ms avg total FAIL)
- corridor|acker|J0: 111.50 ms
- corridor|acker|J1: 107.44 ms
- corridor|acker|J2: 103.22 ms
- corridor|acker|J3: 110.16 ms
- corridor|diff|J1: 119.32 ms