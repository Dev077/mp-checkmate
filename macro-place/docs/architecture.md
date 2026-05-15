# Architecture

Seven-stage pipeline plus a diagnostic harness, dispatched on benchmark
structural properties.

## Strategic targets

- **Tier 1 (proxy cost):** 0.95–0.97 average. Beats the current leaderboard
  (1.01) with margin, qualifies for top 7.
- **Tier 2 (Grand Prize, $20K):** Pass the SCORING.md feasibility gate on
  every NG45 design — `WNS_sub ≥ min(WNS_SA, WNS_RP)` and
  `TNS_sub ≥ min(TNS_SA, TNS_RP)`. Then maximize the weighted geometric
  mean `(R_WNS^3 × R_TNS^2 × R_Area^1)^(1/6)`. Push past zero-slack on
  closable designs for unbounded WNS credit.
- **Innovation Award:** not targeted. The pipeline is solid execution of
  known techniques; calling it innovative would be inflating it.

## Stage map

| Stage | Module | Purpose | Runtime |
|---|---|---|---|
| -1 | `diagnostic/` | Evaluate any placement | seconds when called |
|  0 | `stage0_classify` | Benchmark classification + dispatch | <1 min |
|  1 | `stage1_global` | DreamPlace global placement, 2–3 multi-start | 5–8 min |
|  2 | `stage2_legalize` | Hard-macro legalization via slot assignment | <1 min |
|  3 | `stage3_stdcell` | Soft macro re-optimization | 1–2 min |
|  4 | `stage4_orient` | Klein-4 orientation sweep | 2–3 min |
|  5 | `stage5_port` | Port-edge optimization + candidate selection | 1 min |
|  6 | `stage6_refine` | Multi-pass GPU-parallel SA refinement | 30–40 min |
|  7 | `stage7_finalize` | Final pass, verification, save artifacts | 1 min |

## Dispatch criteria

Structural, observable from any benchmark — not benchmark-name-based.

| Property | IBM-regime | NG45-regime |
|---|---|---|
| Canvas dimension | <100 μm | >500 μm |
| Hard-macro count | 246–786 | 20–~150 |
| Soft/hard ratio | 3–4× | likely much smaller |
| Spacing rule | Pack tight | ≥12 μm (mandatory) |
| Refinement λ | ~0.05 | ~0.2 |
| Critical-path method | Structural proxy | DAG traversal (TODO) |

Within IBM-regime, sub-classify by top-10% hard-macro area fraction:
- **sea** (<30%): hard macros nearly identical → symmetry breaking matters
- **anchor** (>50%): a few giants dominate → place them first
- **hybrid** (30–50%): in between

## Stage 6 details (the main grinding step)

Objective: `proxy_cost + λ × critical_path_HPWL`.

Four passes:

1. **WL hotspots (5 min):** propose moves only on macros connected to top-10%
   highest-HPWL nets.
2. **Density hotspots (5 min):** propose moves on macros in top-10% densest
   grid cells; bias toward emptier regions.
3. **Congestion hotspots (5 min):** propose moves on macros contributing to
   worst-congestion regions; bias orientations to reduce local congestion.
4. **General SA (15–20 min):** all move types; proposer weights set adaptively
   from current cost decomposition.

Critical implementation rules:

- `plc.set_use_incremental_cost(True)` for fast move evaluation.
- **Do not stop at WNS surrogate ≥ 0.** SCORING.md's zero-slack tightening
  rule gives unbounded WNS upside on closable designs.
- Track best-ever placement separately from current SA state; restore on
  Stage-6 failure.
- Per-design runtime allocation: scale Stage 6 budget with macro count.

## SCORING.md compliance (strict)

- Hard-macro overlaps disqualify the submission. Stage 7 must verify zero.
- NG45 ≥12 μm spacing — enforce in Stage 2, verify in Stage 7. If verification
  fails, run post-hoc separation; if that fails, accept worse proxy.
- 1-hour runtime cap per benchmark, hard.
- One algorithm per team; dispatch by benchmark structural properties only.

## Open items

- DreamPlace integration (Stage 1)
- DAG-traversal critical-path identification for NG45
- ORFS validation harness for λ tuning
- Per-design runtime allocator
