# macroplace

Macro placement pipeline for the [Partcl/HRT Macro Placement Challenge 2026](https://github.com/partcleda/macro-place-challenge-2026).

## Architecture

Seven-stage pipeline plus a diagnostic harness. See `docs/architecture.md` for the full spec.

| Stage | Module | Purpose |
|---|---|---|
| -1 | `diagnostic/` | Evaluate any placement (used throughout) |
|  0 | `stages/stage0_classify.py` | Benchmark classification and dispatch |
|  1 | `stages/stage1_global.py` | DreamPlace global placement |
|  2 | `stages/stage2_legalize.py` | Hard-macro legalization |
|  3 | `stages/stage3_stdcell.py` | Soft macro re-optimization |
|  4 | `stages/stage4_orient.py` | Klein-4 orientation sweep |
|  5 | `stages/stage5_port.py` | Port-edge optimization + candidate selection |
|  6 | `stages/stage6_refine.py` | Multi-pass GPU-parallel SA refinement |
|  7 | `stages/stage7_finalize.py` | Final pass, verification, save artifacts |

## Quickstart

```bash
# Install
uv sync

# Set up benchmarks (one-time)
./scripts/setup_benchmarks.sh

# Run the diagnostic harness on a benchmark's initial placement
uv run mp-diagnose -b ibm01

# Or the python entry point directly
uv run python scripts/diagnose.py -b ibm01

# (later) Run the full pipeline
uv run mp-pipeline -b ibm01
```

## Directory layout

```
macro-place/
├── src/macroplace/             # the library
│   ├── io/                     # benchmark loading
│   ├── eval/                   # proxy cost, overlap, critical paths
│   ├── stages/                 # the seven pipeline stages
│   ├── diagnostic/             # the diagnostic harness
│   └── pipeline.py             # orchestrates stages end-to-end
├── scripts/                    # CLI entry points
├── tests/                      # pytest tests
├── benchmarks/                 # benchmark data (mostly gitignored)
│   └── external/               # cloned TILOS testcases (gitignored)
├── reports/                    # diagnostic outputs (gitignored)
├── checkpoints/                # intermediate placements (gitignored)
└── docs/
```

## Status

- [x] Repo scaffolding
- [ ] Loader (wraps TILOS PlacementCost)
- [ ] Diagnostic harness
- [ ] Stage 0–7 implementation
