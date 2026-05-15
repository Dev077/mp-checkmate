#!/usr/bin/env python
"""
CLI: run the diagnostic harness on a benchmark.

Examples:
    # Diagnose ibm01's initial placement
    uv run python scripts/diagnose.py -b ibm01

    # NG45 design with the 12 μm spacing check enabled
    uv run python scripts/diagnose.py -b ariane133 --regime NG45 \\
        --base benchmarks/external/MacroPlacement/Testcases/NG45
"""
from __future__ import annotations

import argparse
from pathlib import Path

from macroplace.diagnostic.harness import diagnose_placement, print_report, save_report
from macroplace.diagnostic.visualize import visualize_placement
from macroplace.io.loader import load_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-b", "--benchmark", required=True,
                        help="benchmark name (e.g., ibm01)")
    parser.add_argument(
        "--base",
        default="benchmarks/external/MacroPlacement/Testcases/ICCAD04",
        help="parent directory containing benchmark folders",
    )
    parser.add_argument("--regime", default="IBM", choices=["IBM", "NG45"])
    parser.add_argument("--tag", default=None,
                        help="output tag (defaults to <benchmark>_initial)")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip the PNG visualization")
    parser.add_argument("--outdir", default="reports",
                        help="directory for report outputs")
    args = parser.parse_args()

    bundle = load_benchmark(Path(args.base) / args.benchmark, name=args.benchmark)
    placement = bundle.current_placement()

    tag = args.tag or f"{args.benchmark}_initial"
    report = diagnose_placement(placement, bundle, regime=args.regime)
    print_report(report)
    save_report(report, tag, outdir=args.outdir)

    if not args.no_viz:
        visualize_placement(
            placement, bundle,
            report=report,
            outpath=Path(args.outdir) / f"{tag}.png",
        )


if __name__ == "__main__":
    main()
