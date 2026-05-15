"""
Diagnostic harness.

Single entry point — `diagnose_placement` — that produces a full
`PlacementReport` from a placement.  Used by every stage of the pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from macroplace.diagnostic.report import HotspotCell, PlacementReport
from macroplace.eval.critical_path import (
    critical_path_hpwl,
    identify_critical_edges,
)
from macroplace.eval.overlap import (
    NG45_MIN_SPACING_UM,
    check_overlaps,
    min_macro_spacing,
)
from macroplace.eval.proxy_cost import compute_proxy_cost
from macroplace.io.loader import BenchmarkBundle


# ---------------------------------------------------------------------------
# Hotspot extraction
# ---------------------------------------------------------------------------

def _grid_shape(plc):
    """Return (n_rows, n_cols) for the evaluator's density/congestion grids."""
    return int(plc.grid_row), int(plc.grid_col)


def _extract_grid_hotspots(values: np.ndarray, plc, top_k: int) -> list[dict]:
    """Return the top-K cells of a grid as a list of HotspotCell dicts."""
    if values.size == 0:
        return []
    n_rows, n_cols = _grid_shape(plc)
    if values.ndim == 1:
        values = values.reshape(n_rows, n_cols)
    W, H = plc.get_canvas_width_height()
    cell_w = float(W) / n_cols
    cell_h = float(H) / n_rows

    flat_idx = np.argsort(-values.flatten())[:top_k]
    out = []
    for idx in flat_idx:
        r = idx // n_cols
        c = idx % n_cols
        out.append(
            HotspotCell(
                x_idx=int(c),
                y_idx=int(r),
                x_um=float((c + 0.5) * cell_w),
                y_um=float((r + 0.5) * cell_h),
                value=float(values[r, c]),
            ).to_dict()
        )
    return out


def _density_hotspots(plc, top_k: int) -> list[dict]:
    try:
        values = np.array(plc.get_grid_cells_density(), dtype=np.float64)
        return _extract_grid_hotspots(values, plc, top_k)
    except Exception:
        return []


def _congestion_hotspots(plc, top_k: int) -> list[dict]:
    try:
        h = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float64)
        v = np.array(plc.get_vertical_routing_congestion(), dtype=np.float64)
        n_rows, n_cols = _grid_shape(plc)
        if h.ndim == 1:
            h = h.reshape(n_rows, n_cols)
        if v.ndim == 1:
            v = v.reshape(n_rows, n_cols)
        return _extract_grid_hotspots(np.maximum(h, v), plc, top_k)
    except Exception:
        return []


def _wl_hotspot_nets(bundle: BenchmarkBundle,
                     placement: torch.Tensor,
                     top_k: int) -> list[dict]:
    """Top-K macro pairs by weighted HPWL contribution."""
    try:
        adj = np.array(bundle.plc.get_macro_adjacency(), dtype=np.float64)
    except Exception:
        return []
    n = bundle.num_hard + bundle.num_soft
    A = adj.reshape(n, n)
    rows, cols = np.where(np.triu(A, k=1) > 0)
    if len(rows) == 0:
        return []
    pos = placement.numpy()
    w = A[rows, cols]
    dx = np.abs(pos[rows, 0] - pos[cols, 0])
    dy = np.abs(pos[rows, 1] - pos[cols, 1])
    hpwl = w * (dx + dy)
    order = np.argsort(-hpwl)[:top_k]
    return [
        {
            "macro_i": int(rows[i]),
            "macro_j": int(cols[i]),
            "weight": float(w[i]),
            "hpwl": float(hpwl[i]),
        }
        for i in order
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def diagnose_placement(
    placement: torch.Tensor,
    bundle: BenchmarkBundle,
    regime: str = "IBM",
    critical_top_k: int = 50,
    hotspot_top_k: int = 10,
) -> PlacementReport:
    """
    Full diagnostic on a placement.

    Args:
        placement: [num_hard + num_soft, 2] tensor of macro centers.
        bundle:    loaded benchmark.
        regime:    "IBM" or "NG45" — drives the 12 μm spacing check.

    Returns:
        PlacementReport.
    """
    t0 = time.time()

    cost = compute_proxy_cost(placement, bundle, write_back=True)
    n_overlap, overlap_area = check_overlaps(placement, bundle)
    min_spacing = min_macro_spacing(placement, bundle)

    critical_edges = identify_critical_edges(bundle, top_k=critical_top_k)
    cp_total, cp_max = critical_path_hpwl(placement, critical_edges)

    canvas_area = bundle.canvas_w * bundle.canvas_h
    hard_area = float((bundle.hard_sizes[:, 0] * bundle.hard_sizes[:, 1]).sum())
    soft_area = float((bundle.soft_sizes[:, 0] * bundle.soft_sizes[:, 1]).sum())

    return PlacementReport(
        benchmark_name=bundle.name,
        regime=regime,
        timestamp=t0,
        proxy_cost_total=cost["proxy_cost"],
        wirelength=cost["wirelength"],
        density=cost["density"],
        congestion=cost["congestion"],
        num_overlaps=int(n_overlap),
        overlap_total_area=float(overlap_area),
        min_macro_spacing_um=float(min_spacing),
        spacing_compliant_12um=(
            min_spacing >= NG45_MIN_SPACING_UM if regime == "NG45" else True
        ),
        canvas_width=bundle.canvas_w,
        canvas_height=bundle.canvas_h,
        num_hard_macros=bundle.num_hard,
        num_soft_macros=bundle.num_soft,
        hard_utilization=hard_area / canvas_area,
        soft_utilization=soft_area / canvas_area,
        critical_path_hpwl_total=cp_total,
        critical_path_count=len(critical_edges),
        top_path_hpwl=cp_max,
        wl_hotspot_nets=_wl_hotspot_nets(bundle, placement, hotspot_top_k),
        density_hotspots=_density_hotspots(bundle.plc, hotspot_top_k),
        congestion_hotspots=_congestion_hotspots(bundle.plc, hotspot_top_k),
        eval_time_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: PlacementReport) -> None:
    """Human-readable terminal output."""
    print(f"\n{'=' * 70}")
    print(f"  Placement Diagnostic — {report.benchmark_name} ({report.regime})")
    print(f"{'=' * 70}")

    print(f"\n  PROXY COST:    {report.proxy_cost_total:.4f}")
    print(f"    wirelength:  {report.wirelength:.4f}")
    print(f"    density:     {report.density:.4f}")
    print(f"    congestion:  {report.congestion:.4f}")

    print(f"\n  CONSTRAINTS")
    overlap_marker = "✓" if report.num_overlaps == 0 else "✗ FAIL"
    print(f"    overlaps:           {report.num_overlaps}  {overlap_marker}")
    if report.num_overlaps > 0:
        print(f"    overlap area:       {report.overlap_total_area:.4f}")
    print(f"    min spacing:        {report.min_macro_spacing_um:.3f} μm")
    if report.regime == "NG45":
        marker = "✓" if report.spacing_compliant_12um else "✗ FAIL"
        print(f"    12μm spacing:       {marker}")

    print(f"\n  STRUCTURAL")
    print(f"    canvas:             {report.canvas_width:.1f} × "
          f"{report.canvas_height:.1f} μm")
    print(f"    hard macros:        {report.num_hard_macros}")
    print(f"    soft macros:        {report.num_soft_macros}")
    print(f"    hard utilization:   {100 * report.hard_utilization:.1f}%")
    print(f"    soft utilization:   {100 * report.soft_utilization:.1f}%")

    print(f"\n  CRITICAL PATHS  (top {report.critical_path_count})")
    print(f"    sum HPWL:           {report.critical_path_hpwl_total:.2f}")
    print(f"    worst single:       {report.top_path_hpwl:.3f}")

    print(f"\n  WL HOTSPOTS  (top {len(report.wl_hotspot_nets)} nets)")
    for h in report.wl_hotspot_nets[:5]:
        print(f"    macros {h['macro_i']:4d}↔{h['macro_j']:4d}  "
              f"w={h['weight']:5.1f}  hpwl={h['hpwl']:.3f}")

    print(f"\n  DENSITY HOTSPOTS  (top {len(report.density_hotspots)} cells)")
    for h in report.density_hotspots[:5]:
        print(f"    cell ({h['x_idx']:2d},{h['y_idx']:2d})  "
              f"({h['x_um']:5.1f},{h['y_um']:5.1f}) μm  "
              f"density={h['value']:.3f}")

    print(f"\n  CONGESTION HOTSPOTS  (top {len(report.congestion_hotspots)} cells)")
    for h in report.congestion_hotspots[:5]:
        print(f"    cell ({h['x_idx']:2d},{h['y_idx']:2d})  "
              f"({h['x_um']:5.1f},{h['y_um']:5.1f}) μm  "
              f"max(H,V)={h['value']:.3f}")

    print(f"\n  eval time:            {report.eval_time_s:.2f}s")
    print()


def save_report(
    report: PlacementReport,
    tag: str,
    outdir: str | Path = "reports",
) -> Path:
    """Serialize a report to JSON."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"saved: {json_path}")
    return json_path
