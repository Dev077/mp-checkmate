"""
Three-panel placement visualization: placement, density, congestion.

Same layout as the visualizations the challenge starter repo produces, so we
can eyeball-compare our placements against theirs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from macroplace.diagnostic.report import PlacementReport
from macroplace.io.loader import BenchmarkBundle


def visualize_placement(
    placement: torch.Tensor,
    bundle: BenchmarkBundle,
    report: Optional[PlacementReport] = None,
    outpath: str | Path = "placement.png",
) -> Path:
    """Render the three-panel diagnostic plot."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    W = bundle.canvas_w
    H = bundle.canvas_h
    pos = placement.numpy()
    hard_sizes = bundle.hard_sizes.numpy()
    soft_sizes = bundle.soft_sizes.numpy()

    # --- Placement panel ---
    ax = axes[0]
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    title = f"{bundle.name} — Placement"
    if report is not None:
        title += f"  (proxy={report.proxy_cost_total:.3f})"
    ax.set_title(title)
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")

    # Soft macros (background)
    for i in range(bundle.num_soft):
        x, y = pos[bundle.num_hard + i]
        w, h = soft_sizes[i]
        ax.add_patch(patches.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor="#e8e8f0", edgecolor="#888899",
            linewidth=0.2, alpha=0.5,
        ))
    # Hard macros (foreground)
    for i in range(bundle.num_hard):
        x, y = pos[i]
        w, h = hard_sizes[i]
        ax.add_patch(patches.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor="#4060c0", edgecolor="#202060",
            linewidth=0.3, alpha=0.8,
        ))

    # --- Density panel ---
    ax = axes[1]
    ax.set_title("Density")
    ax.set_aspect("equal")
    try:
        n_rows = bundle.plc.grid_row
        n_cols = bundle.plc.grid_col
        density = np.array(bundle.plc.get_grid_cells_density(), dtype=np.float64)
        if density.ndim == 1:
            density = density.reshape(n_rows, n_cols)
        im = ax.imshow(
            density,
            origin="lower",
            extent=(0, W, 0, H),
            cmap="Blues",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, fraction=0.046)
    except Exception as e:
        ax.text(0.5, 0.5, f"density unavailable\n{e}",
                transform=ax.transAxes, ha="center", va="center")

    # --- Congestion panel ---
    ax = axes[2]
    ax.set_title("Congestion (max H,V)")
    ax.set_aspect("equal")
    try:
        h_cong = np.array(bundle.plc.get_horizontal_routing_congestion(),
                          dtype=np.float64)
        v_cong = np.array(bundle.plc.get_vertical_routing_congestion(),
                          dtype=np.float64)
        if h_cong.ndim == 1:
            h_cong = h_cong.reshape(n_rows, n_cols)
            v_cong = v_cong.reshape(n_rows, n_cols)
        cong = np.maximum(h_cong, v_cong)
        im = ax.imshow(
            cong,
            origin="lower",
            extent=(0, W, 0, H),
            cmap="hot",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, fraction=0.046)
    except Exception as e:
        ax.text(0.5, 0.5, f"congestion unavailable\n{e}",
                transform=ax.transAxes, ha="center", va="center")

    plt.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {outpath}")
    return outpath
