"""
Overlap and spacing checks.

Hard-hard macro overlap is a hard fail (submission is rejected).  Spacing
matters specifically for NG45-regime benchmarks where the ORFS auto-push
enforces ≥12 μm clearance.  See SCORING.md.
"""
from __future__ import annotations

import torch

from macroplace.io.loader import BenchmarkBundle


# TILOS uses 0.004 as the overlap tolerance (per the .plc header banner).
DEFAULT_OVERLAP_TOLERANCE = 0.004

# ORFS power-delivery channel routing needs ~10 μm; the evaluator pushes
# macros to ≥12 μm in Tier 2.  Leaving 12 μm in our submitted placement keeps
# control over Tier 2 positions.
NG45_MIN_SPACING_UM = 12.0


def check_overlaps(
    placement: torch.Tensor,
    bundle: BenchmarkBundle,
    tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
) -> tuple[int, float]:
    """
    Count hard-macro pairs whose overlap exceeds the tolerance.

    Returns:
        (num_overlap_pairs, total_overlap_area_um2)

    Only hard-hard overlaps are flagged.  Soft macros are abstractions over
    standard-cell clusters and may overlap with anything; only hard-hard
    overlaps disqualify a submission.
    """
    hard_positions = placement[: bundle.num_hard].numpy()
    hard_sizes = bundle.hard_sizes.numpy()
    n = hard_positions.shape[0]
    if n < 2:
        return 0, 0.0

    hw = hard_sizes[:, 0] / 2
    hh = hard_sizes[:, 1] / 2

    num_overlap = 0
    total_area = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(hard_positions[i, 0] - hard_positions[j, 0])
            dy = abs(hard_positions[i, 1] - hard_positions[j, 1])
            ox = (hw[i] + hw[j]) - dx
            oy = (hh[i] + hh[j]) - dy
            if ox > tolerance and oy > tolerance:
                num_overlap += 1
                total_area += float(ox * oy)
    return num_overlap, total_area


def min_macro_spacing(
    placement: torch.Tensor,
    bundle: BenchmarkBundle,
) -> float:
    """
    Smallest edge-to-edge distance between any pair of hard macros in μm.

    Negative values indicate overlap (the magnitude is the overlap depth).
    Returns inf if there are fewer than 2 hard macros.
    """
    hard_positions = placement[: bundle.num_hard].numpy()
    hard_sizes = bundle.hard_sizes.numpy()
    n = hard_positions.shape[0]
    if n < 2:
        return float("inf")

    hw = hard_sizes[:, 0] / 2
    hh = hard_sizes[:, 1] / 2

    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(hard_positions[i, 0] - hard_positions[j, 0]) - (hw[i] + hw[j])
            dy = abs(hard_positions[i, 1] - hard_positions[j, 1]) - (hh[i] + hh[j])
            if dx < 0 and dy < 0:
                spacing = min(dx, dy)  # overlap, negative
            elif dx < 0:
                spacing = dy
            elif dy < 0:
                spacing = dx
            else:
                spacing = (dx * dx + dy * dy) ** 0.5
            if spacing < best:
                best = float(spacing)
    return best
