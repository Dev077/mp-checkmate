"""
PlacementReport — structured output of the diagnostic harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class HotspotCell:
    """Single hotspot grid cell."""
    x_idx: int
    y_idx: int
    x_um: float
    y_um: float
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlacementReport:
    """One-page diagnostic on a placement."""

    # Identification
    benchmark_name: str
    regime: str  # "IBM" or "NG45"
    timestamp: float

    # Cost
    proxy_cost_total: float
    wirelength: float
    density: float
    congestion: float

    # Constraints
    num_overlaps: int
    overlap_total_area: float
    min_macro_spacing_um: float
    spacing_compliant_12um: bool  # meaningful only for NG45

    # Structural
    canvas_width: float
    canvas_height: float
    num_hard_macros: int
    num_soft_macros: int
    hard_utilization: float
    soft_utilization: float

    # Critical-path proxy
    critical_path_hpwl_total: float
    critical_path_count: int
    top_path_hpwl: float

    # Hotspots
    wl_hotspot_nets: list[dict] = field(default_factory=list)
    density_hotspots: list[dict] = field(default_factory=list)
    congestion_hotspots: list[dict] = field(default_factory=list)

    # Timing
    eval_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
