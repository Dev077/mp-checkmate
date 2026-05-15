"""Placement diagnostic harness — the tool used to evaluate every placement."""

from macroplace.diagnostic.report import PlacementReport, HotspotCell
from macroplace.diagnostic.harness import diagnose_placement, print_report, save_report

__all__ = [
    "PlacementReport",
    "HotspotCell",
    "diagnose_placement",
    "print_report",
    "save_report",
]
