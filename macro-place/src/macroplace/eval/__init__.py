"""Evaluation utilities — proxy cost, overlap checks, critical-path analysis."""

from macroplace.eval.proxy_cost import compute_proxy_cost
from macroplace.eval.overlap import check_overlaps, min_macro_spacing

__all__ = ["compute_proxy_cost", "check_overlaps", "min_macro_spacing"]
