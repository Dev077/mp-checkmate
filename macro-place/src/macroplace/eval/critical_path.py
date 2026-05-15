"""
Critical-net identification for the Stage 6 refinement regularizer.

For IBM-regime benchmarks (derived from synthetic netlists) we use a
structural proxy: top-K macro pairs by  weight / min(endpoint_degree).
High-weight + low-degree = short dense connections that look like
combinational-path bottlenecks.

For NG45-regime benchmarks (real designs with meaningful logic depth) we
should ideally do a BFS through the netlist DAG to find longest weighted
paths.  That's a TODO — the structural proxy is the v1 we ship for both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from macroplace.io.loader import BenchmarkBundle


@dataclass
class CriticalEdge:
    """A pairwise 'critical' connection (in v1, an adjacency edge)."""
    macro_i: int
    macro_j: int
    weight: float
    score: float


def identify_critical_edges(
    bundle: BenchmarkBundle,
    top_k: int = 50,
) -> list[CriticalEdge]:
    """
    Return the top-K macro-pair edges most likely to be on critical paths.

    Args:
        bundle:  loaded benchmark.
        top_k:   how many edges to return.

    Returns:
        list of CriticalEdge, sorted by descending criticality score.
    """
    plc = bundle.plc

    # plc.get_macro_adjacency returns a flat length-N*N array where
    # N = num_hard + num_soft.
    adj_flat = plc.get_macro_adjacency()
    n_macros = bundle.num_hard + bundle.num_soft
    A = np.array(adj_flat, dtype=np.float64).reshape(n_macros, n_macros)

    degree = (A > 0).sum(axis=1)
    rows, cols = np.where(np.triu(A, k=1) > 0)
    if len(rows) == 0:
        return []

    weights = A[rows, cols]
    pair_min_deg = np.maximum(np.minimum(degree[rows], degree[cols]), 1)
    scores = weights / pair_min_deg

    order = np.argsort(-scores)[:top_k]
    return [
        CriticalEdge(
            macro_i=int(rows[i]),
            macro_j=int(cols[i]),
            weight=float(weights[i]),
            score=float(scores[i]),
        )
        for i in order
    ]


def critical_path_hpwl(
    placement,
    edges: list[CriticalEdge],
) -> tuple[float, float]:
    """
    Sum HPWL across the critical edges.

    Returns:
        (total_hpwl, max_single_hpwl)
    """
    if not edges:
        return 0.0, 0.0

    import torch
    if isinstance(placement, torch.Tensor):
        pos = placement.numpy()
    else:
        pos = np.asarray(placement)

    hpwls = []
    for e in edges:
        dx = abs(pos[e.macro_i, 0] - pos[e.macro_j, 0])
        dy = abs(pos[e.macro_i, 1] - pos[e.macro_j, 1])
        hpwls.append(float(dx + dy))
    return float(sum(hpwls)), float(max(hpwls))
