"""
Proxy cost evaluation.

Thin wrapper over the TILOS PlacementCost `get_cost()` family so callers get
a stable dict-shaped result regardless of how the underlying API evolves.

The proxy cost is defined as:
    proxy = 1.0 * wirelength_cost
          + 0.5 * density_cost
          + 0.5 * congestion_cost

These are exactly the weights specified in the challenge SCORING rules and
the TILOS evaluator's own conventions.
"""
from __future__ import annotations

from typing import TypedDict

import torch

from macroplace.io.loader import BenchmarkBundle


WIRELENGTH_WEIGHT = 1.0
DENSITY_WEIGHT = 0.5
CONGESTION_WEIGHT = 0.5


class ProxyCost(TypedDict):
    proxy_cost: float
    wirelength: float
    density: float
    congestion: float
    horizontal_congestion: float
    vertical_congestion: float


def compute_proxy_cost(
    placement: torch.Tensor,
    bundle: BenchmarkBundle,
    write_back: bool = True,
) -> ProxyCost:
    """
    Compute proxy cost for a placement.

    Args:
        placement:  [num_hard + num_soft, 2] tensor of macro centers.
        bundle:     loaded benchmark.
        write_back: if True, push the placement into the plc evaluator first.
                    Set False if the placement is already applied (saves time
                    in tight inner loops).

    Returns:
        ProxyCost dict with the components.
    """
    if write_back:
        bundle.apply_placement(placement)

    plc = bundle.plc

    wl = float(plc.get_wirelength())
    dens = float(plc.get_density_cost())
    cong = float(plc.get_congestion_cost())

    # Some versions of the evaluator expose H/V congestion separately for
    # diagnostics; treat as optional.
    try:
        h_cong = float(plc.get_H_congestion_cost())
        v_cong = float(plc.get_V_congestion_cost())
    except AttributeError:
        h_cong = float("nan")
        v_cong = float("nan")

    proxy = (
        WIRELENGTH_WEIGHT * wl
        + DENSITY_WEIGHT * dens
        + CONGESTION_WEIGHT * cong
    )
    return ProxyCost(
        proxy_cost=proxy,
        wirelength=wl,
        density=dens,
        congestion=cong,
        horizontal_congestion=h_cong,
        vertical_congestion=v_cong,
    )
