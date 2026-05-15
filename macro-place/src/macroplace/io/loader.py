"""
Benchmark loader.

Wraps the TILOS PlacementCost evaluator so the rest of the codebase doesn't
have to know where it lives.  We support two installation modes:

1.  TILOS source vendored at  benchmarks/external/MacroPlacement/...
    (this is what the challenge starter repo does — we recommend it).

2.  TILOS package importable via `circuit_training.environment.placement_util`.

Both end up giving us the same `PlacementCost` object plus a small Benchmark
dataclass we control.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

def _parse_routes_per_micron(plc_file: Path) -> tuple[float, float]:
    """Pull H/V routes-per-micron out of a .plc file header."""
    h, v = 0.0, 0.0
    with open(plc_file) as f:
        for line in f:
            if "ROUTES PER MICRON" in line:
                # Format: "#[ROUTES PER MICRON] Hor: 65.96, Ver: 106.96"
                import re
                m = re.search(r"Hor:\s*([\d.]+),\s*Ver:\s*([\d.]+)", line)
                if m:
                    h, v = float(m.group(1)), float(m.group(2))
                    break
            if not line.startswith("#"):
                break  # past the header
    return h, v

# ---------------------------------------------------------------------------
# Locate TILOS sources
# ---------------------------------------------------------------------------

def _ensure_tilos_on_path() -> Optional[Path]:
    """
    Try to make the TILOS MacroPlacement Python modules importable.

    Looks for them in (in order):
        $TILOS_MACRO_PLACEMENT_ROOT
        ./benchmarks/external/MacroPlacement
        ../MacroPlacement
        (already importable — nothing to do)

    Returns the path that was added (or None if already importable / nothing
    was found).
    """
    # If already importable, nothing to do
    try:
        import circuit_training  # noqa: F401
        return None
    except ImportError:
        pass

    candidates = []
    env = os.environ.get("TILOS_MACRO_PLACEMENT_ROOT")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("benchmarks/external/MacroPlacement"),
        Path("../MacroPlacement"),
        Path("../../MacroPlacement"),
    ]

    for root in candidates:
        root = root.resolve()
        if not root.exists():
            continue
        # MacroPlacement's Python code lives under  CodeElements/Plc_client/
        # for the proto/eval bindings and under various other subdirs for
        # circuit_training.  We add both candidate paths defensively.
        added = []
        for sub in [
            "CodeElements/Plc_client",
            "CodeElements",
            "",  # the repo root itself, just in case
        ]:
            p = root / sub if sub else root
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
                added.append(str(p))
        if added:
            return root
    return None


# ---------------------------------------------------------------------------
# Benchmark bundle — what the rest of the codebase consumes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkBundle:
    """
    Everything downstream stages need to know about a benchmark.

    Attributes:
        name:          short benchmark identifier (e.g., "ibm01")
        plc:           TILOS PlacementCost evaluator (opaque object)
        canvas_w:      canvas width in μm
        canvas_h:      canvas height in μm
        num_hard:      hard macro count
        num_soft:      soft macro count
        num_ports:     port (IO) count
        hard_indices:  indices in plc.modules_w_pins for hard macros
        soft_indices:  indices for soft macros
        port_indices:  indices for ports
        hard_sizes:    [num_hard, 2] tensor of (w, h) per hard macro in μm
        soft_sizes:    [num_soft, 2] tensor of (w, h) per soft macro
    """
    name: str
    plc: object
    canvas_w: float
    canvas_h: float
    num_hard: int
    num_soft: int
    num_ports: int
    hard_indices: list[int]
    soft_indices: list[int]
    port_indices: list[int]
    hard_sizes: torch.Tensor
    soft_sizes: torch.Tensor

    # ---- convenience methods downstream stages call ----

    def current_placement(self) -> torch.Tensor:
        """Return [num_hard + num_soft, 2] tensor of macro center positions."""
        positions = []
        for idx in self.hard_indices + self.soft_indices:
            mod = self.plc.modules_w_pins[idx]
            x, y = mod.get_pos()
            positions.append([float(x), float(y)])
        return torch.tensor(positions, dtype=torch.float32)

    def apply_placement(self, placement: torch.Tensor) -> None:
        all_indices = self.hard_indices + self.soft_indices
        assert placement.shape[0] == len(all_indices), (
            f"placement has {placement.shape[0]} rows but expected "
            f"{len(all_indices)}"
        )
        for k, idx in enumerate(all_indices):
            x = float(placement[k, 0])
            y = float(placement[k, 1])
            # TILOS uses module indices directly here.
            self.plc.update_node_coords(idx, x, y)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_benchmark(
    benchmark_dir: str | Path,
    name: Optional[str] = None,
) -> BenchmarkBundle:
    """
    Load a benchmark directory (containing a netlist.pb.txt + .plc).

    Args:
        benchmark_dir: path to the benchmark directory (e.g.,
            "benchmarks/external/MacroPlacement/Testcases/ICCAD04/ibm01")
        name: short name for reports (defaults to the directory's basename)

    Returns:
        BenchmarkBundle ready to use by downstream stages.
    """
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"benchmark not found: {benchmark_dir}")

    _ensure_tilos_on_path()

    # Import TILOS plc reader lazily so the rest of the package is importable
    # even before TILOS is wired up.
    try:
        import plc_client_os as plc_client  # the actual TILOS module name
    except ImportError as e:
        raise ImportError(
            "Could not import TILOS PlacementCost bindings (plc_client_os). "
            "Make sure benchmarks/external/MacroPlacement/CodeElements/Plc_client "
            "is on sys.path, or set TILOS_MACRO_PLACEMENT_ROOT."
        ) from e

    netlist_pb = benchmark_dir / "netlist.pb.txt"
    plc_file = next(benchmark_dir.glob("*.plc"), None)
    if not netlist_pb.exists():
        raise FileNotFoundError(f"netlist.pb.txt missing in {benchmark_dir}")
    if plc_file is None:
        raise FileNotFoundError(f"no .plc file in {benchmark_dir}")

    plc = plc_client.PlacementCost(
        netlist_file=str(netlist_pb),
        macro_macro_x_spacing=0.0,
        macro_macro_y_spacing=0.0,
    )
    plc.set_canvas_boundary_check(True)
    plc.restore_placement(str(plc_file))

    # TILOS doesn't always pick up routes_per_micron from the .plc header on
    # restore. Read it directly and set explicitly so congestion is defined.
    h_routes, v_routes = _parse_routes_per_micron(plc_file)
    if h_routes and v_routes:
        plc.set_routes_per_micron(h_routes, v_routes)

    # Catalogue modules
    hard_indices = list(plc.hard_macro_indices)
    soft_indices = list(plc.soft_macro_indices)
    port_indices = list(plc.port_indices)
    canvas_w, canvas_h = plc.get_canvas_width_height()

    def _sizes_for(indices):
        rows = []
        for i in indices:
            mod = plc.modules_w_pins[i]
            rows.append([float(mod.get_width()), float(mod.get_height())])
        return torch.tensor(rows, dtype=torch.float32) if rows \
            else torch.zeros((0, 2), dtype=torch.float32)

    return BenchmarkBundle(
        name=name or benchmark_dir.name,
        plc=plc,
        canvas_w=float(canvas_w),
        canvas_h=float(canvas_h),
        num_hard=len(hard_indices),
        num_soft=len(soft_indices),
        num_ports=len(port_indices),
        hard_indices=hard_indices,
        soft_indices=soft_indices,
        port_indices=port_indices,
        hard_sizes=_sizes_for(hard_indices),
        soft_sizes=_sizes_for(soft_indices),
    )
