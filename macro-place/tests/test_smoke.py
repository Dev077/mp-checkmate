"""Smoke tests — make sure the package imports cleanly."""

def test_imports():
    """The whole package should import without TILOS being installed yet."""
    import macroplace
    from macroplace.diagnostic.report import PlacementReport, HotspotCell
    # These need TILOS to actually run, but should import fine.
    from macroplace.io import loader  # noqa: F401
    from macroplace.eval import overlap, critical_path, proxy_cost  # noqa: F401
    from macroplace.diagnostic import harness, visualize  # noqa: F401
    assert macroplace.__version__ == "0.1.0"


def test_report_serialization():
    """PlacementReport round-trips through to_dict()."""
    from macroplace.diagnostic.report import PlacementReport

    r = PlacementReport(
        benchmark_name="test",
        regime="IBM",
        timestamp=0.0,
        proxy_cost_total=1.5,
        wirelength=1.0,
        density=0.5,
        congestion=0.5,
        num_overlaps=0,
        overlap_total_area=0.0,
        min_macro_spacing_um=0.1,
        spacing_compliant_12um=True,
        canvas_width=22.95,
        canvas_height=23.04,
        num_hard_macros=246,
        num_soft_macros=894,
        hard_utilization=0.428,
        soft_utilization=0.372,
        critical_path_hpwl_total=0.0,
        critical_path_count=0,
        top_path_hpwl=0.0,
    )
    d = r.to_dict()
    assert d["benchmark_name"] == "test"
    assert d["proxy_cost_total"] == 1.5
