"""Package-level entry points for CLI scripts."""

import runpy
from pathlib import Path


def _run_script(name: str):
    """Run scripts/<name>.py with package context."""
    script = Path(__file__).resolve().parents[2] / "scripts" / name
    if not script.exists():
        # Fallback: maybe we're installed and scripts aren't shipped — point user.
        raise FileNotFoundError(
            f"Could not find {script}.  Run from the repo root, or use "
            f"`python -m macroplace.{name.replace('.py', '')}` if available."
        )
    runpy.run_path(str(script), run_name="__main__")


def diagnose_main():
    _run_script("diagnose.py")


def pipeline_main():
    _run_script("run_pipeline.py")
