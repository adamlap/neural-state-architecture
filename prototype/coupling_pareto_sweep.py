"""Compatibility shim — script moved to prototype/experiments/coupling_pareto_sweep.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "experiments" / "coupling_pareto_sweep.py"
    runpy.run_path(str(target), run_name="__main__")
