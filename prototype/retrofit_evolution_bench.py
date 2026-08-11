"""Compatibility shim — script moved to prototype/retrofit/retrofit_evolution_bench.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "retrofit" / "retrofit_evolution_bench.py"
    runpy.run_path(str(target), run_name="__main__")
