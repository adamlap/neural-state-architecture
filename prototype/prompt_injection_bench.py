"""Compatibility shim — script moved to prototype/pillars/prompt_injection_bench.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "pillars" / "prompt_injection_bench.py"
    runpy.run_path(str(target), run_name="__main__")
