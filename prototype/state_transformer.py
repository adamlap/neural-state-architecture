"""Compatibility shim — script moved to prototype/experiments/state_transformer.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "experiments" / "state_transformer.py"
    runpy.run_path(str(target), run_name="__main__")
