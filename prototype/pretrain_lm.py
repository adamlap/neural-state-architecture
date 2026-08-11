"""Compatibility shim — script moved to prototype/pillars/pretrain_lm.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "pillars" / "pretrain_lm.py"
    runpy.run_path(str(target), run_name="__main__")
