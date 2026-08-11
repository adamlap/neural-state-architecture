"""Compatibility shim — script moved to prototype/demos/web_demo.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "demos" / "web_demo.py"
    runpy.run_path(str(target), run_name="__main__")
