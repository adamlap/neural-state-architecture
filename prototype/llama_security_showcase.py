"""Compatibility shim — script moved to prototype/retrofit/llama_security_showcase.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "retrofit" / "llama_security_showcase.py"
    runpy.run_path(str(target), run_name="__main__")
