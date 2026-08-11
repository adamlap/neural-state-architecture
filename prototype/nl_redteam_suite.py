"""Compatibility shim — script moved to prototype/security/nl_redteam_suite.py"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "security" / "nl_redteam_suite.py"
    runpy.run_path(str(target), run_name="__main__")
