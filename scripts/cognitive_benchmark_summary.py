"""Print a one-line PASS/RESEARCH_GATE_NOT_YET_MET summary for each cognitive
benchmark result written to results/*.json.

Used by `make cognitive-benchmarks`; not a benchmark itself.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULT_FILES = [
    "results/cognitive_architecture_benchmark.json",
    "results/dynamic_cognition_benchmark.json",
    "results/state_compression_benchmark.json",
    "results/sufficient_state_dynamics_benchmark.json",
    "results/governance_temptation_benchmark.json",
]


def main() -> None:
    width = max(len(Path(f).stem) for f in RESULT_FILES)
    for path_str in RESULT_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"  {path.stem:<{width}}  MISSING ({path})")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        status = report.get("status", "UNKNOWN")
        failed = [name for name, ok in report.get("gates", {}).items() if not ok]
        detail = "" if not failed else f"  failed: {', '.join(failed)}"
        print(f"  {path.stem:<{width}}  {status}{detail}")


if __name__ == "__main__":
    main()
