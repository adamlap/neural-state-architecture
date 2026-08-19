"""Aggregate JSON experiment outputs from a directory."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .results import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", default="aggregate.json")
    args = parser.parse_args()
    results = []
    for path in sorted(glob.glob(str(Path(args.directory) / "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    aggregate = summarize(results)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
