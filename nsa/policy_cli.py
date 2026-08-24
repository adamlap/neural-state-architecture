"""Practical CLI for inspecting and validating NSA safety policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nsa.policy import NSAPolicy, PolicyCompiler


def load_policy(path: Path) -> NSAPolicy:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return NSAPolicy.from_yaml(path)
    if suffix == ".json":
        return NSAPolicy.from_json(path)
    raise ValueError("policy must be .yaml, .yml, or .json")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSA declarative safety policy tool")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "inspect"):
        p = sub.add_parser(command)
        p.add_argument("policy", type=Path)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    engine = PolicyCompiler.compile(policy)
    if args.command == "validate":
        print(json.dumps({"valid": True, "name": policy.name, "rules": len(policy.prohibited), "compiled": type(engine).__name__}, indent=2))
    else:
        print(json.dumps(policy.to_mapping(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
