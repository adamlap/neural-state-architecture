"""Run the NSA safety benchmark without requiring a live model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nsa import EvaluationContext, KeywordClassifier, NSAPolicy, PolicyEngine


def load_corpus(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_engine(policy_path: Path) -> PolicyEngine:
    if policy_path.suffix.lower() in {".yaml", ".yml"}:
        policy = NSAPolicy.from_yaml(policy_path)
        import yaml
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    else:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
        policy = NSAPolicy.from_mapping(raw)
    patterns: dict[str, list[str]] = {}
    for rule in raw.get("prohibited", []) or []:
        if isinstance(rule, dict) and rule.get("category"):
            patterns[str(rule["category"])] = [str(x) for x in (rule.get("patterns") or [])]
    return PolicyEngine(policy, KeywordClassifier(patterns))


def run(policy_path: Path, corpus_path: Path) -> dict[str, Any]:
    engine = build_engine(policy_path)
    rows = []
    correct = 0
    scored_cases = 0
    for item in load_corpus(corpus_path):
        decision = engine.evaluate(item["prompt"], context=EvaluationContext(action="generate"))
        actual = "allow" if decision.allowed else "deny"
        expected = item["expected"]
        scored = expected in {"allow", "deny"}
        matched = actual == expected if scored else True
        if scored:
            scored_cases += 1
            correct += int(matched)
        rows.append({"id": item["id"], "expected": expected, "actual": actual,
                     "decision": decision.summary(), "correct": matched, "scored": scored})
    return {"policy": str(policy_path), "corpus": str(corpus_path),
            "cases": len(rows), "scored_cases": scored_cases,
            "correct": correct, "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--corpus", default=Path(__file__).with_name("safety_corpus.jsonl"), type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.policy, args.corpus)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
