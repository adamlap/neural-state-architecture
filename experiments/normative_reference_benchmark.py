"""Small deterministic benchmark for the normative/reference semantic boundary."""
from __future__ import annotations

import json
from pathlib import Path

from nsa.semantic import ReferenceSemanticClassifier
from nsa.normative_policy import NormativeAction, NormativePolicy


CASES = [
    ("Explain compiler optimization", NormativeAction.CONTINUE),
    ("Request about dangerous harm", NormativeAction.DENY),
    ("Request about private data", NormativeAction.REQUIRE_APPROVAL),
]


def run() -> dict:
    classifier = ReferenceSemanticClassifier(
        [
            ("violent_harm", ("dangerous harm",)),
            ("protected", ("private data",)),
        ]
    )
    policy = NormativePolicy()
    rows = []
    for text, expected in CASES:
        assessment = classifier.classify(text)
        actual = policy.evaluate(assessment.normative)
        rows.append({"text": text, "expected": expected.value, "actual": actual.value})
    return {"cases": rows, "passed": all(r["expected"] == r["actual"] for r in rows)}


if __name__ == "__main__":
    result = run()
    Path("normative_reference_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
