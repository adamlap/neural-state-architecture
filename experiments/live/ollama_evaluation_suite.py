"""Matched evaluation suite for a real Ollama model.

This module measures externally observable behavior of a baseline model versus
an NSA-wrapped instance using the same live Ollama backend. It intentionally
keeps the scientific claim narrow: runtime state invariants are structural
properties of the wrapper, while task scores are empirical behavioral results.

The suite is safe to import in CI because it does not contact Ollama until
``run`` is called. Live execution always uses ``mode=ollama`` and fails closed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.typed_runtime import NSATypedRuntime


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    prompt: str
    expected: str


CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase("arith-1", "reasoning", "Compute exactly: 37 * 19. Reply with only the integer.", "703"),
    EvaluationCase("arith-2", "reasoning", "Compute exactly: 144 / 12. Reply with only the integer.", "12"),
    EvaluationCase("logic-1", "reasoning", "If all A are B and all B are C, are all A necessarily C? Reply yes or no.", "yes"),
    EvaluationCase("logic-2", "reasoning", "If some A are B and all B are C, must some A be C? Reply yes or no.", "yes"),
    EvaluationCase("constraint-1", "constraint", "Reply with exactly three words: neural state safety.", "neural state safety"),
    EvaluationCase("error-1", "error_detection", "A calculator reports 19 * 4 = 86. Is that result correct? Reply yes or no.", "no"),
    EvaluationCase("error-2", "error_detection", "A report says every square has three sides. Is that claim correct? Reply yes or no.", "no"),
    EvaluationCase("planning-1", "planning", "You need to charge a phone before a two-hour trip. It is at 10 percent and the charger is available. What should you do first? Reply: charge phone.", "charge phone"),
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def score(text: str, expected: str) -> bool:
    return normalize(text) == normalize(expected)


def _authority_unchanged(runtime: NSATypedRuntime) -> bool:
    return torch.equal(runtime.activation.state.authority_state, torch.zeros_like(runtime.activation.state.authority_state))


def _category_accuracy(rows: Iterable[Dict[str, object]], condition: str) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = {}
    for row in rows:
        category = str(row["category"])
        grouped.setdefault(category, []).append(bool(row[condition]["correct"]))
    return {category: sum(values) / len(values) for category, values in sorted(grouped.items())}


def run(model: str, repetitions: int = 1, max_tokens: int = 64, temperature: float = 0.0) -> Dict[str, object]:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    baseline_backend = OllamaInferenceBackend(model_name=model, mode="ollama")
    nsa_backend = OllamaInferenceBackend(model_name=baseline_backend.model_name, mode="ollama")
    if baseline_backend.model_name != nsa_backend.model_name:
        raise RuntimeError("Baseline and NSA must resolve to the same Ollama model")

    runtime = NSATypedRuntime(nsa_backend, goal_id="phase18-evaluation")
    rows: List[Dict[str, object]] = []

    for repetition in range(repetitions):
        for case in CASES:
            started = time.perf_counter()
            baseline = baseline_backend.generate(case.prompt, max_tokens=max_tokens, temperature=temperature)
            baseline_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            wrapped = runtime.generate(case.prompt, max_tokens=max_tokens, temperature=temperature)
            nsa_ms = (time.perf_counter() - started) * 1000.0

            rows.append({
                "repetition": repetition,
                "id": case.case_id,
                "category": case.category,
                "expected": case.expected,
                "baseline": {"text": baseline.text, "correct": score(baseline.text, case.expected), "latency_ms": baseline_ms},
                "nsa": {
                    "text": wrapped.output.text,
                    "correct": score(wrapped.output.text, case.expected),
                    "latency_ms": nsa_ms,
                    "state_step": wrapped.state.state.temporal_state.step_index,
                    "provenance": wrapped.state.state.provenance_state.record_id,
                    "hard_authority_unchanged": _authority_unchanged(runtime),
                },
            })

    total = len(rows)
    baseline_correct = sum(bool(row["baseline"]["correct"]) for row in rows)
    nsa_correct = sum(bool(row["nsa"]["correct"]) for row in rows)
    baseline_accuracy = baseline_correct / total
    nsa_accuracy = nsa_correct / total
    baseline_latency = sum(float(row["baseline"]["latency_ms"]) for row in rows) / total
    nsa_latency = sum(float(row["nsa"]["latency_ms"]) for row in rows) / total

    return {
        "model": nsa_backend.model_name,
        "backend_mode": "ollama",
        "repetitions": repetitions,
        "cases_per_repetition": len(CASES),
        "total_trials": total,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "baseline_accuracy": baseline_accuracy,
        "nsa_accuracy": nsa_accuracy,
        "accuracy_delta": nsa_accuracy - baseline_accuracy,
        "baseline_mean_latency_ms": baseline_latency,
        "nsa_mean_latency_ms": nsa_latency,
        "latency_delta_ms": nsa_latency - baseline_latency,
        "baseline_category_accuracy": _category_accuracy(rows, "baseline"),
        "nsa_category_accuracy": _category_accuracy(rows, "nsa"),
        "all_hard_authority_unchanged": all(bool(row["nsa"]["hard_authority_unchanged"]) for row in rows),
        "strictly_monotonic_state_steps": [row["nsa"]["state_step"] for row in rows] == list(range(1, total + 1)),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.model, repetitions=args.repetitions, max_tokens=args.max_tokens, temperature=args.temperature)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
