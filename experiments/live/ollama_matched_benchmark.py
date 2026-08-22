"""Matched baseline-vs-NSA benchmark for a real Ollama model.

The same model, prompts, token budget and sampling settings are used for both
conditions. The NSA condition adds only the canonical typed-state runtime
boundary. Results are emitted as JSON so CI/local runs can archive them.

The benchmark reports both strict normalized accuracy and a punctuation-tolerant
accuracy. This avoids treating harmless formatting such as ``Yes.`` vs ``yes``
as a cognitive failure while retaining the strict metric for regression tests.
It still does not claim that textual outputs prove metacognition or consciousness.

IMPORTANT: this is a real-model benchmark only. It always constructs the
Ollama backend in ``mode=ollama`` and therefore fails closed if Ollama/model
access is unavailable; it never falls back to the mock backend.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path
from typing import Dict, List

import torch

from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.typed_runtime import NSATypedRuntime


CASES = [
    {"id": "arithmetic", "prompt": "Compute exactly: 37 * 19. Reply with only the integer.", "answer": "703"},
    {"id": "sequence", "prompt": "What is the next number? 2, 4, 8, 16, ?. Reply with only the integer.", "answer": "32"},
    {"id": "constraint", "prompt": "Reply with exactly three words: neural state safety.", "answer": "neural state safety"},
    {"id": "logic", "prompt": "If all A are B and all B are C, are all A necessarily C? Reply yes or no.", "answer": "yes"},
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_loose(text: str) -> str:
    """Normalize case/whitespace and harmless surrounding punctuation."""
    return normalize(text).translate(str.maketrans("", "", string.punctuation))


def score(text: str, expected: str) -> bool:
    return normalize(text) == normalize(expected)


def score_loose(text: str, expected: str) -> bool:
    return normalize_loose(text) == normalize_loose(expected)


def _authority_is_unchanged(runtime: NSATypedRuntime) -> bool:
    return torch.equal(
        runtime.activation.state.authority_state,
        torch.zeros_like(runtime.activation.state.authority_state),
    )


def run(model: str, max_tokens: int, temperature: float) -> Dict[str, object]:
    baseline_backend = OllamaInferenceBackend(model_name=model, mode="ollama")
    nsa_backend = OllamaInferenceBackend(model_name=baseline_backend.model_name, mode="ollama")
    nsa_runtime = NSATypedRuntime(nsa_backend, goal_id="matched-benchmark")

    if baseline_backend.model_name != nsa_backend.model_name:
        raise RuntimeError("Matched benchmark requires identical resolved Ollama model names")

    rows: List[Dict[str, object]] = []
    for case in CASES:
        started = time.perf_counter()
        baseline = baseline_backend.generate(
            case["prompt"], max_tokens=max_tokens, temperature=temperature
        )
        baseline_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        wrapped = nsa_runtime.generate(
            case["prompt"], max_tokens=max_tokens, temperature=temperature
        )
        nsa_ms = (time.perf_counter() - started) * 1000.0

        rows.append(
            {
                "id": case["id"],
                "expected": case["answer"],
                "baseline": {
                    "text": baseline.text,
                    "correct": score(baseline.text, case["answer"]),
                    "normalized_correct": score_loose(baseline.text, case["answer"]),
                    "latency_ms": baseline_ms,
                },
                "nsa": {
                    "text": wrapped.output.text,
                    "correct": score(wrapped.output.text, case["answer"]),
                    "normalized_correct": score_loose(wrapped.output.text, case["answer"]),
                    "latency_ms": nsa_ms,
                    "state_step": wrapped.state.state.temporal_state.step_index,
                    "provenance": wrapped.state.state.provenance_state.record_id,
                    "hard_authority_unchanged": _authority_is_unchanged(nsa_runtime),
                },
            }
        )

    baseline_accuracy = sum(bool(r["baseline"]["correct"]) for r in rows) / len(rows)
    nsa_accuracy = sum(bool(r["nsa"]["correct"]) for r in rows) / len(rows)
    baseline_normalized_accuracy = sum(bool(r["baseline"]["normalized_correct"]) for r in rows) / len(rows)
    nsa_normalized_accuracy = sum(bool(r["nsa"]["normalized_correct"]) for r in rows) / len(rows)
    return {
        "model": nsa_backend.model_name,
        "backend_mode": "ollama",
        "cases": len(rows),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "baseline_accuracy": baseline_accuracy,
        "nsa_accuracy": nsa_accuracy,
        "accuracy_delta": nsa_accuracy - baseline_accuracy,
        "baseline_normalized_accuracy": baseline_normalized_accuracy,
        "nsa_normalized_accuracy": nsa_normalized_accuracy,
        "normalized_accuracy_delta": nsa_normalized_accuracy - baseline_normalized_accuracy,
        "all_hard_authority_unchanged": all(bool(r["nsa"]["hard_authority_unchanged"]) for r in rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.model, args.max_tokens, args.temperature)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
