from pathlib import Path

from benchmarks.run_safety_benchmark import run


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    # Use the JSON policy in the deterministic unit tests so the benchmark
    # remains backend/dependency independent. YAML loading is covered by the
    # CLI/integration path when PyYAML is installed.
    return root / "policies" / "strict.json", root / "benchmarks" / "safety_corpus.jsonl"


def test_safety_benchmark_runs_against_reference_policy():
    policy, corpus = _paths()
    result = run(policy, corpus)
    assert result["cases"] == 7
    assert result["scored_cases"] == 6
    assert result["correct"] == 6


def test_benchmark_does_not_execute_model_code():
    # The deterministic benchmark is intentionally independent of an inference
    # backend so CI can validate the scoring contract without an Ollama service.
    policy, corpus = _paths()
    result = run(policy, corpus)
    assert all("actual" in row for row in result["results"])
