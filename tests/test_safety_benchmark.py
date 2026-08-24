from pathlib import Path

from benchmarks.run_safety_benchmark import run


def test_safety_benchmark_runs_against_reference_policy():
    root = Path(__file__).resolve().parents[1]
    result = run(root / "policies" / "strict.yaml", root / "benchmarks" / "safety_corpus.jsonl")
    assert result["cases"] == 7
    assert result["scored_cases"] == 6
    assert result["correct"] == 6


def test_benchmark_does_not_execute_model_code():
    # The deterministic benchmark is intentionally independent of an inference
    # backend so CI can validate the scoring contract without an Ollama service.
    root = Path(__file__).resolve().parents[1]
    result = run(root / "policies" / "strict.yaml", root / "benchmarks" / "safety_corpus.jsonl")
    assert all("actual" in row for row in result["results"])
