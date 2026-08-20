from experiments.live.ollama_evaluation_suite import CASES, _category_accuracy, normalize, score


def test_normalize_collapses_whitespace_and_case() -> None:
    assert normalize("  Neural\nState   Safety ") == "neural state safety"


def test_exact_score_uses_normalized_text() -> None:
    assert score("YES", "yes")
    assert not score("no", "yes")


def test_case_suite_covers_required_behavioral_categories() -> None:
    categories = {case.category for case in CASES}
    assert {"reasoning", "constraint", "error_detection", "planning"} <= categories


def test_category_accuracy_is_grouped_independently() -> None:
    rows = [
        {"category": "reasoning", "baseline": {"correct": True}, "nsa": {"correct": False}},
        {"category": "reasoning", "baseline": {"correct": False}, "nsa": {"correct": True}},
        {"category": "planning", "baseline": {"correct": True}, "nsa": {"correct": True}},
    ]
    assert _category_accuracy(rows, "baseline") == {"planning": 1.0, "reasoning": 0.5}
    assert _category_accuracy(rows, "nsa") == {"planning": 1.0, "reasoning": 0.5}
