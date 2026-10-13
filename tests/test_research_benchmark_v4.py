from experiments.research_v4.temporal_semantics_benchmark import (
    assess_memory_retrieval,
    build_temporal_memory,
    make_episode,
    render_temporal,
)


def test_current_render_exposes_canonical_semantics():
    observations, _, _, key, _ = make_episode(7, "supersession_current", 1000)
    store = build_temporal_memory(observations)
    rendered = render_temporal(store, [key], history=False, limit=3)
    assert "STATUS=CURRENT" in rendered
    assert "VERSION=2" in rendered
    assert "CURRENT" in rendered
    assert "V007_000" not in rendered


def test_history_render_exposes_superseded_semantics():
    observations, _, expected, key, _ = make_episode(7, "supersession_history", 1000)
    store = build_temporal_memory(observations)
    rendered = render_temporal(store, [key], history=True, limit=3)
    assert f"{expected}" in rendered
    assert "STATUS=SUPERSEDED" in rendered
    assert "CURRENT_VALUE=V007_CURRENT" in rendered


def test_retrieval_correctness_is_separate_from_model_answer():
    observations, _, expected, key, _ = make_episode(17, "supersession_history", 1000)
    correct, stale = assess_memory_retrieval(
        "temporal_memory",
        observations,
        f"What was the previous value of {key}?",
        expected,
        "supersession_history",
        3,
    )
    assert correct is True
    assert stale is False


def test_current_retrieval_rejects_superseded_value():
    observations, _, expected, key, _ = make_episode(17, "supersession_current", 1000)
    correct, stale = assess_memory_retrieval(
        "temporal_memory",
        observations,
        f"What is the current value of {key}?",
        expected,
        "supersession_current",
        3,
    )
    assert correct is True
    assert stale is False
