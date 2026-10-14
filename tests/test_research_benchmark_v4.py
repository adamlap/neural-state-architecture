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
    assert "CURRENT_VALUE" not in rendered and "V007_CURRENT" not in rendered


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


# ---- 4.1: matched control, resume guard, end-to-end smoke ------------------------------
import json
from types import SimpleNamespace

import pytest

from experiments.research_v4 import temporal_semantics_benchmark as v4


class _FakeInner:
    """Answers with the first V-token found in a CURRENT line, else a fixed wrong value."""

    model = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="VALUE=V000_000", raw_response={"prompt_eval_count": 10, "eval_count": 3})


def _args(tmp_path, **over):
    base = dict(models=["fake"], seeds=[7], distractors=[3], trials=1, history_limit=4, memory_limit=3,
                timeout=1.0, no_resume=False, generator="unique", instruction="plain",
                out=str(tmp_path / "out"), ollama_url=None, summarize_only=False)
    base.update(over)
    return SimpleNamespace(**base)


def test_nsa_history_control_sees_history_but_no_state():
    inner = _FakeInner()
    backend = v4.FixedDecodingBackend(inner)
    obs = ["FACT: ITEM_001 has value V001_000.", "DISTRACTOR: ITEM_002 has value V002_000."]
    v4.answer_for("nsa_history", backend, obs, "What is ITEM_001?", "V001_000", "recall", 4, 3)
    history_prompt = backend.last.prompt
    v4.answer_for("nsa_state", backend, obs, "What is ITEM_001?", "V001_000", "recall", 4, 3)
    state_prompt = backend.last.prompt
    assert history_prompt.startswith("RECENT_OBSERVATIONS=") and "FACT: ITEM_001" in history_prompt
    assert "CURRENT_STATE" not in history_prompt and "COGNITIVE_STATE" not in history_prompt
    assert "CURRENT_STATE" in state_prompt and "FACT: ITEM_001" in state_prompt


def test_every_condition_uses_identical_decoding():
    inner = _FakeInner()
    backend = v4.FixedDecodingBackend(inner, **v4.DECODING)
    obs = ["FACT: ITEM_001 has value V001_000.", "UPDATE: ITEM_001 now has value V001_NEW; the previous value is obsolete."]
    for condition in v4.CONDITIONS:
        v4.answer_for(condition, backend, obs, "What is ITEM_001?", "V001_NEW", "supersession_current", 4, 3)
    assert len(inner.calls) == len(v4.CONDITIONS)
    for call in inner.calls:
        assert (call["temperature"], call["max_tokens"], call["num_ctx"]) == (0.0, 32, 32768)


def test_run_writes_manifest_and_guards_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(v4, "OllamaInferenceBackend", lambda **kw: _FakeInner())
    args = _args(tmp_path)
    manifest = v4.run(args)
    total = len(v4.TASKS) * len(v4.CONDITIONS)
    assert manifest["episodes_recorded"] == total
    assert manifest["version"] == "4.2"
    assert "paired_tests_lenient" in manifest and "accuracy_ci95" in manifest["summary"]["raw"]
    # resume: nothing re-run
    assert v4.run(args)["episodes_recorded"] == total
    # different configuration must not silently mix into the same raw.jsonl
    with pytest.raises(SystemExit):
        v4.run(_args(tmp_path, history_limit=9))
    # summarize-only works without a backend
    again = v4.run(_args(tmp_path, summarize_only=True))
    assert again["episodes_recorded"] == total
    rows = [json.loads(l) for l in (tmp_path / "out" / "raw.jsonl").read_text().splitlines()]
    assert all("prompt_tokens" in r and "timed_out" in r for r in rows)


def test_nohist_condition_drops_history_only_for_memory_prompts():
    backend = v4.FixedDecodingBackend(_FakeInner())
    obs = ["FACT: ITEM_001 has value V001_000.", "DISTRACTOR: ITEM_002 has value V002_000."]
    v4.answer_for("nsa_temporal_memory", backend, obs, "What is ITEM_001?", "V001_000", "recall", 4, 3)
    with_hist = backend.last.prompt
    v4.answer_for("nsa_temporal_memory_nohist", backend, obs, "What is ITEM_001?", "V001_000", "recall", 4, 3)
    without = backend.last.prompt
    assert "DISTRACTOR: ITEM_002" in with_hist and "RECENT_OBSERVATIONS" in with_hist
    assert "RECENT_OBSERVATIONS" not in without and "RETRIEVED_MEMORY=" in without


def test_carry_over_reuses_only_unchanged_conditions_and_checks_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr(v4, "OllamaInferenceBackend", lambda **kw: _FakeInner())
    first = _args(tmp_path, out=str(tmp_path / "old"))
    v4.run(first)
    second = _args(tmp_path, out=str(tmp_path / "new"), carry_over=str(tmp_path / "old"))
    manifest = v4.run(second)
    n_tasks = len(v4.TASKS)
    assert manifest["carried_over"]["records"] == n_tasks * len(v4.UNCHANGED_SINCE_4_1)
    assert set(manifest["carried_over"]["conditions"]) == set(v4.UNCHANGED_SINCE_4_1)
    assert manifest["episodes_recorded"] == n_tasks * len(v4.CONDITIONS)
    with pytest.raises(SystemExit):
        v4.run(_args(tmp_path, out=str(tmp_path / "bad"), history_limit=9, carry_over=str(tmp_path / "old")))
