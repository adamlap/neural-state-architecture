from types import SimpleNamespace

from experiments.research_v7 import calibrated_abstention_benchmark as v7


class _Counting:
    model = "fake"

    def __init__(self):
        self.calls = 0
        self.last = None

    def generate(self, prompt, **kw):
        self.calls += 1
        return SimpleNamespace(text="VALUE=V001_000")


def test_unanswerable_key_never_appears_in_the_stream():
    lines, _, expected, key = v7.make_episode(7, "unanswerable", 30)
    assert expected == "UNKNOWN" and not any(key in line for line in lines)


def test_answerable_key_is_in_the_stream_once_or_twice():
    for kind, n in (("recall", 1), ("supersession", 2)):
        lines, _, _, key = v7.make_episode(7, kind, 30)
        assert sum(key in line for line in lines) == n


def test_state_gated_abstains_without_calling_the_model_when_evidence_is_absent():
    backend = _Counting()
    lines, question, _, key = v7.make_episode(7, "unanswerable", 10)
    text, called, uncertainty = v7.answer_for("state_gated", backend, lines, question, key)
    assert text == "VALUE=UNKNOWN" and not called and uncertainty == 1.0 and backend.calls == 0


def test_state_gated_answers_normally_when_evidence_exists():
    backend = _Counting()
    lines, question, _, key = v7.make_episode(7, "recall", 10)
    _, called, uncertainty = v7.answer_for("state_gated", backend, lines, question, key)
    assert called and uncertainty == 0.0 and backend.calls == 1


def test_grading_separates_hallucination_from_over_abstention():
    assert v7.grade("unanswerable", "UNKNOWN", "VALUE=V007_000", False) == (False, False, True)
    assert v7.grade("unanswerable", "UNKNOWN", "VALUE=UNKNOWN", False) == (True, True, False)
    assert v7.grade("recall", "V007_000", "VALUE=UNKNOWN", False) == (True, False, False)
    assert v7.grade("recall", "V007_000", "VALUE=V007_000", False) == (False, True, False)


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(v7, "OllamaInferenceBackend", lambda **kw: _Counting())
    args = SimpleNamespace(models=["fake"], seeds=[7], distractors=[5], trials=1, timeout=1.0, out=str(tmp_path), ollama_url=None, summarize_only=False)
    manifest = v7.run(args)
    assert manifest["episodes_recorded"] == len(v7.KINDS) * len(v7.CONDITIONS)
    assert manifest["by_condition"]["state_gated"]["hallucination_rate"] == 0.0


def test_state_gated_prompt_has_no_escape_clause_but_the_baselines_do():
    seen = {}

    class _Spy(_Counting):
        def generate(self, prompt, **kw):
            seen["prompt"] = prompt
            return super().generate(prompt)

    lines, question, _, key = v7.make_episode(7, "recall", 10)
    for condition in ("state_gated", "memory_generic", "memory_keyed", "transcript"):
        v7.answer_for(condition, _Spy(), lines, question, key)
        seen[condition] = seen["prompt"]
    assert "UNKNOWN" not in seen["state_gated"]
    assert "UNKNOWN" in seen["memory_generic"] and "NO_RELEVANT_MEMORY, reply VALUE=UNKNOWN" in seen["memory_keyed"]
