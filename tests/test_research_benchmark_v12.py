import pytest

pytest.importorskip("numpy")

from types import SimpleNamespace

from experiments.research_v12 import gated_calls_benchmark as v12


class _Oracle:
    """Answers with the first value-looking token in the memory block (a perfect reader), to test the harness and the gating, not a model."""
    model = "oracle"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, **kw):
        import re
        self.calls += 1
        m = re.search(r"RETRIEVED_MEMORY=\n- [^:]+: (\S+)", prompt)
        return SimpleNamespace(text=f"VALUE={m.group(1)}" if m else "VALUE=NOPE", raw_response={})


def test_session_has_updates_unknown_entities_and_varied_wording():
    facts, values, events = v12.make_session(3)
    kinds = [e[0] for e in events]
    assert kinds.count("update") == 6 and kinds.count("ask") >= 60 and len(facts) == 12
    assert any(not (e[2], e[3]) in values for e in events if e[0] == "ask")


def test_gating_saves_calls_with_no_stale_reuse_and_no_hallucination_on_a_perfect_reader():
    always = v12.run_session(_Oracle(), 1, gated=False)
    b = _Oracle()
    gated = v12.run_session(b, 1, gated=True)
    rows_g, rows_a = gated["rows"], always["rows"]
    assert sum(r["stale_reuse"] for r in rows_g) == 0
    assert gated["call_stats"]["model_calls"] < 0.8 * always["call_stats"]["model_calls"]
    assert gated["call_stats"]["reused"] > 0 and gated["call_stats"]["abstained"] > 0
    assert all(not r["hallucinated"] for r in rows_g)
    ok_g = sum(r["correct"] for r in rows_g if r["known"]) / sum(r["known"] for r in rows_g)
    ok_a = sum(r["correct"] for r in rows_a if r["known"]) / sum(r["known"] for r in rows_a)
    assert ok_g >= ok_a - 0.02


def test_corrupt_changes_one_character_and_summary_shapes():
    import random
    q = "What is the database port for staging?"
    assert len(v12.corrupt(q, random.Random(1))) == len(q) - 1
    rows = v12.run_session(_Oracle(), 2, True)["rows"] + v12.run_session(_Oracle(), 2, False)["rows"]
    assert set(v12.summarize(rows)) == set(v12.CONDITIONS)
