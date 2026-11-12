import pytest

pytest.importorskip("numpy")

from types import SimpleNamespace

from experiments.research_v11 import long_noisy_benchmark as v11


def test_long_stream_scales_and_keeps_proportions_and_all_keys():
    for lines in (100, 300, 1000):
        events, truth, typos = v11.long_stream(1, lines)
        assert abs(len(events) - lines) <= 0.1 * lines and len(truth) == 80 and typos


def test_routed_rule_memory_beats_unrouted_on_a_long_noisy_stream_without_any_model():
    events, truth, typos = v11.long_stream(1, 300)
    scores = {}
    for route in (False, True):
        a = v11.build_agent(events, route)
        ok = 0
        for key, v in truth.items():
            cur = a.memory.current(key)
            ok += bool(cur) and cur.content["value"].lower() == v.lower()
        scores[route] = ok / len(truth)
    assert scores[True] > scores[False] + 0.2


class _Fake:
    model = "fake"

    def __init__(self):
        self.num_ctx = 8192
        self.last = SimpleNamespace(truncated=False, prompt_tokens=5, prompt_eval_ms=1.0)

    def generate(self, prompt, **kw):
        return SimpleNamespace(text="VALUE=NOPE")


def test_episode_skips_oversized_transcripts_and_runs_other_conditions():
    rows = v11.run_episode(_Fake(), 1, 300, 8192, cap=500, n_questions=2)
    assert {r["condition"] for r in rows} == set(v11.CONDITIONS) and len(rows) == 8
    assert all(r["skipped"] for r in rows if r["condition"] == "transcript") and not any(r["skipped"] for r in rows if r["condition"] != "transcript")
    summary = v11.summarize(rows)
    assert summary["300|8192|transcript"]["accuracy"] is None and summary["300|8192|rag_overlap"]["n"] == 2
