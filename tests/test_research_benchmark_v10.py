import pytest

np = pytest.importorskip("numpy")

from experiments.research_v10 import gated_memory_benchmark as v10


def test_stream_has_reminders_with_typos_genuine_updates_and_an_untrusted_update():
    events, truth, typos = v10.make_stream(3, "train")
    assert typos and len(truth) == 20
    assert any(s == v10.UNTRUSTED_SOURCE for _, s in events)
    assert all(v not in typos for v in truth.values())


def test_ungated_rules_extractor_is_corrupted_by_typos_and_the_gate_reduces_work_and_corruption():
    ungated = [v10.run_one("rules", None, "train", s) for s in range(4)]
    gated = [v10.run_one("rules", 0.75, "train", s) for s in range(4)]
    assert sum(r.corrupted for r in ungated) > 0 and sum(r.corrupted for r in gated) < sum(r.corrupted for r in ungated)
    assert sum(r.lines_extracted for r in gated) < 0.8 * sum(r.lines_extracted for r in ungated)


def test_run_validates_and_unknown_extractor_is_rejected():
    with pytest.raises(ValueError):
        v10.run(seeds=())
    with pytest.raises(ValueError):
        v10.run_one("nope", None, "train", 0)


def test_end_to_end_episode_runs_all_conditions_with_a_fake_model():
    from types import SimpleNamespace

    from experiments.research_v10 import end_to_end as e2e

    class _Fake:
        model = "fake"
        last = None

        def generate(self, prompt, **kw):
            return SimpleNamespace(text="VALUE=NOPE" if "Extract every fact" not in prompt and "Example 1" not in prompt else "NONE", raw_response={})

    rows = e2e.run_episode(_Fake(), 1, "train", n_questions=2)
    assert {r["condition"] for r in rows} == set(e2e.CONDITIONS) and len(rows) == 8
