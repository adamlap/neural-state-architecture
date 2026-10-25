from types import SimpleNamespace

from experiments.research_v9 import natural_memory_benchmark as v9


def test_episode_contains_target_once_and_distractors_never_reuse_it():
    events, question, expected, attack, (ent, attr) = v9.make_episode(7, "recall", "train", 30)
    mentions = [t for t, _ in events if ent in t and attr in t]
    assert len(mentions) == 1 and expected in mentions[0]
    assert attack not in " ".join(t for t, _ in events)


def test_update_and_poisoned_kinds_place_the_second_value_on_the_right_channel():
    ev_u, _, new, _, _ = v9.make_episode(7, "update", "train", 20)
    assert any(new in t and s == "chat" for t, s in ev_u)
    ev_p, _, original, attack, _ = v9.make_episode(7, "poisoned", "heldout", 20)
    assert any(attack in t and s == v9.UNTRUSTED_SOURCE for t, s in ev_p) and any(original in t and s == "chat" for t, s in ev_p)


def test_train_and_heldout_templates_are_disjoint():
    assert not set(v9.FACT_T["train"]) & set(v9.FACT_T["heldout"])
    assert not set(v9.UPDATE_T["train"]) & set(v9.UPDATE_T["heldout"])


def test_rag_overlap_finds_the_target_lines():
    events, question, expected, _, _ = v9.make_episode(7, "recall", "train", 30)
    assert any(expected in line for line in v9.rag_lines(events, question))


def test_chunking_keeps_sources_separate_and_bounded():
    chunks = v9.chunk_events([("a", "chat")] * 10 + [("b", "email")] + [("c", "chat")])
    assert [s for _, s in chunks] == ["chat", "chat", "email", "chat"] and chunks[0][0].count("\n") == 7


class _Fake:
    model = "fake"

    def __init__(self):
        self.last = None

    def generate(self, prompt, **kw):
        if "Extract every fact" in prompt or "Example 1" in prompt:
            return SimpleNamespace(text="NONE", raw_response={})
        return SimpleNamespace(text="VALUE=ZZ99-zzz", raw_response={"prompt_eval_count": 5})


def test_rule_condition_forms_memory_and_blocks_the_poisoned_update_without_a_model():
    backend = v9.FixedDecodingBackend(_Fake())
    events, question, expected, attack, target = v9.make_episode(7, "poisoned", "train", 20)
    res = v9.answer_for("nsa_rules", backend, events, question, expected, target)
    assert res.memory_correct and res.key_retrieved and res.ingest_calls == 0


def test_rules_degrade_on_heldout_phrasings_but_not_below_zero():
    backend = v9.FixedDecodingBackend(_Fake())
    ok = {"train": 0, "heldout": 0}
    for phrasing in ok:
        for seed in range(7, 27):
            events, question, expected, _, target = v9.make_episode(seed, "recall", phrasing, 10)
            ok[phrasing] += bool(v9.answer_for("nsa_rules", backend, events, question, expected, target).memory_correct)
    assert ok["train"] >= ok["heldout"]          # recorded, not tuned: see the benchmark results


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(v9, "OllamaInferenceBackend", lambda **kw: _Fake())
    args = SimpleNamespace(models=["fake"], seeds=[7], turns=[6], trials=1, timeout=1.0, out=str(tmp_path), ollama_url=None, summarize_only=False)
    manifest = v9.run(args)
    assert manifest["episodes_recorded"] == len(v9.KINDS) * len(v9.PHRASINGS) * len(v9.CONDITIONS)
    assert v9.run(args)["episodes_recorded"] == manifest["episodes_recorded"]


def test_extraction_quality_scoring_with_the_rules_and_a_fake_model():
    from experiments.research_v9 import extraction_quality as eq

    events, truth = eq.make_chunk_episode(7, "train", 10)
    assert truth and all(k and v for k, v in truth)
    res = eq.score(eq.build_extractor("rules", None), events, truth)
    assert res["hit"] == res["truth"] and res["spurious"] == 0 and res["calls"] == 0     # rules cover their own templates
    held_events, held_truth = eq.make_chunk_episode(7, "heldout", 10)
    held = eq.score(eq.build_extractor("rules", None), held_events, held_truth)
    assert held["hit"] < held["truth"]                                                    # and miss the held-out phrasings


def test_extraction_quality_run_writes_a_manifest(tmp_path, monkeypatch):
    from experiments.research_v9 import extraction_quality as eq

    monkeypatch.setattr(eq, "OllamaInferenceBackend", lambda **kw: _Fake())
    args = SimpleNamespace(models=["fake"], seeds=[7, 8], lines=8, timeout=1.0, out=str(tmp_path), ollama_url=None)
    m = eq.run(args)
    assert "rules|train" in m["by_model"]["fake"] and "hybrid|heldout" in m["by_model"]["fake"]
