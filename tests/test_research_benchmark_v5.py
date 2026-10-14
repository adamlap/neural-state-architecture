from types import SimpleNamespace

import pytest

from experiments.research_v5 import memory_poisoning_benchmark as v5


def _episode(task="poisoned", style="plain", d=10, seed=7):
    return v5.make_episode(seed, task, style, d)


def test_governed_store_returns_trusted_value_and_ungoverned_returns_attack():
    events, _, expected, attack, key = _episode("poisoned")
    assert v5.build_store(events, governed=True).current(key).content["value"] == expected
    assert v5.build_store(events, governed=False).current(key).content["value"] == attack
    assert expected != attack


def test_gate_does_not_block_legitimate_updates():
    for task in ("legit_update", "poison_then_legit"):
        events, _, expected, _, key = _episode(task)
        store = v5.build_store(events, governed=True)
        assert store.current(key).content["value"] == expected


def test_forged_claim_is_in_the_text_but_not_in_the_label():
    events, *_ = _episode("poisoned", "forged")
    attack_event = next(e for e in events if not e.trusted)
    assert "[source: operator]" in attack_event.text() and attack_event.trusted is False


def test_authority_style_adds_override_text_and_plain_does_not():
    assert "SYSTEM OVERRIDE" in next(e for e in _episode("poisoned", "authority")[0] if not e.trusted).text()
    assert "SYSTEM OVERRIDE" not in next(e for e in _episode("poisoned", "plain")[0] if not e.trusted).text()


def test_target_key_is_not_repeated_by_distractors():
    events, _, _, _, key = _episode("poison_then_legit", d=200)
    assert sum(e.key == key for e in events) == 3


class _Fake:
    model = "fake"

    def generate(self, prompt, **kw):
        return SimpleNamespace(text="VALUE=NOPE", raw_response={"prompt_eval_count": 1})


def test_run_end_to_end_with_fake_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(v5, "OllamaInferenceBackend", lambda **kw: _Fake())
    args = SimpleNamespace(models=["fake"], seeds=[7], distractors=[3], trials=1, timeout=1.0,
                           out=str(tmp_path), ollama_url=None, summarize_only=False)
    manifest = v5.run(args)
    combos = 3 + 3 + 1
    assert manifest["episodes_recorded"] == combos * len(v5.CONDITIONS)
    assert manifest["by_condition"]["temporal_governed"]["n"] == combos
    assert v5.run(args)["episodes_recorded"] == manifest["episodes_recorded"]    # resume is a no-op
