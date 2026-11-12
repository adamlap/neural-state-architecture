import numpy as np
import pytest

from experiments.organic.novelty_gate import NoveltyGate, auc, call_budget, cue, run, scores_for, stream, typo


def test_cue_is_unit_norm_and_similar_strings_have_similar_cues():
    a, b, c = cue("the port for staging is k3m9x2ab"), cue("the port for staging is k3m9x2ac"), cue("lunch at noon?")
    assert float(a @ a) == pytest.approx(1.0, abs=1e-5)
    assert float(a @ b) > 0.8 > float(a @ c) + 0.3


def test_gate_finds_repeats_familiar_and_new_content_novel_with_fixed_size():
    g = NoveltyGate()
    first = g.observe("The port for staging is k3m9x2ab.")
    again = g.observe("The port for staging is k3m9x2ab.")
    new = g.observe("We changed billing's admin email to q7w8e9rt")
    assert again < 0.6 * first and new > 1.5 * again and g.floats == 256 * 256


def test_stream_labels_first_mentions_and_updates_important_and_reminders_not():
    data = stream(3, 200)
    assert 0.1 < sum(l for _, l in data) / 200 < 0.6
    assert all(not l for t, l in data if t in ("sounds good", "lunch at noon?"))


def test_auc_and_call_budget_sanity():
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0 and auc([0.5] * 4, [True, False, True, False]) == 0.5
    b = call_budget([0.9, 0.1, 0.8, 0.2, 0.3], [True, False, True, False, False], 0.4)
    assert b["recall"] == 1.0 and b["false_call_rate"] == 0.0


def test_field_gate_beats_random_and_does_not_grow():
    lines = [t for t, _ in stream(1, 300)]
    labels = [l for _, l in stream(1, 300)]
    f, fl = scores_for("field", lines, 1)
    r, _ = scores_for("random", lines, 1)
    assert auc(f, labels) > auc(r, labels) + 0.2 and fl == 65536
    assert scores_for("field", [t for t, _ in stream(1, 1000)], 1)[1] == 65536


def test_typo_changes_one_character_and_run_validates():
    import random
    assert sum(x != y for x, y in zip("abcdefgh", typo("abcdefgh", random.Random(1)))) <= 1
    with pytest.raises(ValueError):
        run(seeds=())
