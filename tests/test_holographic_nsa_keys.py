import pytest

from experiments.holographic_state_field.nsa_keys_benchmark import episode, key_name, mean_cosine, run, run_one, token_cue


def test_keys_sharing_tokens_are_correlated_and_unrelated_keys_are_not():
    a, b, c = key_name("billing", "admin email"), key_name("staging", "admin email"), key_name("auth", "log level")
    cues = {k: token_cue(k, 256, 7) for k in (a, b, c)}
    assert float(cues[a] @ cues[b]) > 0.3 > abs(float(cues[a] @ cues[c]))


def test_episode_updates_half_of_the_keys_with_new_values():
    keys, original, final, updated = episode(7, 40)
    assert len(updated) == 20 and all(final[k] != original[k] for k in updated) and all(final[k] == original[k] for k in keys if k not in updated)


def test_exact_dictionary_is_perfect_and_smoke_run_works():
    assert run_one("exact", 7, 40, 16, 16, 16)["accuracy"] == 1.0
    out = run(seeds=(7,), key_counts=(20,), shapes=((16, 16, 16),), methods=("exact", "hsf_sep_homeo_replay"))
    assert len(out["rows"]) == 2
    with pytest.raises(ValueError):
        run(seeds=())


def test_mean_cosine_of_identical_cues_is_one():
    k = key_name("auth", "log level")
    assert mean_cosine({"a": token_cue(k, 64, 1), "b": token_cue(k, 64, 1)}, ["a", "b"]) == pytest.approx(1.0, abs=1e-5)
