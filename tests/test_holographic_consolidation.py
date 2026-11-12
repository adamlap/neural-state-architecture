import numpy as np
import pytest

from experiments.holographic_state_field.consolidation_benchmark import PatternSeparator, code_overlap, run, run_one


def test_pattern_separation_makes_similar_cues_less_similar_and_codes_sparse():
    out = code_overlap(7, 256, 64, 0.9)
    assert out["separated_cosine"] < out["input_cosine"] - 0.2
    code = PatternSeparator(256, 256, 7)(np.ones(256, dtype=np.float32))
    assert np.count_nonzero(code) == 13 and np.linalg.norm(code) == pytest.approx(1.0, abs=1e-5)


def test_whitened_write_is_exact_for_highly_correlated_keys_within_capacity():
    r = run_one("hsf_whitened", 7, 128, 0.9, 16, 16, 16)
    assert r["overall_accuracy"] == 1.0 and r["floats"] == 16 * 16 * 16 + 256 * 256


def test_whitened_write_degrades_past_the_key_dimension():
    assert run_one("hsf_whitened", 7, 512, 0.0, 16, 16, 16)["overall_accuracy"] < 0.9


def test_equal_replay_lets_flat_memories_match_the_delta_rule_at_moderate_similarity():
    hsf = run_one("hsf_delta_replay", 7, 64, 0.5, 16, 16, 16)["overall_accuracy"]
    hdc = run_one("hdc_replay", 7, 64, 0.5, 16, 16, 16)["overall_accuracy"]
    assert abs(hsf - hdc) < 0.1 and hdc > 0.9


def test_separation_rescues_replay_at_high_similarity():
    plain = run_one("hsf_delta_replay", 7, 32, 0.9, 16, 16, 16)["overall_accuracy"]
    separated = run_one("hsf_sep_delta_replay", 7, 32, 0.9, 16, 16, 16)["overall_accuracy"]
    assert separated > plain + 0.2


def test_run_smoke_and_validation():
    out = run(seeds=(7,), counts=(32,), similarities=(0.5,), methods=("hsf_whitened", "hdc_replay"))
    assert len(out["runs"]) == 2 and "0.5" in out["pattern_separation_overlap"]
    with pytest.raises(ValueError):
        run(seeds=())


def test_homeostatic_write_improves_on_plain_delta_in_the_separated_space():
    plain = run_one("hsf_sep_delta", 7, 64, 0.9, 16, 16, 16)["overall_accuracy"]
    homeo = run_one("hsf_sep_homeo", 7, 64, 0.9, 16, 16, 16)
    assert homeo["overall_accuracy"] > plain and homeo["floats"] == 16 * 16 * 16 + 256


def test_wide_key_plane_with_separation_homeostasis_and_replay_beats_hdc_at_high_similarity():
    hsf = run_one("hsf_sep_homeo_replay", 7, 128, 0.9, 16, 32, 8)["overall_accuracy"]
    hdc = run_one("hdc_sep_replay", 7, 128, 0.9, 16, 32, 8)["overall_accuracy"]
    assert hsf > hdc
