import pytest

from experiments.holographic_state_field import consolidation_benchmark as cb
from experiments.holographic_state_field.replication_sweep import run


def test_selection_and_evaluation_seeds_must_be_disjoint():
    with pytest.raises(ValueError):
        run(select_seeds=(1, 2), eval_seeds=(2, 3))


def test_sweep_selects_on_one_seed_set_evaluates_on_another_and_restores_sparsity():
    before = cb.SPARSITY
    out = run(counts=(32,), similarities=(0.9,), select_seeds=(1,), eval_seeds=(11, 23), sparsities=(0.05, 0.10),
              shapes=((16, 16, 16), (16, 32, 8), (32, 32, 4)))
    assert cb.SPARSITY == before
    assert out["shapes"] == [[16, 16, 16], [16, 32, 8]]                     # C=4 < vocabulary size is dropped
    (row,) = out["rows"]
    assert row["paired_wins_hsf"] + row["paired_ties"] + row["paired_wins_hdc"] == 2
    assert row["hsf_best_sparsity"] in (0.05, 0.10) and row["hdc_best_sparsity"] in (0.05, 0.10)


def test_bounded_replay_is_monotone_in_the_buffer_size_for_the_separated_delta_field():
    from experiments.holographic_state_field.consolidation_benchmark import run_one

    acc = {f: run_one("hsf_sep_delta_replay", 7, 64, 0.9, 16, 16, 16, replay_fraction=f)["overall_accuracy"] for f in (0.1, 1.0)}
    assert acc[1.0] > acc[0.1]


def test_priority_replay_beats_random_replay_with_a_small_buffer_and_validates_policy():
    from experiments.holographic_state_field.consolidation_benchmark import run_one

    kw = dict(seed=7, count=64, rho=0.9, h=16, w=32, c=8, replay_fraction=0.125)
    random_ = run_one("hsf_sep_homeo_replay", replay_policy="random", **kw)["overall_accuracy"]
    priority = run_one("hsf_sep_homeo_replay", replay_policy="priority", **kw)["overall_accuracy"]
    assert priority > random_
    with pytest.raises(ValueError):
        run_one("hsf_sep_homeo_replay", replay_policy="nope", **kw)


def test_larger_vocabulary_runs_and_a_field_needs_enough_channels():
    from experiments.holographic_state_field.consolidation_benchmark import run_one

    ok = run_one("hsf_sep_homeo_replay", 7, 32, 0.5, 8, 16, 32, vocab_size=32)["overall_accuracy"]
    starved = run_one("hsf_sep_homeo_replay", 7, 32, 0.5, 32, 16, 8, vocab_size=32)["overall_accuracy"]
    assert ok > starved
