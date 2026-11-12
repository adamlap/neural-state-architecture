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
