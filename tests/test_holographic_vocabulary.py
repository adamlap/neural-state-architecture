import pytest

from experiments.holographic_state_field.vocabulary_sweep import plane, run, shapes_for


def test_shapes_keep_the_budget_and_need_enough_channels():
    for h, w, c in shapes_for(32, 16384):
        assert h * w * c == 16384 and c >= 32
    assert shapes_for(128, 4096) == [(2 ** 2, 8, 128)] or all(c >= 128 for _, _, c in shapes_for(128, 4096))
    assert plane(512) == (16, 32)


def test_seed_sets_must_be_disjoint_and_sweep_smoke():
    with pytest.raises(ValueError):
        run(select_seeds=(1,), eval_seeds=(1,))
    out = run(vocabs=(8,), budgets=(4096,), counts=(32,), sparsities=(0.05,), select_seeds=(1,), eval_seeds=(11,))
    (row,) = out["rows"]
    assert row["hsf_shape"][0] * row["hsf_shape"][1] * row["hsf_shape"][2] == 4096


def test_distributed_codes_use_short_value_codes_independent_of_vocabulary_size():
    for h, w, c in shapes_for(1024, 16384, distributed=True):
        assert c in (16, 32, 64) and h * w * c == 16384


def test_distributed_field_decodes_a_large_vocabulary_with_short_codes():
    from experiments.holographic_state_field.consolidation_benchmark import run_one

    acc = run_one("hsf_sep_homeo_replay", 7, 32, 0.0, 16, 16, 32, vocab_size=512)["overall_accuracy"]
    assert acc > 0.8
