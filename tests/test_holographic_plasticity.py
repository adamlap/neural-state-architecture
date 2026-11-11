"""Tests for the isolated competitive-Hebbian self-organisation experiment."""
import numpy as np
import pytest

from experiments.holographic_state_field.plasticity import CompetitiveHebbianMap
from experiments.holographic_state_field.plasticity_benchmark import make_family_data, run


def test_learning_changes_weights_and_preserves_unit_norms():
    x, _, _, _ = make_family_data(seed=7, dimension=64, train_per_family=8,
                                   test_per_family=4, rho=0.5)
    learner = CompetitiveHebbianMap(units=4, dimension=64, seed=7)
    before = learner.weights
    learner.fit(x, epochs=2, seed=11)
    after = learner.weights
    assert learner.updates == len(x) * 2
    assert not np.allclose(before, after)
    np.testing.assert_allclose(np.linalg.norm(after, axis=1), 1.0, atol=1e-5)


def test_learning_api_does_not_accept_labels():
    learner = CompetitiveHebbianMap(units=4, dimension=16)
    with pytest.raises(TypeError):
        learner.learn_one(np.ones(16, dtype=np.float32), label=2)


def test_invalid_cues_are_rejected():
    learner = CompetitiveHebbianMap(units=4, dimension=16)
    with pytest.raises(ValueError):
        learner.learn_one(np.zeros(16, dtype=np.float32))
    with pytest.raises(ValueError):
        learner.learn_one(np.ones(15, dtype=np.float32))


def test_plasticity_benchmark_is_reproducible_and_reports_heldout_scores():
    kwargs = dict(seeds=(7,), similarities=(0.5,), dimension=64,
                  train_per_family=8, test_per_family=4, epochs=2)
    first, second = run(**kwargs), run(**kwargs)
    assert first == second
    row = first["runs"][0]
    assert 0.0 <= row["hebbian_test_accuracy"] <= 1.0
    assert row["updates"] == 4 * 8 * 2
    assert row["weight_change_l2"] > 0
