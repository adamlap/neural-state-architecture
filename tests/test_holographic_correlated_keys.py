"""Tests for controlled correlated-key interference experiments."""
import numpy as np
import pytest

from experiments.holographic_state_field.baselines import HDCMemory, HRRMemory
from experiments.holographic_state_field.correlated_key_benchmark import correlated_cues, run
from experiments.holographic_state_field.field import HolographicStateField


VOCAB = ("RED", "GREEN", "BLUE", "AMBER")


def test_correlated_cues_are_distinct_and_approximate_target_similarity():
    names, cues, realised = correlated_cues(seed=17, dimension=4096, count=64, rho=0.5)
    assert len(names) == len(set(names)) == len(cues)
    assert len({tuple(cues[name]) for name in names}) == len(names)
    assert realised == pytest.approx(0.5, abs=0.04)


def test_correlated_cue_generation_is_reproducible():
    first = correlated_cues(seed=37, dimension=512, count=32, rho=0.75)
    second = correlated_cues(seed=37, dimension=512, count=32, rho=0.75)
    assert first[0] == second[0]
    assert first[2] == second[2]
    for name in first[0]:
        np.testing.assert_array_equal(first[1][name], second[1][name])


def test_hsf_accepts_external_patterns_without_changing_field_size():
    field = HolographicStateField(VOCAB, height=8, width=8, channels=16)
    pattern = np.ones(64, dtype=np.float32)
    initial_bytes = field.memory_bytes
    field.write_pattern(pattern, "BLUE")
    assert field.retrieve_pattern(pattern).value == "BLUE"
    assert field.memory_bytes == initial_bytes


@pytest.mark.parametrize("memory_type", [HRRMemory, HDCMemory])
def test_vector_baselines_accept_external_cues(memory_type):
    memory = memory_type(VOCAB, dimension=1024, seed=9)
    cue = np.ones(1024, dtype=np.float32)
    memory.write_vector(cue, "GREEN")
    assert memory.retrieve_vector(cue).value == "GREEN"


def test_external_cue_validation():
    field = HolographicStateField(VOCAB, height=4, width=4, channels=8)
    with pytest.raises(ValueError):
        field.write_pattern(np.zeros(16, dtype=np.float32), "RED")
    memory = HDCMemory(VOCAB, dimension=16, seed=1)
    with pytest.raises(ValueError):
        memory.write_vector(np.zeros(15, dtype=np.float32), "RED")


def test_correlated_benchmark_smoke_run_has_expected_grid():
    result = run(seeds=(7,), counts=(16,), similarities=(0.0, 0.5),
                 height=4, width=4, channels=4)
    assert len(result["runs"]) == 2
    assert len(result["aggregates"]) == 2
    for row in result["aggregates"]:
        assert set(row) >= {"target_within_family_cosine", "memories", "hsf", "hrr", "hdc"}
