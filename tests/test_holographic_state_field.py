"""Tests for the experimental Holographic State Field."""
import numpy as np
import pytest

from experiments.holographic_state_field import HolographicStateField

VOCAB = ("RED", "GREEN", "BLUE", "AMBER")


def make_field(seed=7):
    return HolographicStateField(VOCAB, height=16, width=16, channels=32, seed=seed)


def test_single_association_can_be_reconstructed():
    field = make_field()
    field.write("person:ada:employer", "AMBER")
    result = field.retrieve("person:ada:employer")
    assert result.value == "AMBER"
    assert result.confidence_margin > 0


def test_overlapping_memories_share_one_fixed_field():
    field = make_field()
    initial_shape, initial_bytes = field.shape, field.memory_bytes
    for i in range(100):
        field.write(f"entity:{i}", VOCAB[i % len(VOCAB)])
    assert field.shape == initial_shape
    assert field.memory_bytes == initial_bytes
    assert field.writes == 100


def test_superposition_is_order_independent():
    records = [
        ("person:ada:employer", "AMBER"),
        ("person:ada:skill", "BLUE"),
        ("person:grace:employer", "GREEN"),
    ]
    first, second = make_field(), make_field()
    for key, value in records:
        first.write(key, value)
    for key, value in reversed(records):
        second.write(key, value)
    np.testing.assert_allclose(first._field, second._field, rtol=0, atol=1e-6)


def test_associative_retrieval_survives_moderate_interference():
    field = HolographicStateField(VOCAB, height=32, width=32, channels=64, seed=37)
    records = [(f"entity:{i}", VOCAB[(i * 3) % len(VOCAB)]) for i in range(128)]
    for key, value in records:
        field.write(key, value)
    accuracy = sum(field.retrieve(k).value == v for k, v in records) / len(records)
    assert accuracy >= 0.95


def test_reinforcement_and_decay_change_field_energy():
    field = make_field()
    field.write("topic:memory", "BLUE")
    initial = field.field_energy
    field.reinforce("topic:memory", "BLUE", amount=0.5)
    assert field.field_energy > initial
    field.decay(0.5)
    assert field.field_energy < initial * 2


def test_invalid_values_and_decay_are_rejected():
    field = make_field()
    with pytest.raises(KeyError):
        field.write("key", "NOT_IN_VOCAB")
    with pytest.raises(ValueError):
        field.decay(1.5)
