"""Correctness tests for comparative associative-memory baselines."""
import pytest
from experiments.holographic_state_field.baselines import ExactDictionary, HDCMemory, HRRMemory

VOCAB = ("RED", "GREEN", "BLUE", "AMBER")

@pytest.mark.parametrize("memory_type", [HRRMemory, HDCMemory])
def test_vector_memory_recalls_one_association(memory_type):
    memory = memory_type(VOCAB, dimension=2048, seed=7)
    memory.write("person:ada:employer", "AMBER")
    assert memory.retrieve("person:ada:employer").value == "AMBER"

def test_exact_dictionary_is_perfect_and_overwrites_explicitly():
    memory = ExactDictionary()
    memory.write("same-key", "RED")
    memory.write("same-key", "BLUE")
    assert memory.retrieve("same-key").value == "BLUE"
    assert memory.record_count == 1

@pytest.mark.parametrize("memory_type", [HRRMemory, HDCMemory])
def test_vector_memory_has_fixed_numeric_payload(memory_type):
    memory = memory_type(VOCAB, dimension=512, seed=17)
    initial_bytes = memory.memory_bytes
    for index in range(100):
        memory.write(f"entity:{index}", VOCAB[index % len(VOCAB)])
    assert memory.memory_bytes == initial_bytes
    assert memory.writes == 100

def test_unknown_values_are_rejected():
    memory = HDCMemory(VOCAB, dimension=128, seed=7)
    with pytest.raises(KeyError):
        memory.write("key", "UNKNOWN")
