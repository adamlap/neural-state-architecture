"""Tests for the dimension-controlled HSF diagnostic."""
from experiments.holographic_state_field.dimension_control_benchmark import run


def test_dimension_control_benchmark_smoke_and_dimensions():
    result = run(seeds=(7,), counts=(8,), similarities=(0.5,),
                 height=4, width=4, channels=8)
    assert len(result["runs"]) == 6
    assert len(result["aggregates"]) == 6
    rows = {(row["arm"], row["method"]): row for row in result["aggregates"]}
    assert rows[("key_dimension_matched", "hsf")]["key_dimension"] == 16
    assert rows[("key_dimension_matched", "hrr")]["key_dimension"] == 16
    assert rows[("numeric_payload_matched", "hrr")]["key_dimension"] == 128
    assert rows[("numeric_payload_matched", "hsf")]["value_dimension"] == 8
    assert rows[("numeric_payload_matched", "hsf")]["numeric_payload_bytes"] > (
        16 * 8 * 4
    )


def test_dimension_control_benchmark_is_reproducible():
    kwargs = dict(seeds=(17,), counts=(8,), similarities=(0.0,),
                  height=4, width=4, channels=8)
    first = run(**kwargs)
    second = run(**kwargs)
    assert first["aggregates"] == second["aggregates"]
