"""Tests for the fixed-budget HSF factorisation sweep."""
import pytest

from experiments.holographic_state_field.allocation_benchmark import run


def test_allocation_sweep_holds_total_elements_constant_and_reports_dimensions():
    result = run(
        seeds=(7,),
        counts=(8,),
        similarities=(0.0,),
        shapes=((4, 8, 16), (8, 4, 16)),
    )
    assert result["total_dimensions"] == 256
    hsf = [row for row in result["aggregates"] if row["method"] == "hsf"]
    assert len(hsf) == 2
    assert {row["key_dimension"] for row in hsf} == {16, 32}
    assert {row["value_dimension"] for row in hsf} == {16}
    assert all(row["numeric_payload_bytes"] > 0 for row in hsf)


def test_allocation_sweep_rejects_unequal_budgets():
    with pytest.raises(ValueError, match="same H\*W\*C"):
        run(seeds=(7,), counts=(8,), similarities=(0.0,),
            shapes=((4, 8, 16), (8, 4, 32)))


def test_allocation_sweep_requires_enough_value_channels():
    with pytest.raises(ValueError, match="channels must be >= 16"):
        run(seeds=(7,), counts=(8,), similarities=(0.0,),
            shapes=((8, 8, 8),))
