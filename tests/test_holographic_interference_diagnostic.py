"""Tests for HSF interference-source decomposition."""
from experiments.holographic_state_field.interference_diagnostic import run


def test_interference_diagnostic_decomposes_all_associations():
    result = run(seeds=(7,), counts=(8,), similarities=(0.5,),
                 height=4, width=4, channels=8)
    assert len(result["runs"]) == 1
    row = result["aggregates"][0]
    assert row["component_accuracy"]["target_only"] == 1.0
    assert 0.0 <= row["component_accuracy"]["full"] <= 1.0
    assert row["same_family_interference_to_signal"] >= 0.0
    assert row["cross_family_interference_to_signal"] >= 0.0


def test_interference_diagnostic_is_reproducible():
    kwargs = dict(seeds=(17,), counts=(8,), similarities=(0.75,),
                  height=4, width=4, channels=8)
    assert run(**kwargs)["aggregates"] == run(**kwargs)["aggregates"]
