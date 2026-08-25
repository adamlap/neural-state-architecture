from experiments.cognitive.state_compression_benchmark import CONDITIONS, run


def test_state_compression_is_reproducible():
    a = run([7, 17], horizon=40, context_window=8)
    b = run([7, 17], horizon=40, context_window=8)
    assert a["aggregates"] == b["aggregates"]
    assert a["gates"]["authority_zero_violation"]


def test_all_controls_and_memory_budgets_are_present():
    report = run([7], horizon=30, context_window=8)
    assert tuple(report["conditions"]) == CONDITIONS
    assert set(report["aggregates"]) == set(CONDITIONS)
    assert report["aggregates"]["predictive_cce"]["memory_units"] == 3
    assert report["aggregates"]["full_context"]["memory_units"] == 30


def test_scientific_gate_is_fail_honest():
    report = run([7], horizon=30, context_window=8)
    assert report["status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
    if not all(report["gates"].values()):
        assert report["status"] == "RESEARCH_GATE_NOT_YET_MET"
