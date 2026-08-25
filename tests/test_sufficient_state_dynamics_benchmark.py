from experiments.cognitive.sufficient_state_benchmark import CONDITIONS, run


def test_sufficient_state_benchmark_is_reproducible():
    a = run([7, 17], horizon=40)
    b = run([7, 17], horizon=40)
    assert a["aggregates"] == b["aggregates"]
    assert a["gates"]["authority_zero_violation"]


def test_all_conditions_and_compression_budget_are_reported():
    report = run([7], horizon=40)
    assert tuple(report["conditions"]) == CONDITIONS
    assert set(report["aggregates"]) == set(CONDITIONS)
    assert report["aggregates"]["predictive_state"]["memory_units"] < report["aggregates"]["full_context"]["memory_units"]


def test_failed_scientific_gate_is_fail_honest():
    report = run([7], horizon=40)
    assert report["status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
    if not all(report["gates"].values()):
        assert report["status"] == "RESEARCH_GATE_NOT_YET_MET"
