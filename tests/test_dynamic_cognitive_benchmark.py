from experiments.cognitive.dynamic_benchmark import CONDITIONS, run


def test_dynamic_benchmark_is_reproducible():
    a = run([7, 17], horizon=20)
    b = run([7, 17], horizon=20)
    assert a["aggregates"] == b["aggregates"]
    assert a["gates"]["authority_zero_violation"]


def test_dynamic_benchmark_has_all_matched_conditions():
    report = run([7], horizon=20)
    assert tuple(report["conditions"]) == CONDITIONS
    assert set(report["aggregates"]) == set(CONDITIONS)
    assert len(report["episodes"]) == len(CONDITIONS)


def test_failed_scientific_gate_is_fail_honest():
    report = run([7], horizon=20)
    assert report["status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
    if not all(report["gates"].values()):
        assert report["status"] == "RESEARCH_GATE_NOT_YET_MET"
