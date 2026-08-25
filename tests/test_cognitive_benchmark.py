from experiments.cognitive.benchmark import CONDITIONS, TASKS, run


def test_benchmark_is_reproducible():
    a = run([7, 17], horizon=30)
    b = run([7, 17], horizon=30)
    assert a["aggregates"] == b["aggregates"]
    assert a["gates"]["authority_zero_violation"]


def test_all_control_conditions_are_present():
    report = run([7], horizon=20)
    assert tuple(report["conditions"]) == CONDITIONS
    assert tuple(report["tasks"]) == TASKS
    assert set(report["aggregates"]) == set(CONDITIONS)


def test_benchmark_does_not_claim_success_when_gates_fail():
    report = run([7], horizon=20)
    assert report["status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
    if not all(report["gates"].values()):
        assert report["status"] == "RESEARCH_GATE_NOT_YET_MET"
