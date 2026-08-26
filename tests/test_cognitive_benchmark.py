from experiments.cognitive.benchmark import CONDITIONS, TASKS, run


def test_default_configuration_passes_all_research_gates():
    # Regression guard: this benchmark used to have an unfalsifiable ceiling
    # where context_memory and predictive_cce both scored a perfect 1.0, and a
    # Kalman-filter bootstrap bug that could permanently reject valid
    # observations. Both are fixed; this locks in that the fix holds.
    report = run([7, 17, 37, 73, 137], horizon=80)
    assert report["status"] == "PASS", report["gates"]


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
