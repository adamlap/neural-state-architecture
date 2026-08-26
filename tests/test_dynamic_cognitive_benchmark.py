from experiments.cognitive.dynamic_benchmark import CONDITIONS, run


def test_default_configuration_passes_all_research_gates():
    # Regression guard: the decisions gate is a near-zero-crossing binary metric
    # that is too noisy at n=5; the benchmark's default seed list was widened to
    # 10 seeds specifically so this gate is stable. Keep this in sync with the
    # CLI default in dynamic_benchmark.main().
    report = run([7, 17, 37, 73, 137, 211, 307, 401, 503, 601], horizon=60)
    assert report["status"] == "PASS", report["gates"]


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
