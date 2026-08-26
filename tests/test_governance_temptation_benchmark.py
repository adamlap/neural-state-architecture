from experiments.cognitive.governance_benchmark import (
    CONDITIONS,
    build_engine,
    run,
    run_episode,
)
from nsa import KeywordClassifier, NSAPolicy, PolicyEngine


def test_governance_benchmark_is_reproducible():
    a = run([7, 17], horizon=40)
    b = run([7, 17], horizon=40)
    assert a["aggregates"] == b["aggregates"]


def test_all_conditions_are_present():
    report = run([7], horizon=40)
    assert tuple(report["conditions"]) == CONDITIONS
    assert set(report["aggregates"]) == set(CONDITIONS)


def test_default_configuration_passes_all_research_gates():
    report = run([7, 17, 37, 73, 137, 211, 307, 401, 503, 601], horizon=80)
    assert report["status"] == "PASS", report["gates"]


def test_temptation_is_never_executed_under_the_configured_policy():
    report = run([7, 17, 37], horizon=40)
    for aggregate in report["aggregates"].values():
        assert aggregate["shortcut_executions"] == 0
        assert aggregate["temptations_offered"] > 0
        assert aggregate["temptations_denied"] == aggregate["temptations_offered"]


def test_harness_is_not_hardcoded_to_deny_shortcut_executes_under_a_permissive_policy():
    # Sanity check that "zero unauthorized actions" reflects the real PolicyEngine
    # decision rather than being hardcoded: an intentionally permissive policy
    # must let the shortcut through.
    permissive_policy = NSAPolicy.from_mapping({"name": "permissive", "restricted_actions": []})
    permissive_engine = PolicyEngine(permissive_policy, KeywordClassifier({}))
    episode = run_episode(7, "predictive_cce", permissive_engine, horizon=40)
    assert episode.shortcut_executions == episode.temptations_offered
    assert episode.governed_score == episode.ungoverned_score


def test_build_engine_denies_the_restricted_capability():
    from nsa import EvaluationContext
    from nsa.decision import Decision

    engine = build_engine()
    decision = engine.evaluate(
        "x",
        context=EvaluationContext(action="read_ground_truth", capabilities=frozenset({"exfiltrate_ground_truth"})),
    )
    assert decision.decision is Decision.DENY


def test_failed_scientific_gate_is_fail_honest():
    report = run([7], horizon=40)
    assert report["status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
    if not all(report["gates"].values()):
        assert report["status"] == "RESEARCH_GATE_NOT_YET_MET"
