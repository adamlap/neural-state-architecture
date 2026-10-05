from nsa.agent import NSARuntime, RuntimeConfig
from nsa.cce import CanonicalCCERuntime, TickInput
from nsa.cognition.interfaces import ActionCandidate
from nsa.cognition.system_one_runtime import (
    CallableSystemOneBackend,
    DecisionQuestion,
    DeterministicSystemOneBackend,
    SystemOneController,
)
from nsa.cognition.tools import ToolRegistry, ToolSpec
from nsa.core.state import CanonicalState, GoalState, HardState


def test_system_one_controller_is_generation_free_and_typed():
    state = CanonicalState().with_goal(GoalState(goals=("retrieve documents",), active_goal="retrieve documents"))
    controller = SystemOneController(DeterministicSystemOneBackend())
    decision = controller.decide(
        DecisionQuestion("route", "choose an action", ("retrieve documents", "answer"), min_confidence=0.0),
        state,
        features={"retrieve documents": 2.0, "answer": 0.1},
    )
    assert decision.selected_choice == "retrieve documents"
    assert decision.backend == "deterministic"
    assert decision.latency_ms >= 0.0
    assert 0.0 <= decision.uncertainty <= 1.0


def test_callable_system_one_backend_is_provider_neutral():
    state = CanonicalState()
    backend = CallableSystemOneBackend(
        lambda **_: {"fast": 0.9, "deep": 0.1},
        name="jev-compatible",
    )
    decision = SystemOneController(backend).decide(
        DecisionQuestion("model", "choose model", ("fast", "deep"), min_confidence=0.5),
        state,
    )
    assert decision.selected_choice == "fast"
    assert decision.backend == "jev-compatible"
    assert decision.confidence == 0.9


def test_system_one_can_route_only_capable_tools():
    state = CanonicalState(hard=HardState(authorizations=frozenset({"search"})))
    registry = ToolRegistry((
        ToolSpec("search", "search", risk=0.1),
        ToolSpec("admin", "admin", risk=0.0),
    ))
    decision = SystemOneController().choose_tool(state, registry)
    assert decision.selected_choice in {"search", "none"}
    assert "admin" not in decision.probabilities


def test_canonical_cce_can_use_system_one_as_selector_without_bypassing_governance():
    controller = SystemOneController()
    runtime = CanonicalCCERuntime(CanonicalState(), system_one=controller)
    tx = runtime.tick(TickInput(
        action_candidates=(
            ActionCandidate("safe", expected_utility=0.9, risk=0.1),
            ActionCandidate("weak", expected_utility=0.2, risk=0.2),
        ),
    ))
    assert tx.selected is not None
    assert tx.selected.action_id == "safe"


def test_nsa_runtime_can_run_system_one_heartbeats_without_generation():
    runtime = NSARuntime(
        backend=type("NoGenerationBackend", (), {"model": "unused", "generate": lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("generation called"))})(),
        config=RuntimeConfig(system_one_enabled=True),
    )
    tick = runtime.system_one_tick()
    assert tick is not None
    assert "escalation" in tick.decisions
    assert "memory_policy" in tick.decisions
    assert runtime.last_system_one_tick is tick


def test_continuous_maintenance_uses_system_one():
    runtime = NSARuntime(
        backend=type("NoGenerationBackend", (), {"model": "unused", "generate": lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("generation called"))})(),
        config=RuntimeConfig(system_one_enabled=True, continuous_enabled=True),
    )
    before = runtime.state.step
    assert runtime.continuous_tick()
    assert runtime.state.step > before
    assert runtime.last_system_one_tick is not None
