from nsa.runtime.continuous_engine import ContinuousCognitiveEngine


def test_transition_failure_disables_engine_by_default() -> None:
    def fail(_: int) -> int:
        raise RuntimeError("unsafe transition path")

    engine = ContinuousCognitiveEngine(10, fail, enabled=True)

    assert engine.tick() is False
    status = engine.status()
    assert status.enabled is False
    assert status.running is False
    assert status.tick_count == 0
    assert engine.state == 10
    assert status.last_error == "RuntimeError: unsafe transition path"


def test_non_fail_closed_mode_can_be_used_for_research() -> None:
    calls = 0

    def fail_once(state: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return state + 1

    engine = ContinuousCognitiveEngine(10, fail_once, enabled=True, fail_closed=False)

    assert engine.tick() is False
    assert engine.status().enabled is True
    assert engine.tick() is True
    assert engine.state == 11
