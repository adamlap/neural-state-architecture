from __future__ import annotations

import time

from nsa.runtime.continuous_engine import ContinuousCognitiveEngine


def test_disabled_engine_has_no_ticks() -> None:
    calls = []
    engine = ContinuousCognitiveEngine(0, lambda state: calls.append(state) or state + 1)

    assert engine.start() is False
    assert engine.tick() is False
    assert engine.state == 0
    assert calls == []
    assert engine.status().tick_count == 0


def test_manual_tick_is_opt_in_and_stateful() -> None:
    engine = ContinuousCognitiveEngine(0, lambda state: state + 1, enabled=True)

    assert engine.tick() is True
    assert engine.tick() is True
    assert engine.state == 2
    status = engine.status()
    assert status.enabled is True
    assert status.tick_count == 2
    assert status.last_tick_monotonic is not None


def test_background_loop_can_be_started_and_stopped() -> None:
    engine = ContinuousCognitiveEngine(0, lambda state: state + 1, interval_seconds=0.01, enabled=True)

    assert engine.start() is True
    time.sleep(0.06)
    assert engine.stop(timeout=1.0) is True
    assert engine.state >= 2
    assert engine.status().running is False


def test_disabling_stops_future_ticks() -> None:
    engine = ContinuousCognitiveEngine(0, lambda state: state + 1, interval_seconds=0.01, enabled=True)

    engine.start()
    time.sleep(0.03)
    engine.set_enabled(False)
    frozen = engine.state
    time.sleep(0.03)

    assert engine.state == frozen
    assert engine.status().enabled is False
    assert engine.status().running is False


def test_transition_errors_are_contained_and_reported() -> None:
    def failing_step(_: int) -> int:
        raise RuntimeError("synthetic transition failure")

    engine = ContinuousCognitiveEngine(7, failing_step, enabled=True)

    assert engine.tick() is False
    assert engine.state == 7
    assert engine.status().tick_count == 0
    assert engine.status().last_error == "RuntimeError: synthetic transition failure"
