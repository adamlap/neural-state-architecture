import threading
import time

from nsa.cce import ContinuousCognitiveEngine


def test_set_state_during_a_step_is_not_overwritten_by_the_stale_result():
    """Regression: a restore that landed mid-step used to be clobbered by the step's output."""
    entered, release = threading.Event(), threading.Event()

    def step(state):
        entered.set()
        assert release.wait(5)
        return state + 1

    engine = ContinuousCognitiveEngine(10, step, enabled=True)
    result = []
    worker = threading.Thread(target=lambda: result.append(engine.tick()))
    worker.start()
    assert entered.wait(5)
    engine.set_state(500)          # e.g. restoring a checkpoint while a tick is in flight
    release.set()
    worker.join(5)

    assert engine.state == 500     # the restore wins; the stale transition is discarded
    assert result == [False]
    assert engine.status().tick_count == 0


def test_tick_commits_normally_when_nothing_intervenes():
    engine = ContinuousCognitiveEngine(1, lambda s: s + 1, enabled=True)
    assert engine.tick() and engine.tick()
    assert engine.state == 3 and engine.status().tick_count == 2


def test_stop_timeout_does_not_allow_a_second_loop_thread():
    release = threading.Event()
    started = threading.Event()

    def slow_step(state):
        started.set()
        release.wait(5)
        return state + 1

    engine = ContinuousCognitiveEngine(0, slow_step, interval_seconds=0.01, enabled=True)
    assert engine.start()
    assert started.wait(5)
    engine.stop(timeout=0.05)              # times out: the loop thread is still inside slow_step
    assert engine.start() is False         # must not spawn a second loop / clear the stop flag
    release.set()
    engine.stop(timeout=5)
    time.sleep(0.05)
    assert not engine.status().running
    threads = [t for t in threading.enumerate() if t.name == "nsa-cce"]
    assert threads == []


def test_checkpoint_store_concurrent_saves_stay_valid_and_leave_no_temp_files(tmp_path):
    from nsa.cce import StateCheckpointStore

    store = StateCheckpointStore(tmp_path / "state.json")
    errors = []

    def writer(n):
        try:
            for i in range(20):
                store.save({"writer": n, "i": i})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert store.load().state["i"] == 19          # integrity hash verifies: never a torn file
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_checkpoint_store_cleans_up_when_serialisation_fails(tmp_path):
    import pytest
    from nsa.cce import StateCheckpointStore

    store = StateCheckpointStore(tmp_path / "state.json")
    with pytest.raises(TypeError):
        store.save({"bad": object()})
    assert list(tmp_path.iterdir()) == []
