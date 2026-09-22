from types import SimpleNamespace

import pytest

from nsa.cce import TwoPhaseExecutor


class Effect:
    def __init__(self, commit_error=None, abort_error=None):
        self.commit_error, self.abort_error, self.aborted = commit_error, abort_error, False

    def prepare(self, action, state):
        return "prepared"

    def commit(self, prepared):
        if self.commit_error:
            raise self.commit_error
        return {"done": prepared}

    def abort(self, prepared):
        self.aborted = True
        if self.abort_error:
            raise self.abort_error

    def compensate(self, result):
        pass


ACTION = SimpleNamespace(action_id="act")


def test_commit_failure_aborts_and_reports():
    effect = Effect(commit_error=RuntimeError("boom"))
    executor = TwoPhaseExecutor(effect)
    effect_id, prepared = executor.prepare(ACTION, None)
    receipt = executor.commit(effect_id, prepared, ACTION)
    assert effect.aborted and not receipt.committed
    assert receipt.reason == "effect commit failed: RuntimeError: boom"


def test_abort_failure_is_recorded_not_swallowed_silently():
    effect = Effect(commit_error=RuntimeError("boom"), abort_error=OSError("disk"))
    executor = TwoPhaseExecutor(effect)
    receipt = executor.commit("e1", "p", ACTION)
    assert not receipt.committed
    assert "boom" in receipt.reason and "abort failed: OSError: disk" in receipt.reason


def test_keyboard_interrupt_during_abort_is_not_swallowed():
    """Regression: `return` inside `finally` used to discard BaseExceptions from abort()."""
    effect = Effect(commit_error=RuntimeError("boom"), abort_error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        TwoPhaseExecutor(effect).commit("e1", "p", ACTION)


def test_success_path_returns_committed_receipt():
    receipt = TwoPhaseExecutor(Effect()).commit("e1", "p", ACTION)
    assert receipt.committed and receipt.result == {"done": "p"}
