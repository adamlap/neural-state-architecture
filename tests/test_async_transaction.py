from __future__ import annotations

import asyncio

from nsa.cce import AsyncCognitiveTransactionEngine
from nsa.cognition import ActionCandidate
from nsa.core.state import CanonicalState


def test_async_effect_commits_only_after_success():
    engine = AsyncCognitiveTransactionEngine(CanonicalState())

    async def execute(action, state):
        await asyncio.sleep(0)
        return {"ok": True, "action": action.action_id}

    result = asyncio.run(
        engine.tick_async(
            action_candidates=(ActionCandidate("observe", expected_utility=0.8, risk=0.1),),
            semantic_update={"result": "success"},
            executor=execute,
        )
    )

    assert result.executed is True
    assert result.execution_result["ok"] is True
    assert result.receipt.committed is True
    assert engine.state.step == 1


def test_async_effect_failure_does_not_commit_state():
    engine = AsyncCognitiveTransactionEngine(CanonicalState())

    async def execute(action, state):
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    result = asyncio.run(
        engine.tick_async(
            action_candidates=(ActionCandidate("danger", expected_utility=0.8, risk=0.1),),
            semantic_update={"must_not_commit": True},
            executor=execute,
        )
    )

    assert result.executed is False
    assert result.receipt.committed is False
    assert engine.state.step == 0
