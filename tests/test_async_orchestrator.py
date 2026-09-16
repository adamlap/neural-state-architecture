import asyncio

from nsa.cce import AsyncCognitiveOrchestrator, AsyncCognitiveTransactionEngine
from nsa.cognition import ActionCandidate, CognitiveContext, CognitiveProposal
from nsa.core.state import CanonicalState


class FakeModel:
    async def propose(self, context):
        return CognitiveProposal(
            action_candidates=(
                ActionCandidate(
                    action_id="inspect",
                    payload={"target": "demo"},
                    expected_utility=0.9,
                    risk=0.1,
                    reversible=True,
                ),
            ),
            rationale="inspect evidence",
            confidence=0.8,
            metadata={"provider": "test"},
        )


def test_async_orchestrator_commits_only_after_effect():
    async def run():
        runtime = AsyncCognitiveTransactionEngine(CanonicalState())
        calls = []

        async def effect(action, state):
            await asyncio.sleep(0)
            calls.append(action.action_id)
            return {"ok": True}

        result = await AsyncCognitiveOrchestrator(runtime, FakeModel()).cycle(
            CognitiveContext(state=runtime.state, observations=("evidence",)),
            observation={"kind": "test"},
            executor=effect,
        )
        assert calls == ["inspect"]
        assert result.transaction.executed is True
        assert result.transaction.receipt.committed is True
        assert runtime.state.step == 1

    asyncio.run(run())


def test_async_orchestrator_does_not_commit_failed_effect():
    async def run():
        runtime = AsyncCognitiveTransactionEngine(CanonicalState())

        async def effect(action, state):
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        result = await AsyncCognitiveOrchestrator(runtime, FakeModel()).cycle(
            CognitiveContext(state=runtime.state),
            executor=effect,
        )
        assert result.transaction.executed is False
        assert result.transaction.receipt.committed is False
        assert runtime.state.step == 0

    asyncio.run(run())
