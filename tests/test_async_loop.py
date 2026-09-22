import asyncio

from nsa.cce import AsyncCognitiveLoop, AsyncCognitiveTransactionEngine
from nsa.core.state import CanonicalState
from nsa.cognition.interfaces import ActionCandidate, Prediction


def test_async_loop_commits_after_effect():
    engine = AsyncCognitiveTransactionEngine(CanonicalState())

    async def effect(action, state):
        await asyncio.sleep(0)
        return {"ok": True, "action": action.action_id}

    loop = AsyncCognitiveLoop(engine)
    result = asyncio.run(
        loop.cycle(
            {"temperature": 21},
            [ActionCandidate("observe", expected_utility=0.8, risk=0.1, reversible=True)],
            executor=effect,
            reason="test epistemic cycle",
        )
    )

    assert result.transaction.executed is True
    assert result.transaction.receipt.committed is True
    assert engine.state.step == 1


def test_async_loop_records_prediction_error():
    engine = AsyncCognitiveTransactionEngine(CanonicalState())

    class Predictor:
        def predict(self, belief, horizon=1):
            return Prediction(value=10, confidence=0.8, horizon=horizon)

    loop = AsyncCognitiveLoop(engine, predictor=Predictor())
    result = asyncio.run(loop.cycle(7, reason="prediction test"))

    assert result.prediction_error is not None
    assert result.prediction_error.magnitude == 3
    assert engine.state.soft.uncertainty == 1.0
