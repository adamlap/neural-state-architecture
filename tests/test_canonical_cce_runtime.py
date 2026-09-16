from nsa.cce import CanonicalCCERuntime, TickInput
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.state import CanonicalState


def test_runtime_routes_tick_through_transaction_engine():
    runtime = CanonicalCCERuntime(CanonicalState())
    tx = runtime.tick(TickInput(
        observation={"temperature": 20},
        semantic_update={"temperature": 20},
        soft_updates={"confidence": 0.8},
        provenance_source="sensor",
        action_candidates=(ActionCandidate("inspect", expected_utility=0.8),),
    ))
    assert tx is not None
    assert tx.receipt.committed
    assert runtime.state.step == 1
    assert runtime.trajectory.latest.state_digest == tx.receipt.target_digest


def test_runtime_scheduler_is_opt_in():
    runtime = CanonicalCCERuntime(CanonicalState(), enabled=False)
    assert not runtime.start()
    assert runtime.status().running is False
