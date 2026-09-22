"""Regression tests for capability handling in the governed transaction engines."""
import asyncio
import threading
import warnings

import pytest

from nsa.cce import AsyncCognitiveTransactionEngine, CognitiveTransactionEngine
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityAuthority, TrustTier
from nsa.core.state import CanonicalState

SECRET = b"unit-test-secret"


def _authority():
    return CapabilityAuthority(SECRET)


def _two_capability_action():
    return ActionCandidate("deploy", expected_utility=1.0, risk=0.1, reversible=True,
                           required_capabilities=("deploy.write", "deploy.read"))


def test_one_token_can_cover_several_required_capabilities_sync():
    authority = _authority()
    token = authority.mint_capability("agent", "deploy", "ops", TrustTier.T2_REVERSIBLE)
    engine = CognitiveTransactionEngine(
        CanonicalState(), capability_authority=authority, capability_tokens={"deploy": token},
    )
    result = engine.tick(action_candidates=[_two_capability_action()])
    assert result.receipt.committed, result.receipt.reason


def test_one_token_can_cover_several_required_capabilities_async():
    authority = _authority()
    token = authority.mint_capability("agent", "deploy", "ops", TrustTier.T2_REVERSIBLE)
    engine = AsyncCognitiveTransactionEngine(
        CanonicalState(), capability_authority=authority, capability_tokens={"deploy": token},
    )

    async def effect(action, state):
        return "done"

    result = asyncio.run(engine.tick_async(action_candidates=[_two_capability_action()], executor=effect))
    assert result.receipt.committed, result.receipt.reason


def test_cancelled_effect_releases_the_capability_reservation():
    """Regression: CancelledError is a BaseException, so reservations leaked forever."""
    authority = _authority()
    token = authority.mint_capability("agent", "act", "ops", TrustTier.T2_REVERSIBLE)
    engine = AsyncCognitiveTransactionEngine(
        CanonicalState(), capability_authority=authority, capability_tokens={"act": token},
    )
    action = ActionCandidate("act", expected_utility=1.0, risk=0.1, required_capabilities=("act.run",))

    async def scenario():
        started = asyncio.Event()

        async def slow(action, state):
            started.set()
            await asyncio.sleep(60)

        task = asyncio.ensure_future(engine.tick_async(action_candidates=[action], executor=slow))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async def quick(action, state):
            return "ok"

        return await engine.tick_async(action_candidates=[action], executor=quick)

    retry = asyncio.run(scenario())
    assert retry.receipt.committed, retry.receipt.reason  # token was not burned by the cancellation


def test_transaction_reports_executed_when_effect_ran_but_consumption_failed():
    """The audit record must not claim 'not executed' for an effect that happened."""
    authority = _authority()
    token = authority.mint_capability("agent", "act", "ops", TrustTier.T2_REVERSIBLE)
    engine = AsyncCognitiveTransactionEngine(
        CanonicalState(), capability_authority=authority, capability_tokens={"act": token},
    )
    action = ActionCandidate("act", expected_utility=1.0, risk=0.1, required_capabilities=("act.run",))
    ran = []

    async def effect(action, state):
        ran.append(True)
        authority._reserved_nonces.clear()  # simulate the reservation being lost mid-effect
        return "side effect"

    result = asyncio.run(engine.tick_async(action_candidates=[action], executor=effect))
    assert ran and not result.receipt.committed
    assert result.executed is True
    assert result.execution_result == "side effect"


def test_minted_nonces_are_unique_and_unpredictable():
    authority = _authority()
    nonces = {authority.mint_capability("p", "a", "s", TrustTier.T1_INFO_GATHER).nonce for _ in range(500)}
    assert len(nonces) == 500


def test_a_token_can_only_be_consumed_once_across_threads():
    authority = _authority()
    token = authority.mint_capability("p", "a", "s", TrustTier.T2_REVERSIBLE)
    wins, barrier = [], threading.Barrier(16)

    def attempt():
        barrier.wait()
        ok, _ = authority.verify_and_consume_capability(token, "a", TrustTier.T2_REVERSIBLE)
        if ok:
            wins.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(wins) == 1


def test_default_demo_secret_warns():
    with pytest.warns(RuntimeWarning, match="demo secret"):
        CapabilityAuthority()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CapabilityAuthority(b"explicit-secret")  # explicit key: no warning


def test_library_defaults_do_not_use_the_public_demo_secret():
    from nsa.core.safety_kernel import ImmutableSafetyKernel
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kernel = ImmutableSafetyKernel()
    forged = CapabilityAuthority.DEMO_SECRET
    token = CapabilityAuthority(forged).mint_capability("p", "a", "s", TrustTier.T2_REVERSIBLE)
    ok, _ = kernel.capability_authority.verify_capability(token, "a", TrustTier.T2_REVERSIBLE)
    assert not ok  # a token signed with the well-known secret is rejected
