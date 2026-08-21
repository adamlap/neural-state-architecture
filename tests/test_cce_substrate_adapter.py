from types import SimpleNamespace

import torch

from nsa.runtime.cce_adapter import SubstrateTransition, SubstrateTransitionConfig


def test_adapter_delegates_one_authoritative_transition() -> None:
    omega = object()
    next_omega = object()
    action = ("noop", torch.zeros(1), 0.1, 0.1, True)
    calls = []

    class FakeSubstrate:
        def step(self, received_omega, candidates, *, user_clearance_limit, target_action_risk):
            calls.append(
                (received_omega, candidates, user_clearance_limit, target_action_risk)
            )
            return SimpleNamespace(new_omega=next_omega)

    adapter = SubstrateTransition(
        FakeSubstrate(),
        lambda received_omega: [action],
        config=SubstrateTransitionConfig(
            user_clearance_limit=0.25,
            target_action_risk=0.75,
        ),
    )

    assert adapter(omega) is next_omega
    assert calls == [(omega, [action], 0.25, 0.75)]


def test_adapter_rejects_empty_candidate_sets_before_substrate() -> None:
    class FailingSubstrate:
        def step(self, *args, **kwargs):
            raise AssertionError("substrate must not run without candidates")

    adapter = SubstrateTransition(FailingSubstrate(), lambda _: [])

    try:
        adapter(object())
    except ValueError as exc:
        assert str(exc) == "candidate_provider returned no actions"
    else:
        raise AssertionError("expected ValueError")
