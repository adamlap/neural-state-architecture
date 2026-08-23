from __future__ import annotations

from dataclasses import dataclass

import torch

from nsa.runtime.cce_adapter import ContinuousSubstrateRuntime, SubstrateTransition


@dataclass
class Result:
    new_omega: object


class FakeSubstrate:
    def __init__(self) -> None:
        self.calls = 0
        self.last_state = None
        self.last_candidates = None

    def step(self, omega, candidates, *, user_clearance_limit, target_action_risk):
        self.calls += 1
        self.last_state = omega
        self.last_candidates = candidates
        return Result(new_omega=f"committed-{self.calls}")


def candidate_provider(state):
    return [("observe", torch.zeros(1, 2), 0.0, 0.0, True)]


def test_substrate_transition_returns_only_committed_state() -> None:
    substrate = FakeSubstrate()
    transition = SubstrateTransition(substrate, candidate_provider)

    assert transition("omega-0") == "committed-1"
    assert substrate.calls == 1
    assert substrate.last_state == "omega-0"
    assert len(substrate.last_candidates) == 1


def test_empty_candidate_set_fails_closed() -> None:
    substrate = FakeSubstrate()
    transition = SubstrateTransition(substrate, lambda _: [])

    try:
        transition("omega-0")
    except ValueError as exc:
        assert "no actions" in str(exc)
    else:
        raise AssertionError("empty candidate set must not reach the substrate")

    assert substrate.calls == 0


def test_continuous_runtime_is_disabled_by_default_and_commits_substrate_state() -> None:
    substrate = FakeSubstrate()
    runtime = ContinuousSubstrateRuntime(
        "omega-0",
        substrate,
        candidate_provider,
    )

    assert runtime.tick() is False
    assert runtime.state == "omega-0"
    assert substrate.calls == 0

    runtime.set_enabled(True)
    assert runtime.tick() is True
    assert runtime.state == "committed-1"
    assert substrate.calls == 1
    assert runtime.status().tick_count == 1


def test_continuous_runtime_stops_on_transition_failure() -> None:
    class FailingSubstrate:
        def step(self, *args, **kwargs):
            raise RuntimeError("kernel boundary failure")

    runtime = ContinuousSubstrateRuntime(
        "omega-0",
        FailingSubstrate(),
        candidate_provider,
        enabled=True,
    )

    assert runtime.tick() is False
    status = runtime.status()
    assert runtime.state == "omega-0"
    assert status.tick_count == 0
    assert status.enabled is False
    assert status.last_error == "RuntimeError: kernel boundary failure"
