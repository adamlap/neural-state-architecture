"""Tests for CCE Composite Trajectory and Normative State Integration (Phase D)."""
from __future__ import annotations

import torch
from nsa.algebra import ConfidentialityLabel, IntegrityLabel
from nsa.core.state import HardState
from nsa.normative.state import NormativeState
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def test_composite_state_trajectory_creation_and_preservation():
    nu = NormativeState(values={"harm": 0.05, "sensitivity": 0.2}, confidence=0.95)
    cce = PersistentCognitiveState(dimension=4, initial_normative=nu)

    cce.update_memory("session_topic", 1.0)
    # Step continuous dynamics
    cce.observe(torch.tensor([0.1, 0.2, 0.3, 0.4]), dt=0.5)

    sigma_h = HardState(
        confidentiality=ConfidentialityLabel.TRUSTED,
        integrity=IntegrityLabel.TRUSTED,
        authorizations=frozenset(["read_metrics"]),
    )

    composite = cce.composite_state(sigma_h)
    assert composite.sigma_h == sigma_h
    assert composite.nu == nu
    assert composite.memory["session_topic"] == 1.0
    assert composite.cognitive.update_count == 1

    # Invariance check
    assert composite.is_hard_invariant_preserved(sigma_h) is True
