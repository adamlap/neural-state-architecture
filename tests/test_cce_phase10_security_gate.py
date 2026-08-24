"""Tests for ContinuousHardAuthorityMonitor & Security Invariants (Phase CCE-10)."""
from __future__ import annotations

import pytest
import torch
from nsa.algebra import ConfidentialityLabel, IntegrityLabel
from nsa.core.state import HardState
from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.cce_security_monitor import ContinuousHardAuthorityMonitor


def test_continuous_security_monitor_preserves_hard_state():
    baseline = HardState(confidentiality=ConfidentialityLabel.CONFIDENTIAL, integrity=IntegrityLabel.TRUSTED)
    monitor = ContinuousHardAuthorityMonitor(baseline_hard_state=baseline)

    state = PersistentCognitiveState(dimension=4)
    snap = state.observe(torch.tensor([0.1, -0.2, 0.3, -0.4]), dt=0.5)

    rec = monitor.verify_tick(baseline, snap)
    assert rec.violation_detected is False
    assert monitor.total_violations == 0

    # Tampered hard authority (lowered confidentiality) -> MUST RAISE PermissionError
    tampered = HardState(confidentiality=ConfidentialityLabel.PUBLIC, integrity=IntegrityLabel.TRUSTED)
    with pytest.raises(PermissionError) as exc:
        monitor.verify_tick(tampered, snap)

    assert "Hard authority state mutated" in str(exc.value)
    assert monitor.total_violations == 1
