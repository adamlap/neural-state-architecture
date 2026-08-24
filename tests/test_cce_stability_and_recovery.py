"""Tests for CCE Stability Monitor, Perturbation Recovery, and Drift (Phase CCE-4)."""
from __future__ import annotations

import torch
from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.cce_stability import CCEStabilityMonitor


def test_stability_monitor_bounds_and_drift_detection():
    state = PersistentCognitiveState(dimension=4)
    monitor = CCEStabilityMonitor(max_bound=2.0, max_drift_rate=100.0)

    # Within bound
    snap1 = state.observe(torch.tensor([0.5, 0.5, 0.5, 0.5]), dt=0.5)
    m1 = monitor.check_and_record(snap1)
    assert m1.is_bounded is True
    assert m1.anomaly_detected is False

    # Exceeding bound
    snap2 = state.observe(torch.tensor([5.0, 5.0, 5.0, 5.0]), dt=0.5)
    m2 = monitor.check_and_record(snap2)
    assert m2.is_bounded is False
    assert m2.anomaly_detected is True
    assert "State norm exceeded bound" in str(m2.anomaly_reason)


def test_stability_monitor_perturbation_recovery():
    state = PersistentCognitiveState(dimension=4, decay=0.15, learning_rate=0.5)
    monitor = CCEStabilityMonitor()
    res = monitor.measure_perturbation_recovery(state, torch.tensor([1.0, 1.0, 1.0, 1.0]), dt_step=0.1, max_steps=50)
    assert res["recovered"] is True
    assert res["relaxation_steps_to_halflife"] <= 30
