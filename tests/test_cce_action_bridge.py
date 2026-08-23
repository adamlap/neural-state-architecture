"""Unit tests for CCE Action Bridge and Immutable Safety Kernel integration."""

from __future__ import annotations

import pytest
import torch

from nsa.core.capabilities import TrustTier
from nsa.core.omega import ProvenanceRecord, TemporalHorizonState, TeleologicalState, UnifiedCognitiveState
from nsa.core.safety_kernel import ImmutableSafetyKernel
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.runtime.cce_action_bridge import ActionProposal, CCEActionBridge


def make_test_omega(clearance_tier: TrustTier = TrustTier.T1_INFO_GATHER) -> UnifiedCognitiveState:
    return UnifiedCognitiveState(
        semantic_state=torch.zeros(1, 8),
        operational_self_state=torch.zeros(1, 8),
        epistemic_state=EpistemicVector(0.8, 0.2, 1.0, 1.0, 1.0, 1.0, 0.9, EpistemicTier.EMPIRICALLY_VALIDATED),
        authority_state=torch.tensor([float(clearance_tier.value) / 4.0]),
        provenance_state=ProvenanceRecord("prov-test", "test://cce", "hash-0", 1.0),
        temporal_state=TemporalHorizonState(1, 100, 0.1, "ckpt-1"),
        goal_state=TeleologicalState("cce_action_test", 0.9, 0.0, True),
    )


def test_action_bridge_authorized_commit():
    bridge = CCEActionBridge()
    executed = []

    def mock_probe_service(params):
        executed.append(params)
        return {"status": "success", "result": "config_valid"}

    bridge.register_handler("probe_service_config", mock_probe_service)

    proposal = ActionProposal(
        action_name="probe_service_config",
        target="cluster_service",
        parameters={"service": "auth"},
        requested_risk=0.1,
        confidence=0.95,
    )

    omega = make_test_omega(TrustTier.T1_INFO_GATHER)
    record = bridge.evaluate_and_execute(
        proposal,
        omega,
        required_tier=TrustTier.T1_INFO_GATHER,
        user_clearance=TrustTier.T1_INFO_GATHER,
    )

    assert record.allowed is True
    assert record.verdict == "COMMIT"
    assert len(executed) == 1
    assert executed[0]["service"] == "auth"
    assert record.execution_result["status"] == "success"
    assert len(bridge.audit_log) == 1


def test_action_bridge_clearance_rejection():
    bridge = CCEActionBridge()
    executed = []

    def mock_fatal_mutation(params):
        executed.append(params)
        return {"status": "mutated"}

    bridge.register_handler("rm_rf_root_system", mock_fatal_mutation)

    proposal = ActionProposal(
        action_name="rm_rf_root_system",
        target="root",
        parameters={"force": True},
        requested_risk=0.9,
        confidence=0.8,
    )

    # User only has T1 clearance, but proposal requires T4 critical mutation
    omega = make_test_omega(TrustTier.T1_INFO_GATHER)
    record = bridge.evaluate_and_execute(
        proposal,
        omega,
        required_tier=TrustTier.T4_CRITICAL,
        user_clearance=TrustTier.T1_INFO_GATHER,
    )

    assert record.allowed is False
    assert record.verdict == "REJECT"
    assert len(executed) == 0  # Crucial: Handler was never invoked!
    assert "rejection_reason" in record.to_dict() and record.rejection_reason is not None
    assert len(bridge.audit_log) == 1
