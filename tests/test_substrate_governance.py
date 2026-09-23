"""Tests for Governed Substrate Flow Engine."""
import pytest
from nsa.core import CanonicalState, GovernedSubstrateGovernor, HardState, SubstrateState
from nsa.residency.types import MemoryTier, ResidencyEvent, ResidencySnapshot


def test_governed_substrate_lifecycle():
    initial = CanonicalState()
    governor = GovernedSubstrateGovernor(initial)

    snapshot = ResidencySnapshot(
        bytes_by_tier={MemoryTier.VRAM: 1024, MemoryTier.RAM: 2048, MemoryTier.NVME: 4096}
    )

    state = governor.commit_step(
        active_regions=["layer.0.attn", "layer.0.mlp"],
        residency_snapshot=snapshot,
        reason="step-1",
    )

    assert isinstance(state, SubstrateState)
    assert state.step == 1
    assert state.active_regions == ("layer.0.attn", "layer.0.mlp")
    assert state.bytes_by_tier["vram"] == 1024
    assert state.bytes_by_tier["ram"] == 2048
    assert "step-1" in state.canonical.provenance.evidence_ids[0]


def test_residency_invariant_cannot_grant_authority():
    governor = GovernedSubstrateGovernor()
    hard = HardState()

    # Valid physical memory residency event
    valid_event = ResidencyEvent(
        timestamp=1.0,
        region_id="layer.0.expert.1",
        action="resident",
        source=MemoryTier.NVME,
        destination=MemoryTier.RAM,
        bytes_moved=1024,
    )
    assert governor.verify_residency_invariants(valid_event, hard) is True

    # Invalid event attempting to grant capability / bypass policy
    for forbidden_action in ("grant_capability", "elevate_privilege", "bypass_policy"):
        invalid_event = ResidencyEvent(
            timestamp=2.0,
            region_id="layer.0.expert.1",
            action=forbidden_action,
            source=MemoryTier.NVME,
            destination=MemoryTier.VRAM,
        )
        with pytest.raises(PermissionError):
            governor.verify_residency_invariants(invalid_event, hard)