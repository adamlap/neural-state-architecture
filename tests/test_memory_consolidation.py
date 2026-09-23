"""Tests for episodic-to-semantic memory consolidation."""
from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.trajectory import CognitiveTrajectory
from nsa.core.state import CanonicalState
from nsa.core.transition import TransitionReceipt
from nsa.memory.consolidation import MemoryConsolidator
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyPolicy
from nsa.residency.types import MemoryTier, NeuralRegion


def test_memory_consolidation_cycle():
    s0 = CanonicalState(step=0)
    trajectory = CognitiveTrajectory(s0)

    s1 = CanonicalState(step=1)
    r1 = TransitionReceipt(
        action_id="probe_service",
        source_digest="d0",
        target_digest="d1",
        proposal_digest="p1",
        committed=True,
        reason="ok",
        timestamp=1.0,
        step=1,
    )
    trajectory.append(s1, receipt=r1, events=(CognitiveEvent(EventKind.EXECUTION, 1, "e-1"),))

    s2 = CanonicalState(step=2)
    r2 = TransitionReceipt(
        action_id="probe_service",
        source_digest="d1",
        target_digest="d2",
        proposal_digest="p2",
        committed=True,
        reason="ok",
        timestamp=2.0,
        step=2,
    )
    trajectory.append(s2, receipt=r2, events=(CognitiveEvent(EventKind.EXECUTION, 2, "e-2"),))

    s3 = CanonicalState(step=3)
    r3 = TransitionReceipt(
        action_id="hazardous_cmd",
        source_digest="d2",
        target_digest="d2",
        proposal_digest="p3",
        committed=False,
        reason="rejected",
        timestamp=3.0,
        step=3,
    )
    trajectory.append(s3, receipt=r3, events=(CognitiveEvent(EventKind.ERROR, 3, "e-3"),))

    initial_hard = s3.hard

    mgr = NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=10*1024*1024, ram_budget_bytes=20*1024*1024))
    mgr.register([NeuralRegion("layer.0.attn", size_bytes=1024)])
    mgr.record_resident("layer.0.attn", MemoryTier.VRAM)

    consolidator = MemoryConsolidator(min_samples_for_rule=1)
    new_state, report = consolidator.consolidate(trajectory, s3, residency_manager=mgr)

    assert report.replayed_episodes >= 3
    assert len(report.induced_rules) >= 2

    probe_rule = next(r for r in report.induced_rules if r.action_id == "probe_service")
    assert probe_rule.commit_rate == 1.0
    assert probe_rule.sample_count == 2

    hazard_rule = next(r for r in report.induced_rules if r.action_id == "hazardous_cmd")
    assert hazard_rule.commit_rate == 0.0

    assert new_state.hard == initial_hard
    assert "consolidated_rules" in new_state.semantic.value
    assert "probe_service" in new_state.semantic.value["consolidated_rules"]