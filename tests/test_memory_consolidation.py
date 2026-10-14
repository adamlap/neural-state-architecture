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

def test_consolidation_does_not_teach_the_residency_predictor_about_nonexistent_regions():
    """Regression: action ids were fed to observe_state_tags() as if they were region ids."""
    from nsa.cce.trajectory import CognitiveTrajectory
    from nsa.core.state import CanonicalState
    from nsa.memory import MemoryConsolidator
    from nsa.residency import NeuralRegion, NeuralResidencyManager, ResidencyPolicy

    class Recorder:
        def __init__(self):
            self.seen = []

        def predict(self, regions, state, current_region=None):
            return {}

        def observe_state_tags(self, tags, region_id):
            self.seen.append((tuple(tags), region_id))

    from nsa.cce.events import CognitiveEvent, EventKind
    from nsa.core.transition import TransitionReceipt

    state = CanonicalState()
    trajectory = CognitiveTrajectory(state)
    for _ in range(3):  # enough committed samples that a rule (and the old feed) is induced
        receipt = TransitionReceipt("act", "s", "t", "p", True, "ok", 0.0, 0)
        trajectory.append(state, receipt=receipt, events=(CognitiveEvent(EventKind.OBSERVATION, 0, "e", {}),))
    manager = NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100, ram_budget_bytes=100), predictor=Recorder())
    manager.register([NeuralRegion("layer.0", size_bytes=10)])
    _, report = MemoryConsolidator().consolidate(trajectory, state, manager)
    assert [rule.action_id for rule in report.induced_rules] == ["act"]  # the rule is induced...
    assert manager.predictor.seen == []  # ...but never sent to the region predictor
