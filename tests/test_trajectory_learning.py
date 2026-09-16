from nsa.cce import CognitiveTrajectory, CognitiveTransactionEngine, TrajectoryLearner
from nsa.cognition import ActionCandidate
from nsa.core.state import CanonicalState


def test_trajectory_learner_preserves_causal_transition_and_outcome():
    engine = CognitiveTransactionEngine(CanonicalState())
    engine.tick(
        action_candidates=(ActionCandidate("inspect", expected_utility=0.8, risk=0.1),),
        semantic_update={"observed": True},
    )

    learner = TrajectoryLearner()
    examples = learner.examples(engine.trajectory)
    assert len(examples) == 1
    assert examples[0].step == 1
    assert examples[0].previous_digest != examples[0].next_digest
    assert examples[0].action_id == "inspect"
    assert examples[0].committed is True
    assert examples[0].outcome == "committed"
    assert "action_proposal" in examples[0].events

    outcomes = learner.action_outcomes(engine.trajectory)
    assert outcomes["inspect"]["attempts"] == 1.0
    assert outcomes["inspect"]["commit_rate"] == 1.0


def test_trajectory_learner_can_flatten_multiple_trajectories():
    first = CognitiveTrajectory(CanonicalState())
    second = CognitiveTrajectory(CanonicalState())
    learner = TrajectoryLearner()
    assert learner.training_records((first, second)) == ()
