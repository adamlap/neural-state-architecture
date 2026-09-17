from pathlib import Path

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.persistence import TrajectoryJournal
from nsa.core.state import CanonicalState


def test_cognitive_event_kinds_cover_epistemic_chain():
    assert EventKind.INFORMATION_NEED.value == "information_need"
    assert EventKind.DELIBERATION.value == "deliberation"
    assert EventKind.LEARNING_UPDATE.value == "learning_update"


def test_trajectory_journal_recovers_latest_state(tmp_path: Path):
    state=CanonicalState()
    journal=TrajectoryJournal(tmp_path/"trajectory.jsonl")
    from nsa.cce.trajectory import CognitiveTrajectory
    from nsa.core.transition import TransitionReceipt, state_digest
    trajectory=CognitiveTrajectory(state)
    receipt=TransitionReceipt("test",state_digest(state),state_digest(state),"0"*64,False,"test",0.0,state.step)
    trajectory.append(state,receipt=receipt,events=[CognitiveEvent(EventKind.LEARNING_UPDATE,0,"learning-0",{"verified":True})])
    journal.append(trajectory.latest,state=state)
    restored=journal.latest_state()
    assert restored is not None
    assert restored.step == state.step
    assert journal.verify()[0]
