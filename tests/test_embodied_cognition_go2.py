from nsa.cognition.embodied import ActiveCognition, ActiveCognitionState

def test_information_gain_drives_selection():
    scores = ActiveCognition().score_actions(["observe", "respond"], uncertainty=.9, information_gain={"observe":.9,"respond":.1}, expected_utility={"respond":.5}, risk={"respond":.1})
    assert scores[0][0] == "observe"

def test_transition_updates_homeostasis_and_identity():
    state = ActiveCognition().transition(ActiveCognitionState(), uncertainty=.8, candidate_actions=["observe"], information_gain={"observe":.8}, observation="event")
    assert state.cycle == 1 and state.identity.age == 1
    assert state.identity.autobiographical == ("event",)
    assert 0 <= state.homeostasis.stability <= 1

def test_identity_persists_across_cycles():
    active=ActiveCognition(); state=ActiveCognitionState()
    for i in range(5): state=active.transition(state, uncertainty=.2, observation=f"event-{i}")
    assert state.identity.age == 5 and len(state.identity.autobiographical) == 5
