from nsa import CanonicalState, EchoBackend, NSA, RuntimeConfig


def test_agent_persists_typed_state_and_history():
    agent = NSA(EchoBackend(), initial_state={"goal": "test"})
    result = agent.run("hello")
    assert not result.blocked
    assert result.text
    assert isinstance(result.state, CanonicalState)
    assert result.state.goals.active_goal == "test"
    assert len(agent.history) == 1
    assert len(agent.trace) == 1


def test_agent_can_disable_state_prompt_binding():
    agent = NSA(EchoBackend(), config=RuntimeConfig(include_state_in_prompt=False))
    assert agent.run("hello").text == "hello"


def test_agent_observation_updates_uncertainty():
    agent = NSA(EchoBackend())
    agent.observe("sensor", source="sensor", confidence=0.75)
    assert agent.state.soft.confidence == 0.75
    assert agent.state.soft.uncertainty == 0.25
