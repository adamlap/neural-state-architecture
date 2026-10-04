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


def test_public_cce_is_composed_by_nsa_runtime():
    agent = NSA(EchoBackend(), config=RuntimeConfig(continuous_enabled=True))
    assert agent.continuous_status().enabled
    before = agent.state.step
    assert agent.continuous_tick()
    assert agent.state.step == before + 1
    assert agent.continuous_status().tick_count == 1


def test_public_cce_can_use_authoritative_transition_callback():
    seen = []

    def transition(state):
        seen.append(state.step)
        return state

    agent = NSA(EchoBackend(), config=RuntimeConfig(continuous_enabled=True), continuous_transition=transition)
    assert agent.continuous_tick()
    assert seen == [0]
    assert agent.continuous_status().tick_count == 1


def test_agent_save_load_is_lossless(tmp_path):
    from nsa.cce import StateCheckpointStore
    agent = NSA(EchoBackend(), checkpoint=StateCheckpointStore(tmp_path / "state.json"))
    agent.observe({"nested": [1, 2]}, source="test", confidence=0.8)
    before = agent.state
    agent.save()
    agent.observe("changed", confidence=0.2)
    restored = agent.load_state()
    assert restored == before


def test_agent_trace_records_are_snapshot_only():
    agent = NSA(EchoBackend())
    agent.run("hello")
    records = agent.trace_records()
    assert records[0]["blocked"] is False
    records[0]["prompt"] = "changed"
    assert agent.trace[0]["prompt"] == "hello"
