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


class _Recorder:
    """Backend that records every prompt it receives."""

    model = "recorder"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, state=None):
        self.prompts.append(prompt)
        return "VALUE=X"


def test_history_limit_shows_that_many_real_observations_and_the_query_once():
    """Regression: step() recorded the query first, so history_limit=6 showed only 5
    real observations and the query consumed a slot."""
    backend = _Recorder()
    agent = NSA(backend, config=RuntimeConfig(history_limit=6))
    for i in range(10):
        agent.observe(f"OBS_{i}", source="t")
    agent.step("QUERY_TEXT")
    prompt = backend.prompts[-1]
    observed = prompt.split("RECENT_OBSERVATIONS=")[1].split("USER_INPUT=")[0]
    assert [f"OBS_{i}" in observed for i in range(4, 10)] == [True] * 6
    assert "OBS_3" not in observed
    assert "QUERY_TEXT" not in observed
    assert prompt.count("QUERY_TEXT") == 1
    assert len(agent.history) == 11  # the query is still recorded in history/state


def test_history_only_control_sees_the_same_window_without_any_state_text():
    """Regression: include_state_in_prompt=False silently dropped history too, so the
    'no state' control was really 'no context'."""
    backend = _Recorder()
    config = RuntimeConfig(history_limit=3, include_state_in_prompt=False, include_history_in_prompt=True)
    agent = NSA(backend, config=config)
    for i in range(5):
        agent.observe(f"OBS_{i}", source="t")
    agent.step("QUERY_TEXT")
    prompt = backend.prompts[-1]
    assert "OBS_4" in prompt and "OBS_2" in prompt and "OBS_1" not in prompt
    assert "CURRENT_STATE" not in prompt and "COGNITIVE_STATE" not in prompt
    assert prompt.endswith("USER_INPUT=QUERY_TEXT")


def test_default_prompt_behaviour_is_unchanged_when_history_flag_is_left_unset():
    backend = _Recorder()
    agent = NSA(backend, config=RuntimeConfig(include_state_in_prompt=False))
    agent.observe("OBS_0")
    agent.step("hello")
    assert backend.prompts[-1] == "hello"  # no state and no history, as before


def test_state_prompt_size_does_not_grow_with_the_number_of_observations():
    """Regression: CURRENT_STATE embedded every provenance source ever recorded."""
    sizes = {}
    for n in (10, 1000):
        backend = _Recorder()
        agent = NSA(backend, config=RuntimeConfig(history_limit=6))
        for i in range(n):
            agent.observe(f"OBS_{i}", source="benchmark")
        agent.step("q")
        sizes[n] = len(backend.prompts[-1])
    assert sizes[1000] - sizes[10] < 100
    assert "'source_count': 1001" in backend.prompts[-1]


def test_retrieved_memory_is_rendered_with_real_newlines():
    """Regression: the memory block used an escaped \\n, so the model saw literal backslash-n."""
    backend = _Recorder()
    agent = NSA(backend, config=RuntimeConfig(memory_enabled=True, include_state_in_prompt=False))
    agent.remember("K", {"value": "V1"})
    agent.step("What is K?", memory_keys=["K"])
    prompt = backend.prompts[-1]
    assert "\\n" not in prompt
    assert prompt.startswith("RETRIEVED_MEMORY=\n- K: V1")
    assert prompt.endswith("\n\nWhat is K?")


def test_history_can_be_dropped_only_when_memory_is_attached():
    def run(with_history, memory_keys):
        backend = _Recorder()
        agent = NSA(backend, config=RuntimeConfig(memory_enabled=True, include_history_with_memory=with_history))
        agent.remember("K", {"value": "V1"})
        agent.observe("DISTRACTOR: J has value Z9.")
        agent.step("What is K?", memory_keys=memory_keys)
        return backend.prompts[-1]

    kept = run(True, ["K"])
    assert "DISTRACTOR: J" in kept and "RETRIEVED_MEMORY=" in kept
    dropped = run(False, ["K"])
    assert "DISTRACTOR: J" not in dropped and "RETRIEVED_MEMORY=\n- K: V1" in dropped
    assert "CURRENT_STATE=" in dropped                      # state text is unaffected
    assert "DISTRACTOR: J" in run(False, [])                # no memory attached -> history stays


def test_auto_checkpoint_writes_a_format_load_state_can_read(tmp_path):
    """Regression: auto_checkpoint wrote the lossy snapshot(); load_state() then raised
    'unsupported canonical state schema: None'."""
    from nsa.cce import StateCheckpointStore
    agent = NSA(EchoBackend(), checkpoint=StateCheckpointStore(tmp_path / "s.json"),
                config=RuntimeConfig(auto_checkpoint=True))
    agent.step("hello")
    saved = agent.state
    agent.observe("later", confidence=0.1)
    assert agent.load_state() == saved


def test_backend_errors_are_not_retried_even_if_their_message_mentions_state():
    """Regression: a TypeError mentioning 'state' re-invoked generate() a second time."""
    calls = []

    class Broken:
        model = "broken"

        def generate(self, prompt, *, state=None):
            calls.append(1)
            raise TypeError("unsupported operand for bad state")

    import pytest
    with pytest.raises(TypeError):
        NSA(Broken()).step("x")
    assert len(calls) == 1


def test_backends_without_a_state_parameter_still_work():
    class Plain:
        model = "plain"

        def generate(self, prompt):
            return "ok"

    assert NSA(Plain(), config=RuntimeConfig(include_state_in_prompt=False)).step("x").text == "ok"
