"""Tests for the live capability benchmark's deterministic helper logic.

These do not require a running Ollama instance (matching the project's
convention for testing live-model harnesses: `tests/test_live_ollama_benchmark.py`
exercises pure logic against a deterministic mock backend rather than requiring
live network access in the default `make test` / `pytest tests/` run). Live
replication itself is run manually via
`make live-capability-benchmark` (requires a local Ollama server).
"""
from __future__ import annotations

from typing import List

from experiments.live.cce_live_capability_benchmark import (
    CONDITIONS,
    _build_prompt,
    _is_observation_turn,
    _parse_number,
    run,
    run_episode,
)
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class ScriptedBackend(InferenceBackend):
    """Returns a scripted sequence of responses, one per call, cycling if exhausted."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.model_name = "scripted-mock"

    def generate(self, prompt, max_tokens=256, temperature=0.7, extract_hidden=False) -> LLMGenerationOutput:
        text = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return LLMGenerationOutput(text=text, tokens=[], confidence_estimate=1.0)

    def propose_action(self, system_context, task_instruction, available_tools):
        return {"action": available_tools[0]["name"]} if available_tools else {"action": "none"}


def test_parse_number_extracts_leading_numeric_value():
    assert _parse_number("51.15") == 51.15
    assert _parse_number("  -3.5 is my guess") == -3.5
    assert _parse_number("no numbers here") is None


def test_observation_schedule_has_an_anchor_window_and_a_blackout():
    observed = [_is_observation_turn(t) for t in range(12)]
    assert observed[:5] == [True] * 5
    assert observed[5:10] == [False] * 5


def test_build_prompt_differs_by_condition_and_reflects_missing_observation():
    prompt = _build_prompt("stateless", None, [], 0.0, 0.0, 0.0)
    assert "No observation is available this turn" in prompt
    prompt = _build_prompt("predictive_cce", None, [], 0.0, 42.0, 1.5)
    assert "42.00" in prompt and "1.500" in prompt


def test_run_episode_reports_perfect_accuracy_when_the_model_echoes_ground_truth():
    class EchoBackend(InferenceBackend):
        model_name = "echo-mock"

        def generate(self, prompt, max_tokens=256, temperature=0.7, extract_hidden=False):
            # The prompt always contains the ground-truth observation for the
            # stateless condition on observed turns; echoing it back should
            # score near-perfectly on those turns.
            import re
            match = re.search(r"observation: (-?\d+\.\d+)", prompt)
            text = match.group(1) if match else "0"
            return LLMGenerationOutput(text=text, tokens=[], confidence_estimate=1.0)

        def propose_action(self, system_context, task_instruction, available_tools):
            return {"action": "none"}

    episode = run_episode(7, "stateless", EchoBackend(), horizon=5, max_tokens=12, temperature=0.0)
    observed_turns = [t for t in episode.turns if t.observed]
    assert observed_turns
    # Echoing the shown observation back should be close to true_value, bounded
    # only by the environment's own observation noise (+/- 1.2), not exact.
    assert all(t.error < 1.3 for t in observed_turns)


def test_run_episode_caps_error_for_implausible_canary_style_answers():
    backend = ScriptedBackend(["12345"])
    episode = run_episode(7, "persistent_cce", backend, horizon=3, max_tokens=12, temperature=0.0)
    assert episode.implausible_answers == 3
    assert all(t.error == 12.0 for t in episode.turns)


def test_run_reports_all_four_conditions_and_gate_keys():
    backend = ScriptedBackend(["50", "51", "52"])
    report = run([7], model="scripted", horizon=5, backend=backend)
    assert tuple(report["conditions"]) == CONDITIONS
    assert set(report["aggregates"]) == set(CONDITIONS)
    expected_gates = {
        "persistent_beats_stateless",
        "predictive_beats_persistent",
        "predictive_beats_raw_context",
        "predictive_beats_stateless",
    }
    assert set(report["gates"]) == expected_gates
    assert report["status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
