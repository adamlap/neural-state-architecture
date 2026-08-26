"""Live four-way matched cognitive capability benchmark (Ollama).

Bridges the deterministic cognitive benchmark suite (``experiments/cognitive/``)
to a real, live language model. All four conditions share the same model,
prompts, temperature and token budget (matched compute); they differ only in
what state context the trusted runtime provides:

- ``stateless``   -- only this turn's raw observation (if any). No history.
- ``raw_context``  -- the full raw observation transcript so far. The model
  must do its own drift-tracking/extrapolation from an unfiltered transcript.
- ``persistent_cce`` -- a runtime-maintained, Kalman-filtered point estimate
  with no velocity/dynamics term (``a=1``): explicit state, not predictive.
- ``predictive_cce`` -- a runtime-maintained position+velocity Kalman estimate
  (the same ``experiments.cognitive._kalman.ConstantVelocityKalman`` validated
  by the deterministic retention benchmark): explicit *predictive* state.

Every turn, the model's only job is to report its best-guess number for the
current hidden value. The trusted runtime never asks it to do arithmetic over a
raw transcript except in the ``raw_context`` condition -- that is the point
under test: does offloading estimation to a small validated filter beat asking
the model to re-derive it from context every turn?

This is a live-model replication of the deterministic retention benchmark's
``hidden_state`` task (see ``experiments/cognitive/benchmark.py``), checking
whether that result holds as a measurable capability difference for a real
model rather than only in a closed-form simulation.

Scientific boundary: this measures single-model, single-task, CPU-only,
small-sample behavior. It does not establish general superiority across model
families, tasks, or scales. Effect sizes/uncertainty should be read accordingly,
and a null or mixed result is a valid, reportable outcome.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from experiments.cognitive._kalman import ConstantVelocityKalman, ScalarKalman
from nsa.runtime.inference.base import InferenceBackend
from nsa.runtime.inference.ollama import OllamaInferenceBackend

CONDITIONS = ("stateless", "raw_context", "persistent_cce", "predictive_cce")

_OBSERVATION_NOISE = 1.2
_ERROR_NORMALIZER = 12.0
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")
# The environment (not the model) knows the true value's exact generative
# range: it starts in [10, 99] and drifts by at most 0.45/turn. A fixed,
# loose bound (e.g. 300) catches egregious "12345"-style canary output but
# missed a real failure mode: a wrong-but-plausible-looking number like 149
# on an episode whose true value never leaves roughly [10, 25]. Deriving the
# bound from the actual horizon-dependent range catches both.
_PLAUSIBILITY_MARGIN = 10.0

_SYSTEM_PREAMBLE = (
    "You are tracking a single hidden numeric quantity that drifts slightly "
    "each turn. Respond with ONLY your best-guess number for its current "
    "value: a single number, no words, no units, no explanation."
)


@dataclass
class Turn:
    turn: int
    observed: bool
    true_value: float
    raw_text: str
    parsed_value: Optional[float]
    error: float
    latency_ms: float


@dataclass
class LiveEpisode:
    seed: int
    condition: str
    turns: List[Turn]
    mean_error: float
    score: float
    parse_failures: int
    implausible_answers: int


def _plausible_range(horizon: int) -> tuple[float, float]:
    """The true value's exact generative bounds for this horizon: starts in
    [10, 99], drifts by at most 0.45/turn, plus a safety margin for noise."""
    max_drift = 0.45 * horizon
    return (10.0 - max_drift - _PLAUSIBILITY_MARGIN, 99.0 + max_drift + _PLAUSIBILITY_MARGIN)


def _is_observation_turn(t: int) -> bool:
    # A longer anchor window (5 observations, matching the deterministic
    # benchmark's convention) lets the velocity estimate stabilize before the
    # deliberate multi-turn observation blackout (t=5..9) that follows, so a
    # system with no dynamics model (raw_context/persistent_cce) goes
    # noticeably stale while a predictive model can extrapolate through it --
    # mirroring the deterministic benchmark's interruption_recovery task.
    if t < 5:
        return True
    if 5 <= t <= 9:
        return False
    return t % 3 == 0


def _parse_number(text: str) -> Optional[float]:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _stateless_prompt_lines(obs: Optional[float]) -> List[str]:
    if obs is not None:
        return [f"This turn's observation: {obs:.2f}."]
    return ["No observation is available this turn."]


def _raw_context_prompt_lines(obs: Optional[float], history: List[Optional[float]]) -> List[str]:
    transcript = json.dumps([round(v, 2) if v is not None else None for v in history])
    lines = [f"Raw observation history so far, oldest first (null = missing): {transcript}"]
    if obs is not None:
        lines.append(f"This turn's observation: {obs:.2f}.")
    else:
        lines.append("No observation is available this turn.")
    return lines


def _persistent_cce_prompt_lines(persistent_estimate: float) -> List[str]:
    """Tuned independently of predictive_cce's prompt -- editing one must not require editing the other."""
    return [
        "A tracking system maintains a running estimate for you and has "
        f"already updated it for the current turn: {persistent_estimate:.2f}. "
        "This is already the current-turn value -- report it as-is; do not "
        "adjust or extrapolate it further."
    ]


def _predictive_cce_prompt_lines(predictive_estimate: float, predictive_velocity: float) -> List[str]:
    """Tuned independently of persistent_cce's prompt -- editing one must not require editing the other."""
    return [
        "A tracking system maintains a running estimate with a drift model "
        "and has already updated it for the current turn, including any "
        f"drift since the last observation: {predictive_estimate:.2f}. "
        "This is already the current-turn value -- report it as-is; do not "
        "add drift or extrapolate it further yourself. (For reference only, "
        f"the system's estimated drift per turn is {predictive_velocity:.3f}, "
        "already reflected in the value above.)"
    ]


def _build_prompt(condition: str, obs: Optional[float], history: List[Optional[float]],
                   persistent_estimate: float, predictive_estimate: float,
                   predictive_velocity: float) -> str:
    lines = [_SYSTEM_PREAMBLE]
    if condition == "stateless":
        lines.extend(_stateless_prompt_lines(obs))
    elif condition == "raw_context":
        lines.extend(_raw_context_prompt_lines(obs, history))
    elif condition == "persistent_cce":
        lines.extend(_persistent_cce_prompt_lines(persistent_estimate))
    else:
        lines.extend(_predictive_cce_prompt_lines(predictive_estimate, predictive_velocity))
    lines.append("Your best-guess number for the current value:")
    return "\n".join(lines)


def run_episode(seed: int, condition: str, backend: InferenceBackend,
                 horizon: int, max_tokens: int, temperature: float) -> LiveEpisode:
    r = random.Random(seed)
    true_value = float(r.randint(10, 99))
    drift = r.uniform(-0.45, 0.45)
    lower_bound, upper_bound = _plausible_range(horizon)

    history: List[Optional[float]] = []
    persistent = ScalarKalman(measurement_noise=_OBSERVATION_NOISE, outlier_sigma=5.0)
    predictive = ConstantVelocityKalman(measurement_noise=_OBSERVATION_NOISE, outlier_sigma=5.0)

    turns: List[Turn] = []
    parse_failures = 0
    implausible_answers = 0

    for t in range(horizon):
        true_value += drift
        obs = None
        if _is_observation_turn(t):
            obs = true_value + r.uniform(-_OBSERVATION_NOISE, _OBSERVATION_NOISE)
        history.append(obs)

        persistent_estimate = persistent.step(obs)
        predictive_estimate = predictive.step(obs)

        prompt = _build_prompt(condition, obs, history, persistent_estimate,
                                predictive_estimate, predictive.velocity)
        start = time.perf_counter()
        generation = backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        latency_ms = (time.perf_counter() - start) * 1000.0

        parsed = _parse_number(generation.text)
        if parsed is None:
            parse_failures += 1
            error = _ERROR_NORMALIZER  # fully penalize an unparseable answer
        elif parsed < lower_bound or parsed > upper_bound:
            implausible_answers += 1
            error = _ERROR_NORMALIZER  # outside the environment's true generative range
        else:
            error = abs(parsed - true_value)

        turns.append(Turn(t, obs is not None, true_value, generation.text, parsed, error, latency_ms))

    mean_error = statistics.fmean(turn.error for turn in turns)
    score = max(0.0, 1.0 - mean_error / _ERROR_NORMALIZER)
    return LiveEpisode(seed, condition, turns, mean_error, score, parse_failures, implausible_answers)


def run(seeds: Iterable[int], model: str, horizon: int = 18, max_tokens: int = 12,
        temperature: float = 0.0, backend: Optional[InferenceBackend] = None) -> Dict:
    seed_list = list(seeds)
    if backend is None:
        backend = OllamaInferenceBackend(model_name=model)
    episodes = [run_episode(s, c, backend, horizon, max_tokens, temperature)
                for s in seed_list for c in CONDITIONS]

    aggregates: Dict[str, Dict] = {}
    for condition in CONDITIONS:
        rows = [e for e in episodes if e.condition == condition]
        aggregates[condition] = {
            "n": len(rows),
            "mean_score": statistics.fmean(e.score for e in rows),
            "mean_error": statistics.fmean(e.mean_error for e in rows),
            "parse_failures": sum(e.parse_failures for e in rows),
            "implausible_answers": sum(e.implausible_answers for e in rows),
            "total_turns": sum(len(e.turns) for e in rows),
            "mean_latency_ms": statistics.fmean(
                turn.latency_ms for e in rows for turn in e.turns
            ),
        }

    def beats(a: str, b: str) -> bool:
        return aggregates[a]["mean_score"] > aggregates[b]["mean_score"]

    gates = {
        "persistent_beats_stateless": beats("persistent_cce", "stateless"),
        "predictive_beats_persistent": beats("predictive_cce", "persistent_cce"),
        "predictive_beats_raw_context": beats("predictive_cce", "raw_context"),
        "predictive_beats_stateless": beats("predictive_cce", "stateless"),
    }
    return {
        "benchmark": "NSA/CCE Live Capability Benchmark (Ollama)",
        "version": "1.1.0",
        "scientific_boundary": (
            "Single model, single task, small sample, CPU-only live-model "
            "replication of the deterministic retention benchmark's drift-"
            "tracking task. Does not establish general superiority across "
            "models, tasks or scales."
        ),
        "model": backend.model_name,
        "conditions": list(CONDITIONS),
        "seeds": seed_list,
        "horizon": horizon,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "matched": {
            "same_model": True,
            "same_prompts_per_condition": True,
            "same_temperature": True,
            "same_max_tokens": True,
        },
        "episodes": [
            {**{k: v for k, v in asdict(e).items() if k != "turns"},
             "turns": [asdict(t) for t in e.turns]}
            for e in episodes
        ],
        "aggregates": aggregates,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "RESEARCH_GATE_NOT_YET_MET",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37, 73, 137])
    p.add_argument("--model", default="qwen2.5:0.5b")
    p.add_argument("--horizon", type=int, default=18)
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default="results/live_capability_benchmark.json")
    args = p.parse_args()
    report = run(args.seeds, args.model, args.horizon, args.max_tokens, args.temperature)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregates"], indent=2))
    print(json.dumps(report["gates"], indent=2))
    print(f"status={report['status']}")
    print(f"artifact={out}")


if __name__ == "__main__":
    main()
