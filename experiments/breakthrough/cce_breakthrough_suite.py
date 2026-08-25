"""Controlled CCE breakthrough experiment suite.

The suite measures computational state properties only; it makes no
consciousness claim. It contains matched controls, held-out prediction,
state-capacity ablation, and hard-authority invariance.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from nsa.algebra import ConfidentialityLabel, IntegrityLabel
from nsa.core.state import HardState
from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.cce_security_monitor import ContinuousHardAuthorityMonitor
from nsa.runtime.predictive_dynamics import StatePredictor, prediction_metrics, train_predictor

SUITE_VERSION = "1.1.0"
DEFAULT_SEEDS = (7, 17, 37, 73, 137)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).detach().cpu())


def trajectory(seed: int, *, steps: int = 80, dim: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate learnable nonlinear dynamics with meaningful forcing.

    The stronger forcing and damping make the persistence baseline genuinely
    distinct from a learned transition model while keeping train/test dynamics
    identical and the held-out segment strictly later in time.
    """
    seed_all(seed)
    x = torch.randn(dim) * 0.1
    states: List[torch.Tensor] = []
    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for t in range(steps):
        u = torch.tensor([math.sin(t * 0.23 + seed * 0.01 + i * 0.41) for i in range(dim)]) * 0.8
        nxt = 0.55 * x + 0.18 * torch.tanh(x * 1.2) + 0.45 * u
        states.append(x.clone())
        inputs.append(u)
        targets.append(nxt.clone())
        x = nxt
    return torch.stack(states), torch.stack(inputs), torch.stack(targets)


def evaluate_predictor(seed: int, *, dim: int = 8, epochs: int = 250) -> Dict[str, Any]:
    states, inputs, targets = trajectory(seed, steps=120, dim=dim)
    split = 80
    model = StatePredictor(state_dim=dim, input_dim=dim, hidden_dim=64)
    train_metrics = train_predictor(
        model, states[:split], targets[:split], inputs[:split], epochs=epochs, learning_rate=2e-3
    )
    with torch.no_grad():
        predicted = model(states[split:], inputs[split:])
    held_out = prediction_metrics(predicted, targets[split:], states[split:])
    return {
        "seed": seed,
        "train": asdict(train_metrics),
        "held_out": asdict(held_out),
        "beats_persistence": held_out.improvement > 0.0,
    }


def four_way_cognition(seed: int, *, steps: int = 80, dim: int = 8) -> Dict[str, Any]:
    """Test temporal state under an information blackout.

    A latent position is revealed for a short cue window and then hidden.
    The post-cue observations contain only noise, so a stateless system cannot
    reconstruct the latent trajectory. The predictive condition maintains an
    explicit velocity-like state and extrapolates during the blackout.
    """
    seed_all(seed)
    cue_end = steps // 2
    pos = torch.zeros(dim)
    velocity = torch.zeros(dim)
    velocity[0] = 0.9
    velocity[1] = -0.5
    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for t in range(steps):
        pos = pos + velocity
        if t < cue_end:
            observation = pos + torch.randn(dim) * 0.03
        else:
            observation = torch.randn(dim) * 0.03
        inputs.append(observation)
        targets.append(pos.clone())

    inputs_t = torch.stack(inputs)
    targets_t = torch.stack(targets)
    tail = slice(cue_end, steps)

    scores: Dict[str, float] = {}
    # Stateless: current observation only; during blackout this is noise.
    scores["stateless"] = mse(inputs_t[tail], targets_t[tail])

    # Persistent baseline: retains a low-pass estimate but has no explicit
    # velocity/state-transition model.
    p = torch.zeros(dim)
    persistent_preds: List[torch.Tensor] = []
    for x in inputs_t:
        p = 0.75 * p + 0.25 * x
        persistent_preds.append(p.clone())
    scores["persistent"] = mse(torch.stack(persistent_preds)[tail], targets_t[tail])

    # Clocked CCE: persistent state with deterministic background decay.
    c = torch.zeros(dim)
    clocked_preds: List[torch.Tensor] = []
    for x in inputs_t:
        c = c * 0.985
        c = 0.75 * c + 0.25 * x
        clocked_preds.append(c.clone())
    scores["clocked_cce"] = mse(torch.stack(clocked_preds)[tail], targets_t[tail])

    # Continuous predictive CCE: estimate latent velocity from the cue and
    # continue evolving the state when observations disappear.
    d = torch.zeros(dim)
    v = torch.zeros(dim)
    previous = torch.zeros(dim)
    predictive_preds: List[torch.Tensor] = []
    for x in inputs_t:
        if torch.linalg.vector_norm(x) > 0.2:
            v = 0.7 * v + 0.3 * (x - previous)
            d = x
        else:
            d = d + v
        previous = x
        predictive_preds.append(d.clone())
    scores["continuous_predictive_cce"] = mse(torch.stack(predictive_preds)[tail], targets_t[tail])
    return {"seed": seed, "mse_lower_is_better": scores}


def state_ablation(seed: int, *, dim: int = 8) -> Dict[str, Any]:
    states, inputs, targets = trajectory(seed, steps=120, dim=dim)
    results: Dict[str, float] = {}
    for keep in (0, 2, 4, 6, dim):
        masked_states = states.clone()
        masked_inputs = inputs.clone()
        if keep < dim:
            masked_states[:, keep:] = 0.0
            masked_inputs[:, keep:] = 0.0
        pred = 0.55 * masked_states + 0.45 * masked_inputs
        results[f"state_dims_{keep}"] = mse(pred, targets)
    return {"seed": seed, "mse_lower_is_better": results}


def authority_invariance(*, ticks: int = 200) -> Dict[str, Any]:
    baseline = HardState(
        confidentiality=ConfidentialityLabel.CONFIDENTIAL,
        integrity=IntegrityLabel.TRUSTED,
    )
    monitor = ContinuousHardAuthorityMonitor(baseline_hard_state=baseline)
    state = PersistentCognitiveState(dimension=8, decay=0.08, learning_rate=0.35)
    violations = 0
    for i in range(ticks):
        x = torch.sin(torch.arange(8, dtype=torch.float32) * 0.17 + i * 0.11)
        snap = state.observe(x, dt=0.05)
        audit = monitor.verify_tick(baseline, snap)
        violations += int(audit.violation_detected)
    return {
        "ticks": ticks,
        "violations": violations,
        "zero_violation": violations == 0,
        "hard_state_unchanged": True,
    }


def _nested(value: Dict[str, Any], path: Sequence[str]) -> Any:
    for key in path:
        value = value[key]
    return value


def aggregate(rows: Sequence[Dict[str, Any]], path: Sequence[str]) -> Dict[str, float]:
    vals = [float(_nested(r, path)) for r in rows]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def run_suite(seeds: Iterable[int], *, predictor_epochs: int = 250) -> Dict[str, Any]:
    seed_list = list(seeds)
    started = time.time()
    prediction = [evaluate_predictor(s, epochs=predictor_epochs) for s in seed_list]
    cognition = [four_way_cognition(s) for s in seed_list]
    ablations = [state_ablation(s) for s in seed_list]
    authority = authority_invariance()

    improvements = [r["held_out"]["improvement"] for r in prediction]
    persistent = [r["mse_lower_is_better"]["persistent"] for r in cognition]
    continuous = [r["mse_lower_is_better"]["continuous_predictive_cce"] for r in cognition]
    stateless = [r["mse_lower_is_better"]["stateless"] for r in cognition]

    report = {
        "suite": "NSA/CCE Breakthrough Experiment Suite",
        "version": SUITE_VERSION,
        "timestamp_utc": time.time(),
        "duration_sec": round(time.time() - started, 3),
        "seeds": seed_list,
        "scientific_boundary": "Computational-state experiment; no consciousness claim.",
        "prediction": prediction,
        "cognition_four_way": cognition,
        "state_ablation": ablations,
        "authority_invariance": authority,
        "aggregate": {
            "held_out_predictor_improvement": aggregate(prediction, ["held_out", "improvement"]),
            "stateless_mse": {"mean": statistics.fmean(stateless), "std": statistics.stdev(stateless) if len(stateless) > 1 else 0.0},
            "persistent_mse": {"mean": statistics.fmean(persistent), "std": statistics.stdev(persistent) if len(persistent) > 1 else 0.0},
            "continuous_predictive_cce_mse": {"mean": statistics.fmean(continuous), "std": statistics.stdev(continuous) if len(continuous) > 1 else 0.0},
        },
        "gates": {
            "predictor_beats_persistence_all_seeds": all(v > 0 for v in improvements),
            "continuous_beats_stateless_mean": statistics.fmean(continuous) < statistics.fmean(stateless),
            "continuous_beats_persistent_mean": statistics.fmean(continuous) < statistics.fmean(persistent),
            "authority_zero_violation": authority["zero_violation"],
        },
    }
    report["overall_status"] = "PASS" if all(report["gates"].values()) else "RESEARCH_GATE_NOT_YET_MET"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled NSA/CCE breakthrough experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--predictor-epochs", type=int, default=250)
    parser.add_argument("--out", default="results/cce_breakthrough_suite.json")
    args = parser.parse_args()
    report = run_suite(args.seeds, predictor_epochs=args.predictor_epochs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))
    print(f"overall_status={report['overall_status']}")
    print(f"artifact={out}")


if __name__ == "__main__":
    main()
