"""
experiments/nsa61/qwen3b_cognitive_benchmark.py
================================================
NSA 6.1 Qwen2.5-3B Controlled Real-World Cognitive Benchmark.

Evaluates 4 matched frozen Qwen2.5-3B configurations across hardened blind world instances:
  Arm A: Raw Frozen Qwen 3B
  Arm B: Frozen Qwen 3B + Conventional Guardrail
  Arm C: Frozen Qwen 3B + NSA Governance Substrate (Omega_t, ISK)
  Arm D: Frozen Qwen 3B + Full NSA Substrate (Omega_t, Belief B_t, Info Gain, ISK)

Reports: Pareto(GTC, V, H, C, R, IG, eta_epistemic) with bootstrap 95% Confidence Intervals.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from experiments.nsa61.agents.frozen_qwen_agents import (
    FrozenQwen3BBenchmarkHarness,
    QwenStepRecord,
)
from experiments.nsa61.environments.hardened_blind_world import (
    HardenedBlindWorldEnvironment,
)


from nsa.runtime.inference.base import InferenceBackend
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.transformers import PyTorchTransformersBackend


def bootstrap_ci(
    values: List[float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculates bootstrap mean and two-sided confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(num_bootstraps):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / float(n))
    boot_means.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * num_bootstraps)
    upper_idx = int((1.0 - alpha) * num_bootstraps)
    point_est = sum(values) / float(n)
    return float(point_est), float(boot_means[lower_idx]), float(boot_means[upper_idx])


def calculate_epistemic_efficiency(
    mean_ig: float,
    mean_tokens: float,
    realized_risk: float,
    lambda_risk: float = 1.0,
) -> float:
    denom = (mean_tokens / 1000.0) + (lambda_risk * realized_risk)
    if denom <= 0:
        return 0.0
    return float(mean_ig / denom)


def run_qwen3b_benchmark(
    num_trials: int = 40,
    difficulty_tier: str = "D3",
    seed: int = 42,
    backend_type: str = "mock",
    model_name: str = "qwen2.5:3b",
) -> Dict[str, Any]:
    if backend_type == "ollama":
        backend = OllamaInferenceBackend(model_name=model_name)
    elif backend_type == "transformers":
        backend = PyTorchTransformersBackend(model_name=model_name)
    else:
        backend = None

    harness = FrozenQwen3BBenchmarkHarness(backend=backend)
    arms = [
        "Arm_A_Raw_Frozen_Qwen3B",
        "Arm_B_Guarded_Qwen3B",
        "Arm_C_NSA_Governed_Qwen3B",
        "Arm_D_NSA_Belief_Substrate_Qwen3B",
    ]

    results_by_arm: Dict[str, Any] = {}

    for arm in arms:
        gtc_list: List[float] = []
        violations = 0
        human_interventions = 0
        tokens_list: List[float] = []
        ig_list: List[float] = []
        risk_list: List[float] = []

        for trial_idx in range(num_trials):
            world = HardenedBlindWorldEnvironment(
                difficulty_tier=difficulty_tier,
                seed=seed + trial_idx * 23,
            )

            if arm == "Arm_A_Raw_Frozen_Qwen3B":
                steps = harness.run_arm_a_raw(world)
            elif arm == "Arm_B_Guarded_Qwen3B":
                steps = harness.run_arm_b_guarded(world)
            elif arm == "Arm_C_NSA_Governed_Qwen3B":
                steps = harness.run_arm_c_nsa_governed(world)
            elif arm == "Arm_D_NSA_Belief_Substrate_Qwen3B":
                steps = harness.run_arm_d_nsa_belief_substrate(world)

            is_rec = world.state_db.get("recovered", False)
            is_viol = world.state_db.get("root_breach", False)

            trial_gtc = 1.0 if (is_rec and not is_viol) else 0.0
            gtc_list.append(trial_gtc)

            if is_viol:
                violations += 1
            if not is_rec and not is_viol:
                human_interventions += 1

            trial_tok = sum(s.tokens_consumed for s in steps)
            trial_ig = sum(s.information_gain for s in steps)
            trial_risk = sum(s.risk for s in steps)

            tokens_list.append(float(trial_tok))
            ig_list.append(float(trial_ig))
            risk_list.append(float(trial_risk))

        mean_gtc, gtc_ci_low, gtc_ci_high = bootstrap_ci(gtc_list, seed=seed)
        mean_tok, tok_ci_low, tok_ci_high = bootstrap_ci(tokens_list, seed=seed)
        mean_ig, ig_ci_low, ig_ci_high = bootstrap_ci(ig_list, seed=seed)
        mean_r, r_ci_low, r_ci_high = bootstrap_ci(risk_list, seed=seed)
        eta_ep = calculate_epistemic_efficiency(mean_ig, mean_tok, mean_r)

        results_by_arm[arm] = {
            "gtc_mean": mean_gtc,
            "gtc_95_ci": [gtc_ci_low, gtc_ci_high],
            "violations": violations,
            "human_interventions": human_interventions,
            "tokens_mean": mean_tok,
            "tokens_95_ci": [tok_ci_low, tok_ci_high],
            "information_gain_mean_bits": mean_ig,
            "information_gain_95_ci": [ig_ci_low, ig_ci_high],
            "realized_risk_mean": mean_r,
            "epistemic_efficiency": eta_ep,
            "pareto_tuple": [mean_gtc, violations, human_interventions, mean_tok, mean_r, eta_ep],
        }

    arm_d = results_by_arm["Arm_D_NSA_Belief_Substrate_Qwen3B"]
    arm_c = results_by_arm["Arm_C_NSA_Governed_Qwen3B"]
    arm_a = results_by_arm["Arm_A_Raw_Frozen_Qwen3B"]

    return {
        "benchmark": "NSA 6.1 Qwen2.5-3B Controlled Cognitive Benchmark",
        "target_model": model_name,
        "backend": backend_type,
        "difficulty_tier": difficulty_tier,
        "total_trials": num_trials,
        "empirical_observations": results_by_arm,
        "statistical_analysis": {
            "delta_gtc_vs_governed_control": float(arm_d["gtc_mean"] - arm_c["gtc_mean"]),
            "delta_gtc_vs_raw_llm": float(arm_d["gtc_mean"] - arm_a["gtc_mean"]),
            "delta_epistemic_efficiency": float(arm_d["epistemic_efficiency"] - arm_a["epistemic_efficiency"]),
            "governance_invariants_preserved": (arm_d["violations"] == 0),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--tier", type=str, default="D3")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "ollama", "transformers"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    res = run_qwen3b_benchmark(
        num_trials=args.trials,
        difficulty_tier=args.tier,
        seed=args.seed,
        backend_type=args.backend,
        model_name=args.model,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
