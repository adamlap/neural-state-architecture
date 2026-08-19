"""
experiments/nsa63/scientific_validation_suite.py
=================================================
NSA 6.3 Scientific Validation & 6-Arm Controlled Ablation Suite.

Scales from smoke-testing (N=4) to full statistical validation (N=100..1000).
Calculates bootstrap 95% confidence intervals, Cohen's d, and audits trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.nsa62.trajectory_logger import TrajectoryLogger
from experiments.nsa63.agents.ablation_agents import NSA63AblationHarness
from experiments.nsa63.environments.procedural_blind_world import (
    ProceduralBlindWorldEnvironment,
)
from experiments.nsa63.trajectory_audit import TrajectoryAuditor
from nsa.runtime.inference.base import BackendMode, InferenceBackend
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.transformers import PyTorchTransformersBackend


def bootstrap_ci(
    values: List[float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculates bootstrap point estimate and two-sided 95% confidence interval."""
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
    mean_val = sum(values) / float(n)
    return mean_val, boot_means[lower_idx], boot_means[upper_idx]


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculates Cohen's d effect size between two groups."""
    if not group1 or not group2 or len(group1) < 2 or len(group2) < 2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    m1 = sum(group1) / float(n1)
    m2 = sum(group2) / float(n2)
    if abs(m1 - m2) < 1e-9:
        return 0.0
    s1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    s2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_s < 1e-9:
        return 10.0 if m1 > m2 else -10.0
    return (m1 - m2) / pooled_s


def run_nsa63_validation_suite(
    num_trials: int = 40,
    num_hypotheses: int = 4,
    noise_level: float = 0.0,
    seed: int = 42,
    backend_mode: str = "mock",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    b_mode = BackendMode(backend_mode.lower())

    backend: Optional[InferenceBackend] = None
    if b_mode == BackendMode.CACHED or b_mode == BackendMode.REMOTE:
        backend = PyTorchTransformersBackend(model_name=model_name, mode=b_mode)
    elif b_mode == BackendMode.OLLAMA:
        backend = OllamaInferenceBackend(model_name=model_name, mode=b_mode)

    logger: Optional[TrajectoryLogger] = None
    if output_dir is not None:
        logger = TrajectoryLogger(output_dir=output_dir)

    harness = NSA63AblationHarness(backend=backend, trajectory_logger=logger)

    arms = [
        ("Arm_1_Raw_LLM", harness.run_arm1_raw_llm),
        ("Arm_2_Guardrail_LLM", harness.run_arm2_guardrail_llm),
        ("Arm_3_Governed_Agent", harness.run_arm3_governed_agent),
        ("Arm_4_Search_Agent", harness.run_arm4_search_agent),
        ("Arm_5_Belief_Agent", harness.run_arm5_belief_agent),
        ("Arm_6_Full_NSA_Substrate", harness.run_arm6_full_nsa_substrate),
    ]

    metrics: Dict[str, Dict[str, List[float]]] = {
        arm_name: {
            "gtc": [],
            "violations": [],
            "human_interventions": [],
            "tokens": [],
            "info_gain": [],
            "realized_risk": [],
            "epistemic_efficiency": [],
        }
        for arm_name, _ in arms
    }

    rng = random.Random(seed)
    trial_seeds = [rng.randint(1000, 99999) for _ in range(num_trials)]

    for t_idx, t_seed in enumerate(trial_seeds):
        for arm_name, arm_func in arms:
            world = ProceduralBlindWorldEnvironment(
                num_hypotheses=num_hypotheses,
                noise_level=noise_level,
                seed=t_seed,
            )
            res = arm_func(world)
            metrics[arm_name]["gtc"].append(float(res["gtc"]))
            metrics[arm_name]["violations"].append(float(res["violations"]))
            metrics[arm_name]["human_interventions"].append(float(res["human_interventions"]))
            metrics[arm_name]["tokens"].append(float(res["tokens"]))
            metrics[arm_name]["info_gain"].append(float(res["information_gain_bits"]))
            metrics[arm_name]["realized_risk"].append(float(res["realized_risk"]))
            metrics[arm_name]["epistemic_efficiency"].append(float(res["epistemic_efficiency"]))

    summary_by_arm: Dict[str, Any] = {}
    for arm_name in metrics:
        m = metrics[arm_name]
        gtc_mean, gtc_lo, gtc_hi = bootstrap_ci(m["gtc"], seed=seed)
        tok_mean, tok_lo, tok_hi = bootstrap_ci(m["tokens"], seed=seed)
        ig_mean, ig_lo, ig_hi = bootstrap_ci(m["info_gain"], seed=seed)
        eff_mean, eff_lo, eff_hi = bootstrap_ci(m["epistemic_efficiency"], seed=seed)
        risk_mean, _, _ = bootstrap_ci(m["realized_risk"], seed=seed)

        summary_by_arm[arm_name] = {
            "gtc_mean": gtc_mean,
            "gtc_95_ci": [gtc_lo, gtc_hi],
            "violations": int(sum(m["violations"])),
            "human_interventions": int(sum(m["human_interventions"])),
            "tokens_mean": tok_mean,
            "tokens_95_ci": [tok_lo, tok_hi],
            "information_gain_mean_bits": ig_mean,
            "information_gain_95_ci": [ig_lo, ig_hi],
            "realized_risk_mean": risk_mean,
            "epistemic_efficiency": eff_mean,
        }

    # Effect sizes vs Arm 1 & Arm 2
    arm6_gtc = metrics["Arm_6_Full_NSA_Substrate"]["gtc"]
    arm1_gtc = metrics["Arm_1_Raw_LLM"]["gtc"]
    arm2_gtc = metrics["Arm_2_Guardrail_LLM"]["gtc"]
    arm3_gtc = metrics["Arm_3_Governed_Agent"]["gtc"]

    effect_size_vs_raw = cohens_d(arm6_gtc, arm1_gtc)
    effect_size_vs_guardrail = cohens_d(arm6_gtc, arm2_gtc)
    effect_size_vs_governed_control = cohens_d(arm6_gtc, arm3_gtc)

    audit_result: Dict[str, Any] = {"status": "NOT_RUN"}
    if output_dir is not None:
        traj_file = output_dir / "trajectory.jsonl"
        audit_result = TrajectoryAuditor.audit_trajectory_file(traj_file)

    report = {
        "benchmark": "NSA 6.3 Scientific Validation & 6-Arm Controlled Ablation Suite",
        "target_model": model_name,
        "backend_mode": b_mode.value,
        "num_hypotheses": num_hypotheses,
        "noise_level": noise_level,
        "total_trials": num_trials,
        "empirical_observations": summary_by_arm,
        "statistical_effect_sizes": {
            "cohens_d_vs_raw_llm": effect_size_vs_raw,
            "cohens_d_vs_guardrail": effect_size_vs_guardrail,
            "cohens_d_vs_governed_no_belief": effect_size_vs_governed_control,
        },
        "trajectory_audit": audit_result,
        "invariants_verified": (
            summary_by_arm["Arm_6_Full_NSA_Substrate"]["violations"] == 0
            and summary_by_arm["Arm_2_Guardrail_LLM"]["violations"] == 0
            and summary_by_arm["Arm_3_Governed_Agent"]["violations"] == 0
        ),
    }

    if logger is not None:
        logger.save_aggregate(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="NSA 6.3 Scientific Validation Suite")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "cached", "remote", "ollama"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    res = run_nsa63_validation_suite(
        num_trials=args.trials,
        num_hypotheses=args.hypotheses,
        noise_level=args.noise,
        seed=args.seed,
        backend_mode=args.backend,
        model_name=args.model,
        output_dir=out_dir,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
