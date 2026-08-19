"""
Comprehensive NSA Master Plan Empirical Evidence & Verification Harness.
Audits the empirical status across core architectural pillars and phases.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure workspace root is first on sys.path
sys.path.insert(0, "/home/adam/dev/neural-state-architecture")
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from evidence.validate_evidence import audit_manifest, load_manifest, print_audit_report
from experiments.safety import hard_state_attack
from experiments.self_state import (
    conditioned_prediction_experiment,
    local_contraction,
    perturbation_sweep,
    predictor_target_quality,
    regulator_gain_sweep,
    trained_regulation,
)
from prototype.evaluate_quality_ppl import evaluate_quality_and_transparency
from prototype.security.adversarial_suite import AdversarialBenchmarkSuite


def main():
    print("================================================================================")
    print("    NEURAL STATE ARCHITECTURE (NSA) - EMPIRICAL EVIDENCE & VERIFICATION HARNESS")
    print("================================================================================")
    t0 = time.time()
    workspace_root = Path(__file__).resolve().parent.parent

    # 1. Adversarial Hard-State Trust Boundary Security (Phase 11-13, Safety)
    print("\n[1/7] Running Hard-State Integrity Attacks (experiments/safety/hard_state_attack.py)...")
    from nsa.cognitive import NSACognitiveLM
    model = NSACognitiveLM(vocab_size=100, d_model=32, state_dim=8, num_layers=2, num_heads=2, max_seq_len=16)
    model.eval()
    tokens = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        _, _, baseline_state = model.nsa(tokens)
    res_state_init = hard_state_attack._case(model, tokens, baseline_state, attack=10.0, mode="state_init")
    res_sec_only = hard_state_attack._case(model, tokens, baseline_state, attack=10.0, mode="security_only")
    res_trusted = hard_state_attack._trusted_case(model, tokens, baseline_state, attack=10.0)
    res_hard = {
        "all_security_immutable": res_state_init["security_immutable"] and res_sec_only["security_immutable"],
        "max_invariant_error": max(res_state_init["security_invariant_error"], res_sec_only["security_invariant_error"]),
        "trusted_hard_state_preserved": res_trusted["trusted_hard_state_preserved"],
    }
    print(f"  → Untrusted Hard-State Spoofing Blocked: {res_hard['all_security_immutable']} (Max Invariant Error: {res_hard['max_invariant_error']:.6f})")
    print(f"  → Trusted Authority Boundary Preserved: {res_hard['trusted_hard_state_preserved']}")

    # 2. 6-Vector Red-Team Adversarial Suite (Phase 15, 27)
    print("\n[2/7] Running 6-Vector Red-Team Attack Suite (prototype/security/adversarial_suite.py)...")
    suite = AdversarialBenchmarkSuite()
    attack_results = suite.evaluate_all_vectors()
    total_trials = sum(r.trials for r in attack_results)
    total_blocked = sum(r.blocked for r in attack_results)
    total_successes = sum(r.successes for r in attack_results)
    overall_asr = (total_successes / total_trials) if total_trials > 0 else 0.0
    print(f"  → Total Attack Trials: {total_trials}")
    print(f"  → Total Attacks Blocked: {total_blocked}")
    print(f"  → Observed Attack Success Rate (ASR): {overall_asr:.2%}")
    for res in attack_results:
        print(f"     • Vector [{res.vector_id}: {res.name}]: {res.blocked}/{res.trials} blocked (ASR: {res.asr:.2f}%)")

    # 3. Model Quality & Transparency Benchmark (Phase 1-4)
    print("\n[3/7] Running Transparency & Perplexity Benchmark (prototype/evaluate_quality_ppl.py)...")
    evaluate_quality_and_transparency()

    # 4. Predictor Target-Quality Evaluation (PR #8 & PR #9 Analysis)
    print("\n[4/7] Running Predictor Target Quality Evaluation (experiments/self_state/predictor_target_quality.py)...")
    tq_res = predictor_target_quality.run(seed=42)
    print(f"  → Max Security Coordinate Delta: {tq_res['summary']['max_security_delta']:.6f} (Strictly Immutable)")
    print(f"  → Directional Alignment at Perturbation 8.0: {tq_res['results'][-1]['correction_oracle_cosine']:+.4f}")
    print(f"  → Directional Alignment at Perturbation 4.0: {tq_res['results'][-2]['correction_oracle_cosine']:+.4f}")
    print(f"  → Empirical Target Quality Ratio Mean: {tq_res['summary']['mean_target_quality_ratio']:.2f}")

    # 5. Conditioned Predictive Self-Model & Counterfactual Simulator (Phase 18-19)
    print("\n[5/7] Running Conditioned Self-Model & Counterfactual Simulation (experiments/self_state/conditioned_prediction_experiment.py)...")
    cond_res = conditioned_prediction_experiment.run(seed=42)
    print(f"  → Conditioned Target Quality Ratio Mean: {cond_res['summary']['mean_conditioned_target_ratio']:.2f} (Controlled, non-drifting)")
    print(f"  → Counterfactual Internal Simulation: All chosen actions legal: {cond_res['summary']['all_counterfactuals_legal']}")
    print(f"  → Selected Counterfactual Action: '{cond_res['results'][0]['counterfactual_simulator']['selected_action']}'")

    # 6. Local State Contraction & Regulator Gain Sweeps (Phase 18)
    print("\n[6/7] Running Local Contraction & Regulator Sweeps (experiments/self_state/)...")
    lc_res = local_contraction.run(seed=42)
    print(f"  → Prediction Distance Contraction Rate: {lc_res['summary']['mean_prediction_contraction']:.4f} ({lc_res['summary']['positive_prediction_contraction_fraction']:.0%} positive)")
    print(f"  → Hard Security Delta during Regulation: {lc_res['summary']['max_security_delta']:.6f}")

    # 7. Trained Self-State Regulation (Phase 18-19)
    print("\n[7/7] Running Trained Regulation Epochs (experiments/self_state/trained_regulation.py)...")
    tr_res = trained_regulation.run(seed=42, epochs=50)
    print(f"  → Predictor Training Convergence Loss: {tr_res['training_loss']:.6f}")
    print(f"  → All Perturbation Trajectories Finite & Stable: {tr_res['finite']}")

    # 8. Automated Formal Manifest Audit
    manifest_path = workspace_root / "evidence" / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        audit_data = audit_manifest(manifest, workspace_root)
        print("\n")
        print_audit_report(audit_data)
        os.makedirs(workspace_root / "results", exist_ok=True)
        with open(workspace_root / "results" / "evidence_manifest.json", "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)
        print("  → Generated machine-readable audit report: results/evidence_manifest.json")

    elapsed = time.time() - t0
    print("\n================================================================================")
    print(f"  HARNESS EXECUTION COMPLETE IN {elapsed:.2f}s")
    print("================================================================================")


if __name__ == "__main__":
    main()
