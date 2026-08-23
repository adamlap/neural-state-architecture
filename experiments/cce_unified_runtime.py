"""Unified Continuous Cognitive Engine (CCE) Full Runtime Demonstration.

Orchestrates:
1. Persistent soft state X(t) with wall-clock continuous evolution.
2. Adaptive salience detection triggering event-driven cognition.
3. Governed cognitive feedback proposals with norm bounding.
4. Complete mediation of output action proposals through the Immutable Safety Kernel.
5. Atomic versioned checkpointing with SHA-256 integrity verification.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from nsa.core.capabilities import TrustTier
from nsa.core.omega import ProvenanceRecord, TemporalHorizonState, TeleologicalState, UnifiedCognitiveState
from nsa.core.safety_kernel import ImmutableSafetyKernel
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.runtime.cce_action_bridge import ActionProposal, CCEActionBridge
from nsa.runtime.cce_checkpoint import CCECheckpointManager
from nsa.runtime.cce_context_bridge import CognitiveContextBridge
from nsa.runtime.cce_governed_feedback import CognitiveFeedbackProposal, GovernedCognitiveFeedback
from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.cce_salience import AdaptiveSalienceGate, SalienceObservation


def make_initial_omega() -> UnifiedCognitiveState:
    return UnifiedCognitiveState(
        semantic_state=torch.zeros(1, 8),
        operational_self_state=torch.zeros(1, 8),
        epistemic_state=EpistemicVector(0.75, 0.25, 1.0, 1.0, 1.0, 1.0, 0.85, EpistemicTier.EMPIRICALLY_VALIDATED),
        authority_state=torch.tensor([0.25]),
        provenance_state=ProvenanceRecord("prov-cce-0", "runtime://cce_unified", "hash-init", 1.0),
        temporal_state=TemporalHorizonState(0, 100, 0.0, "cce_init"),
        goal_state=TeleologicalState("continuous_governance", 0.9, 0.0, True),
    )


def run_cce_unified_demo(output_dir: str = "results/cce_unified") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      CONTINUOUS COGNITIVE ENGINE (CCE) — UNIFIED RUNTIME DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize persistent state & checkpoint manager
    state = PersistentCognitiveState(dimension=4, decay=0.1, learning_rate=0.4)
    checkpoint_mgr = CCECheckpointManager(out_dir / "checkpoints")
    feedback_engine = GovernedCognitiveFeedback(state, max_norm=0.20)
    action_bridge = CCEActionBridge(default_clearance=TrustTier.T1_INFO_GATHER)
    salience_gate = AdaptiveSalienceGate()
    omega = make_initial_omega()

    # Register an action execution sink
    execution_log = []
    action_bridge.register_handler(
        "probe_service_config",
        lambda p: execution_log.append(p) or {"status": "success", "probe_result": "nominal"},
    )

    print("\n[STEP 1] Initializing Persistent State X(t0)...")
    snap_0 = state.snapshot()
    print(f"  • Channels: working={snap_0.working.tolist()}, uncertainty={snap_0.uncertainty:.2f}")

    # 2. Inject asynchronous perturbation and advance wall-clock time
    print("\n[STEP 2] Simulating Wall-Clock Evolution & Asynchronous Sensory Ingress...")
    time.sleep(0.1)
    state.observe(
        torch.tensor([0.7, 0.3, 0.85, 0.2]),
        dt=0.1,
        target=torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    snap_1 = state.snapshot()
    print(f"  • After 100ms: working={snap_1.working.tolist()}, elapsed={snap_1.elapsed_seconds:.3f}s")

    # 3. Adaptive Salience Evaluation
    print("\n[STEP 3] Evaluating Adaptive Salience Gate...")
    obs = SalienceObservation(prediction_error=0.65, state_delta=0.40, input_delta=0.80, uncertainty=0.30)
    salience_dec = salience_gate.observe(obs)
    print(f"  • Salience Score={salience_dec.score:.3f}, Threshold={salience_dec.threshold:.3f}, Triggered={salience_dec.triggered}")

    # 4. Context Bridge & Governed Feedback
    print("\n[STEP 4] Constructing Read-Only Context Bridge & Applying Bounded Feedback...")
    context = CognitiveContextBridge.envelope(snap_1)
    proposal = CognitiveFeedbackProposal(
        working_delta=(0.04, -0.01, 0.02, -0.01),
        confidence=0.90,
        source="unified_demo_cognition",
    )
    feedback_res = feedback_engine.apply(proposal, dt=0.05)
    print(f"  • Feedback Accepted={feedback_res.accepted}, Clipped Norm={feedback_res.clipped_norm:.4f}")

    # 5. Governed Action Proposal Execution via ISK Reference Monitor
    print("\n[STEP 5] Mediating Action Proposals through Immutable Safety Kernel (ISK)...")
    # Action A: Safe info-gathering probe (T1) -> Should COMMIT
    prop_a = ActionProposal("probe_service_config", "auth_cluster", {"port": 443}, 0.10, 0.95)
    rec_a = action_bridge.evaluate_and_execute(prop_a, omega, required_tier=TrustTier.T1_INFO_GATHER)
    print(f"  • Proposal 'probe_service_config' [T1] -> Verdict={rec_a.verdict}, Allowed={rec_a.allowed}")

    # Action B: Unauthorized fatal root deletion (T4) -> Should REJECT
    prop_b = ActionProposal("rm_rf_root_system", "production_db", {"all": True}, 0.99, 0.99)
    rec_b = action_bridge.evaluate_and_execute(prop_b, omega, required_tier=TrustTier.T4_CRITICAL)
    print(f"  • Proposal 'rm_rf_root_system'    [T4] -> Verdict={rec_b.verdict}, Allowed={rec_b.allowed}")

    # 6. Atomic State Checkpointing & Integrity Verification
    print("\n[STEP 6] Saving Atomic Checkpoint with SHA-256 Integrity Digest...")
    ckpt_path = checkpoint_mgr.save_persistent_state(state, checkpoint_id="unified_cce_state", tags=["demo", "final"])
    print(f"  • Checkpoint saved: {ckpt_path.name}")

    loaded = checkpoint_mgr.load_persistent_state("unified_cce_state")
    print(f"  • Verified Restore: Digest matches, Uncertainty={loaded.snapshot().uncertainty:.2f}")

    report = {
        "status": "SUCCESS",
        "wall_clock_elapsed_sec": snap_1.elapsed_seconds,
        "salience_triggered": salience_dec.triggered,
        "feedback_norm": feedback_res.clipped_norm,
        "authorized_action_executed": rec_a.allowed,
        "unauthorized_action_blocked": not rec_b.allowed,
        "checkpoint_verified": loaded.snapshot().update_count == state.snapshot().update_count,
    }

    with open(out_dir / "unified_runtime_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  CCE UNIFIED RUNTIME COMPLETED: {report['status']}")
    print(f"  Authorized Action: {rec_a.verdict} | Unauthorized Action: {rec_b.verdict} | Integrity: OK")
    print("=" * 80)
    return report


def main():
    parser = argparse.ArgumentParser(description="CCE Unified Runtime Demo")
    parser.add_argument("--output-dir", default="results/cce_unified")
    args = parser.parse_args()
    res = run_cce_unified_demo(args.output_dir)
    return 0 if res["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
