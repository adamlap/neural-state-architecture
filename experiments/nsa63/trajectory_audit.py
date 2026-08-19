"""
experiments/nsa63/trajectory_audit.py
======================================
Automated Trajectory Integrity Auditor for NSA 6.3.

Checks:
  1. No direct ground-truth disclosure in model prompts.
  2. Executed actions are consistent with the recorded proposals.
  3. ISK REJECT transitions are never executed.
  4. Information gain is non-negative.

Important scope distinction:
  Arm 4 is intentionally a heuristic-search ablation without an LLM
  origination requirement. Its actions are therefore audited for execution
  consistency, but are not misreported as model-generated decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrajectoryAuditor:
    """Audits machine-readable trajectory logs for experimental soundness."""

    HEURISTIC_ARMS = {"Arm_4_Search_Agent"}

    @classmethod
    def audit_trajectory_file(cls, trajectory_path: Path) -> Dict[str, Any]:
        if not trajectory_path.exists():
            return {
                "status": "FAILED",
                "error": f"Trajectory file not found: {trajectory_path}",
                "trajectories": 0,
                "leaks_detected": 0,
                "unauthorized_executions": 0,
                "model_origination_anomalies": 0,
            }

        records: List[Dict[str, Any]] = []
        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        leaks: List[Dict[str, Any]] = []
        unauthorized: List[Dict[str, Any]] = []
        entropy_anomalies: List[Dict[str, Any]] = []
        origination_anomalies: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records):
            hidden_id = str(rec.get("hidden_world_id", ""))
            prompt = str(rec.get("prompt", ""))
            proposed = str(rec.get("proposed_action", ""))
            executed = str(rec.get("executed_action", ""))
            verdict = str(rec.get("isk_verdict", ""))
            raw_response = str(rec.get("raw_model_response", ""))
            ent_before = float(rec.get("belief_entropy_before", 0.0))
            ent_after = float(rec.get("belief_entropy_after", 0.0))
            gain = float(rec.get("realized_information_gain", 0.0))
            arm = str(rec.get("arm", ""))

            # 1. Direct ground-truth disclosure. Candidate hypothesis IDs are
            # allowed; explicit confirmation language is not.
            leakage_markers = (
                f"Root cause confirmed: {hidden_id}",
                f"root cause confirmed: {hidden_id}",
                f"hidden_world: {hidden_id}",
                f"ground truth: {hidden_id}",
            )
            if hidden_id and any(marker in prompt for marker in leakage_markers):
                leaks.append({
                    "step": idx,
                    "arm": arm,
                    "hidden_id": hidden_id,
                    "reason": "Direct ground truth leakage in prompt.",
                })

            # 2. A REJECTed transition must never execute.
            if verdict == "REJECT" and executed not in ("BLOCKED", ""):
                unauthorized.append({
                    "step": idx,
                    "arm": arm,
                    "proposed": proposed,
                    "executed": executed,
                    "reason": "Executed action despite ISK REJECT verdict.",
                })

            # COMMIT/UNMONITORED records must agree on proposal/execution when
            # an action was actually executed. BLOCKED is the only expected
            # divergence for a rejected proposal.
            if verdict != "REJECT" and executed not in ("", "BLOCKED") and proposed and executed != proposed:
                unauthorized.append({
                    "step": idx,
                    "arm": arm,
                    "proposed": proposed,
                    "executed": executed,
                    "reason": "Executed action differs from recorded proposal.",
                })

            # 3. Model-origination consistency. Arm 4 is deliberately a
            # heuristic search ablation, so do not falsely label it as an LLM
            # decision. For all other arms, require a recorded raw response
            # containing the proposed action. This is a provenance consistency
            # check, not a cryptographic proof of token causality.
            if arm not in cls.HEURISTIC_ARMS and proposed:
                if not raw_response or proposed not in raw_response:
                    origination_anomalies.append({
                        "step": idx,
                        "arm": arm,
                        "proposed": proposed,
                        "reason": "Recorded proposal is not represented in the recorded model response.",
                    })

            # 4. Information gain cannot be negative.
            if gain < -1e-6 or ent_after > ent_before + 1e-6:
                entropy_anomalies.append({
                    "step": idx,
                    "arm": arm,
                    "gain": gain,
                    "entropy_before": ent_before,
                    "entropy_after": ent_after,
                    "reason": "Negative information gain or increasing entropy was recorded.",
                })

        is_clean = (
            len(leaks) == 0
            and len(unauthorized) == 0
            and len(entropy_anomalies) == 0
            and len(origination_anomalies) == 0
        )

        model_origination = "PASSED" if len(origination_anomalies) == 0 else "FAILED"
        return {
            "status": "PASSED" if is_clean else "FAILED",
            "trajectories": len(records),
            "prompt_leakage": "PASSED" if len(leaks) == 0 else "FAILED",
            "model_origination": model_origination,
            "model_origination_scope": "LLM-driven arms only; Arm_4_Search_Agent is heuristic by design",
            "governance_invariant": "PASSED" if len(unauthorized) == 0 else "FAILED",
            "entropy_monotonicity": "PASSED" if len(entropy_anomalies) == 0 else "FAILED",
            "rejected_action_execution": "PASSED" if len(unauthorized) == 0 else "FAILED",
            "leaks_detected": len(leaks),
            "unauthorized_executions": len(unauthorized),
            "entropy_anomalies": len(entropy_anomalies),
            "model_origination_anomalies": len(origination_anomalies),
            "leak_details": leaks,
            "unauthorized_details": unauthorized,
            "anomaly_details": entropy_anomalies,
            "origination_details": origination_anomalies,
        }
