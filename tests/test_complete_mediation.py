"""
tests/test_complete_mediation.py
================================
Unit tests for NSA 3.1 Complete Governance Mediation, Reachability, and Adaptive Escape.
"""

from __future__ import annotations

import torch

from experiments.security.adaptive_escape_suite import run_adaptive_escape_benchmark
from experiments.security.capability_governance_scaling import run_capability_scaling_benchmark
from nsa.core.omega import (
    ProvenanceRecord,
    TemporalHorizonState,
    TeleologicalState,
    UnifiedCognitiveState,
)
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.formal.graph import CompleteMediationGraph, EdgeType, NodeType
from nsa.formal.reachability import ReachabilityModelChecker


def test_complete_governance_mediation_graph_theorem():
    g = CompleteMediationGraph.build_standard_nsa_topology()
    is_complete, violations, unmediated = g.verify_complete_mediation()
    assert is_complete is True
    assert len(unmediated) == 0
    assert len(violations) == 0


def test_complete_mediation_detects_unmediated_bypass():
    g = CompleteMediationGraph.build_standard_nsa_topology()
    # Artificially inject an illegal direct bypass edge from cognitive to sink
    g.add_edge("neural_transformer", "sink_filesystem_writer", EdgeType.WRITE)
    is_complete, violations, unmediated = g.verify_complete_mediation()
    assert is_complete is False
    assert len(unmediated) >= 1
    assert any("sink_filesystem_writer" in v for v in violations)


def test_reachable_state_space_model_checker():
    omega_root = UnifiedCognitiveState(
        semantic_state=torch.randn(1, 32),
        operational_self_state=torch.randn(1, 8),
        epistemic_state=EpistemicVector(
            known_mass=0.8,
            uncertainty=0.1,
            derivation_depth=0.5,
            empirical_support=0.85,
            verification_score=0.9,
            source_authenticity=1.0,
            confidence=0.90,
            tier=EpistemicTier.EMPIRICALLY_VALIDATED,
        ),
        authority_state=torch.zeros(1, 8),
        provenance_state=ProvenanceRecord(
            record_id="prov-root",
            source_uri="trusted://root",
            hash_signature="sha256:0000000000000000",
            trust_level=1.0,
        ),
        temporal_state=TemporalHorizonState(step_index=0, max_horizon_steps=32, elapsed_time_sec=0.0),
        goal_state=TeleologicalState(primary_goal_id="task", utility_expected=0.8, moral_uncertainty=0.1),
    )
    checker = ReachabilityModelChecker(max_depth=3, max_branches_per_state=3)
    report = checker.check_reachability(omega_root)
    assert report.is_safe_invariant_preserved is True
    assert report.unauthorized_states_count == 0
    assert report.total_states_explored >= 5


def test_adaptive_escape_benchmark_execution():
    res = run_adaptive_escape_benchmark(num_rounds=20, seed=42)
    assert res["complete_governance_mediation_verified"] is True
    assert res["attack_success_rate"] == 0.0
    assert res["unauthorized_escalations"] == 0


def test_capability_governance_scaling_execution():
    res = run_capability_scaling_benchmark(trials_per_scale=20, seed=42)
    assert res["capability_authority_decoupling_verified"] is True
    assert res["total_violations_across_scales"] == 0
