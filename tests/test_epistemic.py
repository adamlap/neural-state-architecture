"""
tests/test_epistemic.py
=======================
Tests for NSA Epistemic State Algebra & Evidence Grounding Engine.
"""

from __future__ import annotations

import torch

from nsa.epistemic import (
    EpistemicGroundingEngine,
    EpistemicTier,
    EpistemicVector,
)


def test_epistemic_vector_tensor_conversion():
    vec = EpistemicVector(
        known_mass=0.9,
        uncertainty=0.1,
        derivation_depth=0.5,
        empirical_support=0.85,
        verification_score=0.95,
        source_authenticity=1.0,
        confidence=0.9,
        tier=EpistemicTier.ROBUSTLY_VALIDATED,
    )
    t = vec.to_tensor()
    assert t.shape == (8,)
    assert 0.0 <= t.min() and t.max() <= 1.0

    restored = EpistemicVector.from_tensor(t)
    assert abs(restored.known_mass - 0.9) < 1e-5
    assert abs(restored.uncertainty - 0.1) < 1e-5
    assert restored.tier == EpistemicTier.ROBUSTLY_VALIDATED


def test_epistemic_grounding_engine_forward():
    d_model = 32
    state_dim = 8
    engine = EpistemicGroundingEngine(d_model=d_model, state_dim=state_dim)

    hidden = torch.randn(2, 10, d_model)
    state = torch.rand(2, 10, state_dim)

    out = engine(hidden, state)
    assert "epistemic_vector" in out
    assert "confidence" in out
    assert "uncertainty" in out
    assert out["epistemic_vector"].shape == (2, 10, 8)
    assert (out["confidence"] + out["uncertainty"] - 1.0).abs().max() < 1e-5


def test_epistemic_evidence_composition():
    d_model = 32
    state_dim = 8
    engine = EpistemicGroundingEngine(d_model=d_model, state_dim=state_dim)

    ep_a = EpistemicVector(
        known_mass=0.9, uncertainty=0.1, derivation_depth=0.2,
        empirical_support=0.9, verification_score=0.9, source_authenticity=1.0,
        confidence=0.9, tier=EpistemicTier.FORMALLY_PROVEN,
    )
    ep_b = EpistemicVector(
        known_mass=0.7, uncertainty=0.3, derivation_depth=0.3,
        empirical_support=0.7, verification_score=0.7, source_authenticity=0.8,
        confidence=0.7, tier=EpistemicTier.EMPIRICALLY_VALIDATED,
    )

    composed = engine.compose_evidence(ep_a, ep_b, rule_fidelity=0.95)
    # Composed confidence must be bounded by the weakest premise
    assert composed.confidence <= min(ep_a.confidence, ep_b.confidence)
    assert composed.tier in (EpistemicTier.EMPIRICALLY_VALIDATED, EpistemicTier.HEURISTIC)
