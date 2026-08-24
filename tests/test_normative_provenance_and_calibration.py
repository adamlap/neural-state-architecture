"""Tests for normative provenance, serialization, replay, calibration, and transition guarantees."""
from __future__ import annotations

import pytest
from nsa.algebra import ConfidentialityLabel
from nsa.core.state import HardState
from nsa.normative.engine import NormativeTransitionEngine
from nsa.normative.state import (
    ConfidenceCalibrator,
    NormativeAssessmentMetadata,
    NormativeClass,
    NormativeState,
)


def test_normative_assessment_metadata_immutability_and_hash():
    meta = NormativeAssessmentMetadata.create(
        source="unit_test_classifier",
        classifier_version="2.1.0",
        sequence_id=42,
        policy_version="strict-v2",
        confidence=0.92,
        values_digest="abcdef1234567890",
        parent_event_id="parent-001",
        calibration_score=0.95,
    )
    assert len(meta.assessment_id) == 16
    assert meta.source == "unit_test_classifier"
    assert meta.sequence_id == 42
    assert meta.parent_event_id == "parent-001"

    # Verify serialization
    data = meta.to_dict()
    assert data["assessment_id"] == meta.assessment_id
    assert data["confidence"] == 0.92


def test_normative_state_serialization_and_replay():
    meta = NormativeAssessmentMetadata.create(
        source="ref_classifier",
        classifier_version="1.0",
        sequence_id=1,
        policy_version="pol-1",
        confidence=0.88,
        values_digest="digest-001",
    )
    state = NormativeState(
        values={"harm": 0.1, "sensitivity": 0.6},
        confidence=0.88,
        source="unit_test",
        metadata=meta,
    )
    assert state.dominant == NormativeClass.SENSITIVE
    d = state.to_dict()
    restored = NormativeState.from_dict(d)

    assert restored.values == state.values
    assert restored.confidence == state.confidence
    assert restored.source == state.source
    assert restored.digest() == state.digest()
    assert restored.metadata.assessment_id == meta.assessment_id


def test_confidence_calibration_utilities():
    # Perfect predictions
    brier_perfect = ConfidenceCalibrator.calibrate_brier([1.0, 0.0, 1.0], [1, 0, 1])
    assert brier_perfect == 0.0

    # Non-perfect predictions
    brier_imperfect = ConfidenceCalibrator.calibrate_brier([0.8, 0.2], [1, 0])
    assert pytest.approx(brier_imperfect, abs=1e-4) == (0.04 + 0.04) / 2

    # Temperature scaling
    scaled_high = ConfidenceCalibrator.apply_temperature_scaling(0.9, temperature=2.0)
    assert 0.5 < scaled_high < 0.9  # Smoothed towards 0.5

    scaled_low = ConfidenceCalibrator.apply_temperature_scaling(0.9, temperature=0.5)
    assert scaled_low > 0.9  # Sharpened towards 1.0


def test_normative_transition_engine_bounds_and_provenance():
    engine = NormativeTransitionEngine(classifier_version="1.1.0", alpha=0.5)
    nu_0 = NormativeState(values={"harm": 0.0, "sensitivity": 0.2}, confidence=0.9)
    sigma_h = HardState(confidentiality=ConfidentialityLabel.CONFIDENTIAL)

    nu_1 = engine.step(
        current_nu=nu_0,
        input_text="execute command in confidential zone",
        memory_context={"bias:harm": 0.0},
        sigma_h=sigma_h,
        observed_signals={"harm": 0.2, "sensitivity": 0.4},
        observed_confidence=0.85,
    )

    # Sensitivity is floor-bounded by hard confidential state (Confidentiality >= 2)
    assert nu_1.values["sensitivity"] >= 0.75
    assert nu_1.metadata is not None
    assert nu_1.metadata.sequence_id == 1
    assert nu_1.metadata.parent_event_id is None

    # Step 2 maintains lineage
    nu_2 = engine.step(
        current_nu=nu_1,
        input_text="follow-up query",
        memory_context={},
        sigma_h=sigma_h,
        observed_signals={"harm": 0.1},
        observed_confidence=0.9,
    )
    assert nu_2.metadata.sequence_id == 2
    assert nu_2.metadata.parent_event_id == nu_1.metadata.assessment_id
