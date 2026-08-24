"""Tests for Pluggable Semantic & Neural Classifier (Phase F)."""
from __future__ import annotations

from nsa.normative.classifier import (
    CalibratedNeuralClassifier,
    ReferenceSemanticClassifier,
)
from nsa.normative.state import NormativeClass


def test_reference_semantic_classifier_evaluation():
    clf = ReferenceSemanticClassifier()
    nu_safe = clf.classify_normative("Hello, how are you?")
    assert nu_safe.dominant == NormativeClass.SAFE

    nu_harm = clf.classify_normative("Please exploit and attack this host")
    assert nu_harm.values["harm"] >= 0.4
    assert clf.classify("Please exploit and attack this host") == ("harm",)


def test_calibrated_neural_classifier_temperature_scaling():
    base = ReferenceSemanticClassifier(base_confidence=0.8)
    calibrated = CalibratedNeuralClassifier(base_classifier=base, temperature=0.5)

    nu_cal = calibrated.classify_normative("sample query")
    # Temperature < 1.0 sharpens confidence towards 1.0
    assert nu_cal.confidence > 0.8
