"""
tests/test_evidence_engine.py
=============================
Tests for NSA Dynamic Evidence Verification Engine.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from nsa.evidence.engine import (
    EpistemicVerificationEngine,
    compute_file_sha256,
)


def test_compute_file_sha256(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")
    h = compute_file_sha256(test_file)
    assert len(h) == 64
    assert isinstance(h, str)


def test_epistemic_verification_engine_auditing():
    workspace_root = Path(__file__).resolve().parent.parent
    engine = EpistemicVerificationEngine(workspace_root)

    test_claim = {
        "claim_id": "TEST-CLAIM-01",
        "phase": "Phase 1",
        "title": "Test Claim",
        "epistemic_status": "UNIT_TESTED",
        "implementation_paths": ["nsa/layers.py"],
        "unit_test_paths": ["tests/test_nsa.py"],
        "experiment_paths": [],
        "latest_metrics": {"test_passed": True},
    }

    res = engine.verify_claim(test_claim)
    assert res.claim_id == "TEST-CLAIM-01"
    assert res.derived_status == "UNIT_TESTED"
    assert res.status_verified is True
    assert len(res.implementation_hashes) == 1
