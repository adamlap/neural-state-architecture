"""
evidence/validate_evidence.py
=============================
Automated Auditor & Evidence Verification Engine for NSA Claims.

Audits `evidence/manifest.json` against the active codebase:
1. Computes SHA-256 artifact hashes for all implementation, test, and experiment files.
2. Derives and verifies epistemic status from active proof and sufficiency criteria.
3. Formats an Epistemic Claim Matrix distinguishing:
   - ROBUSTLY_VALIDATED
   - EMPIRICALLY_VALIDATED
   - UNIT_TESTED
   - IMPLEMENTED
   - FORMALLY_VERIFIED
   - OPEN_RESEARCH
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nsa.evidence.engine import EpistemicVerificationEngine


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_audit_report(audit_data: Dict[str, Any]) -> None:
    print("=" * 110)
    print("                    NSA FORMAL EVIDENCE & EPISTEMIC VERIFICATION MATRIX")
    print("=" * 110)
    print(f"{'Claim ID':<25} | {'Phase':<12} | {'Claimed Tier':<20} | {'Derived Tier':<20} | Status")
    print("-" * 110)

    for res in audit_data["verification_results"]:
        status_tag = "VERIFIED [OK]" if res["status_verified"] else "DISCREPANCY"
        print(
            f"{res['claim_id']:<25} | {res['phase']:<12} | {res['claimed_status']:<20} | "
            f"{res['derived_status']:<20} | {status_tag}"
        )

    print("=" * 110)
    print("Derived Epistemic Breakdown:")
    for status, count in audit_data["derived_status_breakdown"].items():
        print(f"  • {status:<24}: {count} claims")
    print("=" * 110)


def main():
    workspace_root = Path(__file__).resolve().parent.parent
    manifest_path = workspace_root / "evidence" / "manifest.json"

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    engine = EpistemicVerificationEngine(workspace_root)
    audit_data = engine.audit_manifest(manifest)
    print_audit_report(audit_data)

    if not audit_data["all_epistemic_statuses_derived_and_verified"]:
        print("Audit Warning: Epistemic status discrepancies found in manifest!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
