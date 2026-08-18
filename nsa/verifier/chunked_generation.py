"""
nsa.verifier.chunked_generation
===============================
Backward-compatible proxy exporting NSAGenerator and generate_with_auditor.
"""

from nsa.verifier.generation import NSAGenerator, generate_with_auditor

__all__ = ["NSAGenerator", "generate_with_auditor"]
