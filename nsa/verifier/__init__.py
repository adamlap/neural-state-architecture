"""
nsa.verifier
============
Speculative State Auditing, Deep Probing, and Dynamic Generation Engine for NSA 2.0.
"""

from .encoder_head import StateEncoderHead
from .generation import NSAGenerator, generate_with_auditor
from .recovery import (
    AdapterSwitchRecovery,
    HaltRecovery,
    RecoveryPolicy,
    SemanticPivotRecovery,
)
from .router import StreamRouter
from .speculative import AuditResult, MultiLayerStateAuditor, SpeculativeStateAuditor
from .tokenizer import TokenizerAligner
from .tokens import StateControlTokens

__all__ = [
    "AdapterSwitchRecovery",
    "AuditResult",
    "HaltRecovery",
    "MultiLayerStateAuditor",
    "NSAGenerator",
    "RecoveryPolicy",
    "SemanticPivotRecovery",
    "SpeculativeStateAuditor",
    "StateControlTokens",
    "StateEncoderHead",
    "StreamRouter",
    "TokenizerAligner",
    "generate_with_auditor",
]
