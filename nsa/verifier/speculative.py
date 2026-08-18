"""
nsa.verifier.speculative
========================
Speculative State Auditor & Multi-Layer Residual Probing for NSA 2.0.

Provides:
- Checkpoint coverage auditing across intermediate transformer layers L_A = {l_1, ..., l_k}
- Early exit rollback trigger upon detecting lattice order violations
- Multi-batch evaluation (audits all b in [0, B-1])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch

from nsa.algebra import StateLabel

from .encoder_head import StateEncoderHead


@dataclass
class AuditResult:
    """Outcome of a speculative state audit over a generation chunk."""
    is_valid: bool
    violation_token_idx: Optional[int] = None
    violation_layer: Optional[int] = None
    violation_batch_idx: Optional[int] = None
    predicted_states: Optional[List[int]] = None

    def __iter__(self):
        """Enable tuple unpacking for backward compatibility: is_valid, idx, preds, layer."""
        return iter((self.is_valid, self.violation_token_idx, self.predicted_states, self.violation_layer))


class SpeculativeStateAuditor:
    """Evaluates multi-layer intermediate hidden states against state lattice constraints.

    Tier 2 Statistical Monitoring:
        Evaluates hidden states across checkpoint layers L_A = {l_1, ..., l_k}.
        Triggers early exit rollback if an intermediate probe detects an unsafe trajectory.
    """

    def __init__(
        self,
        encoder_head: StateEncoderHead,
        lattice_validator: Optional[Callable[[int], bool]] = None,
        chunk_size: int = 4,
        probe_layers: Optional[List[int]] = None,
    ):
        self.encoder_head = encoder_head
        self.lattice_validator = lattice_validator or (lambda pred: pred != StateLabel.SYSTEM.value)
        self.chunk_size = chunk_size
        self.probe_layers = probe_layers or [-1]

    def _validate_prediction(self, pred: Union[int, torch.Tensor]) -> bool:
        """Evaluate predicate for a state label prediction."""
        if isinstance(pred, torch.Tensor):
            res = self.lattice_validator(pred)
            return bool(torch.all(res).item()) if isinstance(res, torch.Tensor) else bool(res)
        return bool(self.lattice_validator(pred))

    def audit_chunk(
        self,
        hidden_states: torch.Tensor,
        current_state: Union[StateLabel, int] = StateLabel.CONFIDENTIAL,
    ) -> AuditResult:
        """Audit a generated chunk of tokens across residual hidden states for all batch sequences.

        Args:
            hidden_states: Tensor of shape `[batch, K, hidden_size]` or `[batch, K, num_layers, hidden_size]`.
            current_state: Active StateLabel level.

        Returns:
            AuditResult containing validation status, violation token index, layer, and batch index.
        """
        # Multi-Layer 4D Tensor: [batch, K, num_layers, hidden_size]
        if hidden_states.dim() == 4:
            batch_size, k_len, num_layers, _h = hidden_states.shape
            for l_idx in range(num_layers):
                layer_hidden = hidden_states[:, :, l_idx, :]
                logits = self.encoder_head(layer_hidden, async_execution=False)
                predicted = torch.argmax(logits, dim=-1)  # [batch, K]
                layer_id = self.probe_layers[l_idx] if l_idx < len(self.probe_layers) else l_idx

                # Audit across ALL batch sequences b in [0, batch_size-1]
                for b in range(batch_size):
                    for k in range(k_len):
                        pred_val = int(predicted[b, k].item())
                        if not self._validate_prediction(pred_val):
                            return AuditResult(
                                is_valid=False,
                                violation_token_idx=k,
                                violation_layer=layer_id,
                                violation_batch_idx=b,
                                predicted_states=predicted[b].tolist(),
                            )

            return AuditResult(is_valid=True)

        # Single-Layer 3D Tensor: [batch, K, hidden_size]
        batch_size, k_len, _h = hidden_states.shape
        logits = self.encoder_head(hidden_states, async_execution=False)
        predicted = torch.argmax(logits, dim=-1)  # [batch, K]

        # Audit across ALL batch sequences
        for b in range(batch_size):
            for k in range(k_len):
                pred_val = int(predicted[b, k].item())
                if not self._validate_prediction(pred_val):
                    return AuditResult(
                        is_valid=False,
                        violation_token_idx=k,
                        violation_layer=self.probe_layers[-1] if self.probe_layers else -1,
                        violation_batch_idx=b,
                        predicted_states=predicted[b].tolist(),
                    )

        return AuditResult(is_valid=True, predicted_states=predicted[0].tolist())


# Alias for explicit multi-layer semantic usage
MultiLayerStateAuditor = SpeculativeStateAuditor
