"""
nsa.verifier.speculative
========================
Speculative State Auditing & Multi-Layer Residual Stream Deep Probing.

Monitors intermediate and final hidden states across autoregressive generation chunks,
detecting policy violations before they manifest in final token logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch

from nsa.algebra import StateLabel

from .encoder_head import StateEncoderHead


@dataclass
class AuditResult:
    """Structured result of a speculative state audit over a token chunk."""

    is_valid: bool
    violation_token_idx: Optional[int] = None
    violation_layer: Optional[int] = None
    predicted_states: Optional[List[int]] = None

    def __iter__(self):
        """Allow legacy 3-tuple / 4-tuple unpacking for backward compatibility."""
        return iter(
            (self.is_valid, self.violation_token_idx, self.predicted_states, self.violation_layer)
        )

    def __getitem__(self, idx: int):
        return (
            self.is_valid,
            self.violation_token_idx,
            self.predicted_states,
            self.violation_layer,
        )[idx]


class SpeculativeStateAuditor:
    """Audits hidden state representations across token generation chunks.

    Supports:
    1. Single-layer final hidden state classification.
    2. Multi-layer intermediate residual probing with early-exit violation detection.
    3. Custom lattice validator predicates.
    """

    def __init__(
        self,
        encoder_head: StateEncoderHead,
        lattice_validator: Optional[Callable[[Union[int, torch.Tensor]], bool]] = None,
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
        """Audit a generated chunk of tokens across residual hidden states.

        Args:
            hidden_states: Tensor of shape `[batch, K, hidden_size]` or `[batch, K, num_layers, hidden_size]`.
            current_state: Active StateLabel level.

        Returns:
            AuditResult containing validation status, violation index, and probe layer.
        """
        # Multi-Layer 4D Tensor: [batch, K, num_layers, hidden_size]
        if hidden_states.dim() == 4:
            _b, k_len, num_layers, _h = hidden_states.shape
            for l_idx in range(num_layers):
                layer_hidden = hidden_states[:, :, l_idx, :]
                logits = self.encoder_head(layer_hidden, async_execution=False)
                predicted = torch.argmax(logits, dim=-1)  # [batch, K]
                layer_id = self.probe_layers[l_idx] if l_idx < len(self.probe_layers) else l_idx

                for k in range(k_len):
                    pred_val = int(predicted[0, k].item())
                    if not self._validate_prediction(pred_val):
                        return AuditResult(
                            is_valid=False,
                            violation_token_idx=k,
                            violation_layer=layer_id,
                            predicted_states=predicted[0].tolist(),
                        )

            return AuditResult(is_valid=True)

        # Single-Layer 3D Tensor: [batch, K, hidden_size]
        logits = self.encoder_head(hidden_states, async_execution=False)
        predicted = torch.argmax(logits, dim=-1)  # [batch, K]
        k_len = hidden_states.shape[1]

        for k in range(k_len):
            pred_val = int(predicted[0, k].item())
            if not self._validate_prediction(pred_val):
                return AuditResult(
                    is_valid=False,
                    violation_token_idx=k,
                    violation_layer=self.probe_layers[-1] if self.probe_layers else -1,
                    predicted_states=predicted[0].tolist(),
                )

        return AuditResult(is_valid=True, predicted_states=predicted[0].tolist())


# Alias for explicit multi-layer semantic usage
MultiLayerStateAuditor = SpeculativeStateAuditor
