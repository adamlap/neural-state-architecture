"""
nsa.verifier.recovery
=====================
RecoveryPolicy & Native Recovery Adapters for Speculative Violation Interception.

Replaces brittle text-based prompt injection with weight-level adapter hot-swapping
and customizable rollback recovery strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch import nn


class RecoveryPolicy(ABC):
    """Abstract base class for recovery strategies following a lattice violation rollback."""

    @abstractmethod
    def on_violation(
        self,
        model: nn.Module,
        tokenizer: Any,
        violation_idx: int,
        violation_layer: Optional[int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Execute recovery action.

        Returns:
            Tuple[Optional[torch.Tensor], bool]:
                - next_input_ids: Optional input IDs to prime the next forward pass.
                - should_continue: True to continue generation, False to halt immediately.
        """
        raise NotImplementedError


class SemanticPivotRecovery(RecoveryPolicy):
    """Injects a structured system steering sequence into KV context to force refusal."""

    def __init__(self, pivot_text: str, max_pivots: int = 2):
        self.pivot_text = pivot_text
        self.max_pivots = max_pivots
        self._pivot_count = 0

    def on_violation(
        self,
        model: nn.Module,
        tokenizer: Any,
        violation_idx: int,
        violation_layer: Optional[int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if self._pivot_count >= self.max_pivots:
            return None, False

        self._pivot_count += 1
        pivot_ids = tokenizer.encode(
            self.pivot_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        return pivot_ids, True

    def reset(self):
        self._pivot_count = 0


class AdapterSwitchRecovery(RecoveryPolicy):
    """Hot-swaps model weights to a dedicated refusal/recovery adapter at the parameter level."""

    def __init__(
        self,
        recovery_refusal_text: str = "\nI cannot provide this restricted information.",
        adapter_name: Optional[str] = "recovery_adapter",
    ):
        self.recovery_refusal_text = recovery_refusal_text
        self.adapter_name = adapter_name

    def on_violation(
        self,
        model: nn.Module,
        tokenizer: Any,
        violation_idx: int,
        violation_layer: Optional[int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        # If PEFT active adapter switching is supported:
        if hasattr(model, "set_adapter") and self.adapter_name in getattr(model, "peft_config", {}):
            model.set_adapter(self.adapter_name)

        refusal_ids = tokenizer.encode(self.recovery_refusal_text, return_tensors="pt").to(device)
        return refusal_ids, True


class HaltRecovery(RecoveryPolicy):
    """Strict security policy: immediately terminates generation upon first violation."""

    def on_violation(
        self,
        model: nn.Module,
        tokenizer: Any,
        violation_idx: int,
        violation_layer: Optional[int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        return None, False
