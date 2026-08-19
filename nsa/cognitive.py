"""Cognitive extension of the native NSA causal language model.

Adds a predictive self-state loop around the existing NSA transformer without
changing the hard state algebra or authority model. The self-model predicts
the current state from strictly previous information; prediction error is
then fed back into the semantic readout. The loop is opt-in and ablatable.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from nsa.layers import NSACausalLM
from nsa.self_model import CapabilityMonitor, PredictiveSelfState, SelfRegulationController


class NSACognitiveLM(nn.Module):
    """Native NSA LM plus predictive self-state and bounded metacognition."""

    def __init__(self, *args, self_state_feedback: bool = True, **kwargs) -> None:
        super().__init__()
        self.nsa = NSACausalLM(*args, **kwargs)
        self.self_state_feedback = self_state_feedback
        d_model = self.nsa.d_model
        state_dim = self.nsa.state_dim
        self.self_model = PredictiveSelfState(d_model, state_dim)
        self.regulation = SelfRegulationController()
        self.capability = CapabilityMonitor(d_model, state_dim)
        self.error_gate = nn.Sequential(nn.Linear(state_dim, d_model), nn.Tanh())

    def forward(
        self,
        tokens: torch.Tensor,
        state_init: Optional[torch.Tensor] = None,
        self_state_feedback: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        logits, hidden, state = self.nsa(tokens, state_init=state_init)
        enabled = self.self_state_feedback if self_state_feedback is None else self_state_feedback

        # Predict state_t using only information available before position t.
        predicted = torch.zeros_like(state)
        if state.shape[1] > 1:
            predicted[:, 1:] = self.self_model.predict(hidden[:, :-1], state[:, :-1])
        error = state - predicted
        error_signal = self.self_model.error_projection(error)
        if not enabled:
            error_signal = torch.zeros_like(error_signal)

        feedback = self.error_gate(error_signal)
        modulated_hidden = hidden + feedback if enabled else hidden
        logits = self.nsa.lm_head(modulated_hidden)

        regulation = self.regulation(error if enabled else torch.zeros_like(error))
        capability = self.capability(hidden, state)
        return {
            "logits": logits,
            "hidden": modulated_hidden,
            "base_hidden": hidden,
            "state": state,
            "predicted_state": predicted,
            "prediction_error": error,
            "error_signal": error_signal,
            "prediction_mse": error.pow(2).mean(dim=-1, keepdim=True),
            "confidence": regulation.confidence,
            "caution": regulation.caution,
            "request_reassessment": regulation.request_reassessment,
            "capability": capability,
        }


__all__ = ["NSACognitiveLM"]
