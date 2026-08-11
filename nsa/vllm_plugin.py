"""
nsa.vllm_plugin
===============
Prototype helpers for attaching NSA policy masks to third-party inference stacks.

Status
------
This module is a **prototype**, not a production vLLM / SGLang plugin:

* It does not import or register with the real vLLM attention backend.
* ``register_vllm_attention_hook`` attaches generic PyTorch forward pre-hooks
  that merge an NSA state mask when callers pass ``state_vectors`` in kwargs.
* For production HF generate() masking, prefer
  ``prototype/llama_security_showcase.NSAMaskInjector``.
"""

from __future__ import annotations

from typing import Any, List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import StateLattice, DEFAULT_LATTICE, build_label_attention_mask
from nsa.kv_cache import NSAKVCache  # re-export for callers


class NSAvLLMAttentionPlugin(nn.Module):
    """Compute NSA additive policy masks for external attention backends."""

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        num_heads: int = 8,
        temperature: float = 1.0,
        gate_mode: str = "hard",
        lattice: StateLattice = DEFAULT_LATTICE,
        use_discrete_levels: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.gate_mode = gate_mode
        self.lattice = lattice
        self.use_discrete_levels = use_discrete_levels

        self.level_proj = nn.Linear(state_dim, 1, bias=False)
        with torch.no_grad():
            self.level_proj.weight.zero_()
            self.level_proj.weight[0, 0] = 1.0

    def compute_nsa_policy_mask(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return additive mask [B, 1, T_q, T_k]."""
        if self.use_discrete_levels and self.gate_mode == "hard":
            q_lab = query_states[..., 0].round().long().clamp(0, 5)
            k_lab = key_states[..., 0].round().long().clamp(0, 5)
            return build_label_attention_mask(
                q_lab, k_lab, lattice=self.lattice, forbidden_value=float("-inf")
            )

        if self.use_discrete_levels:
            L_q = query_states[..., 0]
            L_k = key_states[..., 0]
        else:
            L_q = self.level_proj(query_states).squeeze(-1)
            L_k = self.level_proj(key_states).squeeze(-1)

        delta = (L_q.unsqueeze(2) - L_k.unsqueeze(1)) / max(self.temperature, 1e-5)
        if self.gate_mode == "hard":
            g = torch.sigmoid(delta).unsqueeze(1)
            return torch.zeros_like(g).masked_fill(g < 0.5, float("-inf"))
        return F.logsigmoid(delta).unsqueeze(1)

    def forward_with_state_governance(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        custom_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SDPA with NSA mask — reference path for engine integration tests."""
        state_mask = self.compute_nsa_policy_mask(query_states, key_states).to(dtype=Q.dtype)
        combined = state_mask if custom_attn_mask is None else state_mask + custom_attn_mask
        try:
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=combined, is_causal=False)
        except Exception:
            scale = Q.shape[-1] ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale + combined
            attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
            return torch.matmul(attn, V)


def register_vllm_attention_hook(
    model: nn.Module,
    state_dim: int = 8,
    gate_mode: str = "hard",
) -> List[Any]:
    """Attach NSA mask pre-hooks to modules whose names look like attention.

    Prototype only: real vLLM integration requires a custom Attention backend.
    """
    warnings.warn(
        "register_vllm_attention_hook is a prototype PyTorch hook helper, "
        "not a production vLLM plugin.",
        UserWarning,
        stacklevel=2,
    )
    hooks: List[Any] = []
    plugin = NSAvLLMAttentionPlugin(state_dim=state_dim, gate_mode=gate_mode)

    for name, module in model.named_modules():
        lname = name.lower()
        if "self_attn" not in lname and "attention" not in lname and not lname.endswith("attn"):
            continue

        def make_pre_hook(p: NSAvLLMAttentionPlugin):
            def pre_hook(mod, args, kwargs):
                states = kwargs.get("state_vectors", None)
                if states is None:
                    return args, kwargs
                policy_mask = p.compute_nsa_policy_mask(states, states)
                if kwargs.get("attention_mask", None) is not None:
                    am = kwargs["attention_mask"]
                    if am.dtype != policy_mask.dtype:
                        policy_mask = policy_mask.to(dtype=am.dtype)
                    kwargs["attention_mask"] = am + policy_mask
                else:
                    kwargs["attention_mask"] = policy_mask
                return args, kwargs

            return pre_hook

        try:
            h = module.register_forward_pre_hook(make_pre_hook(plugin), with_kwargs=True)
            hooks.append(h)
        except TypeError:
            continue

    return hooks
