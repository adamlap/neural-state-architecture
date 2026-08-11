"""
nsa.lora
========
NSA-LoRA: Low-Rank Post-Hoc Retrofitting Adapters for Pre-Trained Transformers.

Concept:
    Allows pre-trained language models (Llama-3, Qwen-2.5, Mistral, GPT-2) to be
    retrofitted with Neural State Architecture policy governance WITHOUT full pre-training.

    Base semantic weights W_0 are FROZEN. Lightweight state transition matrices V
    and low-rank LoRA adapter matrices (A, B) are trained:
        W' = W_0 + (alpha / r) * (B · A)
        σ' = V σ

Trainable Parameter Ratio:
    < 0.5% of model parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import StateLattice, DEFAULT_LATTICE
from nsa.fused_attention import FusedStateAwareAttention


class NSALoRALinear(nn.Module):
    """Low-Rank Adapter wrapper for frozen Linear layers with state operator injection."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        # Freeze base linear layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features, out_features = base_layer.in_features, base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


class NSALoRAAttention(nn.Module):
    """Retrofits standard MultiheadAttention into Fused State-Aware Attention via LoRA."""

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        num_heads: int = 8,
        r: int = 8,
        lora_alpha: float = 16.0,
        gate_mode: str = "hard",
        lattice: StateLattice = DEFAULT_LATTICE,
    ) -> None:
        super().__init__()
        self.fused_attn = FusedStateAwareAttention(
            d_model=d_model,
            state_dim=state_dim,
            num_heads=num_heads,
            gate_mode=gate_mode,
            lattice=lattice,
        )

        # Wrap Q, K, V projections in LoRA adapters
        self.fused_attn.W_q = NSALoRALinear(self.fused_attn.W_q, r=r, lora_alpha=lora_alpha)
        self.fused_attn.W_k = NSALoRALinear(self.fused_attn.W_k, r=r, lora_alpha=lora_alpha)
        self.fused_attn.W_v = NSALoRALinear(self.fused_attn.W_v, r=r, lora_alpha=lora_alpha)
        self.fused_attn.W_o = NSALoRALinear(self.fused_attn.W_o, r=r, lora_alpha=lora_alpha)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fused_attn(x, state, mask=mask)


# Common projection attribute names across HF / custom transformers
_LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "W_q", "W_k", "W_v", "W_o",
    "query", "key", "value", "out_proj",
    "c_attn", "c_proj",
)


def _count_param_stats(model: nn.Module) -> Dict[str, float]:
    """Count total / trainable / frozen params without double-counting."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "pct_trainable": (trainable / max(total, 1)) * 100.0,
    }


def apply_nsa_lora_retrofit(
    model: nn.Module,
    state_dim: int = 8,
    r: int = 8,
    lora_alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    add_state_emb: bool = True,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Freeze base weights and wrap target Linear layers with NSA-LoRA adapters.

    Parameters
    ----------
    model : nn.Module
        Model to retrofit in-place.
    state_dim : int
        Dimensionality of optional state embedding table.
    r, lora_alpha : LoRA rank / scale.
    target_modules : optional list of attribute suffixes to wrap.
        Defaults to common attention projection names.
    add_state_emb : if True, attach ``model.state_emb`` when missing.

    Returns
    -------
    model, param_stats
    """
    targets = tuple(target_modules) if target_modules else _LORA_TARGET_SUFFIXES

    # 1. Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # 2. Walk modules and wrap matching Linear children
    replaced = 0
    for module_name, module in list(model.named_modules()):
        for attr in targets:
            if not hasattr(module, attr):
                continue
            child = getattr(module, attr)
            if isinstance(child, NSALoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, attr, NSALoRALinear(child, r=r, lora_alpha=lora_alpha))
                replaced += 1

    # 3. Optional trainable state embedding for label → σ lookup
    if add_state_emb and not hasattr(model, "state_emb"):
        emb = nn.Embedding(64, state_dim)
        # Canonical init: dim-0 carries discrete label identity for labels 0..5
        with torch.no_grad():
            emb.weight.zero_()
            for lab in range(min(6, emb.num_embeddings)):
                emb.weight[lab, 0] = float(lab)
        model.state_emb = emb
        for p in model.state_emb.parameters():
            p.requires_grad = True

    stats = _count_param_stats(model)
    stats["layers_wrapped"] = float(replaced)
    if replaced == 0:
        # Still valid if caller only wanted freeze + state_emb, but surface a signal
        stats["warning"] = 1.0
    return model, stats


# ---------------------------------------------------------------------------
# DynamicNSARetrofitBlock (Multi-Path Gating: Attention + Residual + FFN)
# ---------------------------------------------------------------------------

class DynamicNSARetrofitBlock(nn.Module):
    """Retrofit block with *selective* multi-path policy interventions.

    Ablation flags (all measured independently):
      gate_attention : if True, hard/soft state mask in attention; else plain residual attn
      gate_residual  : if True, h' = h + sigmoid(W_Γ σ) ⊙ Attn(h); else h + Attn(h)
      gate_ffn       : if True, multiply by sigmoid(W_ffn[σ;h]); else identity
      learn_sigma    : if True, update non-security coords of σ; else pass-through
      fixed_alpha    : if set, use constant coupling α; if None, learn α via sigmoid head

    Coupling (when learn_sigma):
      σ_{l+1}[...,1:] = LN(σ + α · Δ)[...,1:]
      σ_{l+1}[...,0]  = σ[...,0]   # hard security coordinate never moves

    This block is for controlled experiments — not a claim of whole-model NI.
    """

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        num_heads: int = 8,
        r: int = 8,
        lora_alpha: float = 16.0,
        gate_attention: bool = True,
        gate_residual: bool = True,
        gate_ffn: bool = True,
        learn_sigma: bool = True,
        init_alpha: float = 0.01,
        fixed_alpha: Optional[float] = None,
        attn_gate_mode: str = "hard",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.gate_attention = gate_attention
        self.gate_residual = gate_residual
        self.gate_ffn = gate_ffn
        self.learn_sigma = learn_sigma
        self.fixed_alpha = fixed_alpha
        self.attn_gate_mode = attn_gate_mode

        # Attention path: always allocate NSA-LoRA attention so param counts
        # stay comparable; gate_attention toggles whether state mask is applied.
        self.nsa_attn = NSALoRAAttention(
            d_model=d_model,
            state_dim=state_dim,
            num_heads=num_heads,
            r=r,
            lora_alpha=lora_alpha,
            gate_mode=attn_gate_mode,
        )
        if not gate_attention:
            # Disable policy mask while keeping the same LoRA-wrapped projections.
            self.nsa_attn.fused_attn.gate_mode = "off"

        if gate_residual:
            self.residual_gate = nn.Linear(state_dim, d_model)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.zeros_(self.residual_gate.bias)

        if gate_ffn:
            self.ffn_gate = nn.Linear(state_dim + d_model, d_model)
            nn.init.zeros_(self.ffn_gate.weight)
            nn.init.zeros_(self.ffn_gate.bias)

        if learn_sigma:
            if fixed_alpha is None:
                self.state_alpha = nn.Linear(state_dim + d_model, 1)
                a0 = min(max(float(init_alpha), 1e-6), 1.0 - 1e-6)
                init_bias = math.log(a0 / (1.0 - a0))
                nn.init.constant_(self.state_alpha.bias, init_bias)
                nn.init.zeros_(self.state_alpha.weight)
            else:
                self.register_buffer(
                    "_fixed_alpha",
                    torch.tensor(float(fixed_alpha), dtype=torch.float32),
                )
            self.state_transition = nn.Linear(state_dim + d_model, state_dim)
            nn.init.zeros_(self.state_transition.weight)
            nn.init.zeros_(self.state_transition.bias)
            self.state_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        h: torch.Tensor,                # [B, T, d_model]
        sigma: torch.Tensor,            # [B, T, state_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hard_sec = sigma[..., 0:1]

        # 1. Attention (state-masked iff gate_attention)
        attn_out, _ = self.nsa_attn(h, sigma, mask=mask)

        # 2. Residual path
        if self.gate_residual:
            res_gamma = torch.sigmoid(self.residual_gate(sigma))
            h_res = h + res_gamma * attn_out
        else:
            h_res = h + attn_out

        # 3. FFN-style multiplicative gate on residual stream
        if self.gate_ffn:
            ffn_gamma = torch.sigmoid(self.ffn_gate(torch.cat([sigma, h_res], dim=-1)))
            h_out = h_res * ffn_gamma
        else:
            h_out = h_res

        # 4. Optional learned σ (security coord frozen)
        if not self.learn_sigma:
            return h_out, sigma

        if self.fixed_alpha is not None:
            alpha = self._fixed_alpha.to(dtype=h_out.dtype, device=h_out.device)
        else:
            alpha = torch.sigmoid(self.state_alpha(torch.cat([sigma, h_out], dim=-1)))
        delta_sigma = self.state_transition(torch.cat([sigma, h_out], dim=-1))
        sigma_next = self.state_norm(sigma + alpha * delta_sigma)
        sigma_next = torch.cat([hard_sec, sigma_next[..., 1:]], dim=-1)
        return h_out, sigma_next
