"""
nsa.layers
==========
NSATransformerBlock: the core composable building block.

Architecture of one NSA block:
                                    ┌──────────────────────────────┐
  ┌──────┐     ┌─────────────────┐  │  State Manifold              │
  │  m   │────▶│ StateAwareAttn  │  │  σ → V(σ) → σ' via Trans.Op │
  └──────┘     └────────┬────────┘  └──────────────────────────────┘
                        │ m'                        │ σ
               ┌────────▼────────┐        ┌────────▼────────┐
               │   LayerNorm     │        │  StateUpdate     │
               └────────┬────────┘        └────────┬────────┘
                        │                          │ σ'
               ┌────────▼────────┐                 │
               │   FFN + Gate    │◀────────────────┘
               │   Γ(σ') ⊙ FFN  │
               └────────┬────────┘
                        │ m''
                        ▼
                    (m'', σ')

Two coupled streams:
    Semantic stream (m): standard residual transformer blocks, gated by state
    State stream    (σ): a learned transition operator per layer

The streams interact via:
    1. Attention gate:    g(σ_i, σ_j) modifies attention scores
    2. FFN gate:          Γ(σ') = sigmoid(W_s σ') multiplies FFN output
    3. State update:      σ' = V(σ) — state can use m as conditioning
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from nsa.algebra import DEFAULT_LATTICE, StateLattice
from nsa.attention import StateAwareAttention
from nsa.state import SemanticGate, StateTransitionOperator
from nsa.types import TypedTensor

# ---------------------------------------------------------------------------
# State Update Network
# ---------------------------------------------------------------------------


class StateUpdateNetwork(nn.Module):
    """Computes the new state σ' from the current state σ and (optionally) meaning m.

    σ' = LayerNorm(σ + V(σ) + W_mix * pool(m))

    where pool(m) is a mean-pooled summary of the semantic stream injected
    as conditioning. This allows the state to respond to semantic content
    (e.g. if the model "sees" a sensitive keyword, confidence can drop).

    The dominant term is V(σ) — the transition operator — which enforces
    monotone conservation laws. The semantic conditioning is a small residual.
    """

    def __init__(
        self,
        state_dim: int = 8,
        d_model: int = 128,
        condition_on_semantics: bool = True,
    ) -> None:
        super().__init__()
        self.condition_on_semantics = condition_on_semantics
        self.transition = StateTransitionOperator(state_dim=state_dim, monotone_clamp=True)
        self.norm = nn.LayerNorm(state_dim)

        if condition_on_semantics:
            self.mix = nn.Linear(d_model, state_dim, bias=False)
            nn.init.zeros_(self.mix.weight)  # Start neutral; let training decide

    def forward(self, state: torch.Tensor, meaning: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state   : [B, T, state_dim]
        meaning : [B, T, d_model]

        Returns
        -------
        state'  : [B, T, state_dim]

        Security invariant: coordinate 0 (discrete security level) is preserved
        so hard lattice masks remain valid across layers.
        """
        hard_sec = state[..., 0:1]
        delta = self.transition(state)  # [B, T, state_dim]

        if self.condition_on_semantics:
            # Inject a weak semantic conditioning signal
            sem_summary = self.mix(meaning)  # [B, T, state_dim]
            delta = delta + 0.1 * sem_summary  # small coefficient keeps state stable

        updated = self.norm(state + delta)
        # Restore hard security coordinate (non-decreasing optional clamp)
        updated = torch.cat([hard_sec, updated[..., 1:]], dim=-1)
        return updated


# ---------------------------------------------------------------------------
# Feed-Forward Network with State Gate
# ---------------------------------------------------------------------------


class GatedFFN(nn.Module):
    """Position-wise FFN with a state-dependent gate.

    Output = Γ(σ) ⊙ (W_2 GELU(W_1 x))

    The gate Γ(σ) is a sigmoid gating function that controls how much of
    the FFN output propagates, conditioned on the current state.
    Low-trust or high-security states can suppress certain transformations.
    """

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.gate = SemanticGate(d_model, state_dim)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : [B, T, d_model]
        state : [B, T, state_dim]

        Returns
        -------
        Tensor [B, T, d_model]
        """
        h = self.fc2(self.drop(F.gelu(self.fc1(x))))
        h = self.gate(h, state)  # State gates the FFN output
        return self.norm(x + h)  # Residual + norm


# ---------------------------------------------------------------------------
# NSA Transformer Block
# ---------------------------------------------------------------------------


class NSATransformerBlock(nn.Module):
    """One complete NSA transformer block.

    Processes a paired (semantic, state) stream through:
        1. State-aware multi-head attention (semantic + state interact)
        2. State update network (state evolves via transition operator)
        3. Gated feed-forward network (semantic gated by new state)

    Parameters
    ----------
    d_model      : int   — semantic dimension
    state_dim    : int   — state vector dimension
    num_heads    : int   — attention heads
    compat_mode  : str   — attention compatibility: 'dot', 'mlp', 'level'
    gate_mode    : str   — attention gate: 'soft' or 'hard'
    ffn_expansion: int   — FFN hidden expansion factor
    dropout      : float — dropout rate
    lattice      : StateLattice — conservation law lattice
    condition_on_semantics : bool — let state update see semantic stream
    """

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        num_heads: int = 8,
        compat_mode: str = "level",
        gate_mode: str = "hard",
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        lattice: StateLattice = DEFAULT_LATTICE,
        condition_on_semantics: bool = True,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = StateAwareAttention(
            d_model=d_model,
            state_dim=state_dim,
            num_heads=num_heads,
            compat_mode=compat_mode,
            gate_mode=gate_mode,
            lattice=lattice,
        )
        self.state_update = StateUpdateNetwork(
            state_dim=state_dim,
            d_model=d_model,
            condition_on_semantics=condition_on_semantics,
        )
        self.ffn = GatedFFN(
            d_model=d_model,
            state_dim=state_dim,
            expansion=ffn_expansion,
            dropout=dropout,
        )

    def forward(
        self,
        typed_x: TypedTensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TypedTensor:
        """
        Returns
        -------
        TypedTensor — updated semantic and state streams
        """
        x, sigma_h, sigma_s, nu = typed_x.m, typed_x.sigma_h, typed_x.sigma_s, typed_x.nu

        # --- Attention ---
        x_norm = self.attn_norm(x)
        # Pass sigma_h for structural non-interference
        attn_out, _ = self.attn(x_norm, sigma_h, mask=mask)

        # Mathematically strict residual join: m' = m + attn, σ_h' = σ_h ⊔ σ_h_attn
        x_updated = typed_x.join_with(
            TypedTensor(m=attn_out, sigma_h=sigma_h, sigma_s=sigma_s, nu=nu)
        )
        x, sigma_h, sigma_s, nu = x_updated.m, x_updated.sigma_h, x_updated.sigma_s, x_updated.nu

        # --- State update (uses pre-residual meaning as conditioning) ---
        sigma_h = self.state_update(sigma_h, x)

        # --- Gated FFN ---
        # Gated FFN relies on operational confidence (sigma_s) or hard state
        ffn_out = self.ffn(x, sigma_h)

        # Mathematically strict residual join: m'' = m' + ffn, σ'' = σ' ⊔ σ'_ffn
        final_typed = TypedTensor(m=x, sigma_h=sigma_h, sigma_s=sigma_s, nu=nu).join_with(
            TypedTensor(m=ffn_out, sigma_h=sigma_h, sigma_s=sigma_s, nu=nu)
        )

        return final_typed


# ---------------------------------------------------------------------------
# Full NSA Transformer (stack of blocks)
# ---------------------------------------------------------------------------


class NSATransformer(nn.Module):
    """A complete NSA transformer model.

    Wraps N NSATransformerBlock layers with token + positional embeddings.
    Suitable as a drop-in semantic encoder with state tracking.

    Parameters
    ----------
    vocab_size   : int   — vocabulary size
    d_model      : int   — model dimension
    state_dim    : int   — state vector dimension
    num_layers   : int   — number of NSA blocks
    num_heads    : int   — attention heads per block
    max_seq_len  : int   — maximum sequence length (for positional embedding)
    compat_mode  : str   — attention compatibility mode
    gate_mode    : str   — attention gate mode
    dropout      : float — dropout
    lattice      : StateLattice
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        state_dim: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        compat_mode: str = "level",
        gate_mode: str = "hard",
        dropout: float = 0.0,
        lattice: StateLattice = DEFAULT_LATTICE,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # State is initialised as a learned per-position parameter
        self.state_emb = nn.Embedding(max_seq_len, state_dim)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                NSATransformerBlock(
                    d_model=d_model,
                    state_dim=state_dim,
                    num_heads=num_heads,
                    compat_mode=compat_mode,
                    gate_mode=gate_mode,
                    dropout=dropout,
                    lattice=lattice,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,  # [B, T]  (token ids)
        state_init: Optional[torch.Tensor] = None,  # [B, T, state_dim] — optional override
        mask: Optional[torch.Tensor] = None,
    ) -> TypedTensor:
        """
        Returns
        -------
        TypedTensor — final semantic representations and state stream
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)  # [1, T]

        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        sigma = state_init if state_init is not None else self.state_emb(pos).expand(B, T, -1)

        # Initialize default components if not provided
        sigma_s = torch.ones(B, T, 1, device=x.device)
        nu = torch.zeros(B, T, 1, device=x.device)

        typed_x = TypedTensor(m=x, sigma_h=sigma, sigma_s=sigma_s, nu=nu)

        for block in self.blocks:
            typed_x = block(typed_x, mask=mask)

        # Final LayerNorm applies only to semantic stream
        final_x = self.ln_f(typed_x.m)
        return TypedTensor(
            m=final_x, sigma_h=typed_x.sigma_h, sigma_s=typed_x.sigma_s, nu=typed_x.nu
        )


# ---------------------------------------------------------------------------
# NSACausalLM
# ---------------------------------------------------------------------------


class NSACausalLM(nn.Module):
    """Autoregressive Causal Language Model with Dual-Stream State Governance.

    Integrates causal lower-triangular attention masking with state-aware attention.
    Used for language modeling pre-training, evaluation, and generation.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 128,
        state_dim: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        compat_mode: str = "level",
        gate_mode: str = "hard",
        dropout: float = 0.1,
        lattice: StateLattice = DEFAULT_LATTICE,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim

        self.nsa = NSATransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            state_dim=state_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            compat_mode=compat_mode,
            gate_mode=gate_mode,
            dropout=dropout,
            lattice=lattice,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.nsa.tok_emb.weight

    def forward(
        self,
        tokens: torch.Tensor,  # [B, T]
        state_init: Optional[torch.Tensor] = None,  # [B, T, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits     : [B, T, vocab_size] — next-token probability logits
        x          : [B, T, d_model] — semantic hidden state
        final_state: [B, T, state_dim] — state stream after final layer
        """
        B, T = tokens.shape
        device = tokens.device

        # Causal mask: [1, 1, T, T] (1 = allowed, 0 = masked)
        causal_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        typed_final = self.nsa(tokens, state_init=state_init, mask=causal_mask)
        logits = self.lm_head(typed_final.m)
        return logits, typed_final.m, typed_final.sigma_h
