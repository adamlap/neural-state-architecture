"""
nsa.mask_injector
=================
First-class Context Manager for NSA Attention Mask Injection in HuggingFace/PyTorch Models.

Intercepts self-attention modules at runtime, applying additive NSA security lattice masks
to enforce non-interference across confidentiality and integrity levels.

Supports:
1. Static prefill attention masking.
2. Dynamic state tracking expansion (update_state) during autoregressive decoding.
3. Both Eager 4D attention masks and SDPA (Scaled Dot-Product Attention) modes.
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from nsa.algebra import build_level_attention_mask


class NSAMaskInjector:
    """Context manager that injects NSA policy masks via attention-layer hooks.

    Unlike wrapping ``model.forward()`` (which only sees 2D padding masks),
    this hooks each **attention layer** where HuggingFace has already expanded
    the mask to 4D ``[B, 1, T, T]`` — the correct interception point.

    During KV-cached decode steps the mask shape becomes ``[B, 1, 1, T+n]``;
    the hook slices the pre-computed NSA mask to match.

    Usage::

        with NSAMaskInjector(model, state_levels):
            output = model.generate(...)
    """

    def __init__(
        self,
        model: nn.Module,
        state_levels: torch.Tensor,
        decode_row_idx: int = 0,
        gate_mode: str = "hard",
        alpha: float = 10.0,
        temperature: float = 1.0,
    ):
        self.model = model
        self.state_levels = state_levels
        self.decode_row_idx = decode_row_idx
        self.gate_mode = gate_mode
        self.alpha = alpha
        self.temperature = temperature
        self.nsa_mask: Optional[torch.Tensor] = None   # pre-computed [B, 1, T, T]
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def update_state(self, new_level: int):
        """Dynamically append a new state level for newly generated tokens and recompute the mask."""
        device = self.state_levels.device
        new_tensor = torch.tensor([[new_level]], device=device, dtype=self.state_levels.dtype)
        self.state_levels = torch.cat([self.state_levels, new_tensor], dim=1)

        self.nsa_mask = build_level_attention_mask(
            self.state_levels,
            gate_mode=self.gate_mode,
            alpha=self.alpha,
            temperature=self.temperature,
        ).to(device)

        # Update decode row index to point to the newest token
        self.decode_row_idx = self.state_levels.shape[1] - 1

    def snapshot(self) -> Tuple[torch.Tensor, int]:
        """Snapshot injector state levels and decode index for atomic transactional rollback."""
        return (self.state_levels.clone(), self.decode_row_idx)

    def restore(self, snap: Tuple[torch.Tensor, int]) -> None:
        """Restore injector state levels and decode index, recomputing the attention mask."""
        st_levels, idx = snap
        self.state_levels = st_levels.clone()
        self.decode_row_idx = idx
        device = self.state_levels.device
        self.nsa_mask = build_level_attention_mask(
            self.state_levels,
            gate_mode=self.gate_mode,
            alpha=self.alpha,
            temperature=self.temperature,
        ).to(device)

    # ------------------------------------------------------------------ #
    # Hook that merges the NSA mask into the attention_mask kwarg
    # ------------------------------------------------------------------ #
    def _make_hook(self):
        """Return a ``forward_pre_hook`` closure that captures ``self``.

        Handles both cases:
        - **Eager attention**: ``attention_mask`` is a 4D tensor -> merge NSA mask.
        - **SDPA attention**: ``attention_mask`` is ``None`` -> build causal + NSA
          mask from scratch, which forces SDPA to use ``is_causal=False``.
        """
        injector = self

        def hook(_module, args, kwargs):
            if injector.nsa_mask is None:
                return args, kwargs

            attention_mask = kwargs.get("attention_mask", None)
            hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hidden_states is None:
                return args, kwargs
            device = hidden_states.device
            dtype = hidden_states.dtype
            q_len = hidden_states.shape[1]
            prompt_len = injector.nsa_mask.shape[-1]

            # Determine k_len (total key sequence length including KV-cache)
            past_kv = kwargs.get("past_key_value", None)
            cache_pos = kwargs.get("cache_position", None)
            if cache_pos is not None and len(cache_pos) > 0:
                k_len = int(cache_pos[-1].item()) + 1
            elif past_kv is not None:
                if hasattr(past_kv, "get_seq_length"):
                    k_len = past_kv.get_seq_length() + q_len
                elif isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
                    k_len = past_kv[0][0].shape[-2] if isinstance(past_kv[0], (tuple, list)) else past_kv[0].shape[-2]
                    k_len += q_len
                else:
                    k_len = q_len
            else:
                k_len = q_len

            if attention_mask is None:
                # SDPA mode: no explicit mask. Build causal + NSA from scratch
                causal = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
                causal = torch.triu(causal, diagonal=k_len - q_len + 1)
                causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, q, k]

                nsa_component = injector._slice_nsa(q_len, k_len, prompt_len, device, dtype)
                if nsa_component is None:
                    return args, kwargs

                kwargs["attention_mask"] = causal + nsa_component
                return args, kwargs

            if attention_mask.dim() < 4:
                return args, kwargs

            # Eager mode: 4D mask already provided
            _b2, _h, q2, k2 = attention_mask.shape
            nsa_component = injector._slice_nsa(q2, k2, prompt_len, device, dtype)
            if nsa_component is None:
                return args, kwargs

            kwargs["attention_mask"] = attention_mask + nsa_component
            return args, kwargs

        return hook

    def _slice_nsa(self, q_len: int, k_len: int, prompt_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return the NSA mask component shaped ``[1, 1, q_len, k_len]``."""
        if self.nsa_mask is None:
            return None
        if q_len == k_len:
            # Prefill step
            if k_len == prompt_len:
                return self.nsa_mask.to(device=device, dtype=dtype)
            elif k_len > prompt_len:
                return F.pad(self.nsa_mask, (0, k_len - prompt_len, 0, k_len - prompt_len)).to(device=device, dtype=dtype)
            else:
                return self.nsa_mask[:, :, :k_len, :k_len].to(device=device, dtype=dtype)
        elif q_len == 1:
            # KV-cache decode step: selected query row based on decode_row_idx
            row_idx = min(self.decode_row_idx, self.nsa_mask.shape[2] - 1)
            nsa_row = self.nsa_mask[:, :, row_idx:row_idx+1, :].to(device=device, dtype=dtype)
            if k_len > prompt_len:
                pad = torch.zeros(1, 1, 1, k_len - prompt_len, device=device, dtype=dtype)
                return torch.cat([nsa_row, pad], dim=-1)
            else:
                return nsa_row[:, :, :, :k_len]
        return None

    # ------------------------------------------------------------------ #
    # Helpers to locate attention sub-modules
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_attention_modules(model: nn.Module) -> Generator[nn.Module, None, None]:
        """Yield every self-attention sub-module in the model."""
        possible_stacks = [
            ("model", "layers"),
            ("transformer", "h"),
            ("model", "decoder", "layers"),
            ("layers",),
        ]
        found = False
        for path in possible_stacks:
            try:
                stack = model
                for attr in path:
                    stack = getattr(stack, attr)
                for block in stack:
                    for name in ("self_attn", "attn", "attention"):
                        if hasattr(block, name):
                            yield getattr(block, name)
                            found = True
                            break
                if found:
                    return
            except AttributeError:
                continue

        # Fallback: any module whose name ends with self_attn / attn / attention
        for name, mod in model.named_modules():
            if any(name.endswith(suffix) for suffix in ("self_attn", "attn", "attention")):
                yield mod

    # ------------------------------------------------------------------ #
    # Context manager entry / exit
    # ------------------------------------------------------------------ #
    def __enter__(self) -> NSAMaskInjector:
        # Pre-compute NSA additive mask [B, 1, T, T]
        device = next(self.model.parameters()).device
        self.nsa_mask = build_level_attention_mask(
            self.state_levels.to(device),
            gate_mode=self.gate_mode,
            alpha=self.alpha,
            temperature=self.temperature,
        )

        hook_fn = self._make_hook()
        for mod in self._find_attention_modules(self.model):
            handle = mod.register_forward_pre_hook(hook_fn, with_kwargs=True)
            self._hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self.nsa_mask = None
