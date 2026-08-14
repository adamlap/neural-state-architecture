"""
nsa.residual_taint
==================
Residual-stream taint / join tracking for NSA.

Attention hard-masks block *reads*, but residual additions and FFN paths can
still mix representations.  This module tracks a discrete security level per
token position through residual joins:

    level_out = join(level_a, level_b)   # max for the default lattice

and optional declassification checks.  It is an analytical / runtime audit
helper — not a replacement for hard attention masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from nsa.algebra import DEFAULT_LATTICE, StateLabel, StateLattice, DeclassificationCapability


def join_levels(
    a: torch.Tensor,
    b: torch.Tensor,
    lattice: StateLattice = DEFAULT_LATTICE,
) -> torch.Tensor:
    """Elementwise lattice join on integer level tensors (default = max)."""
    if lattice._custom_laws or lattice._override_default:
        # Slow path: pairwise join
        flat_a = a.reshape(-1).long()
        flat_b = b.reshape(-1).long()
        out = torch.empty_like(flat_a)
        for i in range(flat_a.numel()):
            la = StateLabel(int(flat_a[i].item()))
            lb = StateLabel(int(flat_b[i].item()))
            out[i] = lattice.join(la, lb).value
        return out.reshape_as(a).to(dtype=a.dtype)
    return torch.maximum(a, b)


def meet_levels(
    a: torch.Tensor,
    b: torch.Tensor,
    lattice: StateLattice = DEFAULT_LATTICE,
) -> torch.Tensor:
    """Elementwise lattice meet on integer level tensors (default = min)."""
    if lattice._custom_laws or lattice._override_default:
        flat_a = a.reshape(-1).long()
        flat_b = b.reshape(-1).long()
        out = torch.empty_like(flat_a)
        for i in range(flat_a.numel()):
            la = StateLabel(int(flat_a[i].item()))
            lb = StateLabel(int(flat_b[i].item()))
            out[i] = lattice.meet(la, lb).value
        return out.reshape_as(a).to(dtype=a.dtype)
    return torch.minimum(a, b)


@dataclass
class TaintEvent:
    """Record of a residual join that raised a token's taint level."""

    position: Tuple[int, ...]
    before: int
    after: int
    source: str


class ResidualTaintTracker:
    """Track per-token security levels through residual joins.

    Parameters
    ----------
    initial_levels : LongTensor [B, T] (or [T])
        Starting discrete security labels.
    lattice : StateLattice
    """

    def __init__(
        self,
        initial_levels: torch.Tensor,
        lattice: StateLattice = DEFAULT_LATTICE,
    ) -> None:
        if initial_levels.dim() == 1:
            initial_levels = initial_levels.unsqueeze(0)
        self.levels = initial_levels.long().clone()
        self.lattice = lattice
        self.history: List[TaintEvent] = []

    @classmethod
    def from_state(
        cls,
        state: torch.Tensor,
        lattice: StateLattice = DEFAULT_LATTICE,
    ) -> "ResidualTaintTracker":
        """Build tracker from continuous state tensors (uses σ[..., 0])."""
        levels = state[..., 0].round().long().clamp(0, 5)
        return cls(levels, lattice=lattice)

    def residual_add(
        self,
        other_levels: torch.Tensor,
        *,
        source: str = "residual",
    ) -> torch.Tensor:
        """Join current levels with ``other_levels`` (e.g. attention output taint).

        Returns updated levels [B, T].
        """
        if other_levels.dim() == 1:
            other_levels = other_levels.unsqueeze(0)
        other_levels = other_levels.long()
        before = self.levels.clone()
        self.levels = join_levels(self.levels, other_levels, lattice=self.lattice)
        raised = (self.levels > before).nonzero(as_tuple=False)
        for idx in raised.tolist():
            pos = tuple(idx)
            self.history.append(
                TaintEvent(
                    position=pos,
                    before=int(before[pos]),
                    after=int(self.levels[pos]),
                    source=source,
                )
            )
        return self.levels

    def attention_output_taint(
        self,
        query_levels: torch.Tensor,
        key_levels: torch.Tensor,
        attn_probs: Optional[torch.Tensor] = None,
        mass_eps: float = 1e-6,
    ) -> torch.Tensor:
        """Conservative taint of attention outputs.

        Without probs: each query inherits join of all *readable* key levels.
        With probs [B,H,Tq,Tk] or [B,Tq,Tk]: join keys with mass > mass_eps.
        """
        if query_levels.dim() == 1:
            query_levels = query_levels.unsqueeze(0)
        if key_levels.dim() == 1:
            key_levels = key_levels.unsqueeze(0)
        B, Tq = query_levels.shape
        Tk = key_levels.shape[-1]
        device = query_levels.device

        # Readable mask under lattice can_attend
        q = query_levels.unsqueeze(2)  # [B,Tq,1]
        k = key_levels.unsqueeze(1)    # [B,1,Tk]
        readable = q >= k  # default monotone; matches DEFAULT_LATTICE

        if attn_probs is not None:
            if attn_probs.dim() == 4:
                mass = attn_probs.sum(dim=1)  # [B,Tq,Tk]
            else:
                mass = attn_probs
            readable = readable & (mass > mass_eps)

        # For each query, join all readable key levels (and keep at least query)
        out = query_levels.clone()
        for b in range(B):
            for i in range(Tq):
                lvl = int(query_levels[b, i].item())
                for j in range(Tk):
                    if readable[b, i, j]:
                        kj = int(key_levels[b, j].item())
                        if self.lattice._custom_laws or self.lattice._override_default:
                            lvl = self.lattice.join(StateLabel(lvl), StateLabel(kj)).value
                        else:
                            lvl = max(lvl, kj)
                out[b, i] = lvl
        return out.to(device=device)

    def assert_no_write_down(
        self,
        sink_levels: torch.Tensor,
        *,
        name: str = "sink",
    ) -> None:
        """Raise if any tracked token would write-down into ``sink_levels``."""
        if sink_levels.dim() == 1:
            sink_levels = sink_levels.unsqueeze(0)
        for b in range(self.levels.shape[0]):
            for t in range(self.levels.shape[1]):
                w = StateLabel(int(self.levels[b, t].item()))
                s = StateLabel(int(sink_levels[b, t].item()))
                if not self.lattice.can_write(w, s):
                    raise AssertionError(
                        f"write-down blocked at [{b},{t}]: {w.name} -/-> {s.name} ({name})"
                    )

    def declassify(
        self,
        positions: Sequence[Tuple[int, int]],
        target: StateLabel,
        *,
        capability: Optional["DeclassificationCapability"] = None,
    ) -> None:
        """Attempt declassification of selected (batch, time) positions."""
        for b, t in positions:
            src = StateLabel(int(self.levels[b, t].item()))
            if not self.lattice.can_declassify(src, target, capability=capability):
                raise PermissionError(
                    f"declassify {src.name} -> {target.name} denied at [{b},{t}] "
                    f"(capability={capability})"
                )
            self.levels[b, t] = target.value
            self.history.append(
                TaintEvent(
                    position=(b, t),
                    before=src.value,
                    after=target.value,
                    source="declassify" if capability else "declassify_denied",
                )
            )

    def violation_count(self, allowed_ceiling: torch.Tensor) -> int:
        """Count positions where tracked level exceeds an allowed ceiling."""
        if allowed_ceiling.dim() == 1:
            allowed_ceiling = allowed_ceiling.unsqueeze(0)
        return int((self.levels > allowed_ceiling.long()).sum().item())
