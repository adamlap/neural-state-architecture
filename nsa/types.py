"""
nsa.types
=========
Type definitions for Typed Neural Computation (TNC).

Encapsulates semantic data and its structural state metadata into a single
strongly-typed object. Operations on TypedTensor must mathematically respect
the state invariants.
"""

from typing import NamedTuple
import torch

class TypedTensor(NamedTuple):
    """
    A typed neural representation encapsulating semantic state and structural state.

    Attributes:
        m (torch.Tensor): Semantic stream tensor [B, ..., d_model]
        sigma (torch.Tensor): Structural state tensor [B, ..., state_dim]
    """
    m: torch.Tensor
    sigma: torch.Tensor

    def join_with(self, other: "TypedTensor") -> "TypedTensor":
        """
        State composition for residual joins.
        m' = m1 + m2
        sigma' = sigma1 ⊔ sigma2
        """
        new_m = self.m + other.m
        
        # Element-wise maximum performs a lattice join for the standard dimensions
        # (Confidentiality, Integrity, License Tier)
        new_sigma = torch.maximum(self.sigma, other.sigma)
        
        return TypedTensor(m=new_m, sigma=new_sigma)

    def to(self, *args, **kwargs) -> "TypedTensor":
        """Move both tensors to the specified device/dtype."""
        return TypedTensor(
            m=self.m.to(*args, **kwargs),
            sigma=self.sigma.to(*args, **kwargs)
        )
