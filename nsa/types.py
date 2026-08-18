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
        sigma_h (torch.Tensor): Hard structural state (Confidentiality, Integrity) [B, ..., d_state_h]
        sigma_s (torch.Tensor): Soft operational state (Confidence, Risk) [B, ..., d_state_s]
        nu (torch.Tensor): Preference/value alignment [B, ..., d_val]
    """

    m: torch.Tensor
    sigma_h: torch.Tensor
    sigma_s: torch.Tensor
    nu: torch.Tensor

    def join_with(self, other: "TypedTensor") -> "TypedTensor":
        """
        State composition for residual joins.
        m' = m1 + m2
        sigma_h' = max(sigma_h1, sigma_h2)  # Join on hard lattice (most restrictive)
        sigma_s' = min(sigma_s1, sigma_s2)  # Join on soft lattice (minimum confidence)
        nu' = (nu1 + nu2) / 2.0             # Mean pool value alignment
        """
        from nsa.algebra import join_hard_state_tensors, join_soft_state_tensors

        new_m = self.m + other.m

        # Authoritative Hard state product lattice join (supremum - most restrictive)
        new_sigma_h = join_hard_state_tensors(self.sigma_h, other.sigma_h)

        # Authoritative Soft state lattice meet (infimum - least confidence)
        new_sigma_s = join_soft_state_tensors(self.sigma_s, other.sigma_s)

        # Value alignment pooling
        new_nu = (self.nu + other.nu) / 2.0

        return TypedTensor(m=new_m, sigma_h=new_sigma_h, sigma_s=new_sigma_s, nu=new_nu)

    def to(self, *args, **kwargs) -> "TypedTensor":
        """Move all tensors to the specified device/dtype."""
        return TypedTensor(
            m=self.m.to(*args, **kwargs),
            sigma_h=self.sigma_h.to(*args, **kwargs),
            sigma_s=self.sigma_s.to(*args, **kwargs),
            nu=self.nu.to(*args, **kwargs),
        )
