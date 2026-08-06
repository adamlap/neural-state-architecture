"""
nsa.algebra
===========
State algebra: lattice structure, partial order, conservation laws.

The core abstraction is a *state lattice* — a partially ordered set (S, ≤)
equipped with meet (⊓) and join (⊔) operations, forming a bounded lattice.

Every permitted state transition must be monotone with respect to the
lattice order defined by ConservationLaws. Concretely, a law of the form:

    Private → Public  is FORBIDDEN

is encoded by asserting that PRIVATE is *above* PUBLIC in the lattice, so
no downward step is possible without violating the monotone constraint.

The lattice used here is:
    SYSTEM
      │
   PRIVATE
      │
   CONFIDENTIAL
      │
   TRUSTED
      │
   PUBLIC
      │
   UNTRUSTED

Higher nodes are *more restricted*. Transitions may only go toward
*more restricted* states or stay the same (information can be classified
upward but not downward without an explicit declassification gate).

This mirrors the Bell-LaPadula model from information security,
but is formulated as differentiable algebra for use in neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# State Labels
# ---------------------------------------------------------------------------

class StateLabel(IntEnum):
    """Ordered state labels forming the default security lattice.

    Higher integer value = more restricted / higher classification.
    The ordering is: PUBLIC < TRUSTED < CONFIDENTIAL < PRIVATE < SYSTEM.
    """
    UNTRUSTED    = 0
    PUBLIC       = 1
    TRUSTED      = 2
    CONFIDENTIAL = 3
    PRIVATE      = 4
    SYSTEM       = 5

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------
    # Lattice operations (on the default linear order)
    # ------------------------------------------------------------------

    def meet(self, other: "StateLabel") -> "StateLabel":
        """Greatest lower bound (most permissive common state)."""
        return StateLabel(min(self.value, other.value))

    def join(self, other: "StateLabel") -> "StateLabel":
        """Least upper bound (most restrictive common state)."""
        return StateLabel(max(self.value, other.value))

    def allows_transition_to(self, target: "StateLabel") -> bool:
        """By default, information may become *more* restricted but not less.

        i.e. PRIVATE → SYSTEM is allowed (adding restriction).
             PRIVATE → PUBLIC  is forbidden (removing restriction).
        """
        return target.value >= self.value


# ---------------------------------------------------------------------------
# Conservation Law
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConservationLaw:
    """A single conservation law: 'from_state → to_state is {allowed/forbidden}'.

    Laws can express:
        - Monotone constraints: information can only be reclassified upward.
        - Explicit exceptions: a declassification gate may allow PRIVATE → PUBLIC
          iff a certain condition is met (modeled as a soft penalty weight).
    """
    from_label:   StateLabel
    to_label:     StateLabel
    allowed:      bool   = True
    penalty_weight: float = 1.0   # weight in the state constraint loss

    def __str__(self) -> str:
        arrow = "->" if self.allowed else "->X"
        return f"{self.from_label} {arrow} {self.to_label}"

    def is_violated(self, src: StateLabel, dst: StateLabel) -> bool:
        """Return True if a transition src->dst violates this law."""
        if src == self.from_label and dst == self.to_label:
            return not self.allowed
        return False


# ---------------------------------------------------------------------------
# State Lattice
# ---------------------------------------------------------------------------

class StateLattice:
    """A bounded lattice over StateLabels with configurable conservation laws.

    By default, uses the monotone upward-only rule (no downward reclassification).
    Additional custom laws can be added or default laws overridden.

    Examples
    --------
    >>> lattice = StateLattice()
    >>> lattice.is_allowed(StateLabel.PRIVATE, StateLabel.PUBLIC)
    False
    >>> lattice.is_allowed(StateLabel.PUBLIC, StateLabel.PRIVATE)
    True
    >>> lattice.is_allowed(StateLabel.PRIVATE, StateLabel.SYSTEM)
    True
    """

    def __init__(
        self,
        extra_laws: Optional[List[ConservationLaw]] = None,
        override_default: bool = False,
    ) -> None:
        self._custom_laws: List[ConservationLaw] = extra_laws or []
        self._override_default = override_default

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def is_allowed(self, src: StateLabel, dst: StateLabel) -> bool:
        """Check whether the transition src -> dst is permitted.

        Custom laws are checked first; they can explicitly permit or forbid.
        If no custom law matches, the default monotone rule applies.
        """
        for law in self._custom_laws:
            if law.from_label == src and law.to_label == dst:
                return law.allowed

        if self._override_default:
            # If all custom laws are exhausted and default is overridden,
            # fall back to permissive (allow anything not explicitly forbidden).
            return True

        # Default: information may only become *more* restricted.
        return src.allows_transition_to(dst)

    def violation_penalty(self, src: StateLabel, dst: StateLabel) -> float:
        """Return the penalty weight for a forbidden transition (0 if allowed)."""
        if self.is_allowed(src, dst):
            return 0.0
        for law in self._custom_laws:
            if law.from_label == src and law.to_label == dst and not law.allowed:
                return law.penalty_weight
        return 1.0  # default penalty weight

    # ------------------------------------------------------------------
    # Lattice operations
    # ------------------------------------------------------------------

    @staticmethod
    def meet(a: StateLabel, b: StateLabel) -> StateLabel:
        return a.meet(b)

    @staticmethod
    def join(a: StateLabel, b: StateLabel) -> StateLabel:
        return a.join(b)

    # ------------------------------------------------------------------
    # Derived utilities
    # ------------------------------------------------------------------

    def compatible(self, src: StateLabel, dst: StateLabel) -> bool:
        """Return True if src and dst can attend to each other.

        Symmetric: compatible(a, b) == compatible(b, a).
        Two tokens can interact iff *both* transitions are allowed
        (src->dst *and* dst->src). In practice this means they share
        the same level or one is more restrictive and information
        flows *toward* restriction.
        """
        return self.is_allowed(src, dst) and self.is_allowed(dst, src)

    def reachable_from(self, src: StateLabel) -> List[StateLabel]:
        """All labels reachable (via allowed transitions) from src."""
        return [s for s in StateLabel if self.is_allowed(src, s)]

    def summary(self) -> str:
        lines = ["StateLattice transition table:"]
        labels = list(StateLabel)
        header = "FROM\\TO  " + "  ".join(f"{s.name[:6]:>7}" for s in labels)
        lines.append(header)
        for src in labels:
            row = f"{src.name[:8]:<9}"
            for dst in labels:
                sym = "OK" if self.is_allowed(src, dst) else "X"
                row += f"  {sym:>7}"
            lines.append(row)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

DEFAULT_LATTICE = StateLattice()
