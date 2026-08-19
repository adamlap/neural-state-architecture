"""State propagation through the NSA whole-system flow graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from nsa.core.state import CanonicalState, HardState
from nsa.flow.graph import FlowGraph


@dataclass(frozen=True)
class PropagationResult:
    """Result of attempting to propagate selected state dimensions."""

    accepted: bool
    state: CanonicalState
    source: str
    destination: str
    dimensions: frozenset[str]
    reason: str | None = None


class StatePropagationEngine:
    """Enforce graph policy before transferring typed state between boundaries.

    This first implementation is deliberately conservative: it propagates the
    canonical state object unchanged and records the permitted dimensions.
    Transformations/declassification will be introduced explicitly rather
    than silently changing state semantics.
    """

    def __init__(self, graph: FlowGraph) -> None:
        self.graph = graph

    def propagate(
        self,
        state: CanonicalState,
        source: str,
        destination: str,
        dimensions: frozenset[str],
        hard_state: HardState | None = None,
    ) -> PropagationResult:
        violation = self.graph.check_flow(
            source, destination, dimensions, hard_state or state.hard
        )
        if violation is not None:
            return PropagationResult(
                accepted=False,
                state=state,
                source=source,
                destination=destination,
                dimensions=dimensions,
                reason=violation.reason,
            )

        return PropagationResult(
            accepted=True,
            state=state,
            source=source,
            destination=destination,
            dimensions=dimensions,
        )


__all__ = ["StatePropagationEngine", "PropagationResult"]
