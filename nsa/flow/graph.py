"""Typed state-flow graph for whole-system information-flow analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from nsa.core.state import HardState


@dataclass(frozen=True)
class FlowNode:
    """A computational or trust boundary in an NSA system."""

    node_id: str
    kind: str
    required_authorizations: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FlowEdge:
    """An explicitly permitted information/state flow between nodes."""

    source: str
    destination: str
    state_dimensions: frozenset[str]
    requires_authorizations: frozenset[str] = frozenset()
    transform: str | None = None


@dataclass(frozen=True)
class FlowViolation:
    source: str
    destination: str
    reason: str


class FlowGraph:
    """Minimal declarative graph for checking legal state flow.

    The graph is deliberately independent from a model implementation. It
    describes what a system boundary permits; a runtime can later attach
    concrete tensor/activation propagation to the same graph.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, FlowNode] = {}
        self._edges: list[FlowEdge] = []

    @property
    def nodes(self) -> tuple[FlowNode, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[FlowEdge, ...]:
        return tuple(self._edges)

    def add_node(self, node: FlowNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node: {node.node_id}")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: FlowEdge) -> None:
        if edge.source not in self._nodes or edge.destination not in self._nodes:
            raise ValueError("both edge endpoints must exist in the graph")
        self._edges.append(edge)

    def outgoing(self, node_id: str) -> tuple[FlowEdge, ...]:
        return tuple(edge for edge in self._edges if edge.source == node_id)

    def can_flow(
        self,
        source: str,
        destination: str,
        dimensions: Iterable[str],
        hard_state: HardState | None = None,
    ) -> bool:
        requested = frozenset(dimensions)
        return not self.check_flow(source, destination, requested, hard_state)

    def check_flow(
        self,
        source: str,
        destination: str,
        dimensions: frozenset[str],
        hard_state: HardState | None = None,
    ) -> FlowViolation | None:
        candidates = [
            edge
            for edge in self._edges
            if edge.source == source and edge.destination == destination
        ]
        if not candidates:
            return FlowViolation(source, destination, "no permitted edge")

        for edge in candidates:
            if not dimensions.issubset(edge.state_dimensions):
                continue
            required = edge.requires_authorizations | self._nodes[destination].required_authorizations
            if hard_state is not None and not required.issubset(hard_state.authorizations):
                continue
            if required and hard_state is None:
                continue
            return None

        return FlowViolation(
            source,
            destination,
            "requested state dimensions or required authorizations are not permitted",
        )


__all__ = ["FlowNode", "FlowEdge", "FlowGraph", "FlowViolation"]
