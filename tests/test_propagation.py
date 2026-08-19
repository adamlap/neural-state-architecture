"""Tests for graph-enforced state propagation."""

from nsa.core.state import CanonicalState, HardState
from nsa.flow import FlowEdge, FlowGraph, FlowNode
from nsa.flow.propagation import StatePropagationEngine


def make_graph() -> FlowGraph:
    graph = FlowGraph()
    graph.add_node(FlowNode("model", "neural"))
    graph.add_node(FlowNode("memory", "memory"))
    graph.add_node(FlowNode("tool", "action", frozenset({"tool:write"})))
    graph.add_edge(FlowEdge("model", "memory", frozenset({"semantic", "provenance"})))
    graph.add_edge(FlowEdge("model", "tool", frozenset({"semantic"}), frozenset({"tool:write"})))
    return graph


def test_propagation_preserves_state_object():
    state = CanonicalState()
    result = StatePropagationEngine(make_graph()).propagate(
        state, "model", "memory", frozenset({"semantic"})
    )
    assert result.accepted
    assert result.state == state


def test_propagation_blocks_forbidden_dimension():
    result = StatePropagationEngine(make_graph()).propagate(
        CanonicalState(), "model", "memory", frozenset({"hard"})
    )
    assert not result.accepted


def test_propagation_requires_tool_authorization():
    engine = StatePropagationEngine(make_graph())
    denied = engine.propagate(
        CanonicalState(), "model", "tool", frozenset({"semantic"}), HardState()
    )
    assert not denied.accepted

    authorized = CanonicalState(
        hard=HardState(authorizations=frozenset({"tool:write"}))
    )
    allowed = engine.propagate(
        authorized, "model", "tool", frozenset({"semantic"})
    )
    assert allowed.accepted
