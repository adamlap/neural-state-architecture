"""Tests for whole-system NSA information-flow boundaries."""

import pytest

from nsa.core.state import HardState
from nsa.flow import FlowEdge, FlowGraph, FlowNode


def graph() -> FlowGraph:
    g = FlowGraph()
    g.add_node(FlowNode("model", "neural"))
    g.add_node(FlowNode("memory", "memory"))
    g.add_node(FlowNode("tool", "action", frozenset({"tool:write"})))
    g.add_edge(FlowEdge("model", "memory", frozenset({"semantic", "provenance"})))
    g.add_edge(FlowEdge("model", "tool", frozenset({"semantic"}), frozenset({"tool:write"})))
    return g


def test_allowed_semantic_flow():
    assert graph().can_flow("model", "memory", {"semantic"})


def test_unlisted_dimension_is_blocked():
    assert not graph().can_flow("model", "memory", {"hard"})


def test_missing_edge_is_blocked():
    assert not graph().can_flow("memory", "tool", {"semantic"})


def test_action_requires_authorization():
    assert not graph().can_flow("model", "tool", {"semantic"}, HardState())
    state = HardState(authorizations=frozenset({"tool:write"}))
    assert graph().can_flow("model", "tool", {"semantic"}, state)


def test_unknown_node_rejected_when_adding_edge():
    g = FlowGraph()
    g.add_node(FlowNode("model", "neural"))
    with pytest.raises(ValueError):
        g.add_edge(FlowEdge("model", "missing", frozenset({"semantic"})))
