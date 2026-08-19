"""
nsa/formal/graph.py
===================
NSA 3.1 Complete Governance Mediation Graph Analyzer.

Constructs the machine-readable cognitive-execution graph G_NSA = (V, E)
and mathematically verifies the Complete Governance Mediation Theorem:

    forall p: C ~> S_effectful,  K_ISK in p

Every path from the cognitive domain to an effectful protected sink MUST pass
through an authorized decision node of the Immutable Safety Kernel.
"""

from __future__ import annotations

import collections
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


class NodeType(enum.Enum):
    COGNITIVE = "COGNITIVE"        # Neural computation, activations, state
    MEMORY = "MEMORY"              # KV-cache, scratchpad, context window
    GOVERNOR = "GOVERNOR"          # Epistemic governor, simulator
    ISK_MONITOR = "ISK_MONITOR"    # Immutable Safety Kernel Reference Monitor
    SERIALIZER = "SERIALIZER"      # Serialization, deserialization, buffers
    TOOL_WRAPPER = "TOOL_WRAPPER"  # Intermediate agent tool wrappers
    PROTECTED_SINK = "SINK"        # Effectful actuators (filesystem, network, keys)


class EdgeType(enum.Enum):
    INVOKE = "invoke"
    READ = "read"
    WRITE = "write"
    MUTATE = "mutate"
    SERIALIZE = "serialize"
    DELEGATE = "delegate"
    MEDIATE_PASS = "mediate_pass"
    EXECUTE = "execute"


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    description: str


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    is_kernel_mediated: bool = False


class CompleteMediationGraph:
    """Directed graph G_NSA = (V, E) representing the full cognitive-runtime topology."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = collections.defaultdict(list)
        self.in_edges: Dict[str, List[GraphEdge]] = collections.defaultdict(list)

    def add_node(self, node_id: str, node_type: NodeType, description: str = "") -> None:
        self.nodes[node_id] = GraphNode(node_id=node_id, node_type=node_type, description=description)

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, is_kernel_mediated: bool = False) -> None:
        edge = GraphEdge(source_id=source_id, target_id=target_id, edge_type=edge_type, is_kernel_mediated=is_kernel_mediated)
        self.edges[source_id].append(edge)
        self.in_edges[target_id].append(edge)

    @classmethod
    def build_standard_nsa_topology(cls) -> CompleteMediationGraph:
        """Constructs the standard architectural topology of NSA 3.1."""
        g = cls()

        # Cognitive nodes
        g.add_node("neural_transformer", NodeType.COGNITIVE, "Transformer feedforward & attention")
        g.add_node("cognitive_state_omega", NodeType.COGNITIVE, "Omega_t cognitive state vector")
        g.add_node("self_model_predictor", NodeType.COGNITIVE, "Predictive self-state model")
        g.add_node("epistemic_engine", NodeType.COGNITIVE, "Epistemic grounding engine")
        g.add_node("counterfactual_sim", NodeType.COGNITIVE, "Counterfactual internal simulator")

        # Memory nodes
        g.add_node("kv_cache_storage", NodeType.MEMORY, "Attention KV cache storage")
        g.add_node("cognitive_scratchpad", NodeType.MEMORY, "Internal reasoning scratchpad")

        # Governance nodes
        g.add_node("epistemic_governor", NodeType.GOVERNOR, "5-way action governor")
        g.add_node("capability_manager", NodeType.GOVERNOR, "HMAC capability manager")
        g.add_node("isk_reference_monitor", NodeType.ISK_MONITOR, "Immutable Safety Kernel Monitor")

        # Middleware & Tool nodes
        g.add_node("tool_dispatcher", NodeType.TOOL_WRAPPER, "Agent tool dispatcher wrapper")
        g.add_node("json_state_serializer", NodeType.SERIALIZER, "State snapshot serializer")

        # Protected Sinks (Effectful external interfaces)
        g.add_node("sink_filesystem_writer", NodeType.PROTECTED_SINK, "Persistent filesystem modification")
        g.add_node("sink_network_socket", NodeType.PROTECTED_SINK, "External outbound network socket")
        g.add_node("sink_subprocess_exec", NodeType.PROTECTED_SINK, "OS subprocess shell execution")
        g.add_node("sink_cryptographic_keys", NodeType.PROTECTED_SINK, "Key store / root capabilities")

        # Legitimate mediated paths:
        # Cognition -> Governor -> ISK -> Protected Sinks
        g.add_edge("neural_transformer", "cognitive_state_omega", EdgeType.MUTATE)
        g.add_edge("cognitive_state_omega", "self_model_predictor", EdgeType.READ)
        g.add_edge("cognitive_state_omega", "epistemic_engine", EdgeType.READ)
        g.add_edge("cognitive_state_omega", "counterfactual_sim", EdgeType.READ)
        g.add_edge("counterfactual_sim", "cognitive_scratchpad", EdgeType.WRITE)
        g.add_edge("neural_transformer", "kv_cache_storage", EdgeType.WRITE)

        # Action proposals pass to Epistemic Governor
        g.add_edge("neural_transformer", "epistemic_governor", EdgeType.INVOKE)
        g.add_edge("counterfactual_sim", "epistemic_governor", EdgeType.INVOKE)

        # Governor invokes ISK Reference Monitor
        g.add_edge("epistemic_governor", "isk_reference_monitor", EdgeType.INVOKE)
        g.add_edge("capability_manager", "isk_reference_monitor", EdgeType.READ)

        # Tool dispatcher must go through ISK
        g.add_edge("tool_dispatcher", "isk_reference_monitor", EdgeType.INVOKE)

        # ISK alone mediates access to Protected Sinks
        g.add_edge("isk_reference_monitor", "sink_filesystem_writer", EdgeType.EXECUTE, is_kernel_mediated=True)
        g.add_edge("isk_reference_monitor", "sink_network_socket", EdgeType.EXECUTE, is_kernel_mediated=True)
        g.add_edge("isk_reference_monitor", "sink_subprocess_exec", EdgeType.EXECUTE, is_kernel_mediated=True)
        g.add_edge("isk_reference_monitor", "sink_cryptographic_keys", EdgeType.EXECUTE, is_kernel_mediated=True)

        return g

    def verify_complete_mediation(self) -> Tuple[bool, List[str], List[List[str]]]:
        """Verify that EVERY path from any cognitive/memory/tool node to a protected sink

        passes through an ISK_MONITOR node.
        Returns:
            (is_complete, violation_explanations, unmediated_paths)
        """
        cognitive_sources = [
            nid for nid, node in self.nodes.items()
            if node.node_type in (NodeType.COGNITIVE, NodeType.MEMORY, NodeType.TOOL_WRAPPER, NodeType.SERIALIZER)
        ]
        protected_sinks = [
            nid for nid, node in self.nodes.items()
            if node.node_type == NodeType.PROTECTED_SINK
        ]

        violations: List[str] = []
        unmediated_paths: List[List[str]] = []

        for src in cognitive_sources:
            for dst in protected_sinks:
                paths = self._find_all_paths(src, dst)
                for path in paths:
                    has_isk = any(self.nodes[nid].node_type == NodeType.ISK_MONITOR for nid in path)
                    if not has_isk:
                        unmediated_paths.append(path)
                        violations.append(f"Unmediated bypass detected: {' -> '.join(path)}")

        is_complete = (len(unmediated_paths) == 0)
        return is_complete, violations, unmediated_paths

    def _find_all_paths(self, start_id: str, target_id: str, visited: Optional[Set[str]] = None) -> List[List[str]]:
        if visited is None:
            visited = set()
        visited.add(start_id)

        if start_id == target_id:
            return [[start_id]]

        paths = []
        for edge in self.edges.get(start_id, []):
            nxt = edge.target_id
            if nxt not in visited:
                sub_paths = self._find_all_paths(nxt, target_id, visited.copy())
                for sp in sub_paths:
                    paths.append([start_id] + sp)

        return paths
