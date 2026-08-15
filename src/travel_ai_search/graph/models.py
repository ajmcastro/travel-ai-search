"""Graph domain models: nodes, edges, and the adjacency-list graph structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    DESTINATION = "destination"
    AIRPORT = "airport"


class EdgeType(StrEnum):
    SIMILAR_TO = "similar_to"
    FLIES_TO = "flies_to"


@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)


class DestinationGraph:
    """In-memory directed adjacency-list graph of travel entities.

    Nodes are destinations and airports.  Edges are typed directed links.
    SIMILAR_TO edges are added bidirectionally at build time so that
    `neighbors("Mallorca", edge_type=SIMILAR_TO)` returns Menorca without
    needing a separate reverse-lookup pass.

    All operations are O(1) or O(degree) — the graph is small enough (~38
    nodes, ~200 edges) that this never matters in practice, but the data
    structure is correct for larger graphs too.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._adjacency: dict[str, list[tuple[str, EdgeType]]] = {}

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        """Add a directed edge, silently deduplicating identical (source, target, type) triples."""
        if source_id not in self._adjacency:
            self._adjacency[source_id] = []
        for existing_target, existing_type in self._adjacency[source_id]:
            if existing_target == target_id and existing_type == edge_type:
                return
        self._adjacency[source_id].append((target_id, edge_type))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adjacency.values())

    def neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
    ) -> list[GraphNode]:
        """Return direct neighbor nodes reachable from node_id.

        When edge_type is given, only edges of that type are followed.
        Returns an empty list when node_id is not in the graph.
        """
        result: list[GraphNode] = []
        for target_id, et in self._adjacency.get(node_id, []):
            if edge_type is None or et == edge_type:
                node = self._nodes.get(target_id)
                if node is not None:
                    result.append(node)
        return result

    def bfs(
        self,
        start_id: str,
        *,
        max_hops: int = 1,
        edge_type: EdgeType | None = None,
    ) -> list[GraphNode]:
        """BFS from start_id up to max_hops edges, optionally filtered by edge type.

        Returns all reachable nodes (excluding start_id itself) in BFS order.
        Nodes are deduplicated — each appears at most once even if multiple paths
        reach it.  Returns an empty list when start_id is not in the graph.
        """
        if not self.has_node(start_id):
            return []
        visited: set[str] = {start_id}
        queue: list[tuple[str, int]] = [(start_id, 0)]
        result: list[GraphNode] = []
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for neighbor in self.neighbors(current_id, edge_type=edge_type):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    result.append(neighbor)
                    queue.append((neighbor.id, depth + 1))
        return result

    def all_nodes(self, *, node_type: NodeType | None = None) -> list[GraphNode]:
        """Return all nodes, optionally filtered by type."""
        if node_type is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.node_type == node_type]
