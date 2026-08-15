"""Unit tests for the graph-enhanced retrieval module (Milestone 14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from travel_ai_search.graph.builder import (
    _LONG_HAUL_AIRPORTS,
    _LONG_HAUL_DESTINATIONS,
    _UK_AIRPORTS,
    build_destination_graph,
    load_knowledge_docs,
)
from travel_ai_search.graph.models import DestinationGraph, EdgeType, GraphNode, NodeType
from travel_ai_search.rag.knowledge import DestinationKnowledge

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_doc(
    destination: str,
    similar: list[str] | None = None,
    country: str = "Spain",
) -> DestinationKnowledge:
    return DestinationKnowledge(
        id=destination.lower().replace(" ", "-"),
        destination=destination,
        country=country,
        region="Test Region",
        description="A test destination.",
        climate="Sunny and warm.",
        best_months=["July", "August"],
        family_suitability="high",
        nightlife_level="low",
        beach_quality="good",
        activities=["swimming", "hiking"],
        character_tags=["sunny", "relaxed"],
        similar_destinations=similar or [],
        geographic_note="A note.",
    )


@pytest.fixture()
def small_docs() -> list[DestinationKnowledge]:
    """Three docs with known similarity links: A↔B, A↔C (not B↔C)."""
    return [
        _make_doc("Alpha", similar=["Beta", "Gamma"]),
        _make_doc("Beta", similar=["Alpha"]),
        _make_doc("Gamma", similar=["Alpha"]),
    ]


@pytest.fixture()
def small_graph(small_docs: list[DestinationKnowledge]) -> DestinationGraph:
    return build_destination_graph(small_docs)


# ── DestinationGraph unit tests ───────────────────────────────────────────────


class TestDestinationGraph:
    def test_empty_graph_has_no_nodes(self) -> None:
        g = DestinationGraph()
        assert g.node_count() == 0

    def test_empty_graph_has_no_edges(self) -> None:
        g = DestinationGraph()
        assert g.edge_count() == 0

    def test_add_node_stores_node(self) -> None:
        g = DestinationGraph()
        node = GraphNode(id="X", node_type=NodeType.DESTINATION)
        g.add_node(node)
        assert g.has_node("X")

    def test_get_node_returns_node(self) -> None:
        g = DestinationGraph()
        node = GraphNode(id="X", node_type=NodeType.DESTINATION, properties={"k": "v"})
        g.add_node(node)
        retrieved = g.get_node("X")
        assert retrieved is not None
        assert retrieved.id == "X"
        assert retrieved.properties["k"] == "v"

    def test_get_node_returns_none_for_unknown(self) -> None:
        g = DestinationGraph()
        assert g.get_node("missing") is None

    def test_has_node_false_for_unknown(self) -> None:
        g = DestinationGraph()
        assert not g.has_node("missing")

    def test_node_count_increments(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="A", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="B", node_type=NodeType.AIRPORT))
        assert g.node_count() == 2

    def test_add_edge_creates_adjacency(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="A", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="B", node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        assert g.edge_count() == 1

    def test_add_edge_deduplicates(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="A", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="B", node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        assert g.edge_count() == 1

    def test_add_edge_different_types_not_deduplicated(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="A", node_type=NodeType.AIRPORT))
        g.add_node(GraphNode(id="B", node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.FLIES_TO)
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        assert g.edge_count() == 2

    def test_neighbors_returns_connected_nodes(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="A", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="B", node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        neighbors = g.neighbors("A")
        assert len(neighbors) == 1
        assert neighbors[0].id == "B"

    def test_neighbors_filters_by_edge_type(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="AP", node_type=NodeType.AIRPORT))
        g.add_node(GraphNode(id="D1", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="D2", node_type=NodeType.DESTINATION))
        g.add_edge("AP", "D1", EdgeType.FLIES_TO)
        g.add_edge("AP", "D2", EdgeType.SIMILAR_TO)
        flies = g.neighbors("AP", edge_type=EdgeType.FLIES_TO)
        assert len(flies) == 1
        assert flies[0].id == "D1"

    def test_neighbors_unknown_node_returns_empty(self) -> None:
        g = DestinationGraph()
        assert g.neighbors("unknown") == []

    def test_bfs_one_hop_direct_neighbors(self) -> None:
        g = DestinationGraph()
        for name in ["A", "B", "C"]:
            g.add_node(GraphNode(id=name, node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        g.add_edge("A", "C", EdgeType.SIMILAR_TO)
        result = g.bfs("A", max_hops=1, edge_type=EdgeType.SIMILAR_TO)
        assert {n.id for n in result} == {"B", "C"}

    def test_bfs_two_hops_discovers_second_degree(self) -> None:
        g = DestinationGraph()
        for name in ["A", "B", "C"]:
            g.add_node(GraphNode(id=name, node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        g.add_edge("B", "C", EdgeType.SIMILAR_TO)
        result = g.bfs("A", max_hops=2, edge_type=EdgeType.SIMILAR_TO)
        assert {n.id for n in result} == {"B", "C"}

    def test_bfs_excludes_start_node(self) -> None:
        g = DestinationGraph()
        for name in ["A", "B"]:
            g.add_node(GraphNode(id=name, node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        result = g.bfs("A", max_hops=1)
        assert all(n.id != "A" for n in result)

    def test_bfs_unknown_start_returns_empty(self) -> None:
        g = DestinationGraph()
        assert g.bfs("unknown", max_hops=1) == []

    def test_bfs_no_duplicates(self) -> None:
        g = DestinationGraph()
        for name in ["A", "B", "C"]:
            g.add_node(GraphNode(id=name, node_type=NodeType.DESTINATION))
        g.add_edge("A", "B", EdgeType.SIMILAR_TO)
        g.add_edge("A", "C", EdgeType.SIMILAR_TO)
        g.add_edge("B", "C", EdgeType.SIMILAR_TO)
        result = g.bfs("A", max_hops=2)
        ids = [n.id for n in result]
        assert len(ids) == len(set(ids))

    def test_all_nodes_returns_all(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="D", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="AP", node_type=NodeType.AIRPORT))
        assert len(g.all_nodes()) == 2

    def test_all_nodes_filters_by_type(self) -> None:
        g = DestinationGraph()
        g.add_node(GraphNode(id="D", node_type=NodeType.DESTINATION))
        g.add_node(GraphNode(id="AP", node_type=NodeType.AIRPORT))
        destinations = g.all_nodes(node_type=NodeType.DESTINATION)
        airports = g.all_nodes(node_type=NodeType.AIRPORT)
        assert len(destinations) == 1
        assert destinations[0].id == "D"
        assert len(airports) == 1
        assert airports[0].id == "AP"


# ── build_destination_graph tests ─────────────────────────────────────────────


class TestBuildDestinationGraph:
    def test_destination_nodes_count_matches_docs(
        self, small_docs: list[DestinationKnowledge]
    ) -> None:
        g = build_destination_graph(small_docs)
        destinations = g.all_nodes(node_type=NodeType.DESTINATION)
        assert len(destinations) == len(small_docs)

    def test_airport_nodes_added(self, small_graph: DestinationGraph) -> None:
        airports = small_graph.all_nodes(node_type=NodeType.AIRPORT)
        assert len(airports) == len(_UK_AIRPORTS)

    def test_airport_iata_codes_present(self, small_graph: DestinationGraph) -> None:
        for iata in _UK_AIRPORTS:
            assert small_graph.has_node(iata), f"Airport {iata} missing from graph"

    def test_similar_to_edges_exist(self, small_graph: DestinationGraph) -> None:
        # Alpha→Beta and Alpha→Gamma should both be present.
        similar = {n.id for n in small_graph.neighbors("Alpha", edge_type=EdgeType.SIMILAR_TO)}
        assert "Beta" in similar
        assert "Gamma" in similar

    def test_similar_to_edges_are_bidirectional(self, small_graph: DestinationGraph) -> None:
        # Beta→Alpha must also exist.
        similar_from_beta = {
            n.id for n in small_graph.neighbors("Beta", edge_type=EdgeType.SIMILAR_TO)
        }
        assert "Alpha" in similar_from_beta

    def test_similar_to_not_transitive_without_explicit_edge(
        self, small_graph: DestinationGraph
    ) -> None:
        # Beta and Gamma are not directly linked — only via Alpha.
        similar_from_beta = {
            n.id for n in small_graph.neighbors("Beta", edge_type=EdgeType.SIMILAR_TO)
        }
        assert "Gamma" not in similar_from_beta

    def test_flies_to_edges_from_all_airports(self, small_docs: list[DestinationKnowledge]) -> None:
        g = build_destination_graph(small_docs)
        # All three docs are short-haul, so all airports should reach them.
        for iata in _UK_AIRPORTS:
            dests = {n.id for n in g.neighbors(iata, edge_type=EdgeType.FLIES_TO)}
            assert "Alpha" in dests

    def test_long_haul_only_from_hub_airports(self) -> None:
        long_haul_name = next(iter(_LONG_HAUL_DESTINATIONS))
        docs = [_make_doc(long_haul_name, country="Caribbean")]
        g = build_destination_graph(docs)
        hub_airports = _LONG_HAUL_AIRPORTS
        regional_airports = set(_UK_AIRPORTS.keys()) - hub_airports
        for iata in hub_airports:
            dests = {n.id for n in g.neighbors(iata, edge_type=EdgeType.FLIES_TO)}
            assert long_haul_name in dests, f"Hub {iata} should serve {long_haul_name}"
        for iata in regional_airports:
            dests = {n.id for n in g.neighbors(iata, edge_type=EdgeType.FLIES_TO)}
            msg = f"Regional {iata} should not serve {long_haul_name}"
            assert long_haul_name not in dests, msg

    def test_short_haul_from_all_airports(self) -> None:
        short_haul_name = "TestIsland"
        assert short_haul_name not in _LONG_HAUL_DESTINATIONS
        docs = [_make_doc(short_haul_name)]
        g = build_destination_graph(docs)
        for iata in _UK_AIRPORTS:
            dests = {n.id for n in g.neighbors(iata, edge_type=EdgeType.FLIES_TO)}
            assert short_haul_name in dests, (
                f"Airport {iata} should serve short-haul {short_haul_name}"
            )

    def test_invalid_similar_destination_not_added(self) -> None:
        doc = _make_doc("Solo", similar=["DoesNotExist"])
        g = build_destination_graph([doc])
        # No SIMILAR_TO edge should be added because DoesNotExist has no node.
        similar = g.neighbors("Solo", edge_type=EdgeType.SIMILAR_TO)
        assert similar == []

    def test_empty_docs_produces_airport_only_graph(self) -> None:
        g = build_destination_graph([])
        assert g.all_nodes(node_type=NodeType.DESTINATION) == []
        assert len(g.all_nodes(node_type=NodeType.AIRPORT)) == len(_UK_AIRPORTS)

    def test_destination_properties_stored(self) -> None:
        docs = [_make_doc("Tester")]
        g = build_destination_graph(docs)
        node = g.get_node("Tester")
        assert node is not None
        assert node.properties["country"] == "Spain"
        assert "character_tags" in node.properties

    def test_two_hop_bfs_via_similar_to(self, small_docs: list[DestinationKnowledge]) -> None:
        # Beta is 1 hop from Alpha; Gamma is also 1 hop from Alpha.
        # From Beta, Gamma is 2 hops away (Beta→Alpha→Gamma).
        g = build_destination_graph(small_docs)
        result = {n.id for n in g.bfs("Beta", max_hops=2, edge_type=EdgeType.SIMILAR_TO)}
        assert "Alpha" in result
        assert "Gamma" in result


# ── load_knowledge_docs tests ─────────────────────────────────────────────────


class TestLoadKnowledgeDocs:
    def test_loads_jsonl_file(self, tmp_path: Path) -> None:
        doc = _make_doc("Somewhere")
        p = tmp_path / "docs.jsonl"
        p.write_text(doc.model_dump_json() + "\n", encoding="utf-8")
        loaded = load_knowledge_docs(p)
        assert len(loaded) == 1
        assert loaded[0].destination == "Somewhere"

    def test_loads_multiple_lines(self, tmp_path: Path) -> None:
        docs = [_make_doc("A"), _make_doc("B"), _make_doc("C")]
        p = tmp_path / "docs.jsonl"
        p.write_text(
            "\n".join(d.model_dump_json() for d in docs) + "\n",
            encoding="utf-8",
        )
        loaded = load_knowledge_docs(p)
        assert len(loaded) == 3

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        doc = _make_doc("Somewhere")
        p = tmp_path / "docs.jsonl"
        p.write_text("\n" + doc.model_dump_json() + "\n\n", encoding="utf-8")
        loaded = load_knowledge_docs(p)
        assert len(loaded) == 1

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_knowledge_docs(p) == []

    def test_loaded_doc_has_correct_fields(self, tmp_path: Path) -> None:
        doc = _make_doc("Island", similar=["Other"])
        p = tmp_path / "docs.jsonl"
        p.write_text(doc.model_dump_json() + "\n", encoding="utf-8")
        loaded = load_knowledge_docs(p)
        assert loaded[0].similar_destinations == ["Other"]
        assert loaded[0].family_suitability == "high"

    def test_extra_json_fields_are_ignored(self, tmp_path: Path) -> None:
        doc = _make_doc("Island")
        data = json.loads(doc.model_dump_json())
        data["unknown_field"] = "should be ignored"
        p = tmp_path / "docs.jsonl"
        p.write_text(json.dumps(data) + "\n", encoding="utf-8")
        loaded = load_knowledge_docs(p)
        assert len(loaded) == 1
