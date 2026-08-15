"""Build a DestinationGraph from destination knowledge documents."""

from __future__ import annotations

import json
from pathlib import Path

from travel_ai_search.graph.models import DestinationGraph, EdgeType, GraphNode, NodeType
from travel_ai_search.rag.knowledge import DestinationKnowledge

# UK departure airports present in the synthetic hotel dataset.
# Maps IATA code → display name.
_UK_AIRPORTS: dict[str, str] = {
    "LGW": "London Gatwick",
    "LHR": "London Heathrow",
    "MAN": "Manchester",
    "GLA": "Glasgow",
    "EDI": "Edinburgh",
    "BHX": "Birmingham",
    "BRS": "Bristol",
    "NCL": "Newcastle",
}

# Long-haul destinations (Caribbean, Indian Ocean, SE Asia) require a major
# international hub with wide-body aircraft and connecting services.
# Regional UK airports serve only short-haul European/Mediterranean routes.
_LONG_HAUL_DESTINATIONS: frozenset[str] = frozenset(
    {"Barbados", "Cancún", "Maldives", "Phuket", "Koh Samui"}
)

# Only major hubs operate long-haul charter/scheduled services.
_LONG_HAUL_AIRPORTS: frozenset[str] = frozenset({"LGW", "LHR", "MAN"})


def _airport_serves(airport_iata: str, destination_name: str) -> bool:
    """Return True if this UK airport serves the given destination."""
    if destination_name in _LONG_HAUL_DESTINATIONS:
        return airport_iata in _LONG_HAUL_AIRPORTS
    return True  # All UK airports serve short-haul destinations via charter flights.


def build_destination_graph(docs: list[DestinationKnowledge]) -> DestinationGraph:
    """Build an in-memory graph from destination knowledge documents.

    Nodes
    -----
    DESTINATION — one per knowledge document (30 in the default dataset).
    AIRPORT     — one per UK departure airport in the synthetic dataset (8 total).

    Edges
    -----
    SIMILAR_TO (bidirectional): seeded from the similar_destinations field.
      Bidirectional because destination similarity is symmetric — if Menorca is
      similar to Mallorca, Mallorca is similar to Menorca.  Only added when the
      target destination exists as a node (prevents dangling references).

    FLIES_TO (directed, airport → destination): based on realistic UK charter
      flight routes.  Long-haul destinations are restricted to major hub airports
      (LGW, LHR, MAN); short-haul European/Mediterranean destinations are
      reachable from any UK airport.

    The airport→destination model demonstrates a key graph advantage: vector
    search cannot express "which destinations does Glasgow airport serve?" because
    departure-airport reachability is a structural fact, not a semantic concept.
    """
    graph = DestinationGraph()

    # Pass 1: destination nodes.
    for doc in docs:
        graph.add_node(
            GraphNode(
                id=doc.destination,
                node_type=NodeType.DESTINATION,
                properties={
                    "country": doc.country,
                    "region": doc.region,
                    "character_tags": doc.character_tags,
                    "family_suitability": doc.family_suitability,
                    "nightlife_level": doc.nightlife_level,
                    "beach_quality": doc.beach_quality,
                },
            )
        )

    # Pass 2: SIMILAR_TO edges — both directions.
    for doc in docs:
        for similar_name in doc.similar_destinations:
            if graph.has_node(similar_name):
                graph.add_edge(doc.destination, similar_name, EdgeType.SIMILAR_TO)
                graph.add_edge(similar_name, doc.destination, EdgeType.SIMILAR_TO)

    # Pass 3: airport nodes + FLIES_TO edges.
    for iata, full_name in _UK_AIRPORTS.items():
        graph.add_node(
            GraphNode(
                id=iata,
                node_type=NodeType.AIRPORT,
                properties={"name": full_name, "country": "UK"},
            )
        )
        for doc in docs:
            if _airport_serves(iata, doc.destination):
                graph.add_edge(iata, doc.destination, EdgeType.FLIES_TO)

    return graph


def load_knowledge_docs(path: Path) -> list[DestinationKnowledge]:
    """Load destination knowledge documents from a JSONL file.

    Each non-blank line must be a valid JSON object matching DestinationKnowledge.
    """
    docs: list[DestinationKnowledge] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(DestinationKnowledge.model_validate(json.loads(line)))
    return docs
