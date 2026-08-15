"""Graph exploration endpoints (Milestone 14).

These endpoints expose the in-memory destination graph for educational
exploration.  They demonstrate what graph traversal provides that vector
search cannot:

  GET /graph/similar?destination=Mallorca&hops=1
      Follows curated SIMILAR_TO edges — deterministic, zero-latency,
      independent of embedding quality.  hops=2 discovers second-degree
      similarity (e.g. destinations similar to something similar to Mallorca).

  GET /graph/destinations?airport=MAN
      Follows FLIES_TO edges from an airport node.  Answers the structural
      reachability question "where can I fly from Manchester?" — a query that
      has no meaningful embedding representation.

  GET /graph/airports?destination=Barbados
      Reverse FLIES_TO lookup.  Shows which airports serve a destination,
      demonstrating that long-haul routes are restricted to major hubs (LGW,
      LHR, MAN) while short-haul Mediterranean destinations are reachable
      from all UK regional airports.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from travel_ai_search.api.deps import get_destination_graph
from travel_ai_search.api.schemas.graph import (
    AirportsForDestinationResponse,
    DestinationsFromAirportResponse,
    GraphNodeResponse,
    SimilarDestinationsResponse,
)
from travel_ai_search.graph.models import DestinationGraph, EdgeType, NodeType

router = APIRouter(tags=["Graph"])


def _require_graph(graph: object | None) -> DestinationGraph:
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Destination graph is not available. "
                "Set GRAPH_ENABLED=true and ensure the knowledge file exists at "
                "the path configured by KNOWLEDGE_FILE_PATH."
            ),
        )
    assert isinstance(graph, DestinationGraph)
    return graph


@router.get("/similar", response_model=SimilarDestinationsResponse)
def similar_destinations(
    destination: str,
    hops: int = 1,
    graph: object | None = Depends(get_destination_graph),
) -> SimilarDestinationsResponse:
    """Find destinations editorially similar to the given one.

    Traverses SIMILAR_TO edges via BFS up to `hops` levels deep.  The edges
    are curated (seeded from the knowledge base) rather than derived from
    embedding distance, so this returns the same result regardless of model
    or embedding quality.

    hops=1 returns direct neighbours; hops=2 discovers second-degree links
    (e.g. destinations similar to something similar to Mallorca).
    """
    g = _require_graph(graph)
    nodes = g.bfs(destination, max_hops=hops, edge_type=EdgeType.SIMILAR_TO)
    return SimilarDestinationsResponse(
        source=destination,
        hops=hops,
        count=len(nodes),
        similar=[GraphNodeResponse.from_node(n) for n in nodes],
    )


@router.get("/destinations", response_model=DestinationsFromAirportResponse)
def destinations_from_airport(
    airport: str,
    graph: object | None = Depends(get_destination_graph),
) -> DestinationsFromAirportResponse:
    """Find all destinations reachable from the given departure airport.

    Traverses FLIES_TO edges from the airport node.  Long-haul destinations
    (Barbados, Cancún, Maldives, Phuket, Koh Samui) are only reachable from
    major hub airports (LGW, LHR, MAN).  All other destinations are reachable
    from any UK regional airport via charter flights.

    This demonstrates a structural reachability query that vector search
    cannot answer: "what can I actually fly to from Glasgow?" has no
    embedding representation — it is a fact about airline routes.
    """
    g = _require_graph(graph)
    airport_upper = airport.upper()
    airport_node = g.get_node(airport_upper)
    if airport_node is None:
        known = "LGW, LHR, MAN, GLA, EDI, BHX, BRS, NCL"
        raise HTTPException(
            status_code=404,
            detail=f"Airport '{airport_upper}' not found in graph. Known airports: {known}.",
        )
    destinations = g.neighbors(airport_upper, edge_type=EdgeType.FLIES_TO)
    airport_name = str(airport_node.properties.get("name", airport_upper))
    return DestinationsFromAirportResponse(
        airport=airport_upper,
        airport_name=airport_name,
        count=len(destinations),
        destinations=[GraphNodeResponse.from_node(n) for n in destinations],
    )


@router.get("/airports", response_model=AirportsForDestinationResponse)
def airports_for_destination(
    destination: str,
    graph: object | None = Depends(get_destination_graph),
) -> AirportsForDestinationResponse:
    """Find all airports that operate flights to the given destination.

    Reverses the FLIES_TO direction by scanning all airport nodes.  Useful
    for answering "can I fly to Barbados from Glasgow?" — compare the result
    for Barbados (LGW, LHR, MAN only) with Tenerife (all 8 UK airports) to
    see how hub-and-spoke routing restricts long-haul access.
    """
    g = _require_graph(graph)
    if not g.has_node(destination):
        raise HTTPException(
            status_code=404,
            detail=f"Destination '{destination}' not found in graph.",
        )
    serving_airports = [
        node
        for node in g.all_nodes(node_type=NodeType.AIRPORT)
        if any(n.id == destination for n in g.neighbors(node.id, edge_type=EdgeType.FLIES_TO))
    ]
    return AirportsForDestinationResponse(
        destination=destination,
        count=len(serving_airports),
        airports=[GraphNodeResponse.from_node(n) for n in serving_airports],
    )
