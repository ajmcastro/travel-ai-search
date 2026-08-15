"""Pydantic response schemas for the /graph endpoints (Milestone 14)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from travel_ai_search.graph.models import GraphNode


class GraphNodeResponse(BaseModel):
    id: str
    node_type: str
    properties: dict[str, Any]

    @classmethod
    def from_node(cls, node: GraphNode) -> GraphNodeResponse:
        return cls(id=node.id, node_type=node.node_type.value, properties=node.properties)


class SimilarDestinationsResponse(BaseModel):
    source: str
    hops: int
    count: int
    similar: list[GraphNodeResponse]


class DestinationsFromAirportResponse(BaseModel):
    airport: str
    airport_name: str
    count: int
    destinations: list[GraphNodeResponse]


class AirportsForDestinationResponse(BaseModel):
    destination: str
    count: int
    airports: list[GraphNodeResponse]
