"""Pydantic schemas for the search API request/response layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from travel_ai_search.retrieval.lexical import LexicalSearchResult


class SearchHit(BaseModel):
    """A single hotel result returned by a search endpoint.

    Includes all fields relevant for display.  Geographic coordinates and the
    derived geo_point are omitted — they live in the index for distance queries
    but are not useful in a JSON response.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    score: float

    # Display fields
    hotel_name: str
    hotel_description: str
    destination: str
    region: str
    country: str
    star_rating: int
    customer_rating: float
    price_per_person_gbp: float
    family_friendly: bool
    adults_only: bool
    amenities: list[str]
    board_types: list[str]
    beach_distance_km: float
    airport_distance_km: float
    activities: list[str]
    tags: list[str]
    available_departure_airports: list[str]
    available_months: list[str]
    climate_zone: str


class FacetBucket(BaseModel):
    key: str
    count: int


class LexicalSearchResponse(BaseModel):
    hits: list[SearchHit]
    total: int
    took_ms: int
    facets: dict[str, list[FacetBucket]] = {}

    @classmethod
    def from_result(cls, result: LexicalSearchResult) -> LexicalSearchResponse:
        hits = [
            SearchHit.model_validate({"id": hit.id, "score": hit.score, **hit.source})
            for hit in result.hits
        ]
        facets = {
            name: [
                FacetBucket(key=str(b["key"]), count=int(b["doc_count"]))
                for b in agg.get("buckets", [])
            ]
            for name, agg in result.raw_aggregations.items()
        }
        return cls(hits=hits, total=result.total, took_ms=result.took_ms, facets=facets)
