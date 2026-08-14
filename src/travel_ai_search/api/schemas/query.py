"""Pydantic schemas for the query understanding API."""

from __future__ import annotations

from pydantic import BaseModel

from travel_ai_search.query_understanding.models import QueryUnderstanding


class QueryUnderstandRequest(BaseModel):
    query: str


class QueryUnderstandResponse(BaseModel):
    original_query: str
    semantic_query: str
    month: str | None = None
    departure_airport: str | None = None
    max_price: float | None = None
    min_star_rating: int | None = None
    family_friendly: bool | None = None
    adults_only: bool | None = None
    country: str | None = None
    destination: str | None = None
    soft_preferences: list[str] = []
    understanding_took_ms: int = 0

    @classmethod
    def from_understanding(cls, qu: QueryUnderstanding) -> QueryUnderstandResponse:
        return cls(
            original_query=qu.original_query,
            semantic_query=qu.semantic_query,
            month=qu.month,
            departure_airport=qu.departure_airport,
            max_price=qu.max_price,
            min_star_rating=qu.min_star_rating,
            family_friendly=qu.family_friendly,
            adults_only=qu.adults_only,
            country=qu.country,
            destination=qu.destination,
            soft_preferences=qu.soft_preferences,
            understanding_took_ms=qu.understanding_took_ms,
        )
