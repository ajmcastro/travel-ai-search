"""Domain model for a parsed query understanding result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryUnderstanding:
    """Structured representation of a parsed free-text travel query.

    Hard constraints (month, departure_airport, max_price, min_star_rating,
    family_friendly, adults_only, country) are exact filters — they eliminate
    documents that do not match.

    soft_preferences are detected keywords that influence BM25/vector ranking
    but are NOT used as hard filters (they remain in semantic_query).

    semantic_query is the residual text after removing hard-constraint phrases;
    it is used as the free-text input to BM25 and vector retrieval.
    """

    original_query: str
    semantic_query: str

    # Hard constraints — any None means "no filter applied"
    month: str | None = None
    departure_airport: str | None = None  # IATA code (e.g. "MAN")
    max_price: float | None = None
    min_star_rating: int | None = None
    family_friendly: bool | None = None
    adults_only: bool | None = None
    country: str | None = None
    destination: str | None = None  # exact city (rarely extracted by rule-based engine)

    # Soft preferences — for observability; already present in semantic_query
    soft_preferences: list[str] = field(default_factory=list)

    understanding_took_ms: int = 0

    def to_search_filters(self) -> dict[str, Any]:
        """Return a dict of non-None hard constraints suitable for HybridSearchParams."""
        filters: dict[str, Any] = {}
        if self.month is not None:
            filters["month"] = self.month
        if self.departure_airport is not None:
            filters["airport"] = self.departure_airport
        if self.max_price is not None:
            filters["max_price"] = self.max_price
        if self.min_star_rating is not None:
            filters["min_star_rating"] = self.min_star_rating
        if self.family_friendly is not None:
            filters["family_friendly"] = self.family_friendly
        if self.adults_only is not None:
            filters["adults_only"] = self.adults_only
        if self.country is not None:
            filters["country"] = self.country
        if self.destination is not None:
            filters["destination"] = self.destination
        return filters
