"""BM25 lexical retrieval via OpenSearch multi-field queries.

Design notes
------------
_build_query() is a pure function (dict in, dict out) so it can be unit-tested
without a running OpenSearch instance.

Field boosting strategy
  hotel_name^3    — the name is the most specific identifier; exact hotel
                    lookups and brand queries need to win here
  destination^2, region^2, country^2
                  — location is the primary travel search dimension
  hotel_description, activities, amenities, tags.text
                  — full semantic context at base weight (1.0)

Filter vs must
  Hard constraints (country, price, family_friendly …) go in the `filter`
  clause, not `must`. This keeps them out of the BM25 scoring calculation
  and lets OpenSearch cache them as bitsets between requests.

Fuzziness
  AUTO lets OpenSearch apply 1-2 edit distances for terms ≥ 3 characters,
  handling common typos without requiring a full query-understanding pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opensearchpy import OpenSearch

from travel_ai_search.ingestion.index import INDEX_NAME

# Fields included in the BM25 multi-match query, with boost factors.
_SEARCH_FIELDS: list[str] = [
    "hotel_name^3",
    "destination^2",
    "region^2",
    "country^2",
    "hotel_description",
    "activities",
    "tags.text",
    "amenities",
]

# Aggregations returned alongside every search response.
_AGGS: dict[str, Any] = {
    "countries": {"terms": {"field": "country", "size": 30}},
    "star_ratings": {"terms": {"field": "star_rating", "size": 5}},
    "board_types": {"terms": {"field": "board_types", "size": 10}},
    "climate_zones": {"terms": {"field": "climate_zone", "size": 10}},
}


@dataclass
class LexicalSearchParams:
    """All parameters that drive a single lexical search request.

    All filter fields default to None, meaning "no filter applied".
    """

    query: str
    top_k: int = 10
    # ── Filters ─────────────────────────────────────────────────────────────
    country: str | None = None
    destination: str | None = None
    family_friendly: bool | None = None
    adults_only: bool | None = None
    min_star_rating: int | None = None
    max_price: float | None = None
    month: str | None = None
    airport: str | None = None


@dataclass
class Hit:
    """A single document returned from OpenSearch."""

    id: str
    score: float
    source: dict[str, Any]


@dataclass
class LexicalSearchResult:
    hits: list[Hit]
    total: int
    took_ms: int
    raw_aggregations: dict[str, Any] = field(default_factory=dict)


def _build_query(params: LexicalSearchParams) -> dict[str, Any]:
    """Return the OpenSearch query body for the given parameters.

    Pure function — no network calls, fully unit-testable.
    """
    # ── Main text query ──────────────────────────────────────────────────────
    if params.query.strip():
        query_clause: dict[str, Any] = {
            "multi_match": {
                "query": params.query,
                "fields": _SEARCH_FIELDS,
                "type": "best_fields",
                "tie_breaker": 0.3,
                "fuzziness": "AUTO",
                "operator": "or",
            }
        }
    else:
        # Empty query = browsing mode: return all documents ordered by relevance
        # (score will be 1.0 for all; callers may want to add sort by rating or
        # price for a meaningful ordering in pure browsing contexts)
        query_clause = {"match_all": {}}

    # ── Hard-constraint filters ──────────────────────────────────────────────
    filters: list[dict[str, Any]] = []
    if params.country is not None:
        filters.append({"term": {"country": params.country}})
    if params.destination is not None:
        filters.append({"term": {"destination": params.destination}})
    if params.family_friendly is not None:
        filters.append({"term": {"family_friendly": params.family_friendly}})
    if params.adults_only is not None:
        filters.append({"term": {"adults_only": params.adults_only}})
    if params.min_star_rating is not None:
        filters.append({"range": {"star_rating": {"gte": params.min_star_rating}}})
    if params.max_price is not None:
        filters.append({"range": {"price_per_person_gbp": {"lte": params.max_price}}})
    if params.month is not None:
        filters.append({"term": {"available_months": params.month}})
    if params.airport is not None:
        filters.append({"term": {"available_departure_airports": params.airport}})

    bool_clause: dict[str, Any] = {"must": [query_clause]}
    if filters:
        bool_clause["filter"] = filters

    return {
        "size": params.top_k,
        "query": {"bool": bool_clause},
        "aggs": _AGGS,
    }


def lexical_search(
    client: OpenSearch,
    params: LexicalSearchParams,
    index: str = INDEX_NAME,
) -> LexicalSearchResult:
    """Execute a BM25 search and return structured results."""
    body = _build_query(params)
    response = client.search(index=index, body=body)

    hits = [
        Hit(
            id=hit["_id"],
            score=float(hit["_score"] or 0.0),
            source=hit["_source"],
        )
        for hit in response["hits"]["hits"]
    ]

    return LexicalSearchResult(
        hits=hits,
        total=int(response["hits"]["total"]["value"]),
        took_ms=int(response["took"]),
        raw_aggregations=response.get("aggregations", {}),
    )
