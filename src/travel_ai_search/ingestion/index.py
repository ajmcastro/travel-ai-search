"""OpenSearch index definition for travel hotels.

Design rationale for each field type:
  text + english analyser  → BM25 full-text search with stemming and stop-word removal
  keyword                  → exact filters, aggregations, facets
  integer / float          → range queries (price, star_rating, distances)
  boolean                  → family_friendly / adults_only filters
  geo_point                → distance and bounding-box queries (Milestone 5+)
  multi-field (text+kw)    → BM25 search AND exact filtering on the same field
"""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

INDEX_NAME = "travel_hotels"

INDEX_BODY: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        # 0 replicas keeps cluster health green on a single-node dev setup.
        # In production, raise this to 1+.
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            # ── Identifier ──────────────────────────────────────────────────
            "id": {"type": "keyword"},
            # ── Searchable text (inverted index, English analysis) ───────────
            # multi-field: .keyword sub-field enables exact matching / sorting
            "hotel_name": {
                "type": "text",
                "analyzer": "english",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "hotel_description": {
                "type": "text",
                "analyzer": "english",
            },
            "amenities": {
                "type": "text",
                "analyzer": "english",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "activities": {
                "type": "text",
                "analyzer": "english",
                "fields": {"keyword": {"type": "keyword"}},
            },
            # ── Keywords (exact match, filters, facets) ──────────────────────
            # destination has a .text sub-field so BM25 queries can also match it
            "destination": {
                "type": "keyword",
                "fields": {"text": {"type": "text", "analyzer": "standard"}},
            },
            "region": {"type": "keyword"},
            "country": {"type": "keyword"},
            "board_types": {"type": "keyword"},
            "available_departure_airports": {"type": "keyword"},
            "available_months": {"type": "keyword"},
            "climate_zone": {"type": "keyword"},
            "peak_season_months": {"type": "keyword"},
            # tags are categorical but get a .text sub-field for BM25 discovery
            "tags": {
                "type": "keyword",
                "fields": {"text": {"type": "text", "analyzer": "standard"}},
            },
            # ── Numeric / range ──────────────────────────────────────────────
            "star_rating": {"type": "integer"},
            "customer_rating": {"type": "float"},
            "price_per_person_gbp": {"type": "float"},
            "beach_distance_km": {"type": "float"},
            "airport_distance_km": {"type": "float"},
            # ── Boolean ──────────────────────────────────────────────────────
            "family_friendly": {"type": "boolean"},
            "adults_only": {"type": "boolean"},
            # ── Geographic ───────────────────────────────────────────────────
            # latitude / longitude stored as floats for display;
            # location stores both as a geo_point for distance queries (M5+)
            "latitude": {"type": "float"},
            "longitude": {"type": "float"},
            "location": {"type": "geo_point"},
        }
    },
}


def index_exists(client: OpenSearch, index: str = INDEX_NAME) -> bool:
    """Return True if the index exists in OpenSearch."""
    return bool(client.indices.exists(index=index))


def create_index(
    client: OpenSearch,
    *,
    index: str = INDEX_NAME,
    recreate: bool = False,
) -> None:
    """Create the hotels index with the canonical mapping.

    If the index already exists and recreate=False, this is a no-op.
    If recreate=True, the existing index is deleted first (all data lost).
    """
    if index_exists(client, index):
        if not recreate:
            return
        delete_index(client, index)
    client.indices.create(index=index, body=INDEX_BODY)


def delete_index(client: OpenSearch, index: str = INDEX_NAME) -> None:
    """Delete the index, silently ignoring a missing index."""
    try:
        client.indices.delete(index=index)
    except NotFoundError:
        pass
