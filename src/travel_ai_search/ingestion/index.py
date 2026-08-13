"""OpenSearch index definition for travel hotels.

Design rationale for each field type:
  text + english analyser  → BM25 full-text search with stemming and stop-word removal
  keyword                  → exact filters, aggregations, facets
  integer / float          → range queries (price, star_rating, distances)
  boolean                  → family_friendly / adults_only filters
  geo_point                → distance and bounding-box queries
  knn_vector               → HNSW approximate nearest-neighbour search (Milestone 5)
  multi-field (text+kw)    → BM25 search AND exact filtering on the same field
"""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

INDEX_NAME = "travel_hotels"

# Dimension of the embedding model (all-MiniLM-L6-v2 → 384).
# This must match Settings.embedding_dimension and the loaded model.
# Changing the model requires recreating the index (knn_vector dimension is immutable).
EMBEDDING_DIMENSION = 384

INDEX_BODY: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        # 0 replicas keeps cluster health green on a single-node dev setup.
        # In production, raise this to 1+.
        "number_of_replicas": 0,
        # Enable the OpenSearch k-NN plugin for this index.
        # Without this flag, knn_vector fields can be stored but not searched.
        "knn": True,
        # ef_search controls the ANN search quality vs speed tradeoff at query time.
        # Higher values visit more candidate neighbours → better recall, slower queries.
        # This can also be overridden per-query; 100 is a sensible default.
        "knn.algo_param.ef_search": 100,
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
            # location stores both as a geo_point for distance queries
            "latitude": {"type": "float"},
            "longitude": {"type": "float"},
            "location": {"type": "geo_point"},
            # ── Dense vector (HNSW) ──────────────────────────────────────────
            # Stores the output of build_embedding_text() passed through the
            # embedding model.  knn_vector fields are excluded from _source
            # in search responses (large float arrays add no display value).
            #
            # HNSW parameters:
            #   m=16             — edges per node; higher → better recall, more memory
            #   ef_construction  — candidate pool during graph build; higher → better
            #                      graph quality, slower indexing (one-time cost)
            "embedding_vector": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIMENSION,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    # lucene supports efficient filter mode (OpenSearch 2.9+);
                    # nmslib does not support filters in the knn query clause.
                    "engine": "lucene",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16,
                    },
                },
            },
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
