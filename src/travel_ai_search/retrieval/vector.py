"""Vector (ANN) retrieval via OpenSearch k-NN.

Design notes
------------
_build_vector_query() is a pure function (no I/O) so it can be unit-tested
without an OpenSearch instance or an embedding model.

Filter support
  The same 8 hard-constraint filters from lexical search are available here.
  Filters are applied using OpenSearch's efficient knn filter mode (OpenSearch
  2.9+), which prunes the HNSW graph during traversal rather than post-filtering.
  Post-filtering can silently return fewer than k results when many candidates
  fail the filter; efficient filtering avoids that.

Excluded fields
  embedding_vector is excluded from _source in responses — it is a large float
  array (384 × 4 bytes = 1.5 KB per document) with no display value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opensearchpy import OpenSearch

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.ingestion.index import INDEX_NAME
from travel_ai_search.retrieval.fusion import build_filter_clauses
from travel_ai_search.retrieval.types import Hit

# The vector field name must match the knn_vector mapping in index.py.
_VECTOR_FIELD = "embedding_vector"


@dataclass
class VectorSearchParams:
    """Parameters for a single vector (ANN) search request.

    Filter fields are identical to LexicalSearchParams so the evaluation
    framework can pass the same GoldenQuery.filters dict to both strategies.
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
class VectorSearchResult:
    hits: list[Hit]
    total: int
    took_ms: int


def _build_vector_query(
    vector: list[float],
    params: VectorSearchParams,
) -> dict[str, Any]:
    """Return the OpenSearch query body for a k-NN search.

    Pure function — no network calls, fully unit-testable.

    The knn clause uses efficient filter mode when filters are present.
    With no filters, it performs a plain ANN search across all documents.
    """
    filter_clauses = build_filter_clauses(
        country=params.country,
        destination=params.destination,
        family_friendly=params.family_friendly,
        adults_only=params.adults_only,
        min_star_rating=params.min_star_rating,
        max_price=params.max_price,
        month=params.month,
        airport=params.airport,
    )

    knn_clause: dict[str, Any] = {
        "vector": vector,
        "k": params.top_k,
    }
    if filter_clauses:
        # efficient filter: applied during HNSW graph traversal, not post-hoc
        knn_clause["filter"] = {"bool": {"filter": filter_clauses}}

    return {
        "size": params.top_k,
        "query": {"knn": {_VECTOR_FIELD: knn_clause}},
        # Exclude the large embedding vector from the response payload.
        "_source": {"excludes": [_VECTOR_FIELD]},
    }


def vector_search(
    client: OpenSearch,
    embedding_provider: EmbeddingProvider,
    params: VectorSearchParams,
    *,
    index: str = INDEX_NAME,
) -> VectorSearchResult:
    """Execute an ANN search and return structured results.

    Embeds `params.query` with the provider, builds the knn query, and
    returns the top-k nearest neighbours with optional hard filters.
    """
    vector = embedding_provider.embed(params.query)
    body = _build_vector_query(vector, params)
    response = client.search(index=index, body=body)

    hits = [
        Hit(
            id=hit["_id"],
            score=float(hit["_score"] or 0.0),
            source=hit["_source"],
        )
        for hit in response["hits"]["hits"]
    ]

    return VectorSearchResult(
        hits=hits,
        total=int(response["hits"]["total"]["value"]),
        took_ms=int(response["took"]),
    )
