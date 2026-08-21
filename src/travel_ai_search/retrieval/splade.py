"""Learned sparse retrieval (SPLADE) via OpenSearch rank_features — Milestone 17.

SPLADE queries are built as bool.should clauses — one rank_features clause per
non-zero query vocabulary term.  OpenSearch retrieves documents that share at least
one vocabulary term with the query (minimum_should_match: 1) and scores them by
the saturation function applied to document-side term weights, boosted by the
query-side term weight.

Scoring approximation (per matched term t):
    score_t ≈ query_weight_t × sat(doc_weight_t)

This approximates the SPLADE inner product score:
    score(q, d) = Σ_t query_weight_t × doc_weight_t

The saturation function (the OpenSearch default) is a BM25-like saturation that
bounds the influence of very large document-side weights.  The linear function
(exact dot product) is available in later OpenSearch versions.

Documents that have no splade_vector field (not yet encoded) are excluded —
run ``make generate-sparse-embeddings`` to populate the field before querying.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from opensearchpy import OpenSearch

from travel_ai_search.ingestion.index import INDEX_NAME
from travel_ai_search.retrieval.fusion import build_filter_clauses
from travel_ai_search.retrieval.types import Hit

_DEFAULT_FIELD = "splade_vector"


@dataclass
class SpladeSearchParams:
    """Parameters for a SPLADE sparse retrieval request.

    top_k_terms caps the query fan-out: the top_k_terms highest-weight vocabulary
    terms from the encoded query are sent to OpenSearch as rank_features clauses.
    Lower values are faster; higher values give more recall from rare terms.
    """

    query: str
    top_k: int = 10
    top_k_terms: int = 64
    country: str | None = None
    destination: str | None = None
    family_friendly: bool | None = None
    adults_only: bool | None = None
    min_star_rating: int | None = None
    max_price: float | None = None
    month: str | None = None
    airport: str | None = None


@dataclass
class SpladeSearchResult:
    hits: list[Hit]
    total: int
    took_ms: int
    n_query_terms: int  # actual non-zero terms used in the query (≤ top_k_terms)


def _build_splade_query(
    sparse_vector: dict[str, float],
    params: SpladeSearchParams,
    *,
    splade_field: str = _DEFAULT_FIELD,
) -> dict[str, Any]:
    """Return an OpenSearch query body for SPLADE sparse retrieval.

    Pure function — no network calls, fully unit-testable.
    """
    # Truncate to the highest-weight query terms to bound the query fan-out.
    top_terms = sorted(sparse_vector.items(), key=lambda kv: kv[1], reverse=True)[
        : params.top_k_terms
    ]

    # One rank_feature clause per query term; boost = query-side weight.
    # Note: field type is "rank_features" (plural); query clause key is "rank_feature" (singular).
    should_clauses: list[dict[str, Any]] = [
        {"rank_feature": {"field": f"{splade_field}.{term}", "boost": float(weight)}}
        for term, weight in top_terms
    ]

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

    # minimum_should_match: 1 ensures documents with no SPLADE encoding are excluded.
    # (When filter clauses are present OpenSearch would otherwise default to 0.)
    bool_body: dict[str, Any] = {
        "should": should_clauses,
        "minimum_should_match": 1,
    }
    if filter_clauses:
        bool_body["filter"] = filter_clauses

    return {
        "query": {"bool": bool_body},
        "size": params.top_k,
        # Exclude the (potentially large) sparse vector from the _source payload.
        "_source": {"excludes": [splade_field]},
    }


def splade_search(
    client: OpenSearch,
    splade_provider: Any,
    params: SpladeSearchParams,
    *,
    index: str = INDEX_NAME,
    splade_field: str = _DEFAULT_FIELD,
) -> SpladeSearchResult:
    """Execute a SPLADE learned-sparse retrieval query against OpenSearch.

    Encodes params.query to a sparse vocabulary-weight vector then queries
    OpenSearch via rank_features clauses.  Returns an empty result when the
    encoded query vector is empty (graceful degradation).
    """
    t0 = time.perf_counter()

    sparse_vector: dict[str, float] = splade_provider.encode(params.query)
    if not sparse_vector:
        return SpladeSearchResult(hits=[], total=0, took_ms=0, n_query_terms=0)

    body = _build_splade_query(sparse_vector, params, splade_field=splade_field)
    response = client.search(index=index, body=body)

    hits = [
        Hit(id=h["_id"], score=float(h["_score"] or 0.0), source=h["_source"])
        for h in response["hits"]["hits"]
    ]

    return SpladeSearchResult(
        hits=hits,
        total=int(response["hits"]["total"]["value"]),
        took_ms=round((time.perf_counter() - t0) * 1_000),
        n_query_terms=min(len(sparse_vector), params.top_k_terms),
    )
