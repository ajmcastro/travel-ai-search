"""Hybrid retrieval: BM25 + dense vector, fused via min-max normalised weighted sum.

Design notes
------------
Client-side hybrid (chosen over OpenSearch native pipeline)
  OpenSearch 2.x supports search pipelines with a hybrid query type, but
  setting them up requires index-time configuration and a running pipeline
  plugin.  Client-side fusion is more transparent for learning, works on all
  2.x versions, and keeps the normalisation logic fully unit-testable.

Candidate pool (candidate_k > top_k)
  Each retriever fetches `candidate_k` documents independently.  Fusion then
  selects the best `top_k` from the union (≤ 2 × candidate_k unique
  candidates).  A larger pool lets fusion re-rank documents that one retriever
  ranked highly but the other did not see at top_k cutoff.

Missing-retriever penalty
  Documents found by only one retriever receive 0.0 for the missing side.
  This naturally penalises single-retriever results: max score ≤ max_weight,
  while a document found and top-ranked by both can reach 1.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from opensearchpy import OpenSearch

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.ingestion.index import INDEX_NAME
from travel_ai_search.retrieval.fusion import fuse_results
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.types import Hit
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search


@dataclass
class HybridSearchParams:
    """Parameters for hybrid (BM25 + vector) search.

    candidate_k controls the pool size: each retriever fetches `candidate_k`
    documents; fusion then selects the best `top_k` from up to
    2 × candidate_k unique candidates.

    Filter fields are identical to LexicalSearchParams and VectorSearchParams
    so the evaluation framework can pass the same GoldenQuery.filters dict
    to all three strategies.
    """

    query: str
    top_k: int = 10
    candidate_k: int = 50
    lexical_weight: float = 0.5
    vector_weight: float = 0.5
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
class HybridSearchResult:
    hits: list[Hit]
    total: int  # unique candidates in the fusion pool (≤ 2 × candidate_k)
    took_ms: int  # wall-clock time for the entire hybrid call (ms)
    lexical_took_ms: int  # OpenSearch-reported time for the BM25 query (ms)
    vector_took_ms: int  # OpenSearch-reported time for the knn query (ms)


def hybrid_search(
    client: OpenSearch,
    embedding_provider: EmbeddingProvider,
    params: HybridSearchParams,
    *,
    index: str = INDEX_NAME,
) -> HybridSearchResult:
    """Run BM25 + vector search in parallel and merge via weighted score fusion.

    Pipeline
    --------
    1. BM25 lexical search → up to `candidate_k` hits.
    2. Dense vector ANN search → up to `candidate_k` hits.
    3. Min-max normalise each list independently to [0, 1].
    4. For every unique document in the union:
         combined = lexical_weight × norm_bm25 + vector_weight × norm_vector
       (missing side = 0.0)
    5. Sort descending, return the top `top_k` hits.
    """
    t_start = time.monotonic()

    # ── Stage 1: BM25 lexical ─────────────────────────────────────────────────
    lex_params = LexicalSearchParams(
        query=params.query,
        top_k=params.candidate_k,
        country=params.country,
        destination=params.destination,
        family_friendly=params.family_friendly,
        adults_only=params.adults_only,
        min_star_rating=params.min_star_rating,
        max_price=params.max_price,
        month=params.month,
        airport=params.airport,
    )
    lex_result = lexical_search(client, lex_params, index=index)

    # ── Stage 2: dense vector ANN ─────────────────────────────────────────────
    vec_params = VectorSearchParams(
        query=params.query,
        top_k=params.candidate_k,
        country=params.country,
        destination=params.destination,
        family_friendly=params.family_friendly,
        adults_only=params.adults_only,
        min_star_rating=params.min_star_rating,
        max_price=params.max_price,
        month=params.month,
        airport=params.airport,
    )
    vec_result = vector_search(client, embedding_provider, vec_params, index=index)

    # ── Stage 3: fusion ───────────────────────────────────────────────────────
    pool_size = len({h.id for h in lex_result.hits} | {h.id for h in vec_result.hits})
    hits = fuse_results(
        lex_result.hits,
        vec_result.hits,
        lexical_weight=params.lexical_weight,
        vector_weight=params.vector_weight,
        top_k=params.top_k,
    )

    took_ms = int((time.monotonic() - t_start) * 1000)

    return HybridSearchResult(
        hits=hits,
        total=pool_size,
        took_ms=took_ms,
        lexical_took_ms=lex_result.took_ms,
        vector_took_ms=vec_result.took_ms,
    )
