"""Hybrid retrieval: BM25 + dense vector, fused via configurable strategy.

Design notes
------------
Client-side hybrid (chosen over OpenSearch native pipeline)
  OpenSearch 2.x supports search pipelines with a hybrid query type, but
  setting them up requires index-time configuration and a running pipeline
  plugin.  Client-side fusion is more transparent for learning, works on all
  2.x versions, and keeps the normalisation logic fully unit-testable.

Candidate pool (candidate_k > top_k)
  Each retriever fetches `candidate_k` documents independently.  Fusion then
  selects the best `top_k` (or `rerank_k` when reranking is active) from the
  union (≤ 2 × candidate_k unique candidates).  A larger pool lets fusion
  re-rank documents that one retriever ranked highly but the other did not
  see at top_k cutoff.

Missing-retriever penalty
  Documents found by only one retriever receive 0.0 for the missing side.
  This naturally penalises single-retriever results: max score ≤ max_weight,
  while a document found and top-ranked by both can reach 1.0.

Reranking stage (Milestone 8, optional)
  When a Reranker is supplied, fusion returns `rerank_k` candidates instead
  of `top_k`.  The cross-encoder then re-scores each (query, candidate) pair
  and the final `top_k` are returned.  If the reranker raises, the pipeline
  falls back to the fused ranking with a logged warning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from opensearchpy import OpenSearch

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.ingestion.index import INDEX_NAME
from travel_ai_search.reranking.base import Reranker
from travel_ai_search.retrieval.fusion import FusionMethod, fuse_results, rrf_fuse
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.types import Hit
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchParams:
    """Parameters for hybrid (BM25 + vector) search.

    candidate_k controls the retriever pool size: each retriever fetches
    `candidate_k` documents; fusion selects the best from up to
    2 × candidate_k unique candidates.

    rerank_k controls the reranking pool size: when a reranker is supplied,
    fusion returns `rerank_k` candidates (must be ≥ top_k) which the
    cross-encoder then re-scores to produce the final `top_k` results.
    Ignored when no reranker is provided.

    Filter fields are identical to LexicalSearchParams and VectorSearchParams
    so the evaluation framework can pass the same GoldenQuery.filters dict
    to all three strategies.
    """

    query: str
    top_k: int = 10
    candidate_k: int = 50
    # ── Fusion ──────────────────────────────────────────────────────────────
    fusion: FusionMethod = FusionMethod.weighted
    # Weighted-sum only — ignored when fusion=rrf
    lexical_weight: float = 0.5
    vector_weight: float = 0.5
    # RRF only — ignored when fusion=weighted
    rrf_k: int = 60
    # ── Reranking (Milestone 8) ──────────────────────────────────────────────
    # rerank_k: candidates passed to the cross-encoder; ignored when no reranker.
    # Must satisfy top_k ≤ rerank_k ≤ 2 × candidate_k for best results.
    rerank_k: int = 50
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
    reranking_took_ms: int = field(default=0)  # cross-encoder time; 0 when not reranked


def hybrid_search(
    client: OpenSearch,
    embedding_provider: EmbeddingProvider,
    params: HybridSearchParams,
    *,
    index: str = INDEX_NAME,
    reranker: Reranker | None = None,
) -> HybridSearchResult:
    """Run BM25 + vector retrieval, fuse results, and optionally rerank.

    Pipeline
    --------
    1. BM25 lexical search → up to `candidate_k` hits.
    2. Dense vector ANN search → up to `candidate_k` hits.
    3. Fuse (weighted-sum or RRF) → top `fusion_top_k` candidates.
       fusion_top_k = rerank_k when a reranker is supplied, else top_k.
    4. [Optional] Cross-encoder reranking → final top_k.
       Falls back to fused ranking if the reranker raises.
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
    # When reranking is active, fetch a larger candidate pool for the reranker.
    fusion_top_k = max(params.top_k, params.rerank_k) if reranker is not None else params.top_k
    pool_size = len({h.id for h in lex_result.hits} | {h.id for h in vec_result.hits})

    if params.fusion == FusionMethod.rrf:
        hits = rrf_fuse(
            [lex_result.hits, vec_result.hits],
            k=params.rrf_k,
            top_k=fusion_top_k,
        )
    else:
        hits = fuse_results(
            lex_result.hits,
            vec_result.hits,
            lexical_weight=params.lexical_weight,
            vector_weight=params.vector_weight,
            top_k=fusion_top_k,
        )

    # ── Stage 4: optional cross-encoder reranking ─────────────────────────────
    reranking_took_ms = 0
    if reranker is not None:
        t_rerank = time.monotonic()
        try:
            hits = reranker.rerank(params.query, hits, top_k=params.top_k)
        except Exception as exc:
            logger.warning("Reranker failed (%s); falling back to fused results.", exc)
            hits = hits[: params.top_k]
        reranking_took_ms = int((time.monotonic() - t_rerank) * 1000)

    took_ms = int((time.monotonic() - t_start) * 1000)

    return HybridSearchResult(
        hits=hits,
        total=pool_size,
        took_ms=took_ms,
        lexical_took_ms=lex_result.took_ms,
        vector_took_ms=vec_result.took_ms,
        reranking_took_ms=reranking_took_ms,
    )
