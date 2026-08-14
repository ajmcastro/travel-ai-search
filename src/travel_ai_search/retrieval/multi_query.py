"""Multi-query retrieval (Milestone 11).

Runs independent BM25 + vector retrieval for each query variant in the
expanded query set, then fuses all 2N rank lists with RRF for higher recall
than a single-query hybrid search.

Design
------
For N expanded queries:
  - N × BM25 lexical searches → N ranked lists
  - N × dense vector ANN searches → N ranked lists
  - rrf_fuse([lex₁, vec₁, lex₂, vec₂, …, lexₙ, vecₙ]) → top-k candidates
  - [Optional] cross-encoder reranking on primary query → final top-k

The filter fields in `HybridSearchParams` (country, destination,
family_friendly, …) apply identically to every per-query retrieval.  Only
the query text varies.

RRF is the only supported fusion method here: weighted-sum normalisation
requires exactly two lists and cannot extend to 2N lists in a principled way.
If `params.fusion` is set to `weighted`, RRF is used anyway (with `params.rrf_k`)
and a debug-level log is emitted.

Cross-encoder reranking uses `params.query` (the original / rewritten query
before expansion) as the relevance query, so that scores reflect the user's
actual intent rather than any one variant.
"""

from __future__ import annotations

import logging
import time

from opensearchpy import OpenSearch

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.ingestion.index import INDEX_NAME
from travel_ai_search.reranking.base import Reranker
from travel_ai_search.retrieval.fusion import rrf_fuse
from travel_ai_search.retrieval.hybrid import HybridSearchParams, HybridSearchResult
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.types import Hit
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

logger = logging.getLogger(__name__)


def multi_query_search(
    client: OpenSearch,
    embedding_provider: EmbeddingProvider,
    params: HybridSearchParams,
    expanded_queries: list[str],
    *,
    index: str = INDEX_NAME,
    reranker: Reranker | None = None,
) -> HybridSearchResult:
    """Run BM25 + vector retrieval for each expanded query, fuse all results.

    For N expanded queries this function:
      1. Executes N lexical + N vector searches (2N total OpenSearch requests).
      2. Calls rrf_fuse on all 2N ranked lists.
      3. Optionally applies cross-encoder reranking on the primary query.

    Timing fields in the returned HybridSearchResult:
      - lexical_took_ms: sum of OpenSearch-reported times for all BM25 queries.
      - vector_took_ms:  sum of OpenSearch-reported times for all knn queries.
      - took_ms:         total wall-clock time including all queries and fusion.

    Args:
        client: OpenSearch client.
        embedding_provider: Encoder for converting queries to dense vectors.
        params: Search parameters (filters, top_k, candidate_k, rrf_k).
        expanded_queries: List of query strings from a QueryExpander.
            Must include at least one element (the original query).
        index: OpenSearch index name.
        reranker: Optional cross-encoder reranker applied after RRF fusion.
    """
    if not expanded_queries:
        expanded_queries = [params.query]

    t_start = time.monotonic()
    all_lists: list[list[Hit]] = []
    total_lexical_ms = 0
    total_vector_ms = 0

    for query_text in expanded_queries:
        lex_params = LexicalSearchParams(
            query=query_text,
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
        all_lists.append(lex_result.hits)
        total_lexical_ms += lex_result.took_ms

        vec_params = VectorSearchParams(
            query=query_text,
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
        all_lists.append(vec_result.hits)
        total_vector_ms += vec_result.took_ms

    pool_size = len({hit.id for lst in all_lists for hit in lst})

    fusion_top_k = max(params.top_k, params.rerank_k) if reranker is not None else params.top_k
    hits = rrf_fuse(all_lists, k=params.rrf_k, top_k=fusion_top_k)

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
        lexical_took_ms=total_lexical_ms,
        vector_took_ms=total_vector_ms,
        reranking_took_ms=reranking_took_ms,
    )
