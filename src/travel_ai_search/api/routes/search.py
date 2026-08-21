"""Search routes."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Query
from opensearchpy import OpenSearch

from travel_ai_search.api.deps import (
    get_embedding_provider,
    get_knowledge_retriever,
    get_os_client,
    get_query_expander,
    get_query_rewriter,
    get_query_understanding_engine,
    get_rag_synthesizer,
    get_reranker,
    get_settings,
    get_splade_provider,
)
from travel_ai_search.api.schemas.query import QueryUnderstandResponse
from travel_ai_search.api.schemas.search import (
    DestinationContextItem,
    FullSearchRequest,
    FullSearchResponse,
    HybridSearchResponse,
    LexicalSearchResponse,
    SpladeSearchResponse,
    VectorSearchResponse,
)
from travel_ai_search.config.settings import Settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.observability.metrics import (
    search_fallbacks_total,
    search_latency_ms,
    search_requests_total,
)
from travel_ai_search.query_understanding.base import QueryUnderstandingEngine
from travel_ai_search.query_understanding.expander import QueryExpander
from travel_ai_search.query_understanding.rewriter import QueryRewriter
from travel_ai_search.reranking.base import Reranker
from travel_ai_search.retrieval.fusion import FusionMethod
from travel_ai_search.retrieval.hybrid import HybridSearchParams, HybridSearchResult, hybrid_search
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.multi_query import multi_query_search
from travel_ai_search.retrieval.splade import SpladeSearchParams, splade_search
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


@router.get("/lexical", response_model=LexicalSearchResponse)
def lexical_search_endpoint(
    q: str = Query("", description="Free-text search query"),
    top_k: int = Query(None, ge=1, le=100, description="Maximum results to return"),
    country: str | None = Query(None, description="Filter by country name"),
    destination: str | None = Query(None, description="Filter by exact destination name"),
    family_friendly: bool | None = Query(None, description="Filter to family-friendly hotels"),
    adults_only: bool | None = Query(None, description="Filter to adults-only hotels"),
    min_stars: int | None = Query(None, ge=1, le=5, description="Minimum star rating"),
    max_price: float | None = Query(None, gt=0, description="Maximum price per person (GBP)"),
    month: str | None = Query(None, description="Filter to hotels available in this month"),
    airport: str | None = Query(None, description="Filter by departure airport IATA code"),
    client: OpenSearch = Depends(get_os_client),
    settings: Settings = Depends(get_settings),
) -> LexicalSearchResponse:
    """BM25 lexical search across hotel name, description, destination, activities and more.

    Supports free-text queries with optional structured filters. All filters are
    hard constraints — documents not matching a filter are excluded entirely, not
    just ranked lower.

    Example:
        GET /search/lexical?q=family+beach+hotel&country=Spain&max_price=1000&month=July
    """
    params = LexicalSearchParams(
        query=q,
        top_k=top_k if top_k is not None else settings.top_k,
        country=country,
        destination=destination,
        family_friendly=family_friendly,
        adults_only=adults_only,
        min_star_rating=min_stars,
        max_price=max_price,
        month=month,
        airport=airport,
    )
    result = lexical_search(client, params, index=settings.opensearch_index_name)
    return LexicalSearchResponse.from_result(result)


@router.get("/vector", response_model=VectorSearchResponse)
def vector_search_endpoint(
    q: str = Query("", description="Natural-language search query"),
    top_k: int = Query(None, ge=1, le=100, description="Maximum results to return"),
    country: str | None = Query(None, description="Filter by country name"),
    destination: str | None = Query(None, description="Filter by exact destination name"),
    family_friendly: bool | None = Query(None, description="Filter to family-friendly hotels"),
    adults_only: bool | None = Query(None, description="Filter to adults-only hotels"),
    min_stars: int | None = Query(None, ge=1, le=5, description="Minimum star rating"),
    max_price: float | None = Query(None, gt=0, description="Maximum price per person (GBP)"),
    month: str | None = Query(None, description="Filter to hotels available in this month"),
    airport: str | None = Query(None, description="Filter by departure airport IATA code"),
    client: OpenSearch = Depends(get_os_client),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    settings: Settings = Depends(get_settings),
) -> VectorSearchResponse:
    """Dense vector (ANN) search using sentence-transformer embeddings and HNSW.

    The query is encoded into a 384-dimensional vector and compared against
    all indexed hotel embeddings using approximate nearest-neighbour search.
    Semantically similar hotels are returned even with no keyword overlap.

    Example:
        GET /search/vector?q=quiet+adults+retreat+near+the+sea&country=Spain
    """
    params = VectorSearchParams(
        query=q,
        top_k=top_k if top_k is not None else settings.top_k,
        country=country,
        destination=destination,
        family_friendly=family_friendly,
        adults_only=adults_only,
        min_star_rating=min_stars,
        max_price=max_price,
        month=month,
        airport=airport,
    )
    result = vector_search(client, provider, params, index=settings.opensearch_index_name)
    return VectorSearchResponse.from_result(result)


@router.get("/sparse", response_model=SpladeSearchResponse)
def sparse_search_endpoint(
    q: str = Query("", description="Natural-language search query"),
    top_k: int = Query(None, ge=1, le=100, description="Maximum results to return"),
    top_k_terms: int = Query(None, ge=1, le=512, description="Max query vocabulary terms"),
    country: str | None = Query(None, description="Filter by country name"),
    destination: str | None = Query(None, description="Filter by exact destination name"),
    family_friendly: bool | None = Query(None, description="Filter to family-friendly hotels"),
    adults_only: bool | None = Query(None, description="Filter to adults-only hotels"),
    min_stars: int | None = Query(None, ge=1, le=5, description="Minimum star rating"),
    max_price: float | None = Query(None, gt=0, description="Maximum price per person (GBP)"),
    month: str | None = Query(None, description="Filter to hotels available in this month"),
    airport: str | None = Query(None, description="Filter by departure airport IATA code"),
    client: OpenSearch = Depends(get_os_client),
    loaded_splade: object | None = Depends(get_splade_provider),
    settings: Settings = Depends(get_settings),
) -> SpladeSearchResponse:
    """Learned sparse (SPLADE) search using vocabulary-weight vectors stored in rank_features.

    SPLADE encodes the query using a masked-language model head: each vocabulary term
    receives a learned weight proportional to its semantic relevance.  Documents are
    scored by the inner product between query weights and document weights over shared
    vocabulary terms.

    Unlike BM25, SPLADE supports vocabulary expansion: the model can activate "resort"
    when the query says "hotel", even with no surface-form overlap.  Unlike dense
    vector search, results are interpretable — the matched vocabulary terms are visible.

    Requires SPLADE_ENABLED=true in settings and a pre-generated sparse index
    (run ``make generate-sparse-embeddings`` after ``make ingest``).

    Example:
        GET /search/sparse?q=quiet+adults+retreat+near+the+sea
    """
    from fastapi import HTTPException

    if loaded_splade is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "SPLADE provider is not loaded. "
                "Set SPLADE_ENABLED=true and run: make generate-sparse-embeddings"
            ),
        )

    params = SpladeSearchParams(
        query=q,
        top_k=top_k if top_k is not None else settings.top_k,
        top_k_terms=top_k_terms if top_k_terms is not None else settings.splade_top_k_terms,
        country=country,
        destination=destination,
        family_friendly=family_friendly,
        adults_only=adults_only,
        min_star_rating=min_stars,
        max_price=max_price,
        month=month,
        airport=airport,
    )
    result = splade_search(client, loaded_splade, params, index=settings.opensearch_index_name)
    return SpladeSearchResponse.from_result(result)


@router.get("/hybrid", response_model=HybridSearchResponse)
def hybrid_search_endpoint(
    q: str = Query("", description="Free-text or natural-language search query"),
    top_k: int = Query(None, ge=1, le=100, description="Maximum results to return"),
    candidate_k: int = Query(None, ge=1, le=200, description="Candidates per retriever"),
    fusion: FusionMethod | None = Query(None, description="Fusion method: weighted or rrf"),
    lexical_weight: float = Query(None, ge=0.0, le=1.0, description="BM25 score weight"),
    vector_weight: float = Query(None, ge=0.0, le=1.0, description="Vector score weight"),
    rrf_k: int = Query(None, ge=1, description="RRF smoothing constant k (default: 60)"),
    rerank: bool = Query(False, description="Apply cross-encoder reranking to top candidates"),
    rerank_k: int = Query(None, ge=1, le=200, description="Candidates to rerank (default: 50)"),
    country: str | None = Query(None, description="Filter by country name"),
    destination: str | None = Query(None, description="Filter by exact destination name"),
    family_friendly: bool | None = Query(None, description="Filter to family-friendly hotels"),
    adults_only: bool | None = Query(None, description="Filter to adults-only hotels"),
    min_stars: int | None = Query(None, ge=1, le=5, description="Minimum star rating"),
    max_price: float | None = Query(None, gt=0, description="Maximum price per person (GBP)"),
    month: str | None = Query(None, description="Filter to hotels available in this month"),
    airport: str | None = Query(None, description="Filter by departure airport IATA code"),
    client: OpenSearch = Depends(get_os_client),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    loaded_reranker: Reranker | None = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
) -> HybridSearchResponse:
    """Hybrid BM25 + vector search with configurable score fusion and optional reranking.

    Two fusion strategies are available via the `fusion` parameter:

    - **weighted** (default): min-max normalise each list to [0, 1], then
      combine: `combined = lexical_weight × norm_bm25 + vector_weight × norm_vec`
    - **rrf**: Reciprocal Rank Fusion (Cormack et al., 2009).
      `RRF_score(d) = Σ_r 1/(k + rank_r(d))`.  Uses only rank positions,
      so it is robust to retrievers with incomparable score scales.

    Set `rerank=true` to apply cross-encoder reranking to the top `rerank_k`
    fusion candidates.  Requires `reranking_enabled=true` in settings and the
    cross-encoder model loaded at startup; otherwise reranking is silently
    skipped and `reranking_took_ms` will be 0.

    Example (weighted):
        GET /search/hybrid?q=romantic+beach+retreat+in+Spain&country=Spain
    Example (RRF):
        GET /search/hybrid?q=hotels+in+Tenerife&fusion=rrf
    Example (RRF + reranking):
        GET /search/hybrid?q=luxury+adults+spa&fusion=rrf&rerank=true
    """
    params = HybridSearchParams(
        query=q,
        top_k=top_k if top_k is not None else settings.top_k,
        candidate_k=candidate_k if candidate_k is not None else settings.hybrid_candidate_k,
        fusion=fusion if fusion is not None else FusionMethod(settings.hybrid_fusion),
        lexical_weight=(
            lexical_weight if lexical_weight is not None else settings.hybrid_lexical_weight
        ),
        vector_weight=(
            vector_weight if vector_weight is not None else settings.hybrid_vector_weight
        ),
        rrf_k=rrf_k if rrf_k is not None else settings.rrf_k,
        rerank_k=rerank_k if rerank_k is not None else settings.rerank_k,
        country=country,
        destination=destination,
        family_friendly=family_friendly,
        adults_only=adults_only,
        min_star_rating=min_stars,
        max_price=max_price,
        month=month,
        airport=airport,
    )
    active_reranker = loaded_reranker if rerank else None
    result = hybrid_search(
        client,
        provider,
        params,
        index=settings.opensearch_index_name,
        reranker=active_reranker,
    )
    return HybridSearchResponse.from_result(result)


@router.post("", response_model=FullSearchResponse)
def full_search_endpoint(
    body: FullSearchRequest,
    client: OpenSearch = Depends(get_os_client),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    loaded_reranker: Reranker | None = Depends(get_reranker),
    qu_engine: QueryUnderstandingEngine = Depends(get_query_understanding_engine),
    query_rewriter: QueryRewriter | None = Depends(get_query_rewriter),
    query_expander: QueryExpander | None = Depends(get_query_expander),
    knowledge_retriever: object | None = Depends(get_knowledge_retriever),
    rag_synthesizer: object | None = Depends(get_rag_synthesizer),
    settings: Settings = Depends(get_settings),
) -> FullSearchResponse:
    """Full orchestrated search pipeline: QU → optional rewriting → optional expansion → retrieval.

    The query is parsed by the rule-based query understanding engine, which extracts
    hard constraints (month, departure airport, max price, star rating, family/adults
    flags, country) and a clean semantic query.

    When rewrite=true and a query rewriter is configured (query_rewriting_enabled=true),
    the semantic query is expanded or paraphrased by the LLM provider before retrieval.

    When expand=true and a query expander is configured (query_expansion_enabled=true),
    n_queries variant queries are generated from the retrieval query and retrieved
    independently.  All rank lists are fused via RRF, giving broader candidate recall
    than a single-query hybrid search.  Both rewrite and expand may be combined: the
    rewritten query is expanded, not the raw semantic query.

    expanded_queries in the response lists the variants used for observability.

    Example:
        POST /search
        {"query": "family beach holiday in Greece July from Manchester under £2000"}

    Example (with expansion):
        POST /search
        {"query": "quiet luxury spa retreat", "expand": true, "n_queries": 3}
    """
    _t_start = time.perf_counter()

    # ── Stage 1: Query understanding ──────────────────────────────────────
    _t_qu = time.perf_counter()
    qu = qu_engine.understand(body.query)
    qu_took_ms = round((time.perf_counter() - _t_qu) * 1_000)
    qu_response = QueryUnderstandResponse.from_understanding(qu)

    # ── Stage 2: Optional query rewriting ────────────────────────────────
    # Graceful degradation: if the rewriter raises at runtime (e.g. LLM
    # timeout) we log a warning and continue with the original semantic query.
    rewritten_query: str | None = None
    retrieval_query = qu.semantic_query
    rewrite_took_ms = 0
    if body.rewrite and query_rewriter is not None:
        _t_rw = time.perf_counter()
        try:
            rewritten_query = query_rewriter.rewrite(qu.semantic_query)
            retrieval_query = rewritten_query
        except Exception as exc:
            logger.warning(
                "Query rewriting failed at runtime: %s — using original semantic query.", exc
            )
        rewrite_took_ms = round((time.perf_counter() - _t_rw) * 1_000)

    # ── Stage 3: Retrieval ────────────────────────────────────────────────
    candidate_k = body.candidate_k if body.candidate_k is not None else settings.hybrid_candidate_k
    rerank_k = body.rerank_k if body.rerank_k is not None else settings.rerank_k

    params = HybridSearchParams(
        query=retrieval_query,
        top_k=body.top_k,
        candidate_k=candidate_k,
        fusion=body.fusion,
        rrf_k=settings.rrf_k,
        rerank_k=rerank_k,
        **qu.to_search_filters(),
    )
    active_reranker = loaded_reranker if body.rerank else None

    # Determine planned strategy label (used for metrics and response field).
    strategy = "multi_query" if (body.expand and query_expander is not None) else "hybrid"
    fallback_used = False

    expanded_queries: list[str] | None = None
    try:
        if body.expand and query_expander is not None:
            n_queries = max(1, body.n_queries)
            expanded_queries = query_expander.expand(retrieval_query, n_queries)
            result = multi_query_search(
                client,
                provider,
                params,
                expanded_queries,
                index=settings.opensearch_index_name,
                reranker=active_reranker,
            )
        else:
            result = hybrid_search(
                client,
                provider,
                params,
                index=settings.opensearch_index_name,
                reranker=active_reranker,
            )
    except Exception as exc:
        # Graceful degradation: if vector embedding fails (or any other
        # unrecoverable error in hybrid search) fall back to BM25 lexical
        # retrieval so the request still returns useful results.
        logger.warning(
            "Hybrid/vector retrieval failed: %s — falling back to BM25 (lexical_fallback).", exc
        )
        fallback_used = True
        strategy = "lexical_fallback"
        search_fallbacks_total.inc({"reason": "embedding_failure"})

        lex_params = LexicalSearchParams(
            query=retrieval_query,
            top_k=body.top_k,
            **qu.to_search_filters(),
        )
        lex_result = lexical_search(client, lex_params, index=settings.opensearch_index_name)
        result = HybridSearchResult(
            hits=lex_result.hits,
            total=lex_result.total,
            took_ms=lex_result.took_ms,
            lexical_took_ms=lex_result.took_ms,
            vector_took_ms=0,
            reranking_took_ms=0,
        )

    # ── Stage 4: Optional RAG ─────────────────────────────────────────────
    # Runs after hotel retrieval so it does not affect hotel ranking.
    knowledge_context: list[DestinationContextItem] | None = None
    rag_summary: str | None = None
    rag_took_ms = 0
    if body.rag and knowledge_retriever is not None:
        from travel_ai_search.rag.knowledge import DestinationKnowledge
        from travel_ai_search.rag.retriever import KnowledgeRetriever

        _t_rag = time.perf_counter()
        raw_knowledge: list[DestinationKnowledge] = []
        try:
            retriever: KnowledgeRetriever = knowledge_retriever  # type: ignore[assignment]
            raw_knowledge = retriever.retrieve(
                qu.semantic_query,
                country=qu.country,
            )
            knowledge_context = [DestinationContextItem.from_knowledge(k) for k in raw_knowledge]
        except Exception as exc:
            logger.warning("Knowledge retrieval failed: %s — skipping RAG.", exc)

        if raw_knowledge and rag_synthesizer is not None:
            try:
                from travel_ai_search.rag.synthesizer import RAGSynthesizer

                synthesizer: RAGSynthesizer = rag_synthesizer  # type: ignore[assignment]
                rag_summary = synthesizer.synthesize(body.query, result.hits, raw_knowledge)
            except Exception as exc:
                logger.warning("RAG synthesis failed: %s — returning knowledge context only.", exc)

        rag_took_ms = round((time.perf_counter() - _t_rag) * 1_000)

    # ── Observability ─────────────────────────────────────────────────────
    total_took_ms = round((time.perf_counter() - _t_start) * 1_000)

    search_requests_total.inc({"strategy": strategy})
    search_latency_ms.observe(total_took_ms)

    logger.info(
        "search_request_completed",
        extra={
            "query": body.query,
            "strategy": strategy,
            "result_count": len(result.hits),
            "total_took_ms": total_took_ms,
            "qu_took_ms": qu_took_ms,
            "rewrite_took_ms": rewrite_took_ms,
            "lexical_took_ms": result.lexical_took_ms,
            "vector_took_ms": result.vector_took_ms,
            "reranking_took_ms": result.reranking_took_ms,
            "rag_took_ms": rag_took_ms,
            "fallback_used": fallback_used,
            "rewritten": rewritten_query is not None,
            "expanded": expanded_queries is not None,
        },
    )

    # Patch the result's took_ms to reflect the full end-to-end wall-clock time
    # (hybrid_search's own took_ms only covers the retrieval stage).
    result = HybridSearchResult(
        hits=result.hits,
        total=result.total,
        took_ms=total_took_ms,
        lexical_took_ms=result.lexical_took_ms,
        vector_took_ms=result.vector_took_ms,
        reranking_took_ms=result.reranking_took_ms,
    )

    return FullSearchResponse.from_result(
        result,
        qu_response,
        qu_took_ms=qu_took_ms,
        rewrite_took_ms=rewrite_took_ms,
        rag_took_ms=rag_took_ms,
        strategy=strategy,
        fallback_used=fallback_used,
        rewritten_query=rewritten_query,
        expanded_queries=expanded_queries,
        knowledge_context=knowledge_context,
        rag_summary=rag_summary,
    )
