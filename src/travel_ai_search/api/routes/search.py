"""Search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from opensearchpy import OpenSearch

from travel_ai_search.api.deps import (
    get_embedding_provider,
    get_os_client,
    get_reranker,
    get_settings,
)
from travel_ai_search.api.schemas.search import (
    HybridSearchResponse,
    LexicalSearchResponse,
    VectorSearchResponse,
)
from travel_ai_search.config.settings import Settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.reranking.base import Reranker
from travel_ai_search.retrieval.fusion import FusionMethod
from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

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
