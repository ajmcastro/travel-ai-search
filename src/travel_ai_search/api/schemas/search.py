"""Pydantic schemas for the search API request/response layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from travel_ai_search.api.schemas.query import QueryUnderstandResponse
from travel_ai_search.retrieval.fusion import FusionMethod
from travel_ai_search.retrieval.hybrid import HybridSearchResult
from travel_ai_search.retrieval.lexical import LexicalSearchResult
from travel_ai_search.retrieval.vector import VectorSearchResult


class SearchHit(BaseModel):
    """A single hotel result returned by a search endpoint.

    Includes all fields relevant for display.  Geographic coordinates and the
    derived geo_point are omitted — they live in the index for distance queries
    but are not useful in a JSON response.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    score: float

    # Display fields
    hotel_name: str
    hotel_description: str
    destination: str
    region: str
    country: str
    star_rating: int
    customer_rating: float
    price_per_person_gbp: float
    family_friendly: bool
    adults_only: bool
    amenities: list[str]
    board_types: list[str]
    beach_distance_km: float
    airport_distance_km: float
    activities: list[str]
    tags: list[str]
    available_departure_airports: list[str]
    available_months: list[str]
    climate_zone: str


class FacetBucket(BaseModel):
    key: str
    count: int


class LexicalSearchResponse(BaseModel):
    hits: list[SearchHit]
    total: int
    took_ms: int
    facets: dict[str, list[FacetBucket]] = {}

    @classmethod
    def from_result(cls, result: LexicalSearchResult) -> LexicalSearchResponse:
        hits = [
            SearchHit.model_validate({"id": hit.id, "score": hit.score, **hit.source})
            for hit in result.hits
        ]
        facets = {
            name: [
                FacetBucket(key=str(b["key"]), count=int(b["doc_count"]))
                for b in agg.get("buckets", [])
            ]
            for name, agg in result.raw_aggregations.items()
        }
        return cls(hits=hits, total=result.total, took_ms=result.took_ms, facets=facets)


class VectorSearchResponse(BaseModel):
    """Response schema for the vector (ANN) search endpoint.

    No facets — aggregations alongside a knn query are supported by
    OpenSearch but deferred to the hybrid milestone for simplicity.
    """

    hits: list[SearchHit]
    total: int
    took_ms: int

    @classmethod
    def from_result(cls, result: VectorSearchResult) -> VectorSearchResponse:
        hits = [
            SearchHit.model_validate({"id": hit.id, "score": hit.score, **hit.source})
            for hit in result.hits
        ]
        return cls(hits=hits, total=result.total, took_ms=result.took_ms)


class HybridSearchResponse(BaseModel):
    """Response schema for the hybrid (BM25 + vector) search endpoint.

    Includes per-stage timing so callers can see the breakdown between
    the lexical query, the vector query, fusion overhead, and optional
    cross-encoder reranking.  reranking_took_ms is 0 when reranking was
    not applied.
    """

    hits: list[SearchHit]
    total: int  # unique candidates in the fusion pool
    took_ms: int  # total wall-clock time (both queries + fusion + reranking)
    lexical_took_ms: int
    vector_took_ms: int
    reranking_took_ms: int = 0

    @classmethod
    def from_result(cls, result: HybridSearchResult) -> HybridSearchResponse:
        hits = [
            SearchHit.model_validate({"id": hit.id, "score": hit.score, **hit.source})
            for hit in result.hits
        ]
        return cls(
            hits=hits,
            total=result.total,
            took_ms=result.took_ms,
            lexical_took_ms=result.lexical_took_ms,
            vector_took_ms=result.vector_took_ms,
            reranking_took_ms=result.reranking_took_ms,
        )


class FullSearchRequest(BaseModel):
    """Request body for the full orchestrated search pipeline (POST /search).

    The query is parsed by the query understanding engine to extract hard
    constraints (month, airport, price, stars, family/adults flags, country)
    and a clean semantic query.  No explicit filter parameters are accepted
    here — constraints must come from natural language in query.
    """

    query: str
    top_k: int = 10
    candidate_k: int | None = None
    fusion: FusionMethod = FusionMethod.rrf
    rerank: bool = False
    rerank_k: int | None = None
    rewrite: bool = False
    expand: bool = False
    n_queries: int = 3


class FullSearchResponse(BaseModel):
    """Response for POST /search — extends HybridSearchResponse with QU metadata."""

    hits: list[SearchHit]
    total: int
    took_ms: int
    lexical_took_ms: int
    vector_took_ms: int
    reranking_took_ms: int = 0
    query_understanding: QueryUnderstandResponse
    rewritten_query: str | None = None
    expanded_queries: list[str] | None = None

    @classmethod
    def from_result(
        cls,
        result: HybridSearchResult,
        qu_response: QueryUnderstandResponse,
        *,
        rewritten_query: str | None = None,
        expanded_queries: list[str] | None = None,
    ) -> FullSearchResponse:
        hits = [
            SearchHit.model_validate({"id": hit.id, "score": hit.score, **hit.source})
            for hit in result.hits
        ]
        return cls(
            hits=hits,
            total=result.total,
            took_ms=result.took_ms,
            lexical_took_ms=result.lexical_took_ms,
            vector_took_ms=result.vector_took_ms,
            reranking_took_ms=result.reranking_took_ms,
            query_understanding=qu_response,
            rewritten_query=rewritten_query,
            expanded_queries=expanded_queries,
        )
