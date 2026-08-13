"""Search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from opensearchpy import OpenSearch

from travel_ai_search.api.deps import get_os_client, get_settings
from travel_ai_search.api.schemas.search import LexicalSearchResponse
from travel_ai_search.config.settings import Settings
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search

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
