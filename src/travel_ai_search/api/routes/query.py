"""Query understanding routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from travel_ai_search.api.deps import get_query_understanding_engine
from travel_ai_search.api.schemas.query import QueryUnderstandRequest, QueryUnderstandResponse
from travel_ai_search.query_understanding.base import QueryUnderstandingEngine

router = APIRouter(tags=["Query Understanding"])


@router.post("/understand", response_model=QueryUnderstandResponse)
def query_understand_endpoint(
    body: QueryUnderstandRequest,
    engine: QueryUnderstandingEngine = Depends(get_query_understanding_engine),
) -> QueryUnderstandResponse:
    """Parse a free-text travel query into structured constraints and a semantic query.

    Extracts hard constraints (month, departure airport, max price, star rating,
    family/adults flags, country) using pattern matching and keyword lookup.
    The semantic_query field contains the residual text after constraint phrases
    are removed — suitable for BM25 and vector retrieval.

    Useful for inspecting what the query understanding engine extracts before
    sending to the full search pipeline.

    Example:
        POST /query/understand
        {"query": "family beach holiday in Greece in July from Manchester under £2000"}
    """
    qu = engine.understand(body.query)
    return QueryUnderstandResponse.from_understanding(qu)
