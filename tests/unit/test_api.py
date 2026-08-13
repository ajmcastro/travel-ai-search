"""Unit tests for the FastAPI application.

The OpenSearch client is mocked at the lifespan level so these tests run
without any infrastructure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from travel_ai_search.api.app import app


def _mock_os_response(hits: list[dict] | None = None, total: int = 0) -> dict:  # type: ignore[type-arg]
    """Build a minimal OpenSearch search response dict."""
    return {
        "took": 3,
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": hits or [],
        },
        "aggregations": {
            "countries": {"buckets": []},
            "star_ratings": {"buckets": []},
            "board_types": {"buckets": []},
            "climate_zones": {"buckets": []},
        },
    }


# ── Health endpoint ───────────────────────────────────────────────────────────


@patch("travel_ai_search.api.app.create_client", return_value=MagicMock())
def test_health_returns_ok(_: MagicMock) -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Lexical search endpoint ───────────────────────────────────────────────────


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_returns_200(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        response = client.get("/search/lexical?q=beach")
    assert response.status_code == 200


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_response_shape(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        data = client.get("/search/lexical?q=beach").json()
    assert "hits" in data
    assert "total" in data
    assert "took_ms" in data
    assert "facets" in data


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_empty_query_is_accepted(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        response = client.get("/search/lexical")
    assert response.status_code == 200


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_top_k_passed_to_opensearch(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        client.get("/search/lexical?q=beach&top_k=5")

    call_body = mock_os.search.call_args[1]["body"]
    assert call_body["size"] == 5


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_country_filter_passed_to_opensearch(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        client.get("/search/lexical?q=beach&country=Spain")

    call_body = mock_os.search.call_args[1]["body"]
    filters = call_body["query"]["bool"]["filter"]
    assert {"term": {"country": "Spain"}} in filters


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_with_hit_returns_valid_schema(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response(
        hits=[
            {
                "_id": "hotel_000001",
                "_score": 3.14,
                "_source": {
                    "hotel_name": "Test Hotel",
                    "hotel_description": "A nice hotel.",
                    "destination": "Albufeira",
                    "region": "Algarve",
                    "country": "Portugal",
                    "star_rating": 4,
                    "customer_rating": 8.0,
                    "price_per_person_gbp": 749.0,
                    "family_friendly": True,
                    "adults_only": False,
                    "amenities": ["pool"],
                    "board_types": ["half_board"],
                    "beach_distance_km": 0.2,
                    "airport_distance_km": 20.0,
                    "activities": ["swimming"],
                    "tags": ["beach"],
                    "available_departure_airports": ["LGW"],
                    "available_months": ["July"],
                    "climate_zone": "Mediterranean",
                    # Extra fields that should be silently ignored
                    "latitude": 37.09,
                    "longitude": -8.25,
                    "location": {"lat": 37.09, "lon": -8.25},
                },
            }
        ],
        total=1,
    )
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        data = client.get("/search/lexical?q=beach").json()

    assert data["total"] == 1
    hit = data["hits"][0]
    assert hit["id"] == "hotel_000001"
    assert hit["score"] == 3.14
    assert hit["hotel_name"] == "Test Hotel"
    # Extra fields from _source should not appear in the response
    assert "latitude" not in hit
    assert "location" not in hit


@patch("travel_ai_search.api.app.create_client")
def test_lexical_search_facets_in_response(mock_create: MagicMock) -> None:
    mock_os = MagicMock()
    mock_os.search.return_value = {
        "took": 2,
        "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        "aggregations": {
            "countries": {"buckets": [{"key": "Spain", "doc_count": 3}]},
            "star_ratings": {"buckets": [{"key": 4, "doc_count": 2}]},
            "board_types": {"buckets": []},
            "climate_zones": {"buckets": []},
        },
    }
    mock_create.return_value = mock_os

    with TestClient(app) as client:
        data = client.get("/search/lexical?q=test").json()

    assert data["facets"]["countries"][0] == {"key": "Spain", "count": 3}
    assert data["facets"]["star_ratings"][0] == {"key": "4", "count": 2}
