"""Unit tests for BM25 query building.

All tests here are pure — they call _build_query() and inspect the resulting
dict. No OpenSearch instance is required.
"""

from __future__ import annotations

from travel_ai_search.retrieval.lexical import _SEARCH_FIELDS, LexicalSearchParams, _build_query

# ── Helpers ───────────────────────────────────────────────────────────────────


def _params(**kwargs: object) -> LexicalSearchParams:
    defaults: dict[str, object] = {"query": "beach hotel", "top_k": 10}
    defaults.update(kwargs)
    return LexicalSearchParams(**defaults)  # type: ignore[arg-type]


def _get_multi_match(body: dict) -> dict:  # type: ignore[type-arg]
    """Extract the multi_match clause from a built query body."""
    must = body["query"]["bool"]["must"]
    for clause in must:
        if "multi_match" in clause:
            return clause["multi_match"]  # type: ignore[no-any-return]
    raise AssertionError("No multi_match clause found in query")


# ── Query structure ───────────────────────────────────────────────────────────


def test_build_query_has_bool_query() -> None:
    body = _build_query(_params())
    assert "query" in body
    assert "bool" in body["query"]


def test_build_query_size_equals_top_k() -> None:
    body = _build_query(_params(top_k=25))
    assert body["size"] == 25


def test_build_query_includes_aggregations() -> None:
    body = _build_query(_params())
    assert "aggs" in body
    assert "countries" in body["aggs"]
    assert "star_ratings" in body["aggs"]
    assert "board_types" in body["aggs"]
    assert "climate_zones" in body["aggs"]


def test_build_query_non_empty_uses_multi_match() -> None:
    body = _build_query(_params(query="beach family"))
    must = body["query"]["bool"]["must"]
    assert any("multi_match" in clause for clause in must)


def test_build_query_empty_uses_match_all() -> None:
    body = _build_query(_params(query=""))
    must = body["query"]["bool"]["must"]
    assert any("match_all" in clause for clause in must)


def test_build_query_whitespace_only_uses_match_all() -> None:
    body = _build_query(_params(query="   "))
    must = body["query"]["bool"]["must"]
    assert any("match_all" in clause for clause in must)


# ── Field boosts ──────────────────────────────────────────────────────────────


def test_hotel_name_has_higher_boost_than_description() -> None:
    # hotel_name^3 vs hotel_description (no boost = 1.0)
    name_boost = next(float(f.split("^")[1]) for f in _SEARCH_FIELDS if f.startswith("hotel_name"))
    assert name_boost >= 2.0, "hotel_name boost should be at least 2x"


def test_destination_is_boosted() -> None:
    dest_boost = next(float(f.split("^")[1]) for f in _SEARCH_FIELDS if f.startswith("destination"))
    assert dest_boost >= 1.5


def test_search_fields_include_description() -> None:
    assert any("hotel_description" in f for f in _SEARCH_FIELDS)


def test_search_fields_include_activities() -> None:
    assert any("activities" in f for f in _SEARCH_FIELDS)


def test_search_fields_include_tags() -> None:
    assert any("tags" in f for f in _SEARCH_FIELDS)


def test_multi_match_has_tie_breaker() -> None:
    mm = _get_multi_match(_build_query(_params()))
    assert "tie_breaker" in mm
    assert 0 < mm["tie_breaker"] < 1


def test_multi_match_has_fuzziness() -> None:
    mm = _get_multi_match(_build_query(_params()))
    assert mm.get("fuzziness") == "AUTO"


def test_multi_match_type_is_best_fields() -> None:
    mm = _get_multi_match(_build_query(_params()))
    assert mm["type"] == "best_fields"


# ── Filters ───────────────────────────────────────────────────────────────────


def test_no_filters_when_all_none() -> None:
    body = _build_query(_params())
    assert "filter" not in body["query"]["bool"]


def test_country_filter_is_term_query() -> None:
    body = _build_query(_params(country="Spain"))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"country": "Spain"}} for f in filters)


def test_destination_filter_is_term_query() -> None:
    body = _build_query(_params(destination="Marbella"))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"destination": "Marbella"}} for f in filters)


def test_family_friendly_filter_true() -> None:
    body = _build_query(_params(family_friendly=True))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"family_friendly": True}} for f in filters)


def test_family_friendly_filter_false() -> None:
    body = _build_query(_params(family_friendly=False))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"family_friendly": False}} for f in filters)


def test_adults_only_filter() -> None:
    body = _build_query(_params(adults_only=True))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"adults_only": True}} for f in filters)


def test_min_star_rating_filter_is_range() -> None:
    body = _build_query(_params(min_star_rating=4))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"range": {"star_rating": {"gte": 4}}} for f in filters)


def test_max_price_filter_is_range() -> None:
    body = _build_query(_params(max_price=1000.0))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"range": {"price_per_person_gbp": {"lte": 1000.0}}} for f in filters)


def test_month_filter_is_term() -> None:
    body = _build_query(_params(month="July"))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"available_months": "July"}} for f in filters)


def test_airport_filter_is_term() -> None:
    body = _build_query(_params(airport="MAN"))
    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"available_departure_airports": "MAN"}} for f in filters)


def test_multiple_filters_all_present() -> None:
    body = _build_query(_params(country="Spain", family_friendly=True, max_price=800.0))
    filters = body["query"]["bool"]["filter"]
    assert len(filters) == 3
