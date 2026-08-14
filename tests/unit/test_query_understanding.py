"""Unit tests for the query understanding module.

All tests are pure — no OpenSearch, no embedding models, no network calls.
"""

from __future__ import annotations

import pytest

from travel_ai_search.query_understanding.base import QueryUnderstandingEngine
from travel_ai_search.query_understanding.extractor import RuleBasedQueryUnderstandingEngine
from travel_ai_search.query_understanding.models import QueryUnderstanding


@pytest.fixture
def engine() -> RuleBasedQueryUnderstandingEngine:
    return RuleBasedQueryUnderstandingEngine()


# ── Protocol / structural typing ──────────────────────────────────────────────


def test_rule_based_engine_satisfies_protocol() -> None:
    engine = RuleBasedQueryUnderstandingEngine()
    assert isinstance(engine, QueryUnderstandingEngine)


def test_any_object_with_understand_satisfies_protocol() -> None:
    class _Stub:
        def understand(self, query: str) -> QueryUnderstanding:
            return QueryUnderstanding(original_query=query, semantic_query=query)

    assert isinstance(_Stub(), QueryUnderstandingEngine)


# ── Month extraction ──────────────────────────────────────────────────────────


def test_month_full_name(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach holiday in July")
    assert qu.month == "July"


def test_month_abbreviation(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("holidays in oct")
    assert qu.month == "October"


def test_month_no_preposition(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach hotel August")
    assert qu.month == "August"


def test_month_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury spa hotel Spain")
    assert qu.month is None


def test_month_removed_from_semantic_query(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach holiday in October")
    assert "October" not in qu.semantic_query
    assert "october" not in qu.semantic_query.lower()
    assert "beach" in qu.semantic_query.lower()


# ── Price extraction ──────────────────────────────────────────────────────────


def test_price_under_pounds(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family holiday under 2000 pounds")
    assert qu.max_price == 2000.0


def test_price_pound_symbol(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("holiday under £1500 per person")
    assert qu.max_price == 1500.0


def test_price_up_to(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach resort up to £800")
    assert qu.max_price == 800.0


def test_price_max_keyword(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("5 star hotel max £3000")
    assert qu.max_price == 3000.0


def test_price_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury spa resort in Spain")
    assert qu.max_price is None


def test_price_removed_from_semantic_query(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family holiday in Spain under £2000")
    assert "2000" not in qu.semantic_query
    assert "holiday" in qu.semantic_query.lower()


# ── Star rating extraction ────────────────────────────────────────────────────


def test_star_rating_digit(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("5 star luxury hotel Spain")
    assert qu.min_star_rating == 5


def test_star_rating_word(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("four star hotel in Greece")
    assert qu.min_star_rating == 4


def test_star_rating_hyphenated(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("4-star beachfront resort")
    assert qu.min_star_rating == 4


def test_star_rating_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach holiday")
    assert qu.min_star_rating is None


def test_star_rating_removed_from_semantic(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("5 star hotel in Greece")
    assert "5 star" not in qu.semantic_query
    assert "hotel" in qu.semantic_query.lower()


# ── Airport extraction ────────────────────────────────────────────────────────


def test_airport_manchester(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family holiday from Manchester")
    assert qu.departure_airport == "MAN"


def test_airport_departing_from(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach resort departing from Birmingham")
    assert qu.departure_airport == "BHX"


def test_airport_bare_city_name(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("holidays Edinburgh July")
    assert qu.departure_airport == "EDI"


def test_airport_gatwick(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach hotel departing Gatwick")
    assert qu.departure_airport == "LGW"


def test_airport_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury spa resort in Santorini")
    assert qu.departure_airport is None


def test_airport_removed_from_semantic(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family holiday from Manchester in July")
    assert "manchester" not in qu.semantic_query.lower()
    assert "family" not in qu.semantic_query.lower()


# ── Location (country) extraction ────────────────────────────────────────────


def test_country_name_direct(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach hotel in Greece")
    assert qu.country == "Greece"


def test_country_from_region_tenerife(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("hotels in Tenerife")
    assert qu.country == "Spain"


def test_country_from_region_santorini(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("romantic getaway in Santorini")
    assert qu.country == "Greece"


def test_country_from_region_algarve(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family holiday Algarve July")
    assert qu.country == "Portugal"


def test_country_from_region_ibiza(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("party nightlife holiday Ibiza")
    assert qu.country == "Spain"


def test_country_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach holiday in August")
    assert qu.country is None


def test_country_removed_from_semantic(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach holiday in Greece")
    assert "greece" not in qu.semantic_query.lower()
    assert "beach" in qu.semantic_query.lower()


# ── Family / adults-only flags ────────────────────────────────────────────────


def test_family_flag_keyword(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach holiday")
    assert qu.family_friendly is True


def test_family_flag_with_kids(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach resort with kids")
    assert qu.family_friendly is True


def test_family_flag_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury spa adults only")
    assert qu.family_friendly is None


def test_adults_only_flag(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("adults only spa resort")
    assert qu.adults_only is True


def test_adults_only_no_kids(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("quiet hotel no kids pool")
    assert qu.adults_only is True


def test_adults_only_none_when_absent(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach holiday Spain")
    assert qu.adults_only is None


# ── Soft preferences ──────────────────────────────────────────────────────────


def test_soft_pref_beach(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beachfront hotel in Spain")
    assert "beach" in qu.soft_preferences


def test_soft_pref_spa(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury spa retreat adults only")
    assert "spa" in qu.soft_preferences


def test_soft_pref_multiple(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("all-inclusive beach resort with spa and pool")
    assert "all-inclusive" in qu.soft_preferences
    assert "beach" in qu.soft_preferences
    assert "spa" in qu.soft_preferences
    assert "pool" in qu.soft_preferences


def test_soft_prefs_empty_when_none_detected(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("5 star hotel in Greece July Manchester")
    assert isinstance(qu.soft_preferences, list)


# ── Semantic query distillation ───────────────────────────────────────────────


def test_semantic_query_multi_constraint(engine: RuleBasedQueryUnderstandingEngine) -> None:
    """Golden dataset multi_constraint pattern: all constraints removed, semantic remains."""
    qu = engine.understand("family beach hotel Greece July Manchester")
    assert qu.month == "July"
    assert qu.country == "Greece"
    assert qu.departure_airport == "MAN"
    assert qu.family_friendly is True
    # Semantic should keep "beach hotel" and not include constraint phrases
    assert "beach" in qu.semantic_query.lower()
    assert "hotel" in qu.semantic_query.lower()
    assert "greece" not in qu.semantic_query.lower()
    assert "manchester" not in qu.semantic_query.lower()
    assert "july" not in qu.semantic_query.lower()


def test_semantic_query_natural_language(engine: RuleBasedQueryUnderstandingEngine) -> None:
    """Golden dataset natural_language pattern."""
    nl_query = (
        "Find me a family holiday somewhere warm in October"
        " departing from Manchester under 2000 pounds"
    )
    qu = engine.understand(nl_query)
    assert qu.month == "October"
    assert qu.departure_airport == "MAN"
    assert qu.max_price == 2000.0
    assert qu.family_friendly is True
    assert "warm" in qu.semantic_query.lower()


def test_semantic_query_fallback_to_original_when_empty(
    engine: RuleBasedQueryUnderstandingEngine,
) -> None:
    """If all tokens are consumed, fall back to original query rather than returning empty."""
    qu = engine.understand("July")
    # Consumed "July" → semantic would be empty → fall back to original
    assert qu.semantic_query  # must not be empty


def test_original_query_always_preserved(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury 5 star hotel in Greece July from Manchester")
    assert qu.original_query == "luxury 5 star hotel in Greece July from Manchester"


# ── QueryUnderstanding.to_search_filters ─────────────────────────────────────


def test_to_search_filters_full(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("family beach hotel Greece July Manchester under £2000")
    filters = qu.to_search_filters()
    assert filters["country"] == "Greece"
    assert filters["month"] == "July"
    assert filters["airport"] == "MAN"
    assert filters["max_price"] == 2000.0
    assert filters["family_friendly"] is True


def test_to_search_filters_excludes_none_values(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("luxury spa hotel")
    filters = qu.to_search_filters()
    assert "month" not in filters
    assert "airport" not in filters
    assert "max_price" not in filters
    assert "family_friendly" not in filters
    assert "country" not in filters


def test_to_search_filters_empty_for_no_constraints(
    engine: RuleBasedQueryUnderstandingEngine,
) -> None:
    qu = engine.understand("hotel with pool")
    assert qu.to_search_filters() == {}


# ── Understanding timing ──────────────────────────────────────────────────────


def test_understanding_took_ms_is_non_negative(engine: RuleBasedQueryUnderstandingEngine) -> None:
    qu = engine.understand("beach hotel in Greece July")
    assert qu.understanding_took_ms >= 0
