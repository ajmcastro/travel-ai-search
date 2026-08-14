"""Unit tests for Milestone 11 — query expansion / multi-query retrieval.

Covers:
  - QueryExpander protocol satisfaction
  - IdentityQueryExpander behaviour
  - LocalQueryExpander: variant generation, synonym substitution, context elaboration
  - LocalQueryExpander: edge cases (empty query, n_queries extremes)
"""

from __future__ import annotations

from travel_ai_search.query_understanding.expander import (
    IdentityQueryExpander,
    LocalQueryExpander,
    QueryExpander,
)

# ── Protocol satisfaction ──────────────────────────────────────────────────────


class TestQueryExpanderProtocol:
    def test_identity_satisfies_protocol(self) -> None:
        assert isinstance(IdentityQueryExpander(), QueryExpander)

    def test_local_satisfies_protocol(self) -> None:
        assert isinstance(LocalQueryExpander(), QueryExpander)

    def test_custom_class_satisfies_protocol(self) -> None:
        class _Custom:
            def expand(self, semantic_query: str, n_queries: int = 3) -> list[str]:
                return [semantic_query]

        assert isinstance(_Custom(), QueryExpander)

    def test_class_without_expand_fails_protocol(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), QueryExpander)


# ── IdentityQueryExpander ─────────────────────────────────────────────────────


class TestIdentityQueryExpander:
    def setup_method(self) -> None:
        self.expander = IdentityQueryExpander()

    def test_returns_list_with_original(self) -> None:
        result = self.expander.expand("beach holiday")
        assert result == ["beach holiday"]

    def test_always_returns_single_element(self) -> None:
        for n in (1, 2, 3, 5):
            result = self.expander.expand("spa resort", n_queries=n)
            assert result == ["spa resort"], f"Failed for n_queries={n}"

    def test_handles_empty_query(self) -> None:
        result = self.expander.expand("")
        assert result == [""]

    def test_handles_whitespace_query(self) -> None:
        result = self.expander.expand("   ")
        assert result == ["   "]

    def test_first_element_is_original(self) -> None:
        query = "luxury adults spa"
        assert self.expander.expand(query)[0] == query


# ── LocalQueryExpander ────────────────────────────────────────────────────────


class TestLocalQueryExpanderLength:
    def setup_method(self) -> None:
        self.expander = LocalQueryExpander()

    def test_n_queries_1_returns_one_element(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=1)
        assert len(result) == 1

    def test_n_queries_2_returns_two_elements(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=2)
        assert len(result) == 2

    def test_n_queries_3_returns_three_elements(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=3)
        assert len(result) == 3

    def test_n_queries_4_returns_four_elements(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=4)
        assert len(result) == 4

    def test_n_queries_0_returns_one_element(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=0)
        assert len(result) == 1

    def test_n_queries_negative_returns_one_element(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=-1)
        assert len(result) == 1


class TestLocalQueryExpanderFirstElement:
    def setup_method(self) -> None:
        self.expander = LocalQueryExpander()

    def test_first_element_always_original(self) -> None:
        for query in ("beach holiday", "luxury spa", "family resort", "unknown xyz"):
            result = self.expander.expand(query, n_queries=3)
            assert result[0] == query, f"First element was not original for: {query!r}"

    def test_first_element_with_n_queries_1(self) -> None:
        query = "quiet beach"
        result = self.expander.expand(query, n_queries=1)
        assert result[0] == query


class TestLocalQueryExpanderSynonymVariant:
    def setup_method(self) -> None:
        self.expander = LocalQueryExpander()

    def test_beach_substituted(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=2)
        synonym_variant = result[1]
        assert "coastal" in synonym_variant

    def test_luxury_substituted(self) -> None:
        result = self.expander.expand("luxury resort", n_queries=2)
        synonym_variant = result[1]
        assert "premium" in synonym_variant

    def test_family_substituted(self) -> None:
        result = self.expander.expand("family holiday", n_queries=2)
        synonym_variant = result[1]
        assert "child-friendly" in synonym_variant

    def test_spa_substituted(self) -> None:
        result = self.expander.expand("spa retreat", n_queries=2)
        synonym_variant = result[1]
        assert "wellness" in synonym_variant

    def test_quiet_substituted(self) -> None:
        result = self.expander.expand("quiet beach hotel", n_queries=2)
        synonym_variant = result[1]
        assert "peaceful" in synonym_variant

    def test_holiday_substituted(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=2)
        synonym_variant = result[1]
        assert "vacation" in synonym_variant or "coastal" in synonym_variant

    def test_no_known_synonyms_adds_accommodation(self) -> None:
        result = self.expander.expand("train station near city", n_queries=2)
        synonym_variant = result[1]
        assert "accommodation" in synonym_variant

    def test_synonym_variant_differs_from_original(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=2)
        assert result[0] != result[1]


class TestLocalQueryExpanderContextVariant:
    def setup_method(self) -> None:
        self.expander = LocalQueryExpander()

    def test_context_variant_is_longer_than_original(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=3)
        assert len(result[2]) > len(result[0])

    def test_context_variant_starts_with_original(self) -> None:
        query = "family resort"
        result = self.expander.expand(query, n_queries=3)
        assert result[2].startswith(query)

    def test_beach_context_added(self) -> None:
        result = self.expander.expand("beach resort", n_queries=3)
        assert "beachfront" in result[2] or "swimming pool" in result[2]

    def test_family_context_added(self) -> None:
        # Use a query without "beach" so "family" keyword is matched first
        result = self.expander.expand("family resort", n_queries=3)
        assert "kids club" in result[2] or "children activities" in result[2]

    def test_spa_context_added(self) -> None:
        result = self.expander.expand("spa hotel", n_queries=3)
        assert "wellness" in result[2] or "relaxation" in result[2]

    def test_quiet_context_added(self) -> None:
        result = self.expander.expand("quiet resort", n_queries=3)
        assert "peaceful" in result[2] or "nightlife" in result[2]

    def test_unknown_keyword_gets_fallback_context(self) -> None:
        result = self.expander.expand("train station hotel", n_queries=3)
        assert "hotel resort accommodation" in result[2]

    def test_context_variant_differs_from_synonym_variant(self) -> None:
        result = self.expander.expand("beach holiday", n_queries=3)
        assert result[1] != result[2]


class TestLocalQueryExpanderEdgeCases:
    def setup_method(self) -> None:
        self.expander = LocalQueryExpander()

    def test_empty_query_returns_list_of_empties(self) -> None:
        result = self.expander.expand("", n_queries=3)
        assert len(result) == 3
        assert all(q == "" for q in result)

    def test_whitespace_query_returns_list_of_whitespace(self) -> None:
        result = self.expander.expand("   ", n_queries=3)
        assert len(result) == 3
        assert all(q == "   " for q in result)

    def test_extra_slots_filled_with_original(self) -> None:
        query = "beach holiday"
        result = self.expander.expand(query, n_queries=5)
        assert result[0] == query
        assert result[3] == query
        assert result[4] == query

    def test_three_variants_all_distinct_for_known_query(self) -> None:
        result = self.expander.expand("beach family holiday", n_queries=3)
        assert result[0] != result[1] or result[0] != result[2]
