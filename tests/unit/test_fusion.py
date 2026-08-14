"""Unit tests for retrieval.fusion — pure functions, no I/O required."""

from __future__ import annotations

from travel_ai_search.retrieval.fusion import (  # noqa: PLC2701
    _RRF_K_DEFAULT,
    _normalize_scores,
    build_filter_clauses,
    fuse_results,
    rrf_fuse,
)
from travel_ai_search.retrieval.types import Hit

# ── Helpers ───────────────────────────────────────────────────────────────────


def _hit(doc_id: str, score: float, **source_fields: object) -> Hit:
    return Hit(id=doc_id, score=score, source=dict(source_fields))


# ── build_filter_clauses ──────────────────────────────────────────────────────


def test_build_filter_clauses_empty_when_no_args() -> None:
    assert build_filter_clauses() == []


def test_build_filter_clauses_country() -> None:
    clauses = build_filter_clauses(country="Spain")
    assert clauses == [{"term": {"country": "Spain"}}]


def test_build_filter_clauses_destination() -> None:
    clauses = build_filter_clauses(destination="Benidorm")
    assert clauses == [{"term": {"destination": "Benidorm"}}]


def test_build_filter_clauses_family_friendly_true() -> None:
    clauses = build_filter_clauses(family_friendly=True)
    assert clauses == [{"term": {"family_friendly": True}}]


def test_build_filter_clauses_adults_only_false() -> None:
    clauses = build_filter_clauses(adults_only=False)
    assert clauses == [{"term": {"adults_only": False}}]


def test_build_filter_clauses_min_star_rating() -> None:
    clauses = build_filter_clauses(min_star_rating=4)
    assert clauses == [{"range": {"star_rating": {"gte": 4}}}]


def test_build_filter_clauses_max_price() -> None:
    clauses = build_filter_clauses(max_price=1000.0)
    assert clauses == [{"range": {"price_per_person_gbp": {"lte": 1000.0}}}]


def test_build_filter_clauses_month() -> None:
    clauses = build_filter_clauses(month="July")
    assert clauses == [{"term": {"available_months": "July"}}]


def test_build_filter_clauses_airport() -> None:
    clauses = build_filter_clauses(airport="MAN")
    assert clauses == [{"term": {"available_departure_airports": "MAN"}}]


def test_build_filter_clauses_multiple_filters() -> None:
    clauses = build_filter_clauses(country="Spain", family_friendly=True, max_price=800.0)
    assert {"term": {"country": "Spain"}} in clauses
    assert {"term": {"family_friendly": True}} in clauses
    assert {"range": {"price_per_person_gbp": {"lte": 800.0}}} in clauses
    assert len(clauses) == 3


# ── _normalize_scores ─────────────────────────────────────────────────────────


def test_normalize_scores_empty_list() -> None:
    assert _normalize_scores([]) == {}


def test_normalize_scores_single_hit_returns_one() -> None:
    hits = [_hit("a", 5.0)]
    result = _normalize_scores(hits)
    assert result["a"][0] == 1.0


def test_normalize_scores_all_same_score_returns_one() -> None:
    hits = [_hit("a", 3.0), _hit("b", 3.0), _hit("c", 3.0)]
    result = _normalize_scores(hits)
    for doc_id in ("a", "b", "c"):
        assert result[doc_id][0] == 1.0


def test_normalize_scores_maps_range_to_zero_one() -> None:
    hits = [_hit("low", 0.0), _hit("mid", 5.0), _hit("high", 10.0)]
    result = _normalize_scores(hits)
    assert abs(result["low"][0] - 0.0) < 1e-9
    assert abs(result["mid"][0] - 0.5) < 1e-9
    assert abs(result["high"][0] - 1.0) < 1e-9


def test_normalize_scores_preserves_source() -> None:
    hits = [_hit("a", 2.0, hotel_name="Hotel A"), _hit("b", 1.0, hotel_name="Hotel B")]
    result = _normalize_scores(hits)
    assert result["a"][1] == {"hotel_name": "Hotel A"}
    assert result["b"][1] == {"hotel_name": "Hotel B"}


def test_normalize_scores_returns_all_ids() -> None:
    hits = [_hit("x", 1.0), _hit("y", 2.0), _hit("z", 3.0)]
    result = _normalize_scores(hits)
    assert set(result.keys()) == {"x", "y", "z"}


# ── fuse_results ──────────────────────────────────────────────────────────────


def test_fuse_results_both_empty() -> None:
    result = fuse_results([], [], lexical_weight=0.5, vector_weight=0.5, top_k=10)
    assert result == []


def test_fuse_results_lexical_only() -> None:
    lex = [_hit("a", 10.0, hotel_name="A"), _hit("b", 5.0, hotel_name="B")]
    result = fuse_results(lex, [], lexical_weight=0.5, vector_weight=0.5, top_k=10)
    ids = [h.id for h in result]
    assert "a" in ids
    assert "b" in ids


def test_fuse_results_vector_only() -> None:
    vec = [_hit("c", 0.9, hotel_name="C"), _hit("d", 0.6, hotel_name="D")]
    result = fuse_results([], vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    ids = [h.id for h in result]
    assert "c" in ids
    assert "d" in ids


def test_fuse_results_disjoint_sets_top_doc_from_better_retriever() -> None:
    # With equal weights, documents found by only one retriever get 0.5 * 1.0 = 0.5
    # max possible. Disjoint → all tied at 0.5; order is arbitrary but both appear.
    lex = [_hit("lex_top", 10.0)]
    vec = [_hit("vec_top", 0.9)]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    ids = {h.id for h in result}
    assert ids == {"lex_top", "vec_top"}


def test_fuse_results_overlapping_doc_scores_highest() -> None:
    # "shared" appears in both lists ranked #1; it should outscore disjoint docs.
    lex = [_hit("shared", 10.0), _hit("lex_only", 8.0)]
    vec = [_hit("shared", 0.9), _hit("vec_only", 0.7)]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    assert result[0].id == "shared"


def test_fuse_results_top_k_limits_output() -> None:
    lex = [_hit(f"l{i}", float(10 - i)) for i in range(10)]
    vec = [_hit(f"v{i}", float(1.0 - i * 0.05)) for i in range(10)]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=5)
    assert len(result) == 5


def test_fuse_results_sorted_descending() -> None:
    lex = [_hit("a", 10.0), _hit("b", 1.0)]
    vec = [_hit("a", 0.9), _hit("c", 0.5)]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    scores = [h.score for h in result]
    assert scores == sorted(scores, reverse=True)


def test_fuse_results_scores_in_zero_one_range() -> None:
    lex = [_hit("a", 100.0), _hit("b", 50.0)]
    vec = [_hit("a", 0.9), _hit("b", 0.3)]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    for hit in result:
        assert 0.0 <= hit.score <= 1.0


def test_fuse_results_lexical_weight_one_ignores_vector() -> None:
    # With vector_weight=0, only lexical scores determine ranking.
    lex = [_hit("lex_best", 10.0), _hit("lex_worst", 1.0)]
    vec = [_hit("lex_worst", 0.99)]  # lex_worst wins in vector, should be ignored
    result = fuse_results(lex, vec, lexical_weight=1.0, vector_weight=0.0, top_k=10)
    assert result[0].id == "lex_best"


def test_fuse_results_vector_weight_one_ignores_lexical() -> None:
    lex = [_hit("lex_best", 10.0)]  # wins in lexical, should be ignored
    vec = [_hit("vec_best", 0.99), _hit("lex_best", 0.01)]
    result = fuse_results(lex, vec, lexical_weight=0.0, vector_weight=1.0, top_k=10)
    assert result[0].id == "vec_best"


def test_fuse_results_source_preserved_from_lexical_hit() -> None:
    lex = [_hit("a", 5.0, hotel_name="From Lexical")]
    vec = [_hit("b", 0.5, hotel_name="From Vector")]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    sources = {h.id: h.source for h in result}
    assert sources["a"]["hotel_name"] == "From Lexical"
    assert sources["b"]["hotel_name"] == "From Vector"


def test_fuse_results_source_fallback_to_vector_when_lex_missing() -> None:
    # "vec_only" is in vector results only; its source should come from vec.
    lex = [_hit("lex_only", 5.0, hotel_name="Lex Hotel")]
    vec = [_hit("vec_only", 0.8, hotel_name="Vec Hotel")]
    result = fuse_results(lex, vec, lexical_weight=0.5, vector_weight=0.5, top_k=10)
    sources = {h.id: h.source for h in result}
    assert sources["vec_only"]["hotel_name"] == "Vec Hotel"


# ── rrf_fuse ──────────────────────────────────────────────────────────────────


def test_rrf_fuse_no_lists_returns_empty() -> None:
    result = rrf_fuse([], top_k=10)
    assert result == []


def test_rrf_fuse_single_empty_list_returns_empty() -> None:
    result = rrf_fuse([[]], top_k=10)
    assert result == []


def test_rrf_fuse_single_list_single_doc_correct_score() -> None:
    hits = [_hit("a", 5.0)]
    result = rrf_fuse([hits], top_k=10)
    assert len(result) == 1
    assert result[0].id == "a"
    # rank=1, k=60 → score = 1/(60+1)
    expected = 1.0 / (_RRF_K_DEFAULT + 1)
    assert abs(result[0].score - expected) < 1e-12


def test_rrf_fuse_single_list_preserves_relative_order() -> None:
    # RRF of a single list ranks by position, so rank 1 > rank 2.
    hits = [_hit("first", 10.0), _hit("second", 5.0), _hit("third", 1.0)]
    result = rrf_fuse([hits], top_k=10)
    assert [h.id for h in result] == ["first", "second", "third"]


def test_rrf_fuse_two_disjoint_lists_all_docs_present() -> None:
    lex = [_hit("lex_a", 10.0), _hit("lex_b", 8.0)]
    vec = [_hit("vec_a", 0.9), _hit("vec_b", 0.7)]
    result = rrf_fuse([lex, vec], top_k=10)
    ids = {h.id for h in result}
    assert ids == {"lex_a", "lex_b", "vec_a", "vec_b"}


def test_rrf_fuse_doc_in_both_lists_scores_higher_than_single_list() -> None:
    # "shared" appears at rank 1 in both lists; "solo" appears at rank 1 in one.
    lex = [_hit("shared", 10.0), _hit("lex_solo", 8.0)]
    vec = [_hit("shared", 0.9), _hit("vec_solo", 0.8)]
    result = rrf_fuse([lex, vec], top_k=10)
    scores = {h.id: h.score for h in result}
    # shared: 1/(60+1) + 1/(60+1) = 2/(60+1) ≈ 0.0328
    # lex_solo: 1/(60+2) ≈ 0.0161
    assert scores["shared"] > scores["lex_solo"]
    assert scores["shared"] > scores["vec_solo"]


def test_rrf_fuse_rank1_both_lists_score_formula() -> None:
    lex = [_hit("a", 10.0)]
    vec = [_hit("a", 0.9)]
    result = rrf_fuse([lex, vec], top_k=10)
    k = _RRF_K_DEFAULT
    expected = 1.0 / (k + 1) + 1.0 / (k + 1)
    assert abs(result[0].score - expected) < 1e-12


def test_rrf_fuse_top_k_limits_output() -> None:
    lex = [_hit(f"l{i}", float(10 - i)) for i in range(10)]
    vec = [_hit(f"v{i}", float(1.0 - i * 0.05)) for i in range(10)]
    result = rrf_fuse([lex, vec], top_k=5)
    assert len(result) == 5


def test_rrf_fuse_sorted_descending() -> None:
    lex = [_hit("a", 10.0), _hit("b", 8.0), _hit("c", 1.0)]
    vec = [_hit("b", 0.9), _hit("d", 0.5)]
    result = rrf_fuse([lex, vec], top_k=10)
    scores = [h.score for h in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_smaller_k_amplifies_top_rank() -> None:
    # With smaller k, rank-1 contributes relatively more vs rank-N.
    hits = [_hit("rank1", 5.0), _hit("rank2", 3.0)]
    result_small_k = rrf_fuse([hits], k=1, top_k=10)
    result_large_k = rrf_fuse([hits], k=100, top_k=10)
    # score(rank1, k=1) = 1/2;  score(rank2, k=1) = 1/3  → ratio = 1.5
    # score(rank1, k=100) = 1/101; score(rank2, k=100) = 1/102 → ratio ≈ 1.01
    s1_small = result_small_k[0].score
    s2_small = result_small_k[1].score
    s1_large = result_large_k[0].score
    s2_large = result_large_k[1].score
    ratio_small = s1_small / s2_small
    ratio_large = s1_large / s2_large
    assert ratio_small > ratio_large


def test_rrf_fuse_three_lists() -> None:
    a = [_hit("x", 1.0), _hit("y", 0.5)]
    b = [_hit("y", 0.9), _hit("z", 0.4)]
    c = [_hit("x", 0.8), _hit("z", 0.7)]
    result = rrf_fuse([a, b, c], top_k=10)
    ids = {h.id for h in result}
    assert ids == {"x", "y", "z"}
    # x appears in lists a (rank1) and c (rank1) → 2/(k+1)
    # y appears in lists a (rank2) and b (rank1) → 1/(k+2) + 1/(k+1)
    # z appears in lists b (rank2) and c (rank2) → 2/(k+2)
    scores = {h.id: h.score for h in result}
    k = _RRF_K_DEFAULT
    assert abs(scores["x"] - (2.0 / (k + 1))) < 1e-12
    assert abs(scores["y"] - (1.0 / (k + 2) + 1.0 / (k + 1))) < 1e-12
    assert abs(scores["z"] - (2.0 / (k + 2))) < 1e-12


def test_rrf_fuse_source_from_first_occurrence() -> None:
    # Source is captured from whichever ranked list contains the doc first.
    lex = [_hit("a", 5.0, hotel_name="From Lexical")]
    vec = [_hit("a", 0.9, hotel_name="From Vector")]
    result = rrf_fuse([lex, vec], top_k=10)
    assert result[0].source["hotel_name"] == "From Lexical"


def test_rrf_fuse_scores_positive() -> None:
    lex = [_hit("a", 10.0), _hit("b", 5.0)]
    vec = [_hit("a", 0.9), _hit("c", 0.3)]
    result = rrf_fuse([lex, vec], top_k=10)
    for hit in result:
        assert hit.score > 0.0
