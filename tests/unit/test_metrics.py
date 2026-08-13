"""Unit tests for IR evaluation metrics.

All tests use synthetic retrieved/relevant inputs so the expected output
can be computed by hand and verified independently of any search system.
"""

from __future__ import annotations

import math

import pytest

from travel_ai_search.evaluation.metrics import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ── precision_at_k ────────────────────────────────────────────────────────────


def test_precision_at_k_all_relevant() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c", "d", "e"}
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)


def test_precision_at_k_none_relevant() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"x", "y"}
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_precision_at_k_half_relevant() -> None:
    retrieved = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c"}
    # top-4: a, x, b, y  →  2 relevant / 4 = 0.5
    assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(0.5)


def test_precision_at_k_cutoff_before_list_end() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "d", "e"}
    # top-2: a, b → 0 relevant / 2 = 0.0
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], {"a"}, k=5) == pytest.approx(0.0)


def test_precision_at_k_k_larger_than_retrieved() -> None:
    # k=10 but only 3 results; still divide by k (TREC convention)
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(retrieved, relevant, k=10) == pytest.approx(3 / 10)


# ── recall_at_k ───────────────────────────────────────────────────────────────


def test_recall_at_k_all_found() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_none_found() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_recall_at_k_partial() -> None:
    retrieved = ["a", "x", "b", "y"]
    relevant = {"a", "b", "c"}
    # top-3: a, x, b → 2 found / 3 relevant = 2/3
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["a", "b"], set(), k=2) == pytest.approx(0.0)


# ── hit_rate_at_k ─────────────────────────────────────────────────────────────


def test_hit_rate_at_k_first_is_relevant() -> None:
    assert hit_rate_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)


def test_hit_rate_at_k_last_in_top_k_is_relevant() -> None:
    assert hit_rate_at_k(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1.0)


def test_hit_rate_at_k_relevant_outside_cutoff() -> None:
    # "a" is at rank 4 but k=3
    assert hit_rate_at_k(["x", "y", "z", "a"], {"a"}, k=3) == pytest.approx(0.0)


def test_hit_rate_at_k_no_relevant_at_all() -> None:
    assert hit_rate_at_k(["x", "y", "z"], {"a"}, k=3) == pytest.approx(0.0)


def test_hit_rate_at_k_empty_retrieved() -> None:
    assert hit_rate_at_k([], {"a"}, k=5) == pytest.approx(0.0)


# ── reciprocal_rank ───────────────────────────────────────────────────────────


def test_rr_first_result_relevant() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_rr_second_result_relevant() -> None:
    assert reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)


def test_rr_third_result_relevant() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_rr_no_relevant() -> None:
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == pytest.approx(0.0)


def test_rr_empty_retrieved() -> None:
    assert reciprocal_rank([], {"a"}) == pytest.approx(0.0)


# ── average_precision ─────────────────────────────────────────────────────────


def test_ap_all_relevant() -> None:
    # Retrieved: [a, b, c], all relevant
    # Precision at rank 1=1.0, 2=1.0, 3=1.0 → sum/3 = 1.0
    assert average_precision(["a", "b", "c"], {"a", "b", "c"}) == pytest.approx(1.0)


def test_ap_none_relevant() -> None:
    assert average_precision(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_ap_empty_relevant() -> None:
    assert average_precision(["a", "b"], set()) == pytest.approx(0.0)


def test_ap_interleaved() -> None:
    # Retrieved: [a, x, b, y, c], relevant={a, b, c}
    # rank 1: a relevant → P@1=1/1; hits=1
    # rank 3: b relevant → P@3=2/3; hits=2
    # rank 5: c relevant → P@5=3/5; hits=3
    # AP = (1 + 2/3 + 3/5) / 3
    expected = (1.0 + 2 / 3 + 3 / 5) / 3
    assert average_precision(["a", "x", "b", "y", "c"], {"a", "b", "c"}) == pytest.approx(
        expected, rel=1e-6
    )


def test_ap_single_relevant_at_rank_two() -> None:
    # AP = (1/2) / 1 = 0.5
    assert average_precision(["x", "a"], {"a"}) == pytest.approx(0.5)


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────


def test_ndcg_ideal_ranking() -> None:
    # Retrieved matches perfect ideal order → NDCG=1.0
    graded = {"a": 3, "b": 2, "c": 1}
    retrieved = ["a", "b", "c"]
    assert ndcg_at_k(retrieved, graded, k=3) == pytest.approx(1.0)


def test_ndcg_worst_ranking() -> None:
    # Grade-3 doc last, grade-1 first
    graded = {"a": 3, "b": 2, "c": 1}
    retrieved = ["c", "b", "a"]
    # DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3) + (2^3-1)/log2(4)
    #      = 1/1 + 3/1.585 + 7/2 = 1 + 1.893 + 3.5 = 6.393
    # IDCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    #       = 7/1 + 3/1.585 + 1/2 = 7 + 1.893 + 0.5 = 9.393
    dcg = 1 / math.log2(2) + 3 / math.log2(3) + 7 / math.log2(4)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(retrieved, graded, k=3) == pytest.approx(dcg / idcg, rel=1e-6)


def test_ndcg_no_relevant_retrieved() -> None:
    graded = {"a": 3, "b": 2}
    retrieved = ["x", "y", "z"]
    assert ndcg_at_k(retrieved, graded, k=3) == pytest.approx(0.0)


def test_ndcg_empty_graded() -> None:
    assert ndcg_at_k(["a", "b"], {}, k=2) == pytest.approx(0.0)


def test_ndcg_single_doc_at_rank_one() -> None:
    graded = {"a": 2}
    retrieved = ["a"]
    # DCG = (2^2-1)/log2(2) = 3/1 = 3; IDCG = 3 → NDCG = 1.0
    assert ndcg_at_k(retrieved, graded, k=5) == pytest.approx(1.0)


def test_ndcg_single_doc_at_rank_two() -> None:
    graded = {"a": 2}
    retrieved = ["x", "a"]
    # DCG = (2^2-1)/log2(3) = 3/log2(3)
    # IDCG = (2^2-1)/log2(2) = 3
    expected = (3 / math.log2(3)) / 3
    assert ndcg_at_k(retrieved, graded, k=5) == pytest.approx(expected, rel=1e-6)


def test_ndcg_cutoff_excludes_relevant_tail() -> None:
    # "a" is at rank 5 but k=3 — should contribute 0
    graded = {"a": 3}
    retrieved = ["x", "y", "z", "w", "a"]
    assert ndcg_at_k(retrieved, graded, k=3) == pytest.approx(0.0)


# ── mean_reciprocal_rank / mean_average_precision ─────────────────────────────


def test_mrr_basic() -> None:
    rr_scores = [1.0, 0.5, 1 / 3]
    expected = (1.0 + 0.5 + 1 / 3) / 3
    assert mean_reciprocal_rank(rr_scores) == pytest.approx(expected, rel=1e-6)


def test_mrr_empty() -> None:
    assert mean_reciprocal_rank([]) == pytest.approx(0.0)


def test_map_basic() -> None:
    ap_scores = [0.8, 0.6, 1.0]
    expected = (0.8 + 0.6 + 1.0) / 3
    assert mean_average_precision(ap_scores) == pytest.approx(expected, rel=1e-6)


def test_map_empty() -> None:
    assert mean_average_precision([]) == pytest.approx(0.0)
