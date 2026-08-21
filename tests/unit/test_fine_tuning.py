"""Unit tests for fine_tuning.pairs — Milestone 19."""

from __future__ import annotations

import pytest

from travel_ai_search.fine_tuning.pairs import (
    TrainingPair,
    build_training_pairs,
    select_hard_negatives,
)

# ── TrainingPair ──────────────────────────────────────────────────────────────


class TestTrainingPair:
    def test_fields_are_accessible(self) -> None:
        pair = TrainingPair(query="q", positive="p", negative="n")
        assert pair.query == "q"
        assert pair.positive == "p"
        assert pair.negative == "n"

    def test_is_frozen(self) -> None:
        pair = TrainingPair(query="q", positive="p", negative="n")
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            pair.query = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TrainingPair(query="q", positive="p", negative="n")
        b = TrainingPair(query="q", positive="p", negative="n")
        assert a == b

    def test_inequality_on_positive(self) -> None:
        a = TrainingPair(query="q", positive="p1", negative="n")
        b = TrainingPair(query="q", positive="p2", negative="n")
        assert a != b


# ── select_hard_negatives ─────────────────────────────────────────────────────


class TestSelectHardNegatives:
    def test_filters_out_known_relevant(self) -> None:
        bm25_ids = ["h1", "h2", "h3", "h4"]
        result = select_hard_negatives(bm25_ids, {"h1", "h3"})
        assert result == ["h2", "h4"]

    def test_empty_known_returns_all_hits(self) -> None:
        bm25_ids = ["h1", "h2"]
        result = select_hard_negatives(bm25_ids, set())
        assert result == ["h1", "h2"]

    def test_all_known_returns_empty(self) -> None:
        result = select_hard_negatives(["h1", "h2"], {"h1", "h2"})
        assert result == []

    def test_empty_hits_returns_empty(self) -> None:
        result = select_hard_negatives([], {"h1"})
        assert result == []

    def test_order_is_preserved(self) -> None:
        bm25_ids = ["h5", "h3", "h1", "h2", "h4"]
        result = select_hard_negatives(bm25_ids, {"h1", "h3"})
        assert result == ["h5", "h2", "h4"]

    def test_does_not_modify_input(self) -> None:
        bm25_ids = ["h1", "h2", "h3"]
        original = list(bm25_ids)
        select_hard_negatives(bm25_ids, {"h2"})
        assert bm25_ids == original

    def test_returns_list_type(self) -> None:
        result = select_hard_negatives(["h1"], {"h2"})
        assert isinstance(result, list)


# ── build_training_pairs ──────────────────────────────────────────────────────


class TestBuildTrainingPairs:
    def test_empty_positives_returns_empty(self) -> None:
        result = build_training_pairs("q", [], ["neg1"], max_pairs_per_query=5)
        assert result == []

    def test_empty_negatives_returns_empty(self) -> None:
        result = build_training_pairs("q", ["pos1"], [], max_pairs_per_query=5)
        assert result == []

    def test_empty_both_returns_empty(self) -> None:
        result = build_training_pairs("q", [], [], max_pairs_per_query=5)
        assert result == []

    def test_returns_training_pair_instances(self) -> None:
        result = build_training_pairs("q", ["p1"], ["n1"], max_pairs_per_query=5)
        assert all(isinstance(r, TrainingPair) for r in result)

    def test_single_pair_fields(self) -> None:
        result = build_training_pairs(
            "luxury beach holiday",
            ["beach resort Tenerife"],
            ["budget motel Paris"],
            max_pairs_per_query=5,
        )
        assert len(result) == 1
        pair = result[0]
        assert pair.query == "luxury beach holiday"
        assert pair.positive == "beach resort Tenerife"
        assert pair.negative == "budget motel Paris"

    def test_max_pairs_limits_output(self) -> None:
        positives = [f"pos{i}" for i in range(10)]
        negatives = [f"neg{i}" for i in range(10)]
        result = build_training_pairs("q", positives, negatives, max_pairs_per_query=3)
        assert len(result) == 3

    def test_output_bounded_by_positives(self) -> None:
        result = build_training_pairs("q", ["p1"], ["n1", "n2", "n3"], max_pairs_per_query=5)
        assert len(result) == 1

    def test_output_bounded_by_max_pairs(self) -> None:
        positives = ["p1", "p2", "p3", "p4", "p5"]
        result = build_training_pairs("q", positives, ["n1", "n2"], max_pairs_per_query=2)
        assert len(result) == 2

    def test_query_text_preserved_in_all_pairs(self) -> None:
        positives = ["p1", "p2"]
        negatives = ["n1", "n2"]
        result = build_training_pairs(
            "my search query", positives, negatives, max_pairs_per_query=5
        )
        assert all(r.query == "my search query" for r in result)

    def test_negative_cycles_round_robin_single(self) -> None:
        positives = ["p1", "p2", "p3"]
        result = build_training_pairs("q", positives, ["n1"], max_pairs_per_query=3)
        assert len(result) == 3
        assert all(r.negative == "n1" for r in result)

    def test_negative_cycles_round_robin_two(self) -> None:
        positives = ["p1", "p2", "p3", "p4"]
        negatives = ["n1", "n2"]
        result = build_training_pairs("q", positives, negatives, max_pairs_per_query=4)
        assert [r.negative for r in result] == ["n1", "n2", "n1", "n2"]

    def test_positives_assigned_in_order(self) -> None:
        positives = ["beach resort", "spa hotel", "city break"]
        result = build_training_pairs(
            "holiday", positives, ["budget hostel"], max_pairs_per_query=3
        )
        assert [r.positive for r in result] == ["beach resort", "spa hotel", "city break"]

    def test_all_positives_distinct(self) -> None:
        positives = ["spa hotel", "beach club", "mountain lodge"]
        result = build_training_pairs(
            "luxury getaway", positives, ["transit motel"], max_pairs_per_query=5
        )
        positives_used = [r.positive for r in result]
        assert len(set(positives_used)) == len(positives_used)

    def test_max_pairs_zero_returns_empty(self) -> None:
        result = build_training_pairs("q", ["p1"], ["n1"], max_pairs_per_query=0)
        assert result == []
