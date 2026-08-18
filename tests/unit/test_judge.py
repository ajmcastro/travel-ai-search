"""Unit tests for the LLM-as-judge evaluation layer (Milestone 16).

Tests cover:
- JudgeVerdict validation
- EchoJudgeProvider behaviour
- JudgeProvider Protocol compliance (structural typing)
- Prompt construction and response parsing
- LLMEvaluator with a stub judge and stub search function
- Spearman ρ and Kendall τ correlation metrics
- Generator-effect gap calculation
- JudgeReport aggregation
- HumanQuery loading
- API schemas (JudgeRequest, JudgeResponse)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from travel_ai_search.evaluation.judge import (
    EchoJudgeProvider,
    JudgeProvider,
    JudgeVerdict,
    build_judge_prompt,
    parse_judge_response,
)
from travel_ai_search.evaluation.judge_evaluator import (
    HumanQuery,
    JudgeHit,
    JudgeQueryResult,
    JudgeReport,
    LLMEvaluator,
    generator_effect_gap,
    kendall_tau,
    load_human_queries,
    spearman_rho,
)
from travel_ai_search.retrieval.types import Hit

# ── JudgeVerdict ─────────────────────────────────────────────────────────────


class TestJudgeVerdict:
    def test_valid_grades_accepted(self) -> None:
        for grade in (0, 1, 2, 3):
            v = JudgeVerdict(score=grade, rationale="ok")
            assert v.score == grade

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="0–3"):
            JudgeVerdict(score=4, rationale="bad")

    def test_negative_score_raises(self) -> None:
        with pytest.raises(ValueError, match="0–3"):
            JudgeVerdict(score=-1, rationale="bad")

    def test_verdict_is_frozen(self) -> None:
        v = JudgeVerdict(score=2, rationale="test")
        with pytest.raises(Exception):
            v.score = 3  # type: ignore[misc]


# ── EchoJudgeProvider ─────────────────────────────────────────────────────────


class TestEchoJudgeProvider:
    def test_returns_fixed_score(self) -> None:
        provider = EchoJudgeProvider(fixed_score=2)
        verdict = provider.judge("beach holiday", "Grand Hotel", "A lovely beach hotel.")
        assert verdict.score == 2

    def test_custom_fixed_score(self) -> None:
        provider = EchoJudgeProvider(fixed_score=3)
        verdict = provider.judge("any query", "Any Hotel", "Description.")
        assert verdict.score == 3

    def test_score_zero_accepted(self) -> None:
        provider = EchoJudgeProvider(fixed_score=0)
        verdict = provider.judge("q", "h", "d")
        assert verdict.score == 0

    def test_invalid_fixed_score_raises(self) -> None:
        with pytest.raises(ValueError):
            EchoJudgeProvider(fixed_score=4)

    def test_model_id_is_echo(self) -> None:
        assert EchoJudgeProvider.model_id == "echo"

    def test_rationale_is_non_empty(self) -> None:
        verdict = EchoJudgeProvider().judge("q", "h", "d")
        assert verdict.rationale


# ── JudgeProvider Protocol compliance ────────────────────────────────────────


class TestJudgeProviderProtocol:
    def test_echo_satisfies_protocol(self) -> None:
        provider = EchoJudgeProvider()
        assert isinstance(provider, JudgeProvider)

    def test_object_without_model_id_fails(self) -> None:
        class BadJudge:
            def judge(self, q: str, h: str, d: str) -> JudgeVerdict:
                return JudgeVerdict(score=1, rationale="x")

        assert not isinstance(BadJudge(), JudgeProvider)

    def test_object_without_judge_method_fails(self) -> None:
        class BadJudge:
            model_id: str = "bad"

        assert not isinstance(BadJudge(), JudgeProvider)


# ── Prompt construction ───────────────────────────────────────────────────────


class TestBuildJudgePrompt:
    def test_contains_query(self) -> None:
        prompt = build_judge_prompt("family beach", "Sunny Resort", "A beach hotel.")
        assert "family beach" in prompt

    def test_contains_hotel_name(self) -> None:
        prompt = build_judge_prompt("q", "Sunny Resort", "desc")
        assert "Sunny Resort" in prompt

    def test_contains_hotel_description(self) -> None:
        prompt = build_judge_prompt("q", "h", "A wonderful coastal property.")
        assert "A wonderful coastal property." in prompt

    def test_does_not_mention_synthetic(self) -> None:
        prompt = build_judge_prompt("q", "h", "desc")
        assert "synthetic" not in prompt.lower()

    def test_scale_explained(self) -> None:
        prompt = build_judge_prompt("q", "h", "desc")
        # All four grades should be explained
        for grade in ("0", "1", "2", "3"):
            assert grade in prompt

    def test_format_instruction_present(self) -> None:
        prompt = build_judge_prompt("q", "h", "desc")
        assert "SCORE:" in prompt
        assert "RATIONALE:" in prompt


# ── Response parsing ──────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_strict_format_score_3(self) -> None:
        v = parse_judge_response("SCORE: 3 | RATIONALE: Perfect match.")
        assert v.score == 3
        assert "Perfect match" in v.rationale

    def test_strict_format_score_0(self) -> None:
        v = parse_judge_response("SCORE: 0 | RATIONALE: Completely irrelevant.")
        assert v.score == 0

    def test_case_insensitive(self) -> None:
        v = parse_judge_response("score: 2 | rationale: Good match.")
        assert v.score == 2

    def test_loose_fallback_digit(self) -> None:
        v = parse_judge_response("I would give this a 2 out of 3.")
        assert v.score == 2

    def test_unparseable_returns_fallback(self) -> None:
        v = parse_judge_response("I cannot decide.", fallback_score=1)
        assert v.score == 1

    def test_empty_response_returns_fallback(self) -> None:
        v = parse_judge_response("", fallback_score=0)
        assert v.score == 0

    def test_rationale_truncated_on_loose_match(self) -> None:
        long = "x" * 500 + " score is 1"
        v = parse_judge_response(long)
        # score should be found; rationale should not exceed 200 chars
        assert len(v.rationale) <= 200


# ── LLMEvaluator ─────────────────────────────────────────────────────────────


def _make_hit(doc_id: str, name: str, desc: str = "A hotel.") -> Hit:
    return Hit(id=doc_id, score=1.0, source={"hotel_name": name, "hotel_description": desc})


def _make_golden_dataset(
    queries: list[dict],
) -> object:
    """Build a minimal GoldenDataset-like object for testing."""
    from travel_ai_search.evaluation.dataset import GoldenDataset, GoldenQuery, RelevanceJudgment

    golden_queries = []
    for q in queries:
        judgments = tuple(
            RelevanceJudgment(doc_id=j["doc_id"], grade=j["grade"]) for j in q.get("judgments", [])
        )
        golden_queries.append(
            GoldenQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                query_class=q.get("query_class", "test"),
                judgments=judgments,
                filters=q.get("filters", {}),
            )
        )
    return GoldenDataset(queries=tuple(golden_queries))


class TestLLMEvaluator:
    def _make_search_fn(self, hits: list[Hit]):
        def fn(query_text: str, top_k: int, filters: dict) -> list[Hit]:
            return hits[:top_k]

        return fn

    def test_evaluate_generated_returns_report(self) -> None:
        judge = EchoJudgeProvider(fixed_score=2)
        evaluator = LLMEvaluator(judge)
        dataset = _make_golden_dataset(
            [
                {
                    "query_id": "q1",
                    "query_text": "beach holiday",
                    "query_class": "family",
                    "judgments": [{"doc_id": "h1", "grade": 3}],
                }
            ]
        )
        hits = [_make_hit("h1", "Beach Hotel")]
        report = evaluator.evaluate_generated(
            self._make_search_fn(hits), dataset, strategy="rrf", k=5
        )
        assert report.strategy == "rrf"
        assert report.slice_name == "generated"
        assert report.n_queries == 1
        assert report.mean_judge_score == 2.0

    def test_evaluate_generated_stores_golden_grade(self) -> None:
        judge = EchoJudgeProvider(fixed_score=2)
        evaluator = LLMEvaluator(judge)
        dataset = _make_golden_dataset(
            [
                {
                    "query_id": "q1",
                    "query_text": "spa retreat",
                    "query_class": "luxury",
                    "judgments": [{"doc_id": "h1", "grade": 3}, {"doc_id": "h2", "grade": 1}],
                }
            ]
        )
        hits = [_make_hit("h1", "Luxury Spa"), _make_hit("h2", "Budget Inn")]
        report = evaluator.evaluate_generated(
            self._make_search_fn(hits), dataset, strategy="bm25", k=5
        )
        h1_hit = next(h for h in report.per_query[0].hits if h.doc_id == "h1")
        assert h1_hit.golden_grade == 3

    def test_evaluate_generated_skips_no_relevant(self) -> None:
        judge = EchoJudgeProvider(fixed_score=2)
        evaluator = LLMEvaluator(judge)
        # Query with no relevant judgments
        dataset = _make_golden_dataset(
            [
                {
                    "query_id": "q1",
                    "query_text": "no match",
                    "query_class": "test",
                    "judgments": [],
                }
            ]
        )
        report = evaluator.evaluate_generated(
            self._make_search_fn([]), dataset, strategy="rrf", k=5
        )
        assert report.n_queries == 0

    def test_evaluate_human_returns_report(self) -> None:
        judge = EchoJudgeProvider(fixed_score=3)
        evaluator = LLMEvaluator(judge)
        queries = [HumanQuery(query_id="hq1", query_text="beach holiday", query_class="family")]
        hits = [_make_hit("h1", "Beach Resort")]
        report = evaluator.evaluate_human(
            self._make_search_fn(hits), queries, strategy="vector", k=5
        )
        assert report.strategy == "vector"
        assert report.slice_name == "human"
        assert report.n_queries == 1
        assert report.mean_judge_score == 3.0

    def test_evaluate_human_agreement_rate_is_none(self) -> None:
        judge = EchoJudgeProvider(fixed_score=2)
        evaluator = LLMEvaluator(judge)
        queries = [HumanQuery(query_id="hq1", query_text="quiet retreat", query_class="quiet")]
        hits = [_make_hit("h1", "Quiet Hotel")]
        report = evaluator.evaluate_human(self._make_search_fn(hits), queries, strategy="bm25", k=5)
        # No golden grades on human queries → agreement rate is None
        assert report.agreement_rate is None

    def test_agreement_rate_computed_when_golden_grades_present(self) -> None:
        judge = EchoJudgeProvider(fixed_score=2)  # score=2, so grade ≥ 2 agreement
        evaluator = LLMEvaluator(judge)
        dataset = _make_golden_dataset(
            [
                {
                    "query_id": "q1",
                    "query_text": "luxury spa",
                    "query_class": "luxury",
                    # h1: golden=2 → agree (judge=2). h2: golden=1 → comparable, no agree.
                    "judgments": [{"doc_id": "h1", "grade": 2}, {"doc_id": "h2", "grade": 1}],
                }
            ]
        )
        hits = [_make_hit("h1", "Spa Hotel"), _make_hit("h2", "Budget Inn")]
        report = evaluator.evaluate_generated(
            self._make_search_fn(hits), dataset, strategy="rrf", k=5
        )
        # comparable_pairs: both h1 (grade=2>0) and h2 (grade=1>0) are comparable → 2 total.
        # agreement_pairs: only h1 (golden=2 ≥2 AND judge=2 ≥2) → 1 total.
        # agreement_rate = 1/2 = 0.5
        agr = report.agreement_rate
        assert agr is not None
        assert math.isclose(agr, 0.5)


# ── JudgeReport serialisation ─────────────────────────────────────────────────


class TestJudgeReport:
    def _make_report(self, strategy: str = "rrf", mean: float = 2.0) -> JudgeReport:
        report = JudgeReport(
            strategy=strategy,
            judge_model_id="echo",
            k=5,
            slice_name="generated",
        )
        verdict = JudgeVerdict(score=int(mean), rationale="test")
        report.per_query.append(
            JudgeQueryResult(
                query_id="q1",
                query_text="beach",
                query_class="family",
                source="generated",
                hits=[JudgeHit(doc_id="h1", hotel_name="H", rank=1, verdict=verdict)],
            )
        )
        return report

    def test_n_queries(self) -> None:
        assert self._make_report().n_queries == 1

    def test_mean_judge_score(self) -> None:
        report = self._make_report(mean=3.0)
        assert report.mean_judge_score == 3.0

    def test_to_dict_keys_present(self) -> None:
        d = self._make_report().to_dict()
        assert "strategy" in d
        assert "judge_model_id" in d
        assert "mean_judge_score" in d
        assert "per_query" in d


# ── Correlation metrics ───────────────────────────────────────────────────────


class TestSpearmanRho:
    def test_identical_ranks_is_one(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        rho = spearman_rho(xs, xs)
        assert rho is not None
        assert math.isclose(rho, 1.0, abs_tol=1e-9)

    def test_reversed_ranks_is_minus_one(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [4.0, 3.0, 2.0, 1.0]
        rho = spearman_rho(xs, ys)
        assert rho is not None
        assert math.isclose(rho, -1.0, abs_tol=1e-9)

    def test_unequal_lengths_returns_none(self) -> None:
        assert spearman_rho([1.0, 2.0], [1.0]) is None

    def test_single_element_returns_none(self) -> None:
        assert spearman_rho([1.0], [1.0]) is None

    def test_ties_handled(self) -> None:
        xs = [1.0, 1.0, 3.0]
        ys = [2.0, 2.0, 1.0]
        rho = spearman_rho(xs, ys)
        # Both have ties; result should be finite and in [-1, 1] up to float precision
        assert rho is not None
        assert -1.0 - 1e-9 <= rho <= 1.0 + 1e-9

    def test_all_tied_returns_none(self) -> None:
        xs = [2.0, 2.0, 2.0]
        ys = [1.0, 2.0, 3.0]
        # xs has zero variance → Pearson is None
        assert spearman_rho(xs, ys) is None


class TestKendallTau:
    def test_identical_is_one(self) -> None:
        xs = [1.0, 2.0, 3.0]
        tau = kendall_tau(xs, xs)
        assert tau is not None
        assert math.isclose(tau, 1.0, abs_tol=1e-9)

    def test_reversed_is_minus_one(self) -> None:
        xs = [1.0, 2.0, 3.0]
        ys = [3.0, 2.0, 1.0]
        tau = kendall_tau(xs, ys)
        assert tau is not None
        assert math.isclose(tau, -1.0, abs_tol=1e-9)

    def test_single_element_returns_none(self) -> None:
        assert kendall_tau([1.0], [1.0]) is None

    def test_unequal_lengths_returns_none(self) -> None:
        assert kendall_tau([1.0, 2.0], [1.0]) is None


# ── Generator-effect gap ──────────────────────────────────────────────────────


class TestGeneratorEffectGap:
    def _report(self, strategy: str, score: float, slice_name: str) -> JudgeReport:
        r = JudgeReport(strategy=strategy, judge_model_id="echo", k=5, slice_name=slice_name)
        # Inject a synthetic mean by adding one query with the desired mean
        # mean_score = sum(hit.verdict.score)/len → one hit at 'score' level
        r.per_query.append(
            JudgeQueryResult(
                query_id="q",
                query_text="q",
                query_class="test",
                source=slice_name,
                hits=[
                    JudgeHit(
                        doc_id="h",
                        hotel_name="H",
                        rank=1,
                        verdict=JudgeVerdict(score=int(round(score)), rationale="x"),
                    )
                ],
            )
        )
        return r

    def test_returns_error_when_less_than_two_strategies(self) -> None:
        gen = [self._report("rrf", 2.0, "generated")]
        hum = [self._report("rrf", 1.8, "human")]
        gap = generator_effect_gap(gen, hum)
        assert "error" in gap

    def test_returns_spearman_for_two_strategies(self) -> None:
        gen = [self._report("bm25", 1.5, "generated"), self._report("rrf", 2.5, "generated")]
        hum = [self._report("bm25", 1.4, "human"), self._report("rrf", 2.3, "human")]
        gap = generator_effect_gap(gen, hum)
        assert "spearman_rho" in gap
        assert "kendall_tau" in gap

    def test_identical_rankings_gives_rho_one(self) -> None:
        gen = [self._report("bm25", 1.0, "generated"), self._report("rrf", 2.0, "generated")]
        hum = [self._report("bm25", 1.0, "human"), self._report("rrf", 2.0, "human")]
        gap = generator_effect_gap(gen, hum)
        # Both slices agree perfectly → ρ should be 1.0
        rho = gap.get("spearman_rho")
        assert rho is not None
        assert math.isclose(rho, 1.0, abs_tol=1e-6)

    def test_gap_interpretation_present(self) -> None:
        gen = [self._report("bm25", 1.5, "generated"), self._report("rrf", 2.5, "generated")]
        hum = [self._report("bm25", 2.5, "human"), self._report("rrf", 1.5, "human")]
        gap = generator_effect_gap(gen, hum)
        assert "gap_interpretation" in gap
        assert isinstance(gap["gap_interpretation"], str)


# ── HumanQuery loading ────────────────────────────────────────────────────────


class TestLoadHumanQueries:
    def test_loads_valid_jsonl(self) -> None:
        lines = [
            json.dumps(
                {
                    "query_id": "hq1",
                    "query_text": "beach holiday",
                    "query_class": "family",
                    "source": "human",
                }
            ),
            json.dumps(
                {
                    "query_id": "hq2",
                    "query_text": "luxury spa",
                    "query_class": "luxury",
                    "source": "human",
                }
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(lines) + "\n")
            tmp_path = Path(f.name)
        try:
            queries = load_human_queries(tmp_path)
            assert len(queries) == 2
            assert queries[0].query_id == "hq1"
            assert queries[1].query_class == "luxury"
        finally:
            tmp_path.unlink()

    def test_skips_blank_lines(self) -> None:
        content = json.dumps({"query_id": "q1", "query_text": "t", "query_class": "c"}) + "\n\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = Path(f.name)
        try:
            queries = load_human_queries(tmp_path)
            assert len(queries) == 1
        finally:
            tmp_path.unlink()
