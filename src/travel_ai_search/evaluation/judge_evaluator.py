"""LLM-as-judge evaluator — Milestone 16.

This module implements the evaluation workflow for LLM-based relevance scoring:

1. Run a retrieval strategy to get top-K hotels for each query.
2. For each (query, hotel) pair, ask a JudgeProvider to score relevance (0–3).
3. Aggregate into a JudgeReport: mean judge score, label agreement rate,
   and rank correlations between golden-label and judge-score strategy orderings.

Generator-effect measurement
-----------------------------
Run ``llm_evaluate`` on two slices — the generated golden queries and a set of
human-written queries.  Compare the resulting strategy rankings.  If vector
search ranks #1 on the generated slice but #2 on the human slice, the gap is
evidence of the generator effect (dense retrieval benefits from synthetic
paraphrase vocabulary matching the embedding model's training distribution).

Correlation metrics (implemented from first principles)
-------------------------------------------------------
Spearman ρ: rank correlation.  Range [–1, 1].  ρ = 1 means two rankings are
  identical; ρ = 0 means no association; ρ = –1 means reversed.
  Formula: ρ = 1 – 6Σd² / (n(n²–1))  where d = rank difference per item.

Kendall τ: concordance-based correlation.  More robust than Spearman when there
  are many ties.  τ = (concordant – discordant) / (n(n–1)/2).

Both are computed over the per-query mean judge scores, not over individual
hotel scores, to assess *strategy-level* agreement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from travel_ai_search.evaluation.judge import JudgeProvider, JudgeVerdict
from travel_ai_search.retrieval.types import Hit

# A retrieval function for the judge: returns ranked Hit objects so that the
# evaluator has access to hotel_name and hotel_description from hit.source.
# Signature: (query_text, top_k, filters) -> list[Hit]
JudgeFn = Callable[[str, int, dict[str, Any]], list[Hit]]


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class JudgeHit:
    """Judge verdict for one retrieved hotel within a query result."""

    doc_id: str
    hotel_name: str
    rank: int
    verdict: JudgeVerdict
    golden_grade: int = 0


@dataclass
class JudgeQueryResult:
    """Judge results for a single query."""

    query_id: str
    query_text: str
    query_class: str
    source: str  # "generated" or "human"
    hits: list[JudgeHit] = field(default_factory=list)
    took_ms: int = 0

    @property
    def mean_score(self) -> float:
        return sum(h.verdict.score for h in self.hits) / len(self.hits) if self.hits else 0.0

    @property
    def agreement_pairs(self) -> int:
        """(query, hotel) pairs where both golden_grade ≥ 2 and judge score ≥ 2."""
        return sum(1 for h in self.hits if h.golden_grade >= 2 and h.verdict.score >= 2)

    @property
    def comparable_pairs(self) -> int:
        """(query, hotel) pairs that have a positive golden grade (judgeable)."""
        return sum(1 for h in self.hits if h.golden_grade > 0)


@dataclass
class JudgeReport:
    """LLM-as-judge evaluation result for one strategy over one query slice."""

    strategy: str
    judge_model_id: str
    k: int
    slice_name: str
    per_query: list[JudgeQueryResult] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return len(self.per_query)

    @property
    def mean_judge_score(self) -> float:
        return _mean(q.mean_score for q in self.per_query)

    @property
    def agreement_rate(self) -> float | None:
        """Fraction of judged (query, hotel) pairs where both signals say ≥ 2.

        Returns None when no golden grades are available (human slice without
        pre-annotated labels).
        """
        total = sum(q.comparable_pairs for q in self.per_query)
        if total == 0:
            return None
        agreed = sum(q.agreement_pairs for q in self.per_query)
        return agreed / total

    def to_dict(self) -> dict[str, Any]:
        agr = self.agreement_rate
        return {
            "strategy": self.strategy,
            "judge_model_id": self.judge_model_id,
            "k": self.k,
            "slice": self.slice_name,
            "n_queries": self.n_queries,
            "mean_judge_score": round(self.mean_judge_score, 4),
            "agreement_rate": round(agr, 4) if agr is not None else None,
            "per_query": [
                {
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "query_class": q.query_class,
                    "source": q.source,
                    "mean_score": round(q.mean_score, 4),
                    "took_ms": q.took_ms,
                    "hits": [
                        {
                            "doc_id": h.doc_id,
                            "hotel_name": h.hotel_name,
                            "rank": h.rank,
                            "judge_score": h.verdict.score,
                            "rationale": h.verdict.rationale,
                            "golden_grade": h.golden_grade,
                        }
                        for h in q.hits
                    ],
                }
                for q in self.per_query
            ],
        }


# ── Simple human-query type (no pre-annotated judgments) ─────────────────────


@dataclass(frozen=True)
class HumanQuery:
    """A query written by a human who never saw the generation prompts.

    Human queries have no pre-annotated relevance judgments — the LLM judge
    provides the relevance signal at evaluation time.
    """

    query_id: str
    query_text: str
    query_class: str
    source: str = "human"


def load_human_queries(path: Any) -> list[HumanQuery]:
    """Load human-written queries from a JSON Lines file."""
    import json
    from pathlib import Path

    queries: list[HumanQuery] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        queries.append(
            HumanQuery(
                query_id=data["query_id"],
                query_text=data["query_text"],
                query_class=data["query_class"],
                source=data.get("source", "human"),
            )
        )
    return queries


# ── Core evaluator ────────────────────────────────────────────────────────────


class LLMEvaluator:
    """Runs LLM-as-judge scoring over a query set.

    Parameters
    ----------
    judge:
        A JudgeProvider implementation.  Must be from a different model family
        than the one that generated the synthetic corpus.
    """

    def __init__(self, judge: JudgeProvider) -> None:
        self._judge = judge

    def evaluate_generated(
        self,
        search_fn: JudgeFn,
        dataset: Any,  # GoldenDataset
        *,
        strategy: str,
        k: int = 10,
    ) -> JudgeReport:
        """Score the generated golden queries with the LLM judge.

        Golden grades from the dataset are stored alongside judge scores so that
        agreement_rate can be computed (how often both signals agree at grade ≥ 2).
        """
        import time

        report = JudgeReport(
            strategy=strategy,
            judge_model_id=self._judge.model_id,
            k=k,
            slice_name="generated",
        )
        for golden_query in dataset.queries:
            if not golden_query.has_relevant:
                continue
            t0 = time.perf_counter()
            hits = search_fn(golden_query.query_text, k, golden_query.filters)
            took_ms = int((time.perf_counter() - t0) * 1000)

            graded = golden_query.graded_relevance
            judge_hits = []
            for rank, hit in enumerate(hits, start=1):
                name = hit.source.get("hotel_name", hit.id)
                desc = hit.source.get("hotel_description", "")
                verdict = self._judge.judge(golden_query.query_text, name, desc)
                judge_hits.append(
                    JudgeHit(
                        doc_id=hit.id,
                        hotel_name=name,
                        rank=rank,
                        verdict=verdict,
                        golden_grade=graded.get(hit.id, 0),
                    )
                )
            report.per_query.append(
                JudgeQueryResult(
                    query_id=golden_query.query_id,
                    query_text=golden_query.query_text,
                    query_class=golden_query.query_class,
                    source="generated",
                    hits=judge_hits,
                    took_ms=took_ms,
                )
            )
        return report

    def evaluate_human(
        self,
        search_fn: JudgeFn,
        queries: list[HumanQuery],
        *,
        strategy: str,
        k: int = 10,
    ) -> JudgeReport:
        """Score human-written queries with the LLM judge.

        No golden grades are available for the human slice, so agreement_rate
        will be None in the resulting report.  Judge scores are the sole signal.
        """
        import time

        report = JudgeReport(
            strategy=strategy,
            judge_model_id=self._judge.model_id,
            k=k,
            slice_name="human",
        )
        for hq in queries:
            t0 = time.perf_counter()
            hits = search_fn(hq.query_text, k, {})
            took_ms = int((time.perf_counter() - t0) * 1000)

            judge_hits = []
            for rank, hit in enumerate(hits, start=1):
                name = hit.source.get("hotel_name", hit.id)
                desc = hit.source.get("hotel_description", "")
                verdict = self._judge.judge(hq.query_text, name, desc)
                judge_hits.append(
                    JudgeHit(
                        doc_id=hit.id,
                        hotel_name=name,
                        rank=rank,
                        verdict=verdict,
                        golden_grade=0,
                    )
                )
            report.per_query.append(
                JudgeQueryResult(
                    query_id=hq.query_id,
                    query_text=hq.query_text,
                    query_class=hq.query_class,
                    source="human",
                    hits=judge_hits,
                    took_ms=took_ms,
                )
            )
        return report


# ── Correlation metrics (from first principles) ───────────────────────────────


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation between two equally-sized lists of scores.

    Returns None when n < 2 (undefined) or when all values are tied (ρ = NaN).

    Formula: ρ = 1 – 6Σd² / (n(n²–1))

    This implementation converts raw scores to ranks (with average-rank tie
    handling) and then applies the standard Pearson formula on those ranks,
    which is equivalent and handles ties correctly.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None

    def to_ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = to_ranks(xs), to_ranks(ys)
    return _pearson(rx, ry)


def kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    """Kendall τ-b correlation between two equally-sized lists of scores.

    Returns None when n < 2.
    τ-b handles ties: τ = (C – D) / sqrt((C + D + T_x)(C + D + T_y))
    where C = concordant pairs, D = discordant pairs, T_x/T_y = tied-only pairs.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None

    concordant = discordant = tie_x = tie_y = tie_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = xs[i] - xs[j]
            sy = ys[i] - ys[j]
            if sx == 0 and sy == 0:
                tie_both += 1
            elif sx == 0:
                tie_x += 1
            elif sy == 0:
                tie_y += 1
            elif (sx > 0) == (sy > 0):
                concordant += 1
            else:
                discordant += 1

    denom_x = (concordant + discordant + tie_x) * (concordant + discordant + tie_y)
    if denom_x <= 0:
        return None
    import math

    return (concordant - discordant) / math.sqrt(denom_x)


def generator_effect_gap(
    reports_generated: list[JudgeReport],
    reports_human: list[JudgeReport],
) -> dict[str, Any]:
    """Measure the generator effect from paired strategy evaluations.

    Parameters
    ----------
    reports_generated:
        One JudgeReport per strategy, evaluated on the generated query slice.
    reports_human:
        One JudgeReport per strategy, evaluated on the human query slice.

    Returns
    -------
    dict with:
        strategies          — strategy names in rank order (generated slice)
        rank_generated      — mean judge scores on generated slice
        rank_human          — mean judge scores on human slice (matched by strategy)
        spearman_rho        — rank correlation between the two orderings
        kendall_tau         — Kendall τ between the two orderings
        gap_interpretation  — human-readable note about the gap size
    """
    gen_by_name = {r.strategy: r for r in reports_generated}
    hum_by_name = {r.strategy: r for r in reports_human}
    shared = sorted(set(gen_by_name) & set(hum_by_name))

    if len(shared) < 2:
        return {"error": "Need at least 2 strategies evaluated on both slices to measure the gap."}

    gen_scores = [gen_by_name[s].mean_judge_score for s in shared]
    hum_scores = [hum_by_name[s].mean_judge_score for s in shared]

    rho = spearman_rho(gen_scores, hum_scores)
    tau = kendall_tau(gen_scores, hum_scores)

    def interpret(r: float | None) -> str:
        if r is None:
            return "undefined"
        if r >= 0.9:
            return "strong agreement — generator effect is small"
        if r >= 0.7:
            return "moderate agreement — some generator effect present"
        if r >= 0.4:
            return "weak agreement — meaningful generator effect"
        return "poor agreement — strategy rankings differ substantially between slices"

    return {
        "strategies": shared,
        "mean_score_generated": {s: round(gen_by_name[s].mean_judge_score, 4) for s in shared},
        "mean_score_human": {s: round(hum_by_name[s].mean_judge_score, 4) for s in shared},
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "kendall_tau": round(tau, 4) if tau is not None else None,
        "gap_interpretation": interpret(rho),
        "n_generated_queries": reports_generated[0].n_queries if reports_generated else 0,
        "n_human_queries": reports_human[0].n_queries if reports_human else 0,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation (used internally by spearman_rho after rank conversion)."""
    import math

    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)
