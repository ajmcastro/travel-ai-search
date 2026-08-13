"""Strategy-agnostic retrieval evaluator.

Usage
-----
Define a SearchFn — any callable with the signature:

    def search(query_text: str, top_k: int, filters: dict) -> list[str]:
        ...  # returns doc IDs in rank order

Pass it to evaluate() alongside a GoldenDataset. The same evaluator
works for BM25 (Milestone 4), vector (Milestone 5), hybrid (Milestone 6),
and all subsequent strategies — only the SearchFn changes.

Metrics computed per query and aggregated globally and by query class:
    Precision@K, Recall@K, HitRate@K, RR, AP, NDCG@K
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from travel_ai_search.evaluation.dataset import GoldenDataset
from travel_ai_search.evaluation.metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# A retrieval strategy is a plain callable.
SearchFn = Callable[[str, int, dict[str, Any]], list[str]]


@dataclass
class QueryMetrics:
    """Per-query evaluation results."""

    query_id: str
    query_text: str
    query_class: str
    precision: float
    recall: float
    hit_rate: float
    rr: float
    ap: float
    ndcg: float
    took_ms: int
    retrieved_count: int
    relevant_count: int


@dataclass
class ClassSummary:
    """Aggregated metrics for one query class."""

    query_class: str
    n_queries: int
    precision: float
    recall: float
    hit_rate: float
    mrr: float
    map_score: float
    ndcg: float


@dataclass
class EvaluationReport:
    """Full evaluation result for one strategy."""

    strategy: str
    k: int
    per_query: list[QueryMetrics] = field(default_factory=list)

    # ── Overall aggregates (computed lazily via properties) ────────────────

    @property
    def n_queries(self) -> int:
        return len(self.per_query)

    @property
    def overall_precision(self) -> float:
        return _mean(q.precision for q in self.per_query)

    @property
    def overall_recall(self) -> float:
        return _mean(q.recall for q in self.per_query)

    @property
    def overall_hit_rate(self) -> float:
        return _mean(q.hit_rate for q in self.per_query)

    @property
    def mrr(self) -> float:
        return _mean(q.rr for q in self.per_query)

    @property
    def map_score(self) -> float:
        return _mean(q.ap for q in self.per_query)

    @property
    def overall_ndcg(self) -> float:
        return _mean(q.ndcg for q in self.per_query)

    @property
    def latency_p50_ms(self) -> int:
        latencies = sorted(q.took_ms for q in self.per_query)
        return latencies[len(latencies) // 2] if latencies else 0

    @property
    def latency_p95_ms(self) -> int:
        latencies = sorted(q.took_ms for q in self.per_query)
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)] if latencies else 0

    def by_class(self) -> list[ClassSummary]:
        groups: dict[str, list[QueryMetrics]] = defaultdict(list)
        for q in self.per_query:
            groups[q.query_class].append(q)
        summaries = []
        for cls, metrics in sorted(groups.items()):
            summaries.append(
                ClassSummary(
                    query_class=cls,
                    n_queries=len(metrics),
                    precision=_mean(m.precision for m in metrics),
                    recall=_mean(m.recall for m in metrics),
                    hit_rate=_mean(m.hit_rate for m in metrics),
                    mrr=_mean(m.rr for m in metrics),
                    map_score=_mean(m.ap for m in metrics),
                    ndcg=_mean(m.ndcg for m in metrics),
                )
            )
        return summaries

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "k": self.k,
            "n_queries": self.n_queries,
            "overall": {
                f"precision_at_{self.k}": round(self.overall_precision, 4),
                f"recall_at_{self.k}": round(self.overall_recall, 4),
                f"hit_rate_at_{self.k}": round(self.overall_hit_rate, 4),
                "mrr": round(self.mrr, 4),
                "map": round(self.map_score, 4),
                f"ndcg_at_{self.k}": round(self.overall_ndcg, 4),
                "latency_p50_ms": self.latency_p50_ms,
                "latency_p95_ms": self.latency_p95_ms,
            },
            "by_class": [
                {
                    "query_class": s.query_class,
                    "n_queries": s.n_queries,
                    f"precision_at_{self.k}": round(s.precision, 4),
                    f"recall_at_{self.k}": round(s.recall, 4),
                    f"hit_rate_at_{self.k}": round(s.hit_rate, 4),
                    "mrr": round(s.mrr, 4),
                    "map": round(s.map_score, 4),
                    f"ndcg_at_{self.k}": round(s.ndcg, 4),
                }
                for s in self.by_class()
            ],
            "per_query": [
                {
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "query_class": q.query_class,
                    f"precision_at_{self.k}": round(q.precision, 4),
                    f"recall_at_{self.k}": round(q.recall, 4),
                    f"hit_rate_at_{self.k}": round(q.hit_rate, 4),
                    "rr": round(q.rr, 4),
                    "ap": round(q.ap, 4),
                    f"ndcg_at_{self.k}": round(q.ndcg, 4),
                    "took_ms": q.took_ms,
                    "retrieved_count": q.retrieved_count,
                    "relevant_count": q.relevant_count,
                }
                for q in self.per_query
            ],
        }


def evaluate(
    search_fn: SearchFn,
    dataset: GoldenDataset,
    *,
    strategy: str,
    k: int = 10,
) -> EvaluationReport:
    """Evaluate a retrieval strategy against the golden dataset.

    For each query in the dataset:
    - Calls search_fn(query_text, k, filters) to get a ranked list of doc IDs.
    - Computes all metrics using the query's relevance judgments.

    Queries with no relevant judgments (has_relevant=False) are skipped with a
    warning because metrics are undefined when there is nothing to find.
    """
    report = EvaluationReport(strategy=strategy, k=k)

    for golden_query in dataset.queries:
        if not golden_query.has_relevant:
            continue

        t0 = time.perf_counter()
        retrieved = search_fn(golden_query.query_text, k, golden_query.filters)
        took_ms = int((time.perf_counter() - t0) * 1000)

        relevant = golden_query.relevant_ids
        graded = golden_query.graded_relevance

        report.per_query.append(
            QueryMetrics(
                query_id=golden_query.query_id,
                query_text=golden_query.query_text,
                query_class=golden_query.query_class,
                precision=precision_at_k(retrieved, relevant, k),
                recall=recall_at_k(retrieved, relevant, k),
                hit_rate=hit_rate_at_k(retrieved, relevant, k),
                rr=reciprocal_rank(retrieved, relevant),
                ap=average_precision(retrieved, relevant),
                ndcg=ndcg_at_k(retrieved, graded, k),
                took_ms=took_ms,
                retrieved_count=len(retrieved),
                relevant_count=len(relevant),
            )
        )

    return report


# ── Internal helpers ──────────────────────────────────────────────────────────


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
