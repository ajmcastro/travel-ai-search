"""Information-retrieval evaluation metrics, implemented from first principles.

All functions are pure — they take lists/sets/dicts and return floats.
No external dependencies; all are unit-testable without OpenSearch.

Notation
--------
retrieved   : ordered list of document IDs returned by the system (rank 1 first)
relevant    : set of document IDs judged relevant (binary; grade ≥ 1)
graded      : dict mapping doc_id → grade (0–3); only non-zero entries needed
k           : rank cutoff
"""

from __future__ import annotations

import math
from collections.abc import Set as AbstractSet


def precision_at_k(retrieved: list[str], relevant: AbstractSet[str], k: int) -> float:
    """Fraction of the top-K results that are relevant.

    P@K = |relevant ∩ top_K| / K

    Returns 0.0 when retrieved is empty.
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / k


def recall_at_k(retrieved: list[str], relevant: AbstractSet[str], k: int) -> float:
    """Fraction of all relevant documents found in the top-K.

    Recall@K = |relevant ∩ top_K| / |relevant|

    Returns 0.0 when the relevant set is empty.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: AbstractSet[str], k: int) -> float:
    """1.0 if at least one relevant document appears in the top-K, else 0.0.

    Also known as Success@K. The mean across queries equals the fraction of
    queries where the system returned at least one relevant result.
    """
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: AbstractSet[str]) -> float:
    """1 / rank of the first relevant document; 0.0 if none found.

    RR = 1.0 when the top result is relevant; 0.5 when it is second, etc.
    Average across queries gives MRR.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: AbstractSet[str]) -> float:
    """Area under the precision-recall curve for a single query.

    AP = (1/|R|) Σ P@i  for each i where retrieved[i] is relevant

    Average across queries gives MAP.
    Returns 0.0 when relevant is empty.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def _dcg(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    """DCG@K using the exponential gain formula.

    DCG@K = Σ_{i=1}^{K}  (2^rel_i − 1) / log2(i + 1)

    Grade-to-gain: 3→7, 2→3, 1→1, 0→0.
    Unjudged documents (not in graded) are treated as grade 0.
    """
    total = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = graded.get(doc_id, 0)
        if rel > 0:
            total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: list[str], graded: dict[str, int], k: int) -> float:
    """NDCG@K: DCG@K normalised by the ideal DCG@K.

    NDCG@K = DCG@K / IDCG@K    (bounded [0, 1])

    The ideal ranking places all documents in descending grade order.
    Returns 0.0 when no judgments have grade > 0 (nothing to find).
    """
    dcg = _dcg(retrieved, graded, k)
    # Ideal: sort all judged docs by descending grade
    ideal_order = sorted(graded, key=lambda d: graded[d], reverse=True)
    idcg = _dcg(ideal_order, graded, k)
    return dcg / idcg if idcg > 0.0 else 0.0


# ── Aggregate helpers ─────────────────────────────────────────────────────────


def mean_reciprocal_rank(rr_scores: list[float]) -> float:
    """Mean Reciprocal Rank across a list of per-query RR values."""
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


def mean_average_precision(ap_scores: list[float]) -> float:
    """Mean Average Precision across a list of per-query AP values."""
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
