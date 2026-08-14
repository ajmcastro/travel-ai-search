"""Evaluation CLI — runs retrieval strategies against the golden dataset.

Usage:
    uv run python scripts/evaluate.py                      # BM25 (default)
    uv run python scripts/evaluate.py --strategy vector    # Vector ANN
    uv run python scripts/evaluate.py --strategy hybrid    # Hybrid weighted-sum
    uv run python scripts/evaluate.py --strategy rrf       # Hybrid RRF
    uv run python scripts/evaluate.py --strategy rerank    # RRF + cross-encoder reranking
    uv run python scripts/evaluate.py --strategy bm25
    uv run python scripts/evaluate.py --k 10 --no-save     # skip JSON output
    uv run python scripts/evaluate.py --golden data/evaluation/golden_queries.jsonl

Output:
    Console table broken down by query class.
    JSON results saved to data/evaluation/results/<strategy>_<date>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Ensure src is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.evaluation.dataset import load_dataset
from travel_ai_search.evaluation.evaluator import EvaluationReport, evaluate
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.retrieval.fusion import FusionMethod
from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

GOLDEN_DEFAULT = Path("data/evaluation/golden_queries.jsonl")
RESULTS_DIR = Path("data/evaluation/results")


# ── Strategy factories ────────────────────────────────────────────────────────


def make_bm25_fn(client: Any, index: str, **_: Any) -> Any:
    """Return a SearchFn that calls BM25 lexical search."""

    def search(query_text: str, top_k: int, filters: dict[str, Any]) -> list[str]:
        params = LexicalSearchParams(query=query_text, top_k=top_k, **filters)
        result = lexical_search(client, params, index=index)
        return [hit.id for hit in result.hits]

    return search


def make_vector_fn(client: Any, index: str, embedding_provider: Any = None, **_: Any) -> Any:
    """Return a SearchFn that calls dense vector (ANN) search.

    The embedding_provider is loaded once and reused for all queries in the run.
    If not provided (e.g. from a strategy dict), it is created from settings.
    """
    provider = embedding_provider

    def search(query_text: str, top_k: int, filters: dict[str, Any]) -> list[str]:
        params = VectorSearchParams(query=query_text, top_k=top_k, **filters)
        result = vector_search(client, provider, params, index=index)
        return [hit.id for hit in result.hits]

    return search


def make_hybrid_fn(
    client: Any,
    index: str,
    embedding_provider: Any = None,
    lexical_weight: float = 0.5,
    vector_weight: float = 0.5,
    candidate_k: int = 50,
    **_: Any,
) -> Any:
    """Return a SearchFn that calls hybrid (BM25 + vector) search.

    Weights and candidate pool size can be adjusted for experimentation.
    """
    provider = embedding_provider

    def search(query_text: str, top_k: int, filters: dict[str, Any]) -> list[str]:
        params = HybridSearchParams(
            query=query_text,
            top_k=top_k,
            candidate_k=candidate_k,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
            **filters,
        )
        result = hybrid_search(client, provider, params, index=index)
        return [hit.id for hit in result.hits]

    return search


def make_rrf_fn(
    client: Any,
    index: str,
    embedding_provider: Any = None,
    rrf_k: int = 60,
    candidate_k: int = 50,
    **_: Any,
) -> Any:
    """Return a SearchFn that calls hybrid (BM25 + vector) search with RRF fusion.

    rrf_k and candidate_k can be adjusted for experimentation.
    """
    provider = embedding_provider

    def search(query_text: str, top_k: int, filters: dict[str, Any]) -> list[str]:
        params = HybridSearchParams(
            query=query_text,
            top_k=top_k,
            candidate_k=candidate_k,
            fusion=FusionMethod.rrf,
            rrf_k=rrf_k,
            **filters,
        )
        result = hybrid_search(client, provider, params, index=index)
        return [hit.id for hit in result.hits]

    return search


def make_rerank_fn(
    client: Any,
    index: str,
    embedding_provider: Any = None,
    rrf_k: int = 60,
    candidate_k: int = 50,
    rerank_k: int = 50,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **_: Any,
) -> Any:
    """Return a SearchFn that uses RRF fusion followed by cross-encoder reranking.

    Stage 1: BM25 + vector → RRF fusion → top rerank_k candidates.
    Stage 2: Cross-encoder scores each (query, candidate) pair → top_k returned.
    """
    from travel_ai_search.reranking.local import LocalCrossEncoderReranker

    provider = embedding_provider
    reranker = LocalCrossEncoderReranker(reranker_model_name)

    def search(query_text: str, top_k: int, filters: dict[str, Any]) -> list[str]:
        params = HybridSearchParams(
            query=query_text,
            top_k=top_k,
            candidate_k=candidate_k,
            fusion=FusionMethod.rrf,
            rrf_k=rrf_k,
            rerank_k=max(top_k, rerank_k),
            **filters,
        )
        result = hybrid_search(client, provider, params, index=index, reranker=reranker)
        return [hit.id for hit in result.hits]

    return search


STRATEGIES: dict[str, Any] = {
    "bm25": make_bm25_fn,
    "vector": make_vector_fn,
    "hybrid": make_hybrid_fn,
    "rrf": make_rrf_fn,
    "rerank": make_rerank_fn,
}


# ── Formatting ────────────────────────────────────────────────────────────────


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(report: EvaluationReport) -> None:
    k = report.k
    width = 80

    print()
    print("=" * width)
    print(f"  Strategy: {report.strategy.upper()}   |   @K={k}   |   {report.n_queries} queries")
    print("=" * width)

    # Overall metrics
    print()
    print("  OVERALL")
    print(f"  {'NDCG@K':<18} {_fmt(report.overall_ndcg)}  {_bar(report.overall_ndcg)}")
    print(f"  {'MRR':<18} {_fmt(report.mrr)}  {_bar(report.mrr)}")
    print(f"  {'HitRate@K':<18} {_fmt(report.overall_hit_rate)}  {_bar(report.overall_hit_rate)}")
    print(
        f"  {'Precision@K':<18} {_fmt(report.overall_precision)}  {_bar(report.overall_precision)}"
    )
    print(
        f"  {'Recall@K':<18} {_fmt(report.overall_recall)}  (note: low; many judged docs per query)"
    )
    print(f"  {'MAP':<18} {_fmt(report.map_score)}")
    print(f"  {'Latency p50':<18} {report.latency_p50_ms} ms")
    print(f"  {'Latency p95':<18} {report.latency_p95_ms} ms")

    # Per-class breakdown
    print()
    print("  BY QUERY CLASS")
    header = f"  {'Class':<22} {'n':>3}  {'NDCG':>6}  {'MRR':>6}  {'HR':>6}  {'P':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in report.by_class():
        print(
            f"  {cls.query_class:<22} {cls.n_queries:>3}  "
            f"{_fmt(cls.ndcg):>6}  {_fmt(cls.mrr):>6}  "
            f"{_fmt(cls.hit_rate):>6}  {_fmt(cls.precision):>6}"
        )

    # Bottom-5 queries by NDCG
    sorted_queries = sorted(report.per_query, key=lambda q: q.ndcg)
    print()
    print("  LOWEST NDCG@K QUERIES")
    for q in sorted_queries[:5]:
        print(f"  [{q.query_id}] {_fmt(q.ndcg)}  {q.query_text[:60]}")

    print()
    print("=" * width)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a retrieval strategy against the golden relevance dataset."
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="bm25",
        help="Retrieval strategy to evaluate (default: bm25)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DEFAULT,
        help=f"Path to the golden queries JSONL file (default: {GOLDEN_DEFAULT})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank cutoff for metrics (default: 10)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON results to disk",
    )
    args = parser.parse_args(argv)

    # Infrastructure
    settings = get_settings()
    try:
        client = create_client(settings)
        client.info()
    except Exception as exc:
        host = f"{settings.opensearch_host}:{settings.opensearch_port}"
        print(f"\nERROR: Cannot connect to OpenSearch at {host}")
        print(f"  {exc}")
        print("  Start OpenSearch with:  make up")
        sys.exit(1)

    # Load dataset
    if not args.golden.exists():
        print(f"\nERROR: Golden dataset not found: {args.golden}")
        print("  Build it with:  uv run python scripts/build_golden_dataset.py")
        sys.exit(1)

    print(f"\nLoading golden dataset from {args.golden} …")
    dataset = load_dataset(args.golden)
    print(f"  {len(dataset)} queries across {len(dataset.by_class())} classes")

    # Load embedding model if needed
    embedding_provider = None
    if args.strategy in ("vector", "hybrid", "rrf", "rerank"):
        print(f"\nLoading embedding model '{settings.embedding_model_name}' …")
        embedding_provider = LocalEmbeddingProvider(settings.embedding_model_name)
        print(f"  dimension={embedding_provider.dimension}")

    # Build search function
    factory = STRATEGIES[args.strategy]
    search_fn = factory(
        client,
        settings.opensearch_index_name,
        embedding_provider=embedding_provider,
    )

    # Evaluate
    print(f"\nRunning evaluation: strategy={args.strategy}, k={args.k} …")
    report = evaluate(search_fn, dataset, strategy=args.strategy, k=args.k)

    # Print
    print_report(report)

    # Save
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{args.strategy}_{date.today().isoformat()}.json"
        output_path = RESULTS_DIR / filename
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Results saved to: {output_path}")
        print()


if __name__ == "__main__":
    main()
