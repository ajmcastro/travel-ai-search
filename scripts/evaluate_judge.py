"""LLM-as-judge evaluation CLI — Milestone 16.

Runs one or more retrieval strategies against a query slice (generated golden
queries, human-written queries, or both), scores each result with an LLM judge,
and prints a comparison table.  When two or more strategies are compared on both
slices, also prints the generator-effect gap (Spearman ρ between strategy
rankings on generated vs human queries).

Usage
-----
# Dry run with EchoJudgeProvider (no AWS, score = 2 for every hotel)
uv run python scripts/evaluate_judge.py --strategy rrf --slice generated

# Compare BM25 vs vector vs RRF on both slices with the Echo judge
uv run python scripts/evaluate_judge.py --all-strategies --slice both

# Use the real Bedrock judge (requires AWS credentials + BEDROCK_JUDGE_MODEL_ID set)
uv run python scripts/evaluate_judge.py --strategy rrf --judge-provider bedrock --slice both

# Save results to JSON
uv run python scripts/evaluate_judge.py --all-strategies --no-save=false --slice both
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Ensure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.evaluation.dataset import load_dataset
from travel_ai_search.evaluation.judge import EchoJudgeProvider, JudgeProvider
from travel_ai_search.evaluation.judge_evaluator import (
    JudgeReport,
    LLMEvaluator,
    generator_effect_gap,
    load_human_queries,
)
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.retrieval.fusion import FusionMethod
from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.types import Hit
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

GOLDEN_DEFAULT = Path("data/evaluation/golden_queries.jsonl")
HUMAN_DEFAULT = Path("data/evaluation/human_queries.jsonl")
RESULTS_DIR = Path("data/evaluation/results")

ALL_STRATEGIES = ["bm25", "vector", "rrf"]


# ── Judge provider factory ────────────────────────────────────────────────────


def build_judge_provider(name: str, settings: Any) -> JudgeProvider:
    if name == "bedrock":
        try:
            from travel_ai_search.evaluation.judge_bedrock import BedrockJudgeProvider
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client

            client = create_bedrock_client(settings.aws_region)
            provider = BedrockJudgeProvider(client, model_id=settings.bedrock_judge_model_id)
            print(f"Judge: BedrockJudgeProvider ({settings.bedrock_judge_model_id})")
            return provider
        except Exception as exc:
            print(f"WARNING: Bedrock judge failed ({exc}), falling back to EchoJudgeProvider.")
    print("Judge: EchoJudgeProvider (fixed score=2, no LLM call)")
    return EchoJudgeProvider(fixed_score=2)


# ── Strategy search-function factory ─────────────────────────────────────────


def build_search_fn(strategy: str, client: Any, embedding_provider: Any, settings: Any) -> Any:
    index = settings.opensearch_index_name

    if strategy == "bm25":

        def fn(query_text: str, top_k: int, filters: dict[str, Any]) -> list[Hit]:
            params = LexicalSearchParams(query=query_text, top_k=top_k, **filters)
            return lexical_search(client, params, index=index).hits

        return fn

    if strategy == "vector":

        def fn(query_text: str, top_k: int, filters: dict[str, Any]) -> list[Hit]:
            params = VectorSearchParams(query=query_text, top_k=top_k, **filters)
            return vector_search(client, embedding_provider, params, index=index).hits

        return fn

    # rrf (default)
    def fn(query_text: str, top_k: int, filters: dict[str, Any]) -> list[Hit]:
        params = HybridSearchParams(
            query=query_text,
            top_k=top_k,
            fusion=FusionMethod.rrf,
            candidate_k=max(top_k * 5, 50),
            **filters,
        )
        return hybrid_search(client, embedding_provider, params, index=index).hits

    return fn


# ── Console output helpers ────────────────────────────────────────────────────


def _print_report(report: JudgeReport) -> None:
    agr = report.agreement_rate
    agr_str = f"{agr:.4f}" if agr is not None else "N/A"
    print(
        f"\n{'─' * 60}\n"
        f"Strategy  : {report.strategy}\n"
        f"Slice     : {report.slice_name} ({report.n_queries} queries)\n"
        f"Judge     : {report.judge_model_id}\n"
        f"Mean score: {report.mean_judge_score:.4f}  (scale 0–3)\n"
        f"Agreement : {agr_str}  (golden ≥2 ↔ judge ≥2, N/A when no golden labels)\n"
    )
    print(f"{'Query':<45} {'Class':<20} {'Score':>5}")
    print("─" * 75)
    for q in report.per_query:
        label = q.query_text[:44]
        print(f"{label:<45} {q.query_class:<20} {q.mean_score:>5.2f}")


def _print_gap(gap: dict[str, Any]) -> None:
    if "error" in gap:
        print(f"\nGenerator-effect gap: {gap['error']}")
        return
    rho = gap.get("spearman_rho")
    tau = gap.get("kendall_tau")
    print(
        f"\n{'═' * 60}\n"
        f"Generator-effect gap\n"
        f"{'═' * 60}\n"
        f"  Strategies compared : {', '.join(gap['strategies'])}\n"
        f"  Generated queries   : {gap['n_generated_queries']}\n"
        f"  Human queries       : {gap['n_human_queries']}\n"
        f"  Spearman ρ          : {rho if rho is not None else 'N/A'}\n"
        f"  Kendall τ           : {tau if tau is not None else 'N/A'}\n"
        f"  Interpretation      : {gap['gap_interpretation']}\n"
    )
    print(f"\n  {'Strategy':<12} {'Score (generated)':>18} {'Score (human)':>14}")
    print(f"  {'─' * 46}")
    for s in gap["strategies"]:
        gs = gap["mean_score_generated"][s]
        hs = gap["mean_score_human"][s]
        print(f"  {s:<12} {gs:>18.4f} {hs:>14.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation CLI")
    parser.add_argument(
        "--strategy",
        default="rrf",
        choices=[*ALL_STRATEGIES, "all"],
        help="Retrieval strategy to evaluate (default: rrf).",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Evaluate all strategies (bm25, vector, rrf). Overrides --strategy.",
    )
    parser.add_argument(
        "--judge-provider",
        default="echo",
        choices=["echo", "bedrock"],
        help="Judge provider (default: echo).",
    )
    parser.add_argument(
        "--slice",
        default="generated",
        choices=["generated", "human", "both"],
        help="Query slice to evaluate (default: generated).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Hotels to retrieve and judge per query (default: 5).",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DEFAULT,
        help="Path to the generated golden queries JSONL.",
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=HUMAN_DEFAULT,
        help="Path to the human queries JSONL.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON results to data/evaluation/results/.",
    )
    args = parser.parse_args()

    strategies = ALL_STRATEGIES if args.all_strategies else [args.strategy]

    settings = get_settings()
    client = create_client(settings)
    embedding_provider = LocalEmbeddingProvider(settings.embedding_model_name)
    judge = build_judge_provider(args.judge_provider, settings)
    evaluator = LLMEvaluator(judge)

    gen_reports: list[JudgeReport] = []
    hum_reports: list[JudgeReport] = []

    for strategy in strategies:
        search_fn = build_search_fn(strategy, client, embedding_provider, settings)

        if args.slice in ("generated", "both"):
            if not args.golden.exists():
                print(f"ERROR: Golden dataset not found at {args.golden}")
                print("Run: uv run python scripts/build_golden_dataset.py")
                sys.exit(1)
            dataset = load_dataset(args.golden)
            print(f"\nEvaluating '{strategy}' on generated slice ({len(dataset)} queries)…")
            gen_report = evaluator.evaluate_generated(
                search_fn, dataset, strategy=strategy, k=args.k
            )
            gen_reports.append(gen_report)
            _print_report(gen_report)

        if args.slice in ("human", "both"):
            if not args.human.exists():
                print(f"ERROR: Human queries not found at {args.human}")
                sys.exit(1)
            human_qs = load_human_queries(args.human)
            print(f"\nEvaluating '{strategy}' on human slice ({len(human_qs)} queries)…")
            hum_report = evaluator.evaluate_human(search_fn, human_qs, strategy=strategy, k=args.k)
            hum_reports.append(hum_report)
            _print_report(hum_report)

    # Generator-effect gap (only when both slices have ≥2 strategies)
    if args.slice == "both" and len(strategies) >= 2:
        gap = generator_effect_gap(gen_reports, hum_reports)
        _print_gap(gap)

    # Save results
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        for report in gen_reports + hum_reports:
            fname = f"judge_{report.strategy}_{report.slice_name}_{today}.json"
            out = RESULTS_DIR / fname
            out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"\nSaved → {out}")

    client.close()


if __name__ == "__main__":
    main()
