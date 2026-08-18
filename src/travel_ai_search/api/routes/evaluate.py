"""POST /evaluate/judge — LLM-as-judge evaluation endpoint.

Runs a retrieval strategy on a set of caller-supplied queries, then asks a
JudgeProvider to score each retrieved hotel (0–3).  Returns scored results
along with the judge model identity.

The EchoJudgeProvider (default when JUDGE_PROVIDER=echo) requires no AWS
credentials and returns a fixed score of 2 — useful for verifying the pipeline
wiring without real API calls.

To use the Bedrock judge:
  1. Set JUDGE_PROVIDER=bedrock in your .env.
  2. Set BEDROCK_JUDGE_MODEL_ID to a model that is NOT from the same family as
     the model that generated the synthetic corpus (e.g. amazon.nova-lite-v1:0).
  3. Ensure valid AWS credentials are in the environment.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from opensearchpy import OpenSearch

from travel_ai_search.api.deps import get_embedding_provider, get_os_client, get_settings
from travel_ai_search.api.schemas.evaluate import (
    JudgedHit,
    JudgeQueryOutput,
    JudgeRequest,
    JudgeResponse,
)
from travel_ai_search.config.settings import Settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.evaluation.judge import EchoJudgeProvider, JudgeProvider
from travel_ai_search.evaluation.judge_evaluator import JudgeFn
from travel_ai_search.retrieval.fusion import FusionMethod
from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search
from travel_ai_search.retrieval.types import Hit
from travel_ai_search.retrieval.vector import VectorSearchParams, vector_search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])


def _build_judge_provider(provider_name: str, settings: Settings) -> JudgeProvider:
    """Factory: build the appropriate JudgeProvider from the request parameter."""
    if provider_name == "bedrock":
        try:
            from travel_ai_search.evaluation.judge_bedrock import BedrockJudgeProvider
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client

            client = create_bedrock_client(settings.aws_region)
            return BedrockJudgeProvider(client, model_id=settings.bedrock_judge_model_id)
        except Exception as exc:
            logger.warning(
                "Failed to create BedrockJudgeProvider: %s — falling back to EchoJudgeProvider.",
                exc,
            )
    return EchoJudgeProvider(fixed_score=2)


def _build_search_fn(
    strategy: str,
    client: OpenSearch,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
) -> JudgeFn:
    """Return a JudgeFn (query, k, filters) → list[Hit] for the named strategy."""
    index = settings.opensearch_index_name

    if strategy == "bm25":

        def bm25_fn(query_text: str, top_k: int, filters: dict[str, Any]) -> list[Hit]:
            params = LexicalSearchParams(query=query_text, top_k=top_k, **filters)
            return lexical_search(client, params, index=index).hits

        return bm25_fn

    if strategy == "vector":

        def vector_fn(query_text: str, top_k: int, filters: dict[str, Any]) -> list[Hit]:
            params = VectorSearchParams(query=query_text, top_k=top_k, **filters)
            return vector_search(client, embedding_provider, params, index=index).hits

        return vector_fn

    # Default: RRF hybrid
    def rrf_fn(query_text: str, top_k: int, filters: dict[str, Any]) -> list[Hit]:
        params = HybridSearchParams(
            query=query_text,
            top_k=top_k,
            fusion=FusionMethod.rrf,
            candidate_k=max(top_k * 5, 50),
            **filters,
        )
        return hybrid_search(client, embedding_provider, params, index=index).hits

    return rrf_fn


@router.post("/judge", response_model=JudgeResponse)
def judge_endpoint(
    body: JudgeRequest,
    client: OpenSearch = Depends(get_os_client),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    settings: Settings = Depends(get_settings),
) -> JudgeResponse:
    """Run an LLM judge over the top-K results of a retrieval strategy.

    For each query in the request, this endpoint:
    1. Runs the chosen retrieval strategy to get the top-K hotels.
    2. Calls the JudgeProvider to score each hotel (0 = irrelevant, 3 = highly relevant).
    3. Returns the scored results together with the judge model identity.

    The ``judge_provider`` field controls which backend is used:
    - ``"echo"`` — fixed score of 2 for all hotels; no LLM calls; safe for testing.
    - ``"bedrock"`` — AWS Bedrock (see BEDROCK_JUDGE_MODEL_ID in settings).

    **Important:** the judge model recorded in ``judge_model_id`` must differ from the
    model that generated the hotel data to avoid common-mode bias in the scores.
    """
    judge = _build_judge_provider(body.judge_provider, settings)
    search_fn = _build_search_fn(body.strategy, client, embedding_provider, settings)

    results: list[JudgeQueryOutput] = []
    for q in body.queries:
        hits: list[Hit] = search_fn(q.query_text, body.k, {})
        scored: list[JudgedHit] = []
        for rank, hit in enumerate(hits, start=1):
            name = hit.source.get("hotel_name", hit.id)
            desc = hit.source.get("hotel_description", "")
            verdict = judge.judge(q.query_text, name, desc)
            scored.append(
                JudgedHit(
                    doc_id=hit.id,
                    hotel_name=name,
                    rank=rank,
                    judge_score=verdict.score,
                    rationale=verdict.rationale,
                )
            )
        mean_score = sum(h.judge_score for h in scored) / len(scored) if scored else 0.0
        results.append(
            JudgeQueryOutput(
                query_text=q.query_text,
                query_class=q.query_class,
                mean_score=round(mean_score, 4),
                hits=scored,
            )
        )

    overall_mean = sum(r.mean_score for r in results) / len(results) if results else 0.0

    return JudgeResponse(
        strategy=body.strategy,
        judge_model_id=judge.model_id,
        k=body.k,
        n_queries=len(results),
        mean_judge_score=round(overall_mean, 4),
        results=results,
    )
