"""BedrockJudgeProvider — LLM-as-judge using the AWS Bedrock Converse API.

The judge model MUST differ from the model that generated the synthetic hotel
descriptions to avoid common-mode bias.  See the methodology note in
docs/EXPERIMENTS.md §M16 for a full explanation.

Default judge model: ``amazon.nova-lite-v1:0``
  Rationale: Amazon Nova is a distinct model family from Anthropic Claude, which
  generated the synthetic hotel descriptions.  Nova Lite is fast and low-cost;
  its compact size is acceptable for relevance scoring because the task is
  classification (0–3), not long-form generation.

  Alternatives in order of independence from the generator:
    mistral.mistral-large-2402-v1:0  — European independent lab; strong reasoning
    meta.llama3-70b-instruct-v1:0    — Meta; open-weights family
    cohere.command-r-plus-v1:0       — Cohere; retrieval-focused training

  Do NOT use anthropic.claude-* models as the judge when Claude generated the data.

Switching the judge model: set BEDROCK_JUDGE_MODEL_ID in your .env.
Changing it between runs produces incomparable scores — record the model_id
alongside every saved result set.
"""

from __future__ import annotations

import logging
from typing import Any

from travel_ai_search.evaluation.judge import JudgeVerdict, build_judge_prompt, parse_judge_response

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL_ID = "amazon.nova-lite-v1:0"

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert travel search evaluator. "
    "Always follow the exact response format requested by the user."
)


class BedrockJudgeProvider:
    """JudgeProvider backed by the AWS Bedrock Converse API.

    Satisfies the JudgeProvider Protocol via structural typing — no explicit
    inheritance required.

    Parameters
    ----------
    client:
        A boto3 ``bedrock-runtime`` client (from ``create_bedrock_client()``).
    model_id:
        Bedrock model ID.  Must be from a DIFFERENT family than the generator.
        Defaults to ``amazon.nova-lite-v1:0``.
    """

    def __init__(
        self,
        client: Any,
        *,
        model_id: str = DEFAULT_JUDGE_MODEL_ID,
    ) -> None:
        self._client = client
        self.model_id = model_id

    def judge(
        self,
        query: str,
        hotel_name: str,
        hotel_description: str,
    ) -> JudgeVerdict:
        """Score relevance via the Bedrock Converse API.

        On any API error, logs a warning and returns score=0 so a single failed
        call does not abort the entire evaluation run.
        """
        prompt = build_judge_prompt(query, hotel_name, hotel_description)
        try:
            response = self._client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system=[{"text": _JUDGE_SYSTEM_PROMPT}],
            )
            text: str = response["output"]["message"]["content"][0]["text"]
            return parse_judge_response(text)
        except Exception as exc:
            logger.warning(
                "BedrockJudgeProvider.judge failed (model=%s): %s",
                self.model_id,
                exc,
            )
            return JudgeVerdict(score=0, rationale=f"[error] {exc}")
