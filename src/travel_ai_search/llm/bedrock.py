"""AWS Bedrock LLM provider using the Converse API.

The Converse API provides a model-agnostic interface to conversational models
on Bedrock: Anthropic Claude, Amazon Titan Text, Meta Llama, Mistral, etc.
Switching models is a matter of changing BEDROCK_LLM_MODEL_ID — no code change.

Default model: anthropic.claude-haiku-4-5-20251001
  Rationale: fast (< 1 s typical) and low cost; ideal for query rewriting where
  latency matters more than long-form quality.  The full pipeline latency budget
  is dominated by retrieval, so a fast LLM keeps the overhead proportional.
  Alternative: claude-sonnet-4-6 for richer semantic paraphrase quality at ~5×
  the token cost and ~2× the latency.

Converse API shape:
  messages: list of {role, content[{text}]} turns
  system:   optional system-level instructions (maps to LLMProvider.generate(system=))
  output:   response["output"]["message"]["content"][0]["text"]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "anthropic.claude-haiku-4-5-20251001"


class BedrockLLMProvider:
    """LLM provider backed by the AWS Bedrock Converse API.

    Satisfies LLMProvider Protocol (structural typing — no import needed).
    """

    def __init__(
        self,
        client: Any,
        *,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        self._client = client
        self._model_id = model_id

    def generate(self, prompt: str, *, system: str = "") -> str:
        """Call the Bedrock Converse API and return the first response text.

        Exceptions propagate to the caller.  QueryRewriter wraps generate()
        in a try/except and falls back to the original query on failure,
        so no defensive handling is needed here.
        """
        kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self._client.converse(**kwargs)
        return str(response["output"]["message"]["content"][0]["text"])
