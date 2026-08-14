"""Query rewriting using an LLM provider.

The rewriter takes the semantic query produced by QueryUnderstandingEngine
(hard constraints already stripped) and expands or paraphrases it to improve
retrieval vocabulary coverage.

Design notes:
- The original query is always preserved in the response; only the retrieval
  query changes when rewriting is active.
- Any exception from the LLM logs a warning and returns the original semantic
  query unchanged — search never fails due to a rewriting error.
- The system prompt is written for AWS Bedrock (Milestone 12); local providers
  receive it but may ignore it.
"""

from __future__ import annotations

import logging

from travel_ai_search.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a travel search expert. Rewrite the user's travel search query to be "
    "more descriptive and semantically rich, helping surface the most relevant hotels. "
    "Add related travel concepts, domain synonyms, and terms that appear in hotel "
    "descriptions (e.g. 'all-inclusive', 'beachfront', 'wellness spa', 'family club'). "
    "Preserve the original intent exactly — do not invent destinations or constraints. "
    "Return only the rewritten query: no explanation, no quotes, no commentary."
)


class QueryRewriter:
    """Wraps an LLMProvider to perform travel query rewriting.

    Args:
        llm: Language model provider.  Receives the semantic query as the user
             prompt and the rewriting instruction as the system prompt.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def rewrite(self, semantic_query: str) -> str:
        """Rewrite *semantic_query*.  Falls back to the original on any error."""
        if not semantic_query.strip():
            return semantic_query
        try:
            result = self._llm.generate(semantic_query, system=_SYSTEM_PROMPT).strip()
            return result if result else semantic_query
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewriting failed (%s) — using original query.", exc)
            return semantic_query
