"""RAGSynthesizer: LLM-backed synthesis of hotel results and destination knowledge."""

from __future__ import annotations

from travel_ai_search.llm.base import LLMProvider
from travel_ai_search.rag.knowledge import DestinationKnowledge
from travel_ai_search.retrieval.types import Hit

_SYNTHESIS_SYSTEM = (
    "You are a knowledgeable travel expert. "
    "Give specific, helpful recommendations based only on the information provided."
)

_MAX_HOTELS = 5
_MAX_DESC_CHARS = 120


def _format_knowledge(docs: list[DestinationKnowledge]) -> str:
    lines = []
    for k in docs:
        tags = ", ".join(k.character_tags[:5])
        months = ", ".join(k.best_months[:4])
        lines.append(
            f"- {k.destination} ({k.country}): {k.description} "
            f"Best months: {months}. Character: {tags}. "
            f"Family: {k.family_suitability}. Nightlife: {k.nightlife_level}."
        )
    return "\n".join(lines)


def _format_hotels(hits: list[Hit], *, max_hotels: int) -> str:
    lines = []
    for hit in hits[:max_hotels]:
        name = hit.source.get("hotel_name", "Unknown")
        dest = hit.source.get("destination", "")
        desc = hit.source.get("hotel_description", "")
        truncated = desc[:_MAX_DESC_CHARS].rstrip()
        if len(desc) > _MAX_DESC_CHARS:
            truncated += "…"
        lines.append(f"- {name} ({dest}): {truncated}")
    return "\n".join(lines)


class RAGSynthesizer:
    """Synthesises a travel recommendation from hotel hits and destination knowledge.

    The LLM prompt is structured as:
      1. Travel request (original query)
      2. Destination knowledge (from KnowledgeRetriever)
      3. Top matching hotels (from the hybrid/multi-query retrieval)

    The LLM is asked to produce a 2-3 sentence recommendation.  If the LLM
    call raises, the exception propagates to the route handler, which catches
    it and returns knowledge_context without a rag_summary.
    """

    def __init__(self, llm: LLMProvider, *, max_hotels: int = _MAX_HOTELS) -> None:
        self._llm = llm
        self._max_hotels = max_hotels

    def synthesize(
        self,
        query: str,
        hits: list[Hit],
        knowledge: list[DestinationKnowledge],
    ) -> str:
        """Return a 2-3 sentence travel recommendation string."""
        knowledge_text = _format_knowledge(knowledge)
        hotels_text = _format_hotels(hits, max_hotels=self._max_hotels)

        prompt = (
            f"Travel request: {query}\n\n"
            f"Destination knowledge:\n{knowledge_text}\n\n"
            f"Top matching hotels:\n{hotels_text}\n\n"
            "Write a 2-3 sentence travel recommendation explaining what makes "
            "these destinations and hotels a good match for this travel request. "
            "Be specific — mention destination character and one or two hotel highlights."
        )
        return self._llm.generate(prompt, system=_SYNTHESIS_SYSTEM)
