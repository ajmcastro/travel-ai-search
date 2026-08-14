"""Unit tests for the RAG module (Milestone 13).

Tests cover:
  - DestinationKnowledge model validation and embedding text construction
  - KnowledgeRetriever: embedding call, OpenSearch query shape, country filter, result mapping
  - RAGSynthesizer: LLM call, prompt content, edge cases (empty knowledge, empty hits)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from travel_ai_search.rag.knowledge import DestinationKnowledge, build_knowledge_embedding_text
from travel_ai_search.rag.retriever import KnowledgeRetriever
from travel_ai_search.rag.synthesizer import RAGSynthesizer
from travel_ai_search.retrieval.types import Hit

# ── Fixtures ──────────────────────────────────────────────────────────────────

MALLORCA: dict[str, object] = {
    "id": "mallorca",
    "destination": "Mallorca",
    "country": "Spain",
    "region": "Balearic Islands",
    "description": "Mallorca blends mass tourism with dramatic mountains and cycling routes.",
    "climate": "Classic Mediterranean: hot, dry summers and mild winters.",
    "best_months": ["May", "June", "September", "October"],
    "family_suitability": "high",
    "nightlife_level": "moderate",
    "beach_quality": "excellent",
    "activities": ["cycling", "hiking", "sailing", "scuba diving"],
    "character_tags": ["Mediterranean classic", "cycling mecca", "family resort"],
    "similar_destinations": ["Ibiza", "Menorca"],
    "geographic_note": "Largest Balearic Island.",
}

IBIZA: dict[str, object] = {
    "id": "ibiza",
    "destination": "Ibiza",
    "country": "Spain",
    "region": "Balearic Islands",
    "description": "Ibiza is globally synonymous with nightlife and bohemian culture.",
    "climate": "Hot Mediterranean summers and mild winters.",
    "best_months": ["June", "July", "August", "September"],
    "family_suitability": "low",
    "nightlife_level": "high",
    "beach_quality": "excellent",
    "activities": ["clubbing", "beach clubs", "sailing"],
    "character_tags": ["nightlife capital", "party island", "luxury"],
    "similar_destinations": ["Mykonos", "Mallorca"],
    "geographic_note": "Third largest Balearic Island.",
}


def make_destination(**overrides: object) -> DestinationKnowledge:
    return DestinationKnowledge.model_validate({**MALLORCA, **overrides})


def make_hit(
    *,
    hotel_name: str = "Test Hotel",
    destination: str = "Mallorca",
    description: str = "A lovely hotel.",
) -> Hit:
    return Hit(
        id="h1",
        score=0.9,
        source={
            "hotel_name": hotel_name,
            "destination": destination,
            "hotel_description": description,
        },
    )


# ── DestinationKnowledge ──────────────────────────────────────────────────────


class TestDestinationKnowledge:
    def test_model_validate_from_dict(self) -> None:
        dk = DestinationKnowledge.model_validate(MALLORCA)
        assert dk.destination == "Mallorca"
        assert dk.country == "Spain"

    def test_all_required_fields_present(self) -> None:
        dk = DestinationKnowledge.model_validate(MALLORCA)
        assert dk.id == "mallorca"
        assert dk.region == "Balearic Islands"
        assert dk.family_suitability == "high"
        assert dk.nightlife_level == "moderate"
        assert dk.beach_quality == "excellent"
        assert dk.activities == ["cycling", "hiking", "sailing", "scuba diving"]

    def test_extra_fields_ignored(self) -> None:
        dk = DestinationKnowledge.model_validate({**MALLORCA, "unknown_field": "ignored"})
        assert dk.destination == "Mallorca"
        assert not hasattr(dk, "unknown_field")

    def test_similar_destinations_is_list(self) -> None:
        dk = DestinationKnowledge.model_validate(MALLORCA)
        assert isinstance(dk.similar_destinations, list)
        assert "Ibiza" in dk.similar_destinations


class TestBuildKnowledgeEmbeddingText:
    def test_includes_destination_name(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "Mallorca" in text

    def test_includes_country(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "Spain" in text

    def test_includes_description(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "cycling routes" in text

    def test_includes_climate(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "Mediterranean" in text

    def test_includes_activities(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "cycling" in text
        assert "sailing" in text

    def test_includes_character_tags(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "cycling mecca" in text

    def test_includes_family_and_nightlife(self) -> None:
        dk = make_destination()
        text = build_knowledge_embedding_text(dk)
        assert "high" in text  # family_suitability
        assert "moderate" in text  # nightlife_level

    def test_excludes_geographic_note(self) -> None:
        dk = make_destination(geographic_note="Unique fact not for embedding")
        text = build_knowledge_embedding_text(dk)
        assert "Unique fact not for embedding" not in text

    def test_excludes_similar_destinations(self) -> None:
        dk = make_destination(similar_destinations=["ShouldNotAppear"])
        text = build_knowledge_embedding_text(dk)
        assert "ShouldNotAppear" not in text

    def test_returns_non_empty_string(self) -> None:
        dk = make_destination()
        assert len(build_knowledge_embedding_text(dk)) > 50


# ── KnowledgeRetriever ────────────────────────────────────────────────────────


class TestKnowledgeRetriever:
    def _make_retriever(
        self,
        search_hits: list[dict[str, object]] | None = None,
        *,
        top_k: int = 3,
    ) -> tuple[KnowledgeRetriever, MagicMock, MagicMock]:
        mock_client = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [0.1] * 384

        hits = search_hits if search_hits is not None else [{"_source": MALLORCA}]
        mock_client.search.return_value = {"hits": {"hits": hits}}

        retriever = KnowledgeRetriever(mock_client, mock_provider, top_k=top_k)
        return retriever, mock_client, mock_provider

    def test_calls_embed_with_query(self) -> None:
        retriever, _, mock_provider = self._make_retriever()
        retriever.retrieve("beach holiday in Spain")
        mock_provider.embed.assert_called_once_with("beach holiday in Spain")

    def test_calls_search_on_client(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        retriever.retrieve("beach holiday")
        mock_client.search.assert_called_once()

    def test_search_uses_knn_query(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        retriever.retrieve("beach holiday")
        body = mock_client.search.call_args.kwargs["body"]
        assert "knn" in body["query"]
        assert "embedding_vector" in body["query"]["knn"]

    def test_knn_clause_contains_vector(self) -> None:
        retriever, mock_client, mock_provider = self._make_retriever()
        mock_provider.embed.return_value = [0.5] * 384
        retriever.retrieve("beach holiday")
        body = mock_client.search.call_args.kwargs["body"]
        knn_clause = body["query"]["knn"]["embedding_vector"]
        assert knn_clause["vector"] == [0.5] * 384

    def test_knn_clause_contains_k(self) -> None:
        retriever, mock_client, _ = self._make_retriever(top_k=5)
        retriever.retrieve("beach holiday")
        body = mock_client.search.call_args.kwargs["body"]
        knn_clause = body["query"]["knn"]["embedding_vector"]
        assert knn_clause["k"] == 5

    def test_no_country_filter_by_default(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        retriever.retrieve("beach holiday")
        body = mock_client.search.call_args.kwargs["body"]
        knn_clause = body["query"]["knn"]["embedding_vector"]
        assert "filter" not in knn_clause

    def test_country_filter_added_when_provided(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        retriever.retrieve("beach holiday", country="Spain")
        body = mock_client.search.call_args.kwargs["body"]
        knn_clause = body["query"]["knn"]["embedding_vector"]
        assert knn_clause["filter"] == {"term": {"country": "Spain"}}

    def test_returns_list_of_destination_knowledge(self) -> None:
        retriever, _, _ = self._make_retriever([{"_source": MALLORCA}])
        result = retriever.retrieve("beach holiday")
        assert len(result) == 1
        assert isinstance(result[0], DestinationKnowledge)
        assert result[0].destination == "Mallorca"

    def test_empty_hits_returns_empty_list(self) -> None:
        retriever, _, _ = self._make_retriever([])
        result = retriever.retrieve("beach holiday")
        assert result == []

    def test_multiple_hits_returned(self) -> None:
        retriever, _, _ = self._make_retriever([{"_source": MALLORCA}, {"_source": IBIZA}])
        result = retriever.retrieve("beach holiday")
        assert len(result) == 2
        assert result[0].destination == "Mallorca"
        assert result[1].destination == "Ibiza"

    def test_embedding_vector_excluded_from_source(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        retriever.retrieve("beach holiday")
        body = mock_client.search.call_args.kwargs["body"]
        assert body["_source"]["excludes"] == ["embedding_vector"]

    def test_exception_from_client_propagates(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        mock_client.search.side_effect = RuntimeError("index not found")
        with pytest.raises(RuntimeError, match="index not found"):
            retriever.retrieve("beach holiday")

    def test_uses_custom_index_name(self) -> None:
        retriever, mock_client, _ = self._make_retriever()
        retriever._index = "custom_index"
        retriever.retrieve("beach holiday")
        assert mock_client.search.call_args.kwargs["index"] == "custom_index"


# ── RAGSynthesizer ────────────────────────────────────────────────────────────


class TestRAGSynthesizer:
    def _make_synthesizer(
        self, llm_response: str = "A great match."
    ) -> tuple[RAGSynthesizer, MagicMock]:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_response
        synthesizer = RAGSynthesizer(mock_llm)
        return synthesizer, mock_llm

    def test_returns_string(self) -> None:
        synth, _ = self._make_synthesizer("Great recommendation.")
        result = synth.synthesize("beach holiday", [make_hit()], [make_destination()])
        assert isinstance(result, str)
        assert result == "Great recommendation."

    def test_calls_llm_generate(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach holiday", [make_hit()], [make_destination()])
        mock_llm.generate.assert_called_once()

    def test_prompt_contains_query(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("romantic sunset holiday", [make_hit()], [make_destination()])
        prompt = mock_llm.generate.call_args.args[0]
        assert "romantic sunset holiday" in prompt

    def test_prompt_contains_destination_name(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach", [make_hit()], [make_destination(destination="Santorini")])
        prompt = mock_llm.generate.call_args.args[0]
        assert "Santorini" in prompt

    def test_prompt_contains_hotel_name(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach", [make_hit(hotel_name="Azure Palace")], [make_destination()])
        prompt = mock_llm.generate.call_args.args[0]
        assert "Azure Palace" in prompt

    def test_uses_system_prompt(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach", [make_hit()], [make_destination()])
        system = mock_llm.generate.call_args.kwargs.get("system", "")
        assert len(system) > 0
        assert "travel" in system.lower()

    def test_empty_knowledge_still_calls_llm(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach", [make_hit()], [])
        mock_llm.generate.assert_called_once()

    def test_empty_hits_still_calls_llm(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach", [], [make_destination()])
        mock_llm.generate.assert_called_once()

    def test_limits_hotels_to_max_hotels(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth._max_hotels = 2
        many_hits = [make_hit(hotel_name=f"Hotel {i}") for i in range(10)]
        synth.synthesize("beach", many_hits, [make_destination()])
        prompt = mock_llm.generate.call_args.args[0]
        assert "Hotel 0" in prompt
        assert "Hotel 1" in prompt
        assert "Hotel 2" not in prompt

    def test_knowledge_country_in_prompt(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        synth.synthesize("beach", [make_hit()], [make_destination(country="Portugal")])
        prompt = mock_llm.generate.call_args.args[0]
        assert "Portugal" in prompt

    def test_exception_from_llm_propagates(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        mock_llm.generate.side_effect = RuntimeError("LLM unavailable")
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            synth.synthesize("beach", [make_hit()], [make_destination()])

    def test_hotel_description_truncated_in_prompt(self) -> None:
        synth, mock_llm = self._make_synthesizer()
        long_desc = "A" * 300
        synth.synthesize("beach", [make_hit(description=long_desc)], [make_destination()])
        prompt = mock_llm.generate.call_args.args[0]
        # The full 300-char description should not appear verbatim
        assert long_desc not in prompt
        # But a truncated version should
        assert "A" * 50 in prompt
