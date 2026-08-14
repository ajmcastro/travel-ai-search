"""Unit tests for Milestone 10 — query rewriting.

Covers:
  - LLMProvider protocol satisfaction (EchoLLMProvider, LocalLLMProvider)
  - EchoLLMProvider behaviour
  - LocalLLMProvider keyword-expansion behaviour
  - QueryRewriter — happy path, error fallback, edge cases
"""

from __future__ import annotations

from travel_ai_search.llm.base import LLMProvider
from travel_ai_search.llm.local import _MAX_ADDITIONS, EchoLLMProvider, LocalLLMProvider
from travel_ai_search.query_understanding.rewriter import QueryRewriter

# ── Protocol satisfaction ──────────────────────────────────────────────────────


class TestLLMProviderProtocol:
    def test_echo_satisfies_protocol(self) -> None:
        assert isinstance(EchoLLMProvider(), LLMProvider)

    def test_local_satisfies_protocol(self) -> None:
        assert isinstance(LocalLLMProvider(), LLMProvider)

    def test_custom_class_satisfies_protocol(self) -> None:
        class _Custom:
            def generate(self, prompt: str, *, system: str = "") -> str:
                return prompt + " custom"

        assert isinstance(_Custom(), LLMProvider)

    def test_class_without_generate_fails_protocol(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), LLMProvider)


# ── EchoLLMProvider ────────────────────────────────────────────────────────────


class TestEchoLLMProvider:
    def test_returns_prompt_unchanged(self) -> None:
        p = EchoLLMProvider()
        assert p.generate("beach holiday in Spain") == "beach holiday in Spain"

    def test_returns_empty_string_unchanged(self) -> None:
        p = EchoLLMProvider()
        assert p.generate("") == ""

    def test_ignores_system_prompt(self) -> None:
        p = EchoLLMProvider()
        result = p.generate("spa resort", system="You are a travel expert.")
        assert result == "spa resort"

    def test_whitespace_preserved(self) -> None:
        p = EchoLLMProvider()
        assert p.generate("  beach  ") == "  beach  "


# ── LocalLLMProvider ───────────────────────────────────────────────────────────


class TestLocalLLMProvider:
    def setup_method(self) -> None:
        self.llm = LocalLLMProvider()

    def test_known_keyword_gets_synonym(self) -> None:
        result = self.llm.generate("quiet resort")
        assert result.startswith("quiet resort")
        # At least one synonym appended
        assert len(result) > len("quiet resort")

    def test_beach_gets_synonym(self) -> None:
        result = self.llm.generate("beach holiday")
        assert "beach" in result
        # A synonym from the beach list should appear
        has_synonym = any(s in result for s in ["coastal", "seaside", "sandy beach"])
        assert has_synonym

    def test_spa_gets_synonym(self) -> None:
        result = self.llm.generate("spa retreat")
        has_synonym = any(s in result for s in ["wellness", "thalassotherapy"])
        assert has_synonym

    def test_no_duplicate_terms(self) -> None:
        # If the synonym is already in the query, it should not be appended again
        result = self.llm.generate("quiet peaceful resort")
        # "peaceful" is the first synonym for "quiet"; it's already present
        # so it should not be added again
        assert result.count("peaceful") == 1

    def test_at_most_max_additions(self) -> None:
        # A query with many keywords should not exceed _MAX_ADDITIONS synonyms
        query = "beach quiet family spa luxury pool adventure"
        result = self.llm.generate(query)
        extra = result[len(query) :]
        # Count appended words; multi-word synonyms count as multiple tokens
        # but there is at most _MAX_ADDITIONS synonyms, each at most ~3 words.
        n_words = len(extra.strip().split()) if extra.strip() else 0
        assert n_words <= _MAX_ADDITIONS * 3

    def test_unknown_keywords_return_original(self) -> None:
        query = "hotel near the train station"
        result = self.llm.generate(query)
        assert result == query

    def test_empty_prompt_returns_empty(self) -> None:
        result = self.llm.generate("")
        assert result == ""

    def test_preserves_original_at_start(self) -> None:
        query = "luxury hotel"
        result = self.llm.generate(query)
        assert result.startswith(query)

    def test_system_prompt_ignored(self) -> None:
        result_no_sys = self.llm.generate("beach resort")
        result_with_sys = self.llm.generate("beach resort", system="You are an expert.")
        assert result_no_sys == result_with_sys

    def test_family_gets_synonym(self) -> None:
        result = self.llm.generate("family holiday")
        has_synonym = any(s in result for s in ["family-friendly", "children", "kids"])
        assert has_synonym

    def test_adults_gets_synonym(self) -> None:
        result = self.llm.generate("adults retreat")
        has_synonym = any(s in result for s in ["adults-only", "couples"])
        assert has_synonym


# ── QueryRewriter ──────────────────────────────────────────────────────────────


class TestQueryRewriter:
    def test_uses_llm_output(self) -> None:
        class _MockLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                return "expanded " + prompt

        rewriter = QueryRewriter(_MockLLM())
        result = rewriter.rewrite("beach holiday")
        assert result == "expanded beach holiday"

    def test_falls_back_on_llm_exception(self) -> None:
        class _FailingLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                raise RuntimeError("API unavailable")

        rewriter = QueryRewriter(_FailingLLM())
        result = rewriter.rewrite("beach holiday")
        assert result == "beach holiday"

    def test_falls_back_on_empty_llm_output(self) -> None:
        class _EmptyLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                return ""

        rewriter = QueryRewriter(_EmptyLLM())
        result = rewriter.rewrite("beach holiday")
        assert result == "beach holiday"

    def test_falls_back_on_whitespace_only_llm_output(self) -> None:
        class _WhitespaceLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                return "   "

        rewriter = QueryRewriter(_WhitespaceLLM())
        result = rewriter.rewrite("beach holiday")
        assert result == "beach holiday"

    def test_returns_original_for_empty_query(self) -> None:
        rewriter = QueryRewriter(EchoLLMProvider())
        assert rewriter.rewrite("") == ""

    def test_returns_original_for_whitespace_query(self) -> None:
        rewriter = QueryRewriter(EchoLLMProvider())
        assert rewriter.rewrite("   ") == "   "

    def test_strips_llm_output_whitespace(self) -> None:
        class _PaddedLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                return "  expanded beach  "

        rewriter = QueryRewriter(_PaddedLLM())
        result = rewriter.rewrite("beach")
        assert result == "expanded beach"

    def test_echo_provider_produces_identity_rewrite(self) -> None:
        rewriter = QueryRewriter(EchoLLMProvider())
        query = "spa resort near the coast"
        assert rewriter.rewrite(query) == query

    def test_local_provider_expands_query(self) -> None:
        rewriter = QueryRewriter(LocalLLMProvider())
        result = rewriter.rewrite("beach resort")
        # LocalLLMProvider should add at least one synonym
        assert len(result) > len("beach resort")

    def test_exception_logging_does_not_propagate(self) -> None:
        class _NastyLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                raise ValueError("internal error")

        rewriter = QueryRewriter(_NastyLLM())
        # Must not raise
        result = rewriter.rewrite("spa resort")
        assert result == "spa resort"

    def test_rewriter_passes_system_prompt_to_llm(self) -> None:
        received: list[str] = []

        class _CaptureLLM:
            def generate(self, prompt: str, *, system: str = "") -> str:
                received.append(system)
                return prompt

        QueryRewriter(_CaptureLLM()).rewrite("beach")
        assert received, "generate() was not called"
        assert len(received[0]) > 0, "system prompt should not be empty"
