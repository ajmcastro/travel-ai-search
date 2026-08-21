"""Unit tests for Milestone 17: learned sparse retrieval (SPLADE).

Tests are grouped into:
  - _clean_token: pure token normalisation
  - SpladeSearchParams / SpladeSearchResult: dataclass contracts
  - _build_splade_query: pure query-body construction
  - LocalSparseProvider (mocked): SPLADE encoding logic
  - LearnedSparseProvider protocol: structural typing
  - splade_search (mocked client): end-to-end retrieval function
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import torch

from travel_ai_search.embeddings.sparse import (
    LearnedSparseProvider,
    LocalSparseProvider,
    _clean_token,
)
from travel_ai_search.retrieval.splade import (
    SpladeSearchParams,
    SpladeSearchResult,
    _build_splade_query,
    splade_search,
)
from travel_ai_search.retrieval.types import Hit

# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeBatch(dict):
    """dict-like BatchEncoding stub that supports .to(device)."""

    def to(self, device: str) -> _FakeBatch:
        return self


def _make_provider(
    logits: torch.Tensor,
    decode_map: dict[int, str],
    *,
    top_k: int = 128,
) -> LocalSparseProvider:
    """Build a LocalSparseProvider whose transformers components are mocked."""
    batch_size, seq_len, _ = logits.shape
    fake_inputs = _FakeBatch(
        {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }
    )

    with (
        patch("travel_ai_search.embeddings.sparse.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "travel_ai_search.embeddings.sparse.AutoModelForMaskedLM.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_tok = MagicMock()
        mock_tok.return_value = fake_inputs
        mock_tok.decode.side_effect = lambda ids, **_: decode_map.get(
            ids[0] if isinstance(ids, (list, tuple)) else int(ids), "[UNK]"
        )
        mock_tok_cls.return_value = mock_tok

        mock_model = MagicMock()
        mock_model.return_value = MagicMock(logits=logits)
        mock_model_cls.return_value = mock_model

        # Patches only needed during __init__; stored references outlive the block.
        return LocalSparseProvider("fake-model", top_k=top_k)


def _simple_provider(top_k: int = 128) -> LocalSparseProvider:
    """Provider with two non-zero terms: beach=2.0, resort=1.0 at position 0."""
    vocab_size = 600
    logits = torch.zeros(1, 3, vocab_size)
    logits[0, 0, 100] = 2.0  # "beach"  → log(1+2) ≈ 1.099
    logits[0, 1, 200] = 1.0  # "resort" → log(1+1) ≈ 0.693
    logits[0, 2, 300] = -1.0  # negative → ReLU → 0, filtered out
    decode_map = {100: "beach", 200: "resort", 300: "badterm"}
    return _make_provider(logits, decode_map, top_k=top_k)


# ── _clean_token ──────────────────────────────────────────────────────────────


class TestCleanToken:
    def test_plain_word_kept(self) -> None:
        assert _clean_token("beach") == "beach"

    def test_uppercase_lowercased(self) -> None:
        assert _clean_token("Beach") == "beach"

    def test_subword_hash_prefix_dropped(self) -> None:
        assert _clean_token("##ing") == ""

    def test_double_hash_subword_dropped(self) -> None:
        # BERT tokenizer uses ## for subword continuation — must be excluded.
        assert _clean_token("##resort") == ""

    def test_special_token_dropped(self) -> None:
        assert _clean_token("[CLS]") == ""

    def test_single_character_dropped(self) -> None:
        assert _clean_token("a") == ""

    def test_empty_string_dropped(self) -> None:
        assert _clean_token("") == ""

    def test_punctuation_dropped(self) -> None:
        assert _clean_token(".") == ""

    def test_alphanumeric_with_digit_kept(self) -> None:
        # e.g. "co2" is a valid term starting with a letter
        assert _clean_token("co2") == "co2"

    def test_digit_first_token_dropped(self) -> None:
        # Pattern requires [a-z] as first character
        assert _clean_token("4g") == ""

    def test_hyphen_dropped(self) -> None:
        # Hyphens are not in the allowed character set
        assert _clean_token("adults-only") == ""

    def test_whitespace_stripped(self) -> None:
        assert _clean_token("  resort  ") == "resort"


# ── SpladeSearchParams ────────────────────────────────────────────────────────


class TestSpladeSearchParams:
    def test_defaults(self) -> None:
        params = SpladeSearchParams(query="beach hotel")
        assert params.top_k == 10
        assert params.top_k_terms == 64
        assert params.country is None

    def test_custom_values(self) -> None:
        params = SpladeSearchParams(
            query="q", top_k=20, top_k_terms=128, country="Spain", family_friendly=True
        )
        assert params.top_k == 20
        assert params.top_k_terms == 128
        assert params.country == "Spain"
        assert params.family_friendly is True

    def test_all_filter_fields_present(self) -> None:
        params = SpladeSearchParams(
            query="q",
            country="Greece",
            destination="Santorini",
            family_friendly=False,
            adults_only=True,
            min_star_rating=4,
            max_price=2000.0,
            month="August",
            airport="LHR",
        )
        assert params.destination == "Santorini"
        assert params.adults_only is True
        assert params.airport == "LHR"


# ── SpladeSearchResult ────────────────────────────────────────────────────────


class TestSpladeSearchResult:
    def test_empty_result(self) -> None:
        result = SpladeSearchResult(hits=[], total=0, took_ms=0, n_query_terms=0)
        assert result.hits == []
        assert result.n_query_terms == 0

    def test_with_hits(self) -> None:
        hit = Hit(id="hotel-1", score=1.5, source={"hotel_name": "Beach Resort"})
        result = SpladeSearchResult(hits=[hit], total=1, took_ms=12, n_query_terms=38)
        assert len(result.hits) == 1
        assert result.n_query_terms == 38


# ── _build_splade_query ───────────────────────────────────────────────────────


class TestBuildSpladeQuery:
    def _params(self, **kwargs: Any) -> SpladeSearchParams:
        return SpladeSearchParams(query="test", **kwargs)

    def test_should_clauses_present(self) -> None:
        vec = {"beach": 0.82, "resort": 0.61}
        body = _build_splade_query(vec, self._params())
        should = body["query"]["bool"]["should"]
        assert len(should) == 2

    def test_field_naming(self) -> None:
        vec = {"beach": 0.82}
        body = _build_splade_query(vec, self._params(), splade_field="splade_vector")
        clause = body["query"]["bool"]["should"][0]["rank_feature"]
        assert clause["field"] == "splade_vector.beach"

    def test_boost_equals_query_weight(self) -> None:
        vec = {"beach": 0.82}
        body = _build_splade_query(vec, self._params())
        boost = body["query"]["bool"]["should"][0]["rank_feature"]["boost"]
        assert abs(boost - 0.82) < 1e-6

    def test_minimum_should_match_one(self) -> None:
        vec = {"beach": 0.5}
        body = _build_splade_query(vec, self._params())
        assert body["query"]["bool"]["minimum_should_match"] == 1

    def test_top_k_terms_truncation(self) -> None:
        vec = {f"term{i}": float(i) for i in range(100)}
        params = self._params(top_k_terms=10)
        body = _build_splade_query(vec, params)
        should = body["query"]["bool"]["should"]
        assert len(should) == 10

    def test_truncation_keeps_highest_weights(self) -> None:
        vec = {"low": 0.1, "medium": 0.5, "high": 0.9}
        params = self._params(top_k_terms=2)
        body = _build_splade_query(vec, params)
        clauses = body["query"]["bool"]["should"]
        terms = {c["rank_feature"]["field"].split(".")[-1] for c in clauses}
        assert "high" in terms
        assert "medium" in terms
        assert "low" not in terms

    def test_size_set_from_params(self) -> None:
        vec = {"beach": 0.5}
        params = self._params(top_k=25)
        body = _build_splade_query(vec, params)
        assert body["size"] == 25

    def test_splade_field_excluded_from_source(self) -> None:
        vec = {"beach": 0.5}
        body = _build_splade_query(vec, self._params(), splade_field="splade_vector")
        excludes = body["_source"]["excludes"]
        assert "splade_vector" in excludes

    def test_filter_clause_added_when_country_set(self) -> None:
        vec = {"beach": 0.5}
        params = self._params(country="Spain")
        body = _build_splade_query(vec, params)
        filter_clauses = body["query"]["bool"].get("filter", [])
        assert any(c.get("term", {}).get("country") == "Spain" for c in filter_clauses)

    def test_no_filter_clause_when_no_filters(self) -> None:
        vec = {"beach": 0.5}
        body = _build_splade_query(vec, self._params())
        assert "filter" not in body["query"]["bool"]

    def test_custom_splade_field_name(self) -> None:
        vec = {"beach": 0.5}
        body = _build_splade_query(vec, self._params(), splade_field="my_sparse")
        clause = body["query"]["bool"]["should"][0]["rank_feature"]
        assert clause["field"].startswith("my_sparse.")
        assert "splade_vector" not in body["_source"]["excludes"]
        assert "my_sparse" in body["_source"]["excludes"]


# ── LocalSparseProvider ───────────────────────────────────────────────────────


class TestLocalSparseProvider:
    def test_encode_returns_dict(self) -> None:
        provider = _simple_provider()
        result = provider.encode("beach hotel")
        assert isinstance(result, dict)

    def test_encode_all_values_positive(self) -> None:
        provider = _simple_provider()
        result = provider.encode("test")
        assert all(v > 0 for v in result.values())

    def test_negative_logit_filtered_by_relu(self) -> None:
        provider = _simple_provider()
        result = provider.encode("test")
        # "badterm" (token 300) has logit -1.0 → ReLU → 0 → excluded
        assert "badterm" not in result

    def test_known_tokens_present(self) -> None:
        provider = _simple_provider()
        result = provider.encode("test")
        assert "beach" in result
        assert "resort" in result

    def test_weights_match_splade_formula(self) -> None:
        # logit=2.0 → log(1+ReLU(2.0)) = log(3) ≈ 1.0986
        # logit=1.0 → log(1+ReLU(1.0)) = log(2) ≈ 0.6931
        provider = _simple_provider()
        result = provider.encode("test")
        assert abs(result["beach"] - 1.0986) < 1e-3
        assert abs(result["resort"] - 0.6931) < 1e-3

    def test_top_k_limits_output_size(self) -> None:
        # Create provider with top_k=1 — only the highest-weight term survives.
        provider = _simple_provider(top_k=1)
        result = provider.encode("test")
        assert len(result) <= 1
        # "beach" (weight ≈ 1.099) outweighs "resort" (weight ≈ 0.693)
        assert list(result.keys()) == ["beach"]

    def test_encode_batch_returns_list(self) -> None:
        # The mock uses batch_size=1 logits; test with a single text to match the fixture.
        provider = _simple_provider()
        results = provider.encode_batch(["beach hotel"])
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_model_id_stored(self) -> None:
        provider = _simple_provider()
        assert provider.model_id == "fake-model"

    def test_unk_tokens_filtered_out(self) -> None:
        # Tokens not in decode_map decode to "[UNK]", which fails _clean_token.
        vocab_size = 600
        logits = torch.zeros(1, 3, vocab_size)
        logits[0, 0, 500] = 1.5  # token 500 not in decode_map → "[UNK]"
        provider = _make_provider(logits, decode_map={}, top_k=10)
        result = provider.encode("test")
        assert result == {}


# ── LearnedSparseProvider Protocol ───────────────────────────────────────────


class TestLearnedSparseProviderProtocol:
    def test_local_sparse_provider_satisfies_protocol(self) -> None:
        provider = _simple_provider()
        assert isinstance(provider, LearnedSparseProvider)

    def test_protocol_requires_model_id(self) -> None:
        class _GoodStub:
            model_id = "test"

            def encode(self, text: str) -> dict[str, float]:
                return {}

            def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
                return [{}] * len(texts)

        assert isinstance(_GoodStub(), LearnedSparseProvider)

    def test_protocol_not_satisfied_without_methods(self) -> None:
        class _BadStub:
            model_id = "test"

        assert not isinstance(_BadStub(), LearnedSparseProvider)


# ── splade_search ─────────────────────────────────────────────────────────────


def _mock_os_response(hits: list[dict[str, Any]], total: int = None) -> dict[str, Any]:
    if total is None:
        total = len(hits)
    return {
        "hits": {
            "total": {"value": total},
            "hits": [
                {
                    "_id": h["id"],
                    "_score": h.get("score", 1.0),
                    "_source": h.get("source", {"hotel_name": "Test Hotel"}),
                }
                for h in hits
            ],
        }
    }


class TestSpladeSearch:
    def _provider_stub(self, vec: dict[str, float]) -> MagicMock:
        stub = MagicMock()
        stub.encode.return_value = vec
        return stub

    def test_returns_hits_from_opensearch(self) -> None:
        client = MagicMock()
        client.search.return_value = _mock_os_response(
            [{"id": "hotel-1", "score": 0.9, "source": {"hotel_name": "Sunny Beach"}}]
        )
        provider = self._provider_stub({"beach": 0.8, "sunny": 0.5})
        result = splade_search(client, provider, SpladeSearchParams(query="beach"), index="idx")
        assert len(result.hits) == 1
        assert result.hits[0].id == "hotel-1"

    def test_n_query_terms_reported(self) -> None:
        client = MagicMock()
        client.search.return_value = _mock_os_response([])
        provider = self._provider_stub({f"term{i}": float(i) for i in range(80)})
        params = SpladeSearchParams(query="q", top_k_terms=64)
        result = splade_search(client, provider, params, index="idx")
        assert result.n_query_terms == 64  # capped at top_k_terms

    def test_empty_sparse_vector_returns_early(self) -> None:
        client = MagicMock()
        provider = self._provider_stub({})
        result = splade_search(client, provider, SpladeSearchParams(query=""), index="idx")
        assert result.hits == []
        assert result.total == 0
        assert result.n_query_terms == 0
        client.search.assert_not_called()

    def test_took_ms_is_non_negative(self) -> None:
        client = MagicMock()
        client.search.return_value = _mock_os_response([])
        provider = self._provider_stub({"beach": 0.5})
        result = splade_search(client, provider, SpladeSearchParams(query="q"), index="idx")
        assert result.took_ms >= 0
