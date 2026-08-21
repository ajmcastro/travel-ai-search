"""Learned sparse embedding (SPLADE) provider — Milestone 17.

SPLADE uses a BERT masked-language model head to produce sparse vocabulary-weight
vectors.  Unlike BM25, which counts exact term occurrences, SPLADE learns which
vocabulary terms are semantically relevant to a text — enabling vocabulary expansion:
"resort" can be activated by "hotel", "aquatic" by "pool", even with no lexical overlap.

Weight formula for vocabulary term t given input text:
    w_t = max_{position i} log(1 + ReLU(MLM_logit_{i,t}))

The ReLU ensures sparsity; the log compresses large values; the max pools across
positions so the strongest activation of any term anywhere in the sequence is kept.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@runtime_checkable
class LearnedSparseProvider(Protocol):
    """Protocol for learned sparse encoders (analogous to EmbeddingProvider).

    Returns dict[str, float] (decoded vocabulary token → learned weight) instead
    of list[float].  Zero-weight terms are absent from the dict.
    """

    model_id: str

    def encode(self, text: str) -> dict[str, float]: ...

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]: ...


class LocalSparseProvider:
    """SPLADE sparse encoder backed by a local HuggingFace masked-language model.

    At init time the model is downloaded from the HuggingFace Hub if not already
    cached (~300 MB for the distil variant on first run; cached in ~/.cache/huggingface/).

    Args:
        model_name: HuggingFace model ID for any BERT-based MLM, e.g.
            ``"naver/splade-cocondenser-ensemble-distil"``.
        top_k: Maximum number of non-zero vocabulary terms to keep per text.
            Sorted by descending weight so the most informative terms are retained.
            Capped at query time to limit OpenSearch fan-out.
        device: Torch device string (``"cpu"``, ``"cuda"``); auto-detected when None.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensemble-distil",
        *,
        top_k: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_id = model_name
        self._top_k = top_k
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)

    def encode(self, text: str) -> dict[str, float]:
        """Encode a single text to a sparse vocabulary-weight dict."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Encode a list of texts; batching amortises tokeniser and GPU overhead."""
        inputs: Any = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (batch, seq_len, vocab_size)

        # Zero out padding positions before aggregating across the sequence.
        # attention_mask is 0 for padding tokens, 1 for real tokens.
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sparse = torch.log1p(torch.relu(logits)) * mask  # (batch, seq_len, vocab)
        sparse = sparse.max(dim=1).values  # (batch, vocab_size)

        return [self._vec_to_dict(sparse[i]) for i in range(len(texts))]

    def _vec_to_dict(self, vec: torch.Tensor) -> dict[str, float]:
        """Convert a (vocab_size,) sparse weight tensor to a filtered {token: weight} dict."""
        non_zero_idx = vec.nonzero().squeeze(-1)
        if non_zero_idx.numel() == 0:
            return {}

        vals = vec[non_zero_idx]

        # Keep only the top_k terms by weight for OpenSearch query efficiency.
        if non_zero_idx.numel() > self._top_k:
            top_pos = vals.argsort(descending=True)[: self._top_k]
            non_zero_idx = non_zero_idx[top_pos]
            vals = vals[top_pos]

        result: dict[str, float] = {}
        for idx, val in zip(non_zero_idx.tolist(), vals.tolist()):
            token = str(self._tokenizer.decode([idx])).strip()
            cleaned = _clean_token(token)
            if cleaned:
                result[cleaned] = float(val)
        return result


_VALID_KEY: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9]*$")


def _clean_token(token: str) -> str:
    """Normalise a decoded BERT token for use as an OpenSearch rank_features key.

    Keeps only lowercase alphabetic-first tokens (≥ 2 chars).  Drops subwords
    (##prefix), special tokens ([CLS], [SEP]), punctuation, and short tokens.
    The restriction to [a-z][a-z0-9]* avoids dot-separated field names that
    OpenSearch would interpret as nested field paths.
    """
    token = token.strip().lower()
    if token.startswith("#"):
        return ""  # BERT subword continuation token (e.g. ##ing, ##resort)
    return token if len(token) >= 2 and bool(_VALID_KEY.match(token)) else ""
