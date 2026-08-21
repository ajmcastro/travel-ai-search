"""ColBERT late-interaction reranker — Milestone 18.

ColBERT (Khattab & Zaharia, 2020) retains all token-level embeddings rather
than collapsing to a single vector.  Relevance is computed via MaxSim:

    Score(q, d) = Σᵢ  max_j  (qᵢ · dⱼ)

For each query token embedding qᵢ, find the maximum dot product across all
document token embeddings dⱼ, then sum.  This lets "family" seek the childcare
paragraph and "beach" seek the waterfront section independently — without the
compression bottleneck of a single-vector bi-encoder.

This is *late interaction*: query and document are encoded independently (like
a bi-encoder — so document embeddings can be pre-computed offline) but they
interact at the token level at query time (more expressive than a single dot
product).

Two-stage pipeline (used by evaluate.py --strategy colbert):
  1. Candidate generation — ANN + RRF retrieves top-K hits (fast).
  2. ColBERT re-scoring — MaxSim re-ranks the candidates (cheap matrix ops).

Storage: per-document token embeddings saved as numpy arrays at
  data/processed/colbert_embeddings/<hotel_id>.npy
  shape: (d_tokens, dim) float32, L2-normalised per token.

Run ``make generate-colbert-embeddings`` before using the reranker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from travel_ai_search.reranking.local import _build_reranking_text
from travel_ai_search.retrieval.types import Hit

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "colbert-ir/colbertv2.0"
_QUERY_MAXLEN = 32
_DOC_MAXLEN = 128


# ── Core scoring function ──────────────────────────────────────────────────────


def maxsim(query_embs: np.ndarray, doc_embs: np.ndarray) -> float:
    """Compute the ColBERT MaxSim score between query and document token embeddings.

    For each query token, find the maximum dot product against all document
    tokens; sum across query tokens.  When embeddings are L2-normalised the
    dot product equals cosine similarity.

    Args:
        query_embs: float32 array of shape (q_len, dim), one row per query token.
        doc_embs:   float32 array of shape (d_len, dim), one row per doc token.

    Returns:
        Scalar ColBERT score (higher = more relevant).
    """
    # (q_len, d_len) similarity matrix
    sims = query_embs @ doc_embs.T
    # max over document tokens for each query token, then sum
    return float(sims.max(axis=1).sum())


# ── Encoder ───────────────────────────────────────────────────────────────────


class ColBERTEncoder:
    """Token-level encoder that produces L2-normalised embeddings per token.

    Works with any BERT-based HuggingFace model.  When the checkpoint includes
    a ``linear.weight`` key (as in ``colbert-ir/colbertv2.0``), the token
    embeddings are projected from the model's hidden size to a smaller ColBERT
    dimension (128 for v2).  When the projection is absent, the full hidden
    state (e.g. 384-dim for MiniLM) is used as-is.

    For quick local testing without downloading a new model, set
    ``COLBERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`` in .env —
    the already-cached MiniLM weights are used; embeddings will be 384-dim
    (larger storage, same MaxSim mechanism).

    Args:
        model_name: HuggingFace model identifier.
        query_maxlen: Query tokens (truncated/padded to this; ColBERT v2 = 32).
        doc_maxlen: Document tokens (truncated to this; ColBERT v2 = 128).
        device: Torch device; auto-detected when None.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        query_maxlen: int = _QUERY_MAXLEN,
        doc_maxlen: int = _DOC_MAXLEN,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_name
        self._query_maxlen = query_maxlen
        self._doc_maxlen = doc_maxlen

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)

        # Best-effort: load the ColBERT linear projection if present.
        self._linear = _load_linear_layer(model_name, device=self._device)
        self._dim: int
        if self._linear is not None:
            self._dim = int(self._linear.out_features)
            logger.info(
                "ColBERT encoder '%s': linear projection %d → %d dims",
                model_name,
                self._model.config.hidden_size,
                self._dim,
            )
        else:
            self._dim = int(self._model.config.hidden_size)
            logger.info(
                "ColBERT encoder '%s': no projection found — using %d-dim hidden state",
                model_name,
                self._dim,
            )

    @property
    def dim(self) -> int:
        """Output dimension of each token embedding."""
        return self._dim

    def encode_query(self, text: str) -> np.ndarray:
        """Encode query text to L2-normalised token embeddings.

        Returns:
            float32 array of shape (q_len, dim).
        """
        return self._encode(text, maxlen=self._query_maxlen)

    def encode_document(self, text: str) -> np.ndarray:
        """Encode document text to L2-normalised token embeddings.

        Returns:
            float32 array of shape (d_len, dim).
        """
        return self._encode(text, maxlen=self._doc_maxlen)

    def _encode(self, text: str, maxlen: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        inputs: Any = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=maxlen,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # (1, seq_len, hidden_size) → (seq_len, hidden_size)
        hidden = outputs.last_hidden_state[0]

        if self._linear is not None:
            hidden = self._linear(hidden)  # (seq_len, colbert_dim)

        # L2-normalise each token vector so dot product = cosine similarity
        embs = F.normalize(hidden, p=2, dim=-1)
        return embs.detach().cpu().numpy().astype(np.float32)


def _load_linear_layer(model_name: str, device: str = "cpu") -> Any:
    """Load the ColBERT linear projection layer from a HuggingFace checkpoint.

    ColBERT v2 stores a ``linear.weight`` tensor alongside the BERT weights.
    AutoModel loads only the BERT backbone and skips unknown keys, so we load
    the full state dict separately and extract the projection.

    Returns None on any error (missing file, auth failure, key absent).
    """
    import torch.nn as nn

    try:
        state = _load_state_dict(model_name)
    except Exception as exc:
        logger.debug("ColBERT: could not load state dict from '%s': %s", model_name, exc)
        return None

    if state is None or "linear.weight" not in state:
        return None

    w = state["linear.weight"]  # (out_features, in_features)
    linear = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
    linear.weight.data = w.float()
    linear.eval()
    linear.to(device)
    logger.debug("ColBERT: loaded linear projection %d → %d", int(w.shape[1]), int(w.shape[0]))
    return linear


def _load_state_dict(model_name: str) -> Any:
    """Return the full weight dict from a HuggingFace checkpoint (cached)."""
    import torch
    from huggingface_hub import hf_hub_download

    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(model_name, filename)
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file

                return load_file(path)
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            continue
    return None


# ── Reranker ──────────────────────────────────────────────────────────────────


class ColBERTReranker:
    """ColBERT late-interaction reranker — drop-in for LocalCrossEncoderReranker.

    Implements the Reranker Protocol.  Requires per-document token embeddings
    pre-computed by ``scripts/generate_colbert_embeddings.py``.

    Workflow:
      1. encode_query(query) → (q_len, dim) tensor.
      2. For each candidate hit: load <hotel_id>.npy → (d_len, dim).
      3. maxsim(query_embs, doc_embs) → scalar.
      4. Sort descending, return top_k.

    Graceful degradation: a hit with no .npy file keeps its original retrieval
    score so that partial-encoding runs never block retrieval.

    Args:
        model_name: HuggingFace ColBERT checkpoint or any BERT model.
        embeddings_dir: Directory containing ``<hotel_id>.npy`` files.
        query_maxlen: Max query tokens.
        doc_maxlen: Max document tokens (must match generation script setting).
        _encoder: Inject a pre-built encoder (used in unit tests to avoid
            loading a real model — underscore signals non-public API).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        embeddings_dir: str | Path = "data/processed/colbert_embeddings",
        *,
        query_maxlen: int = _QUERY_MAXLEN,
        doc_maxlen: int = _DOC_MAXLEN,
        _encoder: Any = None,
    ) -> None:
        if _encoder is not None:
            self._encoder = _encoder
        else:
            self._encoder = ColBERTEncoder(
                model_name,
                query_maxlen=query_maxlen,
                doc_maxlen=doc_maxlen,
            )
        self._embeddings_dir = Path(embeddings_dir)
        logger.info(
            "ColBERT reranker ready: model=%s dim=%s dir=%s",
            getattr(self._encoder, "model_id", "injected"),
            getattr(self._encoder, "dim", "?"),
            self._embeddings_dir,
        )

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        *,
        top_k: int,
    ) -> list[Hit]:
        """Re-score candidates via MaxSim and return the top_k highest-scoring hits.

        Hits with missing .npy embeddings keep their original retrieval score.
        """
        if not hits:
            return []

        query_embs: np.ndarray = self._encoder.encode_query(query)
        scored: list[tuple[Hit, float]] = []
        missing = 0

        for hit in hits:
            emb_path = self._embeddings_dir / f"{hit.id}.npy"
            if emb_path.exists():
                doc_embs = np.load(str(emb_path))
                score = maxsim(query_embs, doc_embs)
            else:
                score = hit.score
                missing += 1
            scored.append((hit, score))

        if missing:
            logger.warning(
                "ColBERT: %d/%d hits lacked embeddings; original scores used. "
                "Run make generate-colbert-embeddings.",
                missing,
                len(hits),
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        return [Hit(id=h.id, score=s, source=h.source) for h, s in scored[:top_k]]


def build_colbert_document_text(source: dict[str, Any]) -> str:
    """Build the text representation used for ColBERT document encoding.

    Re-uses the same field selection as the cross-encoder reranker so that
    comparison between the two is apples-to-apples.
    """
    return _build_reranking_text(source)
