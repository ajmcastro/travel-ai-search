"""Local embedding provider backed by sentence-transformers.

The model is downloaded once on first use and cached by the
sentence-transformers library (default: ~/.cache/huggingface/hub/).

Model choice: all-MiniLM-L6-v2
  • 384 dimensions — small enough for a single-node OpenSearch HNSW index
  • L2-normalised outputs — cosine similarity equals dot product, which
    OpenSearch can compute without an extra normalisation step
  • ~80 MB download, runs on CPU without GPU in ≈ 5–15 ms per query
  • Industry-standard benchmark quality for semantic similarity tasks

Alternative worth knowing: BAAI/bge-small-en-v1.5 (slightly higher quality,
same size) — swap the model name in settings to compare.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbeddingProvider:
    """Wrap a sentence-transformers model as an EmbeddingProvider.

    The underlying SentenceTransformer is loaded once at construction and
    reused for all subsequent calls.  Load this provider once in application
    startup (app.state) rather than per-request.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        dim = self._model.get_embedding_dimension()
        self._dimension: int = int(dim) if dim is not None else 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text; returns an L2-normalised float list."""
        vector: np.ndarray = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return cast(list[float], vector.tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in one forward pass (more efficient than N×embed)."""
        if not texts:
            return []
        vectors: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return cast(list[list[float]], [v.tolist() for v in vectors])
