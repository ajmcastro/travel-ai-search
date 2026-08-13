"""EmbeddingProvider protocol — the contract every embedding backend must satisfy.

Any class with the right methods qualifies structurally (no inheritance needed).
This makes mocking trivial: just build an object with the right attributes.

Implementations
---------------
LocalEmbeddingProvider   — sentence-transformers (runs fully offline)
BedrockEmbeddingProvider — AWS Bedrock Titan Embeddings (Milestone 12, optional)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense-embedding backends.

    Callers should use embed_batch() when processing many texts — batching
    amortises model overhead and is significantly faster than N×embed().
    """

    @property
    def dimension(self) -> int:
        """Number of floats in each output vector."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return a single L2-normalised dense vector for `text`."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised vector per entry in `texts`.

        The returned list is the same length as `texts`.
        An empty input returns an empty list.
        """
        ...
