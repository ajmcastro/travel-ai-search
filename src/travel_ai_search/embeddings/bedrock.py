"""Amazon Titan Embeddings V2 provider via AWS Bedrock.

Model: amazon.titan-embed-text-v2:0
  Dimensions: 256, 512, or 1024 (configurable — larger = richer representation)
  Normalisation: Titan returns L2-normalised vectors when normalize=True,
    matching the LocalEmbeddingProvider contract.
  Throughput: each embed() call is a separate HTTP round-trip.  Titan V2 has
    no batch endpoint, so embed_batch() loops individually.  For offline
    index generation of 5,000+ hotels, LocalEmbeddingProvider is faster
    (sentence-transformers batching amortises GPU/CPU overhead).

Important: switching from local (384d) to Titan (1024d) requires recreating the
OpenSearch index and re-generating all embeddings, because the vector field
dimension is immutable.  Set both EMBEDDING_DIMENSION and BEDROCK_EMBEDDING_DIMENSION
to the same value, then run: make create-index && make generate-embeddings.

For local experimentation without AWS credentials, use LocalEmbeddingProvider.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_DIMENSION = 1024


class BedrockEmbeddingProvider:
    """Embedding provider backed by Amazon Titan Embeddings V2.

    Satisfies EmbeddingProvider Protocol (structural typing — no import needed).
    """

    def __init__(
        self,
        client: Any,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text via Titan V2 invoke_model.

        Returns an L2-normalised vector of length self.dimension.
        Raises on any boto3 / network error — callers are responsible for
        catching and falling back (e.g. via the SearchService fallback logic).
        """
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dimension,
                "normalize": True,
            }
        )
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result: dict[str, Any] = json.loads(response["body"].read())
        return list(result["embedding"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, one API call per text.

        Titan V2 has no batch endpoint.  For large offline workloads, prefer
        LocalEmbeddingProvider which batches via sentence-transformers.
        """
        if not texts:
            return []
        return [self.embed(t) for t in texts]
