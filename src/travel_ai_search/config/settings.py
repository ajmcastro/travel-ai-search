from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional .env file.

    Precedence (highest first): environment variables → .env file → defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    opensearch_index_name: str = "travel_hotels"

    # Application
    log_level: str = "INFO"
    environment: str = "development"

    # Search
    top_k: int = 10

    # Embeddings
    # Change embedding_model_name to switch models, but also update
    # embedding_dimension to match and recreate the index (dimension is immutable).
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Hybrid retrieval (Milestone 6)
    # lexical_weight + vector_weight need not sum to 1.0; they are applied to
    # independently min-max-normalised scores, so the ratio is what matters.
    hybrid_lexical_weight: float = 0.5
    hybrid_vector_weight: float = 0.5
    hybrid_candidate_k: int = 50

    # Fusion strategy (Milestone 7)
    # hybrid_fusion: default fusion method — "weighted" or "rrf"
    # rrf_k: smoothing constant in RRF formula 1/(k + rank); k=60 is the
    #   empirically robust default from Cormack et al. (2009)
    hybrid_fusion: str = "weighted"
    rrf_k: int = 60

    # Reranking (Milestone 8)
    # reranking_enabled: set True to load the cross-encoder at startup.
    #   When False, the reranker is never loaded and reranking is a no-op
    #   even if rerank=true is passed to the API endpoint.
    # reranker_model_name: any cross-encoder model from HuggingFace Hub.
    #   Smaller/faster: cross-encoder/ms-marco-MiniLM-L-2-v2
    #   Larger/better:  cross-encoder/ms-marco-MiniLM-L-12-v2
    # rerank_k: candidates passed to the cross-encoder (must be ≥ top_k).
    reranking_enabled: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_k: int = 50

    # Query understanding (Milestone 9)
    # query_understanding_enabled: controls whether the QU engine is created at startup.
    #   The rule-based engine is pure Python and costs nothing to load; this flag
    #   exists to allow disabling it (e.g. for A/B testing) without code changes.
    query_understanding_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    The cache ensures settings are parsed from the environment exactly once.
    In tests, call get_settings.cache_clear() after patching env vars.
    """
    return Settings()
