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


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    The cache ensures settings are parsed from the environment exactly once.
    In tests, call get_settings.cache_clear() after patching env vars.
    """
    return Settings()
