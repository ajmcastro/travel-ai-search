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

    # Application
    log_level: str = "INFO"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    The cache ensures settings are parsed from the environment exactly once.
    In tests, call get_settings.cache_clear() after patching env vars.
    """
    return Settings()
