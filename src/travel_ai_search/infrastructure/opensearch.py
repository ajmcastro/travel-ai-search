from opensearchpy import OpenSearch

from travel_ai_search.config.settings import Settings


def create_client(settings: Settings) -> OpenSearch:
    """Create an OpenSearch client from application settings.

    For local development (DISABLE_SECURITY_PLUGIN=true), SSL and auth are
    intentionally disabled. For production, set opensearch_use_ssl=True and
    supply credentials via environment variables.
    """
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_compress=True,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
    )
