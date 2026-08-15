"""FastAPI application entry point.

Resources created once at startup and stored on app.state:
  os_client                  — shared OpenSearch connection pool
  embedding_provider         — sentence-transformers model (loaded once; ~1-2 s on first run)
  reranker                   — cross-encoder model (None when reranking_enabled=False)
  query_understanding_engine — rule-based QU engine (always created; pure Python, no I/O)
  query_rewriter             — LLM-based query rewriter (None when query_rewriting_enabled=False)
  query_expander             — query expander for multi-query retrieval (None when disabled)
  knowledge_retriever        — RAG knowledge retriever (None when rag_enabled=False)
  rag_synthesizer            — RAG LLM synthesizer (None when rag_enabled=False)
  destination_graph          — in-memory travel graph (None when graph_enabled=False)

All route handlers receive these via dependency functions in deps.py.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from travel_ai_search.api.routes import graph as graph_router
from travel_ai_search.api.routes import health as health_router
from travel_ai_search.api.routes import query as query_router
from travel_ai_search.api.routes import search as search_router
from travel_ai_search.config.settings import Settings, get_settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.llm.base import LLMProvider
from travel_ai_search.query_understanding.base import QueryUnderstandingEngine
from travel_ai_search.query_understanding.extractor import RuleBasedQueryUnderstandingEngine
from travel_ai_search.query_understanding.rewriter import QueryRewriter
from travel_ai_search.reranking.base import Reranker

# Configure application logging.
#
# Uvicorn's dictConfig only configures uvicorn.* loggers and leaves the root
# logger without a handler.  Records that propagate from travel_ai_search.*
# loggers reach Python's "handler of last resort" (level=WARNING), so INFO
# messages are silently dropped without the lines below.
#
# When LOG_FORMAT=json, StructuredFormatter replaces the plain-text handler so
# that every travel_ai_search.* record is emitted as one line of NDJSON — ideal
# for log aggregation systems (Loki, CloudWatch, Splunk).  Otherwise a simple
# human-readable format is used for local development.
_settings_for_logging = get_settings()
if _settings_for_logging.log_format.lower() == "json":
    from travel_ai_search.observability.logging import configure_structured_logging

    configure_structured_logging("travel_ai_search", _settings_for_logging.log_level)
else:
    logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s")
    logging.getLogger("travel_ai_search").setLevel(_settings_for_logging.log_level.upper())

logger = logging.getLogger(__name__)


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory — separated from lifespan so tests can patch it cheaply.

    Graceful degradation: if the Bedrock provider fails to initialise (missing
    credentials, unavailable region, etc.) log a warning and fall back to local.
    """
    if settings.embedding_provider == "bedrock":
        try:
            from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client

            client = create_bedrock_client(settings.aws_region)
            logger.info(
                "Bedrock embedding provider: %s (dim=%d)",
                settings.bedrock_embedding_model_id,
                settings.bedrock_embedding_dimension,
            )
            return BedrockEmbeddingProvider(
                client,
                model_id=settings.bedrock_embedding_model_id,
                dimension=settings.bedrock_embedding_dimension,
            )
        except Exception as exc:
            logger.warning(
                "Failed to create Bedrock embedding provider: %s — falling back to local.",
                exc,
            )
    return LocalEmbeddingProvider(settings.embedding_model_name)


def _create_query_understanding_engine(settings: Settings) -> QueryUnderstandingEngine:
    """Factory — separated from lifespan so tests can patch it cheaply."""
    return RuleBasedQueryUnderstandingEngine()


def _create_query_rewriter(settings: Settings) -> QueryRewriter | None:
    """Factory — returns None when query rewriting is disabled.

    Graceful degradation: a failed provider load logs a warning and the system
    continues without rewriting.  The llm_provider setting selects the backend:
    'local' (keyword expansion, no deps), 'echo' (identity stub, for testing),
    'bedrock' (Claude via Converse API, requires AWS credentials).
    """
    if not settings.query_rewriting_enabled:
        return None
    try:
        llm: LLMProvider
        if settings.llm_provider == "bedrock":
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client
            from travel_ai_search.llm.bedrock import BedrockLLMProvider

            bedrock_client = create_bedrock_client(settings.aws_region)
            llm = BedrockLLMProvider(bedrock_client, model_id=settings.bedrock_llm_model_id)
            logger.info("Bedrock LLM provider: %s", settings.bedrock_llm_model_id)
        elif settings.llm_provider == "echo":
            from travel_ai_search.llm.local import EchoLLMProvider

            llm = EchoLLMProvider()
        else:
            from travel_ai_search.llm.local import LocalLLMProvider

            llm = LocalLLMProvider()
        return QueryRewriter(llm)
    except Exception as exc:
        logger.warning(
            "Failed to create query rewriter ('%s'): %s — rewriting disabled.",
            settings.llm_provider,
            exc,
        )
        return None


def _create_query_expander(settings: Settings) -> object | None:
    """Factory — returns None when query expansion is disabled.

    Graceful degradation: a failed expander load logs a warning and the system
    continues without expansion.  LocalQueryExpander is pure Python with no
    external dependencies, so failure is unlikely in practice.
    """
    if not settings.query_expansion_enabled:
        return None
    try:
        from travel_ai_search.query_understanding.expander import LocalQueryExpander

        return LocalQueryExpander()
    except Exception as exc:
        logger.warning(
            "Failed to create query expander: %s — expansion disabled.",
            exc,
        )
        return None


def _create_reranker(settings: Settings) -> Reranker | None:
    """Factory — returns None when reranking is disabled or the model fails to load.

    Graceful degradation: a failed reranker load logs a warning and the system
    continues without reranking rather than refusing to start.  If the Bedrock
    reranker fails, falls back to the local cross-encoder.
    """
    if not settings.reranking_enabled:
        return None
    if settings.reranker_provider == "bedrock":
        try:
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client
            from travel_ai_search.reranking.bedrock import BedrockReranker

            client = create_bedrock_client(settings.aws_region)
            logger.info("Bedrock reranker: %s", settings.bedrock_reranker_model_id)
            return BedrockReranker(client, model_id=settings.bedrock_reranker_model_id)
        except Exception as exc:
            logger.warning(
                "Failed to create Bedrock reranker: %s — falling back to local.",
                exc,
            )
    try:
        from travel_ai_search.reranking.local import LocalCrossEncoderReranker

        return LocalCrossEncoderReranker(settings.reranker_model_name)
    except Exception as exc:
        logger.warning(
            "Failed to load reranker '%s': %s — reranking disabled.",
            settings.reranker_model_name,
            exc,
        )
        return None


def _create_knowledge_retriever(
    settings: Settings,
    os_client: object,
    embedding_provider: EmbeddingProvider,
) -> object | None:
    """Factory — returns None when rag_enabled=False or the retriever fails to initialise.

    Graceful degradation: a failed retriever logs a warning and RAG is disabled
    for this server instance.  The most common failure is the knowledge index not
    existing yet — run `make ingest-knowledge` to populate it.
    """
    if not settings.rag_enabled:
        return None
    try:
        from travel_ai_search.rag.retriever import KnowledgeRetriever

        return KnowledgeRetriever(
            os_client,  # type: ignore[arg-type]
            embedding_provider,
            index=settings.knowledge_index_name,
            top_k=settings.rag_context_k,
        )
    except Exception as exc:
        logger.warning(
            "Failed to create KnowledgeRetriever: %s — RAG disabled.",
            exc,
        )
        return None


def _create_destination_graph(settings: Settings) -> object | None:
    """Factory — returns None when graph_enabled=False or the knowledge file is missing.

    The graph is built entirely from the local JSONL file — no OpenSearch required.
    Graceful degradation: a missing file or parse error logs a warning and all
    /graph/* endpoints return HTTP 503 for this server instance.
    """
    if not settings.graph_enabled:
        return None
    try:
        from pathlib import Path

        from travel_ai_search.graph.builder import build_destination_graph, load_knowledge_docs

        docs = load_knowledge_docs(Path(settings.knowledge_file_path))
        graph = build_destination_graph(docs)
        logger.info(
            "Destination graph ready: %d nodes, %d edges.",
            graph.node_count(),
            graph.edge_count(),
        )
        return graph
    except Exception as exc:
        logger.warning(
            "Failed to build destination graph from '%s': %s — /graph/* endpoints disabled.",
            settings.knowledge_file_path,
            exc,
        )
        return None


def _create_rag_synthesizer(settings: Settings) -> object | None:
    """Factory — returns None when rag_enabled=False or the LLM fails to initialise.

    Reuses the llm_provider setting: "bedrock" uses Claude via Converse API,
    "echo" returns the prompt as-is (useful for local RAG pipeline testing),
    "local" uses the keyword-substitution stub (limited synthesis quality).
    """
    if not settings.rag_enabled:
        return None
    try:
        llm: LLMProvider
        if settings.llm_provider == "bedrock":
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client
            from travel_ai_search.llm.bedrock import BedrockLLMProvider

            bedrock_client = create_bedrock_client(settings.aws_region)
            llm = BedrockLLMProvider(bedrock_client, model_id=settings.bedrock_llm_model_id)
            logger.info(
                "RAG synthesizer using Bedrock LLM: %s",
                settings.bedrock_llm_model_id,
            )
        elif settings.llm_provider == "echo":
            from travel_ai_search.llm.local import EchoLLMProvider

            llm = EchoLLMProvider()
        else:
            from travel_ai_search.llm.local import LocalLLMProvider

            llm = LocalLLMProvider()
        from travel_ai_search.rag.synthesizer import RAGSynthesizer

        return RAGSynthesizer(llm)
    except Exception as exc:
        logger.warning(
            "Failed to create RAGSynthesizer ('%s'): %s — synthesis disabled.",
            settings.llm_provider,
            exc,
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.os_client = create_client(settings)
    app.state.embedding_provider = _create_embedding_provider(settings)
    app.state.reranker = _create_reranker(settings)
    app.state.query_understanding_engine = _create_query_understanding_engine(settings)
    app.state.query_rewriter = _create_query_rewriter(settings)
    app.state.query_expander = _create_query_expander(settings)
    app.state.knowledge_retriever = _create_knowledge_retriever(
        settings,
        app.state.os_client,
        app.state.embedding_provider,
    )
    app.state.rag_synthesizer = _create_rag_synthesizer(settings)
    app.state.destination_graph = _create_destination_graph(settings)
    yield
    app.state.os_client.close()


app = FastAPI(
    title="Travel AI Search",
    description="Educational AI search system for travel products",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search_router.router, prefix="/search")
app.include_router(query_router.router, prefix="/query")
app.include_router(graph_router.router, prefix="/graph")
app.include_router(health_router.router, prefix="/health")


@app.get("/metrics", tags=["Observability"], include_in_schema=False)
def metrics_endpoint() -> object:
    """Expose in-memory counters and histograms in Prometheus text format.

    Prometheus (or any compatible scraper) should poll this endpoint on a
    regular interval (e.g. every 15 s) to collect:
      travel_search_requests_total{strategy=...}   — request counts by strategy
      travel_search_latency_ms_bucket{le=...}      — latency distribution
      travel_search_fallbacks_total{reason=...}    — degradation events

    Example (manual scrape):
        curl http://localhost:8000/metrics
    """
    from fastapi.responses import Response

    from travel_ai_search.observability.metrics import _registry

    return Response(content=_registry.render_all(), media_type="text/plain; version=0.0.4")
