"""FastAPI application entry point.

The OpenSearch client is created once at startup and stored on app.state.
All route handlers receive it via the get_os_client() dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from travel_ai_search.api.routes import search as search_router
from travel_ai_search.config.settings import get_settings
from travel_ai_search.infrastructure.opensearch import create_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.os_client = create_client(settings)
    yield
    app.state.os_client.close()


app = FastAPI(
    title="Travel AI Search",
    description="Educational AI search system for travel products",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search_router.router, prefix="/search")


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}
