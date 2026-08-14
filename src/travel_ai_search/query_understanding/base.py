"""QueryUnderstandingEngine Protocol — structural typing for all understanding backends.

Any object that implements understand(query: str) -> QueryUnderstanding satisfies
this Protocol at runtime (via isinstance checks) without inheritance.

Milestone 12 will add BedrockQueryUnderstandingEngine; it slots in here with no
changes to the routes or the orchestration layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from travel_ai_search.query_understanding.models import QueryUnderstanding


@runtime_checkable
class QueryUnderstandingEngine(Protocol):
    def understand(self, query: str) -> QueryUnderstanding: ...
