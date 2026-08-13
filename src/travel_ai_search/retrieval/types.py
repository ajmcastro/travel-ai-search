"""Shared retrieval types — imported by lexical, vector, fusion, and hybrid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Hit:
    """A single document returned from OpenSearch."""

    id: str
    score: float
    source: dict[str, Any]
