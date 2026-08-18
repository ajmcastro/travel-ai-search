"""Pydantic schemas for the POST /evaluate/judge endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeQueryInput(BaseModel):
    """A single query to be judged."""

    query_text: str = Field(..., min_length=1, description="Natural-language query.")
    query_class: str = Field("unspecified", description="Optional query class label.")


class JudgeRequest(BaseModel):
    """Request body for POST /evaluate/judge."""

    queries: list[JudgeQueryInput] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Queries to evaluate. Each query is run through the retrieval strategy "
        "and then scored by the LLM judge.",
    )
    strategy: str = Field(
        "rrf",
        description="Retrieval strategy: 'bm25', 'vector', 'rrf'.",
    )
    k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of hotels to retrieve and judge per query. "
        "Keep small (≤10) to limit LLM judge API calls.",
    )
    judge_provider: str = Field(
        "echo",
        description="Judge provider: 'echo' (fixed score, no API call) or 'bedrock'.",
    )


class JudgedHit(BaseModel):
    """One hotel result with its LLM judge score."""

    doc_id: str
    hotel_name: str
    rank: int
    judge_score: int = Field(..., ge=0, le=3)
    rationale: str


class JudgeQueryOutput(BaseModel):
    """Judge results for one query."""

    query_text: str
    query_class: str
    mean_score: float
    hits: list[JudgedHit]


class JudgeResponse(BaseModel):
    """Response body for POST /evaluate/judge."""

    strategy: str
    judge_model_id: str
    k: int
    n_queries: int
    mean_judge_score: float
    results: list[JudgeQueryOutput]

    # Methodology reminder published with every response.
    methodology_note: str = (
        "Judge model identity is recorded in judge_model_id. "
        "Scores from different judge models are not directly comparable. "
        "The judge model should differ from the model that generated the corpus "
        "to avoid common-mode bias."
    )
