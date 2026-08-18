"""JudgeProvider protocol and local implementations.

A JudgeProvider reads a (query, hotel_name, hotel_description) triple and
returns a JudgeVerdict: a numeric relevance score (0–3) and a one-sentence
rationale.  This is the LLM-as-judge interface for Milestone 16.

Implementations
---------------
EchoJudgeProvider      — fixed score; used in tests and dry runs.
BedrockJudgeProvider   — AWS Bedrock Converse API (judge_bedrock.py).
                         The judge model MUST differ from the generator model.

Methodology note
----------------
Common-mode bias: if the model that generated the synthetic hotel descriptions
is the same as the judge, their agreement reads as high relevance but is partly
structural (shared vocabulary, shared notion of "good hotel text").  Always use
a different model family for judging.  See docs/EXPERIMENTS.md §M16.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeVerdict:
    """Relevance judgment produced by a JudgeProvider for one (query, hotel) pair.

    Attributes
    ----------
    score:     0 = irrelevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant.
    rationale: One sentence explaining the judgment.  Used for qualitative inspection.
    """

    score: int
    rationale: str

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 3:
            raise ValueError(f"JudgeVerdict.score must be 0–3, got {self.score!r}")


@runtime_checkable
class JudgeProvider(Protocol):
    """Interface for LLM-based relevance scoring.

    All concrete implementations must expose a ``model_id`` attribute so that
    the model identity can be recorded alongside every set of judge results.
    This is required by the methodology validity note in the project spec:
    scores from two different judge models are not directly comparable.
    """

    model_id: str

    def judge(
        self,
        query: str,
        hotel_name: str,
        hotel_description: str,
    ) -> JudgeVerdict:
        """Score how relevant ``hotel_name`` / ``hotel_description`` is for ``query``.

        Returns
        -------
        JudgeVerdict with integer score 0–3 and a one-sentence rationale.
        Implementations must not raise; return score=0 on error.
        """
        ...


class EchoJudgeProvider:
    """JudgeProvider that returns a configurable fixed score for all inputs.

    Used in unit tests and dry-run evaluations where no LLM is available.
    ``model_id`` is the literal string "echo" so it is always distinguishable
    in saved results from a real judge.
    """

    model_id: str = "echo"

    def __init__(self, fixed_score: int = 2) -> None:
        if not 0 <= fixed_score <= 3:
            raise ValueError(f"fixed_score must be 0–3, got {fixed_score!r}")
        self._fixed_score = fixed_score

    def judge(
        self,
        query: str,
        hotel_name: str,
        hotel_description: str,
    ) -> JudgeVerdict:
        return JudgeVerdict(
            score=self._fixed_score,
            rationale="Echo provider — fixed score, no real judgment.",
        )


# ── Prompt construction and response parsing ──────────────────────────────────


def build_judge_prompt(query: str, hotel_name: str, hotel_description: str) -> str:
    """Build the relevance-scoring prompt for the judge model.

    Design decisions
    ----------------
    - Uses plain hotel data; never mentions that the data is synthetic.
    - Asks for a numeric grade (0–3) so results are quantitatively aggregable.
    - Asks for a single-sentence rationale for qualitative inspection.
    - Uses a strict output format ("SCORE: N | RATIONALE: ...") for easy parsing.
    - System/user separation is left to the caller; this function returns the
      user-turn content only.
    """
    return (
        "You are an expert travel search evaluator.\n"
        "A user searched for hotels using the query below. "
        "Rate how relevant the following hotel is for that query.\n\n"
        f"QUERY: {query}\n\n"
        f"HOTEL NAME: {hotel_name}\n\n"
        f"HOTEL DESCRIPTION:\n{hotel_description}\n\n"
        "Rate the hotel's relevance on this scale:\n"
        "  0 = Irrelevant — the hotel does not match the query at all.\n"
        "  1 = Marginally relevant — some minor overlap but not a good match.\n"
        "  2 = Relevant — the hotel matches most aspects of the query.\n"
        "  3 = Highly relevant — the hotel is an excellent match for the query.\n\n"
        "Respond ONLY in this exact format (no extra text before or after):\n"
        "SCORE: <0|1|2|3> | RATIONALE: <one sentence explaining your judgment>"
    )


def parse_judge_response(response: str, *, fallback_score: int = 0) -> JudgeVerdict:
    """Parse an LLM response into a JudgeVerdict.

    Tries the strict format first ("SCORE: N | RATIONALE: ..."), then falls
    back to scanning for any digit 0–3, then uses ``fallback_score`` if nothing
    parseable is found.  This tolerates minor model non-compliance without
    crashing the evaluation run.
    """
    text = response.strip()

    # Strict format: SCORE: N | RATIONALE: ...
    strict = re.search(r"SCORE:\s*([0-3])\s*\|\s*RATIONALE:\s*(.*)", text, re.IGNORECASE)
    if strict:
        return JudgeVerdict(
            score=int(strict.group(1)),
            rationale=strict.group(2).strip()[:300],
        )

    # Fallback: extract the first standalone digit 0–3
    loose = re.search(r"\b([0-3])\b", text)
    if loose:
        return JudgeVerdict(
            score=int(loose.group(1)),
            rationale=text[:200],
        )

    return JudgeVerdict(
        score=fallback_score,
        rationale=f"[parse failed] {text[:100]}",
    )
