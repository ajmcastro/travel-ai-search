"""LLMProvider protocol — the interface all language model backends must satisfy.

Implementations:
  LocalLLMProvider  — keyword-expansion stub (M10, no API key required)
  EchoLLMProvider   — identity stub for unit testing (M10)
  BedrockLLMProvider — AWS Bedrock Converse API (M12)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for text generation used in query rewriting and RAG.

    The minimal surface: given a user prompt and optional system instructions,
    return a completion string.  Implementations must not raise on empty input.

    Args:
        prompt: The user message / text to complete or rewrite.
        system: Optional system-level instructions that guide the model's
                behaviour (e.g. persona, output format).  Not all local
                implementations use this; remote ones (Bedrock, OpenAI) do.

    Returns:
        The generated text.  An empty string is a valid response.
    """

    def generate(self, prompt: str, *, system: str = "") -> str: ...
