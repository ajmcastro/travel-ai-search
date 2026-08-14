"""Local LLM provider stubs for development and testing.

Neither provider uses a real language model — they are deterministic, zero-dependency
stubs that demonstrate the LLMProvider interface without requiring a GPU or API key.

  EchoLLMProvider   — returns the input prompt unchanged; use in unit tests
                       to verify pipeline wiring without covering LLM effects.
  LocalLLMProvider  — adds travel-domain synonyms to expand the query vocabulary;
                       useful for local experimentation but NOT a real LLM.

For real LLM quality, plug in BedrockLLMProvider (Milestone 12).
"""

from __future__ import annotations

# Keyword → candidate synonyms to append (first synonym not already in the query wins).
# Only terms where vocabulary mismatch with hotel descriptions is likely to matter.
_EXPANSIONS: dict[str, list[str]] = {
    "beach": ["coastal", "seaside", "sandy beach"],
    "quiet": ["peaceful", "tranquil", "secluded"],
    "peaceful": ["quiet", "tranquil", "serene"],
    "relaxing": ["peaceful", "calm", "tranquil"],
    "family": ["family-friendly", "children", "kids"],
    "kids": ["children", "family-friendly"],
    "children": ["family-friendly", "kids"],
    "luxury": ["five-star", "upscale", "premium"],
    "premium": ["luxury", "five-star"],
    "spa": ["wellness", "thalassotherapy"],
    "wellness": ["spa", "thalassotherapy"],
    "romantic": ["couples retreat", "intimate"],
    "couples": ["adults-only", "romantic"],
    "pool": ["swimming pool", "infinity pool"],
    "all-inclusive": ["full board", "meals included"],
    "inclusive": ["all-inclusive", "full board"],
    "nightlife": ["bars", "clubs", "entertainment"],
    "hiking": ["trekking", "walking trails"],
    "trekking": ["hiking", "mountain trails"],
    "adventure": ["outdoor activities", "sports"],
    "golf": ["golf course", "fairways"],
    "ski": ["snow", "slopes", "winter sports"],
    "boutique": ["charming", "intimate"],
    "budget": ["affordable", "value for money"],
    "affordable": ["budget", "value for money"],
    "adults": ["adults-only", "couples"],
    "island": ["coastal", "mediterranean"],
    "gym": ["fitness centre", "health club"],
    "water": ["watersports", "aquatic"],
    "culture": ["cultural", "heritage", "historic"],
}

# Cap additions so the embedding centroid stays close to the original intent.
_MAX_ADDITIONS = 3


class EchoLLMProvider:
    """Identity stub: returns the prompt unchanged.  For unit testing only."""

    def generate(self, prompt: str, *, system: str = "") -> str:
        return prompt


class LocalLLMProvider:
    """Keyword-expansion provider for local development without an API key.

    Scans the prompt for known travel keywords and appends up to _MAX_ADDITIONS
    synonyms.  The original text is always preserved at the front of the result.
    """

    def generate(self, prompt: str, *, system: str = "") -> str:
        query_lower = prompt.lower()
        additions: list[str] = []
        for keyword, synonyms in _EXPANSIONS.items():
            if len(additions) >= _MAX_ADDITIONS:
                break
            if keyword in query_lower:
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        additions.append(syn)
                        break
        if not additions:
            return prompt
        return prompt + " " + " ".join(additions)
