"""Destination knowledge domain model.

DestinationKnowledge represents structured facts about a travel destination
island or region.  It is the document type stored in the knowledge index and
returned by KnowledgeRetriever.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DestinationKnowledge(BaseModel):
    """Facts about a single travel destination island or region.

    One document per island/region, not per resort cluster — multiple hotel
    clusters within the same island share this knowledge document.  This is
    intentional: "what is Mallorca like?" is answered the same way regardless
    of which resort area the traveller books.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    destination: str
    country: str
    region: str
    description: str
    climate: str
    best_months: list[str]
    family_suitability: str  # "low" | "moderate" | "high"
    nightlife_level: str  # "low" | "moderate" | "high"
    beach_quality: str  # "limited" | "moderate" | "good" | "excellent"
    activities: list[str]
    character_tags: list[str]
    similar_destinations: list[str]
    geographic_note: str


def build_knowledge_embedding_text(k: DestinationKnowledge) -> str:
    """Build the text that will be embedded for semantic search.

    Combines the most semantically informative fields: description, climate,
    activities, and character tags.  The destination name and region are
    included so that exact-name queries also match.

    Excluded: geographic_note (factual, not semantic), best_months (not part
    of the semantic character profile), similar_destinations (would bias the
    vector towards those destinations rather than this one's own character).
    """
    parts = [
        f"{k.destination}, {k.region}, {k.country}.",
        k.description,
        f"Climate: {k.climate}.",
        f"Activities: {', '.join(k.activities)}.",
        f"Character: {', '.join(k.character_tags)}.",
        (
            f"Family suitability: {k.family_suitability}. "
            f"Nightlife: {k.nightlife_level}. "
            f"Beach: {k.beach_quality}."
        ),
    ]
    return " ".join(parts)
