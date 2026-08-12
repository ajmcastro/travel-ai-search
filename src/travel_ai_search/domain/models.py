from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TravelProduct(BaseModel):
    """A searchable travel/hotel product.

    Field roles — drives OpenSearch mapping design in Milestone 2:
      Full-text (analysed, inverted index):
        hotel_name, hotel_description, activities, tags
      Keyword (exact / unanalysed — filters & facets):
        destination, region, country, board_types,
        available_months, available_departure_airports, climate_zone
      Numeric / range:
        star_rating, customer_rating, price_per_person_gbp,
        beach_distance_km, airport_distance_km
      Boolean:
        family_friendly, adults_only
      Geo:
        latitude, longitude
      Embedding input (derived, not stored):
        use build_embedding_text()
    """

    id: str

    # Searchable text
    hotel_name: str
    hotel_description: str

    # Location
    destination: str  # city / resort (e.g. "Playa de las Américas")
    region: str  # island / province (e.g. "Tenerife")
    country: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)

    # Quality
    star_rating: int = Field(ge=1, le=5)
    customer_rating: float = Field(ge=1.0, le=10.0)

    # Character
    family_friendly: bool
    adults_only: bool

    # Facilities
    amenities: list[str]
    board_types: list[str]

    # Proximity in km
    beach_distance_km: float = Field(ge=0.0)
    airport_distance_km: float = Field(ge=0.0)

    # Discovery
    activities: list[str]
    tags: list[str]

    # Offer
    available_departure_airports: list[str]  # IATA codes
    price_per_person_gbp: float = Field(gt=0.0)
    available_months: list[str]

    # Climate
    climate_zone: str
    peak_season_months: list[str]

    @model_validator(mode="after")
    def _adults_only_excludes_family(self) -> TravelProduct:
        if self.family_friendly and self.adults_only:
            raise ValueError("A hotel cannot be both family_friendly and adults_only")
        return self


def build_embedding_text(product: TravelProduct) -> str:
    """Construct the text passed to the embedding model for this product.

    Includes semantically meaningful fields: name, location, description,
    board types, activities, tags, climate, and family/adults character.

    Deliberately excludes structural filter fields — star_rating, price,
    distances, airport codes, and available months — because numbers and
    codes do not improve semantic similarity between products.
    """
    parts: list[str] = [
        f"{product.hotel_name} in {product.destination}, {product.region}, {product.country}.",
        product.hotel_description,
    ]
    if product.board_types:
        boards = [b.replace("_", " ") for b in product.board_types]
        parts.append(f"Board options: {', '.join(boards)}.")
    if product.activities:
        parts.append(f"Activities: {', '.join(product.activities)}.")
    if product.tags:
        parts.append(f"Known for: {', '.join(product.tags)}.")
    parts.append(f"Climate: {product.climate_zone}.")
    if product.adults_only:
        parts.append("Adults only.")
    elif product.family_friendly:
        parts.append("Family friendly.")
    return " ".join(parts)
