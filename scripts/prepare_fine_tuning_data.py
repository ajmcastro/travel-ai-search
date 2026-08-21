"""Prepare training triplets (query, positive, negative) for bi-encoder fine-tuning.

Run this before scripts/fine_tune_embeddings.py.
Requires: OpenSearch running (for BM25 hard negative mining).

Hard negative strategy
----------------------
For each golden query:
  1. Positives: hotels with graded relevance ≥ MIN_POSITIVE_GRADE (default 2).
  2. BM25 candidates: top CANDIDATES_K hits from lexical search.
  3. Hard negatives: candidates NOT present in any graded relevance judgement
     (grade 0 / unrated) — hotels that keyword overlap ranks high but that the
     assessors determined are not relevant.

Why BM25 for mining?
  BM25 retrieves hotels that share keywords with the query.  A hotel retrieved
  this way but not rated relevant is a confusing negative — it looks relevant on
  the surface but is not.  Training on these forces the bi-encoder to learn
  fine-grained distinctions (e.g. "adults-only spa" vs "family beach hotel with
  spa") rather than just separating hotels from noise.

Output
------
    data/evaluation/fine_tuning_pairs.jsonl
    One JSON object per line: {"query": "...", "positive": "...", "negative": "..."}

Usage
-----
    make prepare-fine-tuning-data
    # or directly:
    uv run python scripts/prepare_fine_tuning_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.domain.models import build_embedding_text
from travel_ai_search.evaluation.dataset import load_dataset
from travel_ai_search.fine_tuning.pairs import (
    TrainingPair,
    build_training_pairs,
    select_hard_negatives,
)
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.ingestion.ingestor import load_products
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search

HOTELS_PATH = Path("data/processed/hotels.jsonl")
GOLDEN_PATH = Path("data/evaluation/golden_queries.jsonl")
OUTPUT_PATH = Path("data/evaluation/fine_tuning_pairs.jsonl")

CANDIDATES_K = 50  # BM25 hits to consider as hard negative candidates
MAX_PAIRS_PER_QUERY = 5  # cap training pairs per golden query
MIN_POSITIVE_GRADE = 2  # minimum relevance grade to treat a hotel as positive


def main() -> None:
    settings = get_settings()

    # ── Connect to OpenSearch ─────────────────────────────────────────────────
    print("\nConnecting to OpenSearch …")
    try:
        client = create_client(settings)
        client.info()
        print(f"  Connected to {settings.opensearch_host}:{settings.opensearch_port}")
    except Exception as exc:
        print(f"\nERROR: Cannot connect to OpenSearch: {exc}")
        print("  Start OpenSearch with:  make up")
        sys.exit(1)

    # ── Load hotels into a lookup dict ────────────────────────────────────────
    if not HOTELS_PATH.exists():
        print(f"\nERROR: Hotel dataset not found at {HOTELS_PATH}")
        print("  Generate it with:  make generate-data")
        sys.exit(1)

    print(f"\nLoading hotels from {HOTELS_PATH} …")
    hotels = load_products(HOTELS_PATH)
    hotel_texts: dict[str, str] = {h.id: build_embedding_text(h) for h in hotels}
    print(f"  {len(hotel_texts):,} hotels loaded")

    # ── Load golden dataset ───────────────────────────────────────────────────
    if not GOLDEN_PATH.exists():
        print(f"\nERROR: Golden dataset not found at {GOLDEN_PATH}")
        print("  Build it with:  uv run python scripts/build_golden_dataset.py")
        sys.exit(1)

    print(f"\nLoading golden dataset from {GOLDEN_PATH} …")
    dataset = load_dataset(GOLDEN_PATH)
    print(f"  {len(dataset)} queries loaded")

    # ── Mine hard negatives and build training pairs ──────────────────────────
    index = settings.opensearch_index_name
    all_pairs: list[TrainingPair] = []
    queries_with_pairs = 0
    queries_without_negatives = 0
    queries_without_positives = 0

    print(
        f"\nMining hard negatives"
        f" (BM25 top-{CANDIDATES_K}, grade ≥ {MIN_POSITIVE_GRADE} as positives) …"
    )
    for golden_query in dataset.queries:
        # Positives: hotels with relevance grade ≥ MIN_POSITIVE_GRADE
        positive_ids = [
            doc_id
            for doc_id, grade in golden_query.graded_relevance.items()
            if grade >= MIN_POSITIVE_GRADE
        ]
        positive_texts = [hotel_texts[hid] for hid in positive_ids if hid in hotel_texts]

        if not positive_texts:
            queries_without_positives += 1
            continue

        # BM25 top-CANDIDATES_K for this query (no hard filters — raw lexical)
        bm25_params = LexicalSearchParams(query=golden_query.query_text, top_k=CANDIDATES_K)
        bm25_result = lexical_search(client, bm25_params, index=index)
        bm25_ids = [hit.id for hit in bm25_result.hits]

        # Hard negatives: in BM25 results but NOT in any graded relevance judgement
        known_relevant = set(golden_query.graded_relevance.keys())  # any grade ≥ 1
        hard_negative_ids = select_hard_negatives(bm25_ids, known_relevant)
        negative_texts = [hotel_texts[hid] for hid in hard_negative_ids if hid in hotel_texts]

        if not negative_texts:
            queries_without_negatives += 1
            continue

        pairs = build_training_pairs(
            golden_query.query_text,
            positive_texts,
            negative_texts,
            max_pairs_per_query=MAX_PAIRS_PER_QUERY,
        )
        if pairs:
            all_pairs.extend(pairs)
            queries_with_pairs += 1

    # ── Write output ──────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(
                json.dumps(
                    {"query": pair.query, "positive": pair.positive, "negative": pair.negative},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nDone.")
    print(f"  {len(all_pairs):,} training triplets written to {OUTPUT_PATH}")
    print(f"  {queries_with_pairs} / {len(dataset)} queries contributed pairs")
    if queries_without_positives:
        print(
            f"  INFO: {queries_without_positives} queries had no positives"
            f" at grade ≥ {MIN_POSITIVE_GRADE}"
        )
    if queries_without_negatives:
        print(
            f"  INFO: {queries_without_negatives} queries had no hard negatives"
            f" in BM25 top-{CANDIDATES_K}"
        )
    print()
    print("Next step: make fine-tune-embeddings")
    print()


if __name__ == "__main__":
    main()
