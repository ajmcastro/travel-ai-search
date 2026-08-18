# Annotation guide — human query slice

This guide explains how to write queries and grade hotel relevance for the
human-annotated query slice used in the generator-effect evaluation (Milestone 16).

The purpose of the human slice is to provide an independent evaluation signal
that was not influenced by the same process that generated the synthetic hotel
corpus.  The key requirement is that **annotators must never see the dataset
generation code or prompts**.

---

## Who should annotate

- Anyone who has experience planning or researching travel holidays.
- Annotators must not be the person who wrote `scripts/generate_dataset.py`.
- Multiple annotators for the same query reduce individual bias. If two
  annotators disagree by more than one grade on more than 30% of hotels,
  discuss and resolve before including those judgments.

---

## Step 1 — Read a sample of hotel descriptions

Before writing any queries, read 30–50 hotel descriptions from
`data/processed/hotels.jsonl`.  Get a feel for what kinds of hotels exist,
what language is used, and what distinguishes them.

Do **not** read the generation script or any documentation about how the
descriptions were produced.

---

## Step 2 — Write queries

Write queries as a real traveller would type them into a search box.
Aim for 15–30 queries covering a variety of intent types:

| Query class | Example |
|---|---|
| `exact_destination` | "holiday in Tenerife", "hotels in Mallorca" |
| `family` | "family beach holiday with kids", "all-inclusive family resort" |
| `adults_couples` | "romantic adults-only break", "couples spa retreat" |
| `luxury` | "five star hotel with spa", "luxury beachfront resort" |
| `budget` | "cheap beach holiday under £800", "budget hotel good reviews" |
| `nightlife` | "party resort with clubs and beach", "lively holiday nightlife" |
| `quiet_peaceful` | "peaceful retreat away from crowds", "quiet island no tourists" |
| `activities` | "diving and watersports holiday", "hiking adventure in mountains" |
| `natural_language` | "somewhere relaxing with good food and culture" |
| `multi_constraint` | "family all-inclusive Lanzarote July under £1500" |

Rules for writing queries:
- Write in first person or as a plain search phrase — not a sentence like "I want
  a hotel that has..."
- Do not include field names ("star_rating:", "country:") — write as a user would.
- Include at least two queries per class if possible.
- Cover edge cases: queries unlikely to match anything in the corpus (tests
  calibration of both retrieval and judge).

Save the queries to `data/evaluation/human_queries.jsonl` in this format:

```json
{"query_id": "hq001", "query_text": "beach holiday with kids", "query_class": "family", "source": "human"}
```

---

## Step 3 — Grade the results (optional — needed for agreement-rate measurement)

To compute the `agreement_rate` metric (how often the LLM judge agrees with
human judgment), you need human relevance grades for at least some queries.

For each query, run the default retrieval strategy and look at the top-10 results:

```bash
uv run python scripts/evaluate_judge.py --strategy rrf --slice human --k 10 --no-save
```

For each (query, hotel) pair, assign a grade:

| Grade | Meaning |
|---|---|
| 3 | Highly relevant — excellent match; this is exactly what the query is looking for. |
| 2 | Relevant — matches most of the query intent; a user would likely be satisfied. |
| 1 | Marginally relevant — some overlap but notable gaps or mismatches. |
| 0 | Irrelevant — does not match the query. |

Save grades alongside the query in the JSONL file using the standard judgment
format (same as `golden_queries.jsonl`):

```json
{
  "query_id": "hq001",
  "query_text": "beach holiday with kids",
  "query_class": "family",
  "source": "human",
  "judgments": [
    {"doc_id": "hotel_001234", "grade": 3},
    {"doc_id": "hotel_005678", "grade": 1}
  ]
}
```

If no judgments are provided, the LLM judge scores serve as the sole relevance
signal for the human slice.  `agreement_rate` will be `null` in the results.

---

## Step 4 — Statistical validity note

The human slice is intentionally small (15–30 queries).  At this size:

- Confidence intervals on per-strategy metric differences are wide.
- The Spearman ρ between generated and human slice rankings should be treated
  as an indication of the generator-effect magnitude, not a precise estimate.
- A minimum of 50 human queries per class would be needed for reliable
  per-class comparisons.

Report sample sizes alongside all results.  Never present human-slice rankings
as statistically definitive when n < 30.

---

## Frequently asked questions

**Can I annotate queries I wrote myself?**
Yes, but try to annotate at least some queries written by a different person.
Self-annotation introduces an anchoring bias (you tend to grade highly the
hotels you implicitly thought of when writing the query).

**What if I am not sure whether a hotel is grade 2 or grade 3?**
Use grade 2 when uncertain.  Reserve grade 3 for clear, unambiguous excellent
matches.  The key distinction for the evaluation is grade ≥ 2 vs grade < 2.

**How long does annotation take?**
About 5–10 minutes per query when reading descriptions carefully.  A set of
20 queries with 10 hotels each takes 2–4 hours for careful annotation.
