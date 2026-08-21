"""Fine-tune the dense bi-encoder with contrastive learning — Milestone 19.

Requires:
    make prepare-fine-tuning-data   (writes data/evaluation/fine_tuning_pairs.jsonl)

After training, the model is saved to data/models/bi-encoder-travel/ and can be
loaded by LocalEmbeddingProvider, which accepts local paths and HuggingFace IDs
identically (both are passed to SentenceTransformer()).

Training setup
--------------
Loss: InfoNCE / MultipleNegativesRankingLoss — implemented as a manual PyTorch
training loop to avoid the `datasets` library dependency that model.fit() now
requires in sentence-transformers 5.x.

The loss is:
    cross_entropy(q @ [p; n].T / τ, arange(B))

where:
  q  — (B, d) L2-normalised query embeddings
  p  — (B, d) L2-normalised positive hotel embeddings
  n  — (B, d) L2-normalised hard-negative hotel embeddings
  τ  — temperature (default 0.07)
  ;  — vertical concatenation

  Row i of the similarity matrix q @ [p; n].T has:
    - position i   → sim(q_i, p_i), the true positive (label)
    - all other positions → in-batch negatives (other positives) + hard negatives

Larger batches produce harder training because each anchor sees more in-batch
negatives.  With B=16 each anchor sees 15 in-batch + 1 hard = 16 negatives.

Model: all-MiniLM-L6-v2 (or whatever EMBEDDING_MODEL_NAME is set to in .env).
  - 384-dim, 22 M params, fast on CPU.

After training, verify the improvement:
    FINE_TUNED_EMBEDDING_MODEL_PATH=data/models/bi-encoder-travel
    make evaluate-fine-tuned

Usage
-----
    make fine-tune-embeddings
    # or directly:
    uv run python scripts/fine_tune_embeddings.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings

PAIRS_PATH = Path("data/evaluation/fine_tuning_pairs.jsonl")
OUTPUT_DIR = Path("data/models/bi-encoder-travel")

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
TEMPERATURE = 0.07  # cosine similarity scale before cross-entropy


class _TripletDataset:
    """In-memory dataset of (query, positive, negative) string triplets."""

    def __init__(self, pairs: list[dict[str, str]]) -> None:
        self._data: list[tuple[str, str, str]] = [
            (p["query"], p["positive"], p["negative"]) for p in pairs
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self._data[idx]


def _encode(model: Any, sentences: list[str]) -> Any:
    """Encode sentences with gradient tracking via the underlying HuggingFace model.

    Bypasses sentence-transformers' tokenization API (changed in v5.x) and uses
    the HuggingFace AutoTokenizer/AutoModel directly, then applies mean pooling
    weighted by the attention mask — identical to how all-MiniLM-L6-v2 was
    pre-trained.  Returns an L2-normalised (B, d) tensor connected to the graph.
    """
    import torch
    import torch.nn.functional as F

    # model._first_module() → Transformer (has .tokenizer and .auto_model)
    transformer: Any = model._first_module()
    device = next(model.parameters()).device

    encoded: Any = transformer.tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=getattr(transformer, "max_seq_length", 256),
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    hf_output: Any = transformer.auto_model(**encoded)
    token_embs: Any = hf_output.last_hidden_state  # (B, seq_len, hidden)

    # Weighted mean pooling
    mask: Any = encoded["attention_mask"].unsqueeze(-1).to(torch.float32)
    mean_embs: Any = (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    return F.normalize(mean_embs, p=2, dim=-1)


def main() -> None:
    settings = get_settings()
    model_name = settings.embedding_model_name  # defaults to "all-MiniLM-L6-v2"

    # ── Check prerequisites ───────────────────────────────────────────────────
    if not PAIRS_PATH.exists():
        print(f"\nERROR: Training pairs not found at {PAIRS_PATH}")
        print("  Generate them with:  make prepare-fine-tuning-data")
        sys.exit(1)

    # ── Load training triplets ────────────────────────────────────────────────
    print(f"\nLoading training pairs from {PAIRS_PATH} …")
    raw = [
        json.loads(line)
        for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  {len(raw):,} triplets loaded")

    if not raw:
        print("\nERROR: No training pairs found. Re-run: make prepare-fine-tuning-data")
        sys.exit(1)

    # Lazy imports — torch and sentence-transformers are heavy
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    dataset = _TripletDataset(raw)
    # collate_fn=list: keep each batch as a list[tuple] — no tensor collation
    train_dataloader: DataLoader[tuple[str, str, str]] = DataLoader(
        dataset,  # type: ignore[arg-type]
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=list,
    )

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"\nLoading base model '{model_name}' …")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # ── Optimiser + linear warmup/decay scheduler ─────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        remaining = total_steps - step
        decay_steps = total_steps - warmup_steps
        return max(0.0, float(remaining) / float(max(1, decay_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    print("\nFine-tuning configuration:")
    print(f"  base model:    {model_name}")
    print(f"  triplets:      {len(raw):,}")
    print(f"  batch_size:    {BATCH_SIZE}")
    print(f"  epochs:        {EPOCHS}")
    print(f"  lr:            {LEARNING_RATE}")
    print(f"  temperature:   {TEMPERATURE}")
    print(f"  warmup_steps:  {warmup_steps} / {total_steps} total")
    print(f"  output_dir:    {OUTPUT_DIR}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_train = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            # batch is list[tuple[str, str, str]] due to collate_fn=list
            queries = [item[0] for item in batch]
            positives = [item[1] for item in batch]
            negatives = [item[2] for item in batch]

            # Three forward passes; gradients accumulate into shared parameters
            q_embs = _encode(model, queries)  # (B, d) L2-normalised
            p_embs = _encode(model, positives)  # (B, d)
            n_embs = _encode(model, negatives)  # (B, d)

            # Target matrix: positives (in-batch) + explicit hard negatives
            all_targets = torch.cat([p_embs, n_embs], dim=0)  # (2B, d)
            sim_matrix = (q_embs @ all_targets.T) / TEMPERATURE  # (B, 2B)

            # Correct match for query i is p_embs[i] → position i in targets
            labels = torch.arange(len(queries), device=q_embs.device)
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}  lr={current_lr:.2e}")

    elapsed = time.perf_counter() - t_train
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── Save fine-tuned model ─────────────────────────────────────────────────
    model.save(str(OUTPUT_DIR))
    print(f"  Model saved to: {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print(f"  1. Add to .env:  FINE_TUNED_EMBEDDING_MODEL_PATH={OUTPUT_DIR}")
    print("  2. Run:          make evaluate-fine-tuned")
    print()


if __name__ == "__main__":
    main()
