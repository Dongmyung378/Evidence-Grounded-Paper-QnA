"""Run a tiny, non-production reranker optimizer step with QASPER pairs."""

import argparse
import json
import math
import random
from pathlib import Path

from reranker import DEFAULT_RERANKER_MODEL
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


DEFAULT_PAIRS = PROJECT_ROOT / "data" / "processed" / "reranker_pairs_train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "day25_finetuning_smoke.json"


def balanced_batch(rows, batch_size, seed):
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    if not positives or not negatives:
        raise ValueError("Training smoke test requires positive and negative pairs")
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = max(1, batch_size // 2)
    batch = positives[:half] + negatives[:max(1, batch_size - half)]
    rng.shuffle(batch)
    return batch


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    rows = load_jsonl(args.pairs)
    batch = balanced_batch(rows, args.batch_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to("cpu")
    encoded = tokenizer(
        [row["query"] for row in batch],
        [row["passage"] for row in batch],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([float(row["label"]) for row in batch], dtype=torch.float32)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.eval()
    with torch.no_grad():
        initial_loss = loss_function(model(**encoded).logits.reshape(-1), labels).item()
    model.train()
    optimizer.zero_grad()
    training_loss = loss_function(model(**encoded).logits.reshape(-1), labels)
    training_loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = loss_function(model(**encoded).logits.reshape(-1), labels).item()
    if not all(math.isfinite(value) for value in (initial_loss, training_loss.item(), final_loss)):
        raise RuntimeError("Non-finite loss in reranker smoke training")

    artifact = {
        "schema_version": 1,
        "roadmap_day": 25,
        "status": "completed_non_production_smoke",
        "model": args.model,
        "data_source": "QASPER train only",
        "verified_40_used": False,
        "device": "cpu",
        "optimizer_steps": 1,
        "batch_size": len(batch),
        "labels": {"positive": sum(row["label"] == 1 for row in batch), "negative": sum(row["label"] == 0 for row in batch)},
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "loss": {
            "before_step": initial_loss,
            "training_step": training_loss.item(),
            "after_step_same_batch": final_loss,
        },
        "checkpoint_saved": False,
        "selected_runtime_model": "pretrained",
        "decision": "The optimizer path is verified, but a one-step smoke model is not a valid production checkpoint. Keep the measured pretrained reranker until separate training data and a held-out evaluation justify replacement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
