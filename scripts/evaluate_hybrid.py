"""Evaluate RRF hybrid retrieval and save its metrics as a focused artifact."""

import argparse
import json
from pathlib import Path

from dense_retrieval import DEFAULT_MODEL
from evaluate_retrievers import evaluate_all
from retrieval_common import configure_utf8_stdout


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/hybrid_metrics.json"))
    args = parser.parse_args()

    comparison = evaluate_all(
        args.model,
        args.batch_size,
        args.candidate_k,
        args.rrf_k,
        args.top_k,
        args.bm25_weight,
        args.dense_weight,
    )
    hybrid = {
        "retriever": "hybrid_rrf",
        "config": comparison["config"],
        **comparison["metrics"]["hybrid_rrf"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(hybrid, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
