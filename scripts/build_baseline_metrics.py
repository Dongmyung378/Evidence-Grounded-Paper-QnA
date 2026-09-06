"""Build the Day 18 baseline metric artifact from the fixed comparison run."""

import argparse
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "baseline_metrics.json"
REQUIRED_METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")


def compact_metrics(metrics):
    compact = {
        "questions": metrics["questions"],
        **{name: metrics[name] for name in REQUIRED_METRICS},
    }
    compact["by_language"] = {
        language: {
            "questions": values["questions"],
            **{name: values[name] for name in REQUIRED_METRICS},
        }
        for language, values in metrics["by_language"].items()
    }
    return compact


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    comparison = json.loads(args.input.read_text(encoding="utf-8"))
    config = comparison["config"]
    metrics = comparison["metrics"]
    if set(metrics) != {"bm25", "dense", "hybrid_rrf"}:
        raise ValueError("Expected BM25, dense, and hybrid_rrf metrics")
    if any(values["questions"] != 40 for values in metrics.values()):
        raise ValueError("Day 18 baseline must use all 40 verified questions")

    artifact = {
        "schema_version": 1,
        "roadmap_day": 18,
        "evaluation_set": {
            "name": "verified-40",
            "questions": 40,
            "languages": {"en": 20, "ko": 20},
            "papers": 10,
            "relevance_unit": "gold evidence page",
            "ranking_unit": "unique PDF page",
        },
        "corpus": {"papers": 10, "pages": 188, "chunks": 906},
        "retrieval_config": {
            "model": config["model"],
            "candidate_k": config["candidate_k"],
            "rrf_k": config["rrf_k"],
            "fusion_weights": config.get("fusion_weights", {"bm25": 1.0, "dense": 1.0}),
            "evaluation_top_k": config["evaluation_top_k"],
        },
        "metrics": {
            name: compact_metrics(values)
            for name, values in metrics.items()
        },
        "selected_baseline": "hybrid_rrf",
        "source": str(args.input.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
