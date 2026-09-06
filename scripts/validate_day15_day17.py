"""Validate Day 15–17 retrieval artifacts against the fixed evaluation set."""

import json
from pathlib import Path

from retrieval_common import configure_utf8_stdout


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "data" / "evaluation" / "retrieval_comparison.json"


def main():
    configure_utf8_stdout()
    assert COMPARISON_PATH.exists(), "Run: python scripts/evaluate_retrievers.py"

    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    config = comparison["config"]
    assert config["model"] == "intfloat/multilingual-e5-small"
    assert config["candidate_k"] == 20
    assert config["rrf_k"] == 60
    assert config["evaluation_top_k"] == 10

    for retriever in ("bm25", "dense", "hybrid_rrf"):
        metrics = comparison["metrics"][retriever]
        results = comparison["results"][retriever]
        assert metrics["questions"] == 40
        assert len(results) == 40
        assert set(metrics["by_language"]) == {"en", "ko"}
        assert all(len(row["predicted_pages"]) <= 10 for row in results)

    print("Day 15–17 retrieval validation passed")
    print("questions=40 retrievers=bm25,dense,hybrid_rrf")
    print(
        "recall_at_5="
        f"bm25:{comparison['metrics']['bm25']['recall_at_5']:.3f} "
        f"dense:{comparison['metrics']['dense']['recall_at_5']:.3f} "
        f"hybrid_rrf:{comparison['metrics']['hybrid_rrf']['recall_at_5']:.3f}"
    )


if __name__ == "__main__":
    main()
