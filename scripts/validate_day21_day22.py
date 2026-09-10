"""Validate roadmap completion gates for Days 21 and 22."""

import json

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


METRICS = {"recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"}


def load(name):
    path = PROJECT_ROOT / "data" / "evaluation" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    configure_utf8_stdout()
    demo = load("retrieval_demo.json")
    reranker = load("reranker_metrics.json")

    assert demo["roadmap_day"] == 21
    assert demo["evaluation_set"]["questions"] == 40
    assert set(demo["baseline_table"]) == {"bm25", "dense", "hybrid_rrf"}
    assert demo["selected_baseline"] == "hybrid_rrf"
    assert len(demo["examples"]) == 3
    hybrid_hits = [row["method_results"]["hybrid_rrf"]["hit_at_5"] for row in demo["examples"]]
    assert any(hybrid_hits) and not all(hybrid_hits)
    assert all(row["diagnosis"] and row["next_action"] for row in demo["examples"])

    assert reranker["roadmap_day"] == 22
    assert reranker["evaluation_set"]["questions"] == 40
    assert reranker["evaluation_set"]["languages"] == {"en": 20, "ko": 20}
    assert reranker["config"]["candidate_k"] == 20
    assert reranker["config"]["candidate_source"].startswith("equal-weight RRF")
    assert set(reranker["metrics"]) == {"hybrid_top20", "hybrid_reranker"}
    for method in ("hybrid_top20", "hybrid_reranker"):
        assert reranker["metrics"][method]["questions"] == 40
        assert METRICS <= set(reranker["metrics"][method])
        assert len(reranker["results"][method]) == 40
    before = reranker["metrics"]["hybrid_top20"]
    after = reranker["metrics"]["hybrid_reranker"]
    print("Day 21-22 gate validation passed")
    print("day21=baseline_table+3_case_explainable_demo")
    print(f"day22_questions={after['questions']} reranker={reranker['config']['reranker_model']}")
    print(f"recall_at_5={before['recall_at_5']:.3f}->{after['recall_at_5']:.3f}")
    print(f"mrr={before['mrr']:.3f}->{after['mrr']:.3f}")


if __name__ == "__main__":
    main()
