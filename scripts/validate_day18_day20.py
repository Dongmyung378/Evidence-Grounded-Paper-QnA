"""Validate the completion gates for roadmap Days 18 through 20."""

import json

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


METRIC_NAMES = {"recall_at_1", "recall_at_5", "recall_at_10", "mrr"}
EXPECTED_CATEGORIES = {"chunking", "terminology", "numbers", "multiple_evidence"}


def load(name):
    path = PROJECT_ROOT / "data" / "evaluation" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    configure_utf8_stdout()
    baseline = load("baseline_metrics.json")
    failures = load("failure_analysis.json")
    experiment = load("day20_improvement.json")

    assert baseline["roadmap_day"] == 18
    assert baseline["evaluation_set"]["questions"] == 40
    assert baseline["evaluation_set"]["languages"] == {"en": 20, "ko": 20}
    assert baseline["selected_baseline"] == "hybrid_rrf"
    for method in ("bm25", "dense", "hybrid_rrf"):
        assert baseline["metrics"][method]["questions"] == 40
        assert METRIC_NAMES <= set(baseline["metrics"][method])

    assert failures["roadmap_day"] == 19
    assert failures["reviewed_cases"] == 10
    assert len(failures["cases"]) == 10
    assert set(failures["category_counts"]) == EXPECTED_CATEGORIES
    assert all(case["missed_by"] for case in failures["cases"])

    assert experiment["roadmap_day"] == 20
    assert experiment["evaluation_set"] == baseline["evaluation_set"]
    assert experiment["controlled_change"]["parameter"] == "fusion_weights"
    assert experiment["controlled_change"]["baseline"] == {"bm25": 1.0, "dense": 1.0}
    fixed = experiment["controlled_change"]["fixed"]
    assert fixed["questions"] == 40
    assert fixed["corpus"] == baseline["corpus"]
    assert fixed["model"] == baseline["retrieval_config"]["model"]
    assert fixed["candidate_k"] == baseline["retrieval_config"]["candidate_k"]
    assert fixed["rrf_k"] == baseline["retrieval_config"]["rrf_k"]
    for side in ("baseline", "candidate"):
        assert METRIC_NAMES <= set(experiment[side]["metrics"])

    delta = experiment["delta_candidate_minus_baseline"]["all"]
    print("Day 18-20 gate validation passed")
    print("baseline_metrics=Recall@1/5/10+MRR on 40 questions")
    print(f"failure_cases={failures['reviewed_cases']} categories={','.join(sorted(EXPECTED_CATEGORIES))}")
    print(f"day20_recall_at_5_delta={delta['recall_at_5']:+.3f}")
    print(f"selected={experiment['decision']['selected_configuration']}")


if __name__ == "__main__":
    main()
