"""Run the Day 20 one-variable weighted-RRF experiment."""

import argparse
import json
from pathlib import Path

from dense_retrieval import DEFAULT_MODEL
from evaluate_retrievers import evaluate_all
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


DEFAULT_BASELINE = PROJECT_ROOT / "data" / "evaluation" / "baseline_metrics.json"
DEFAULT_COMPARISON = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "day20_improvement.json"
METRIC_NAMES = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")


def metric_slice(metrics):
    return {
        "questions": metrics["questions"],
        **{name: metrics[name] for name in METRIC_NAMES},
        "by_language": {
            language: {
                "questions": values["questions"],
                **{name: values[name] for name in METRIC_NAMES},
            }
            for language, values in metrics["by_language"].items()
        },
    }


def delta(candidate, baseline):
    return {
        name: round(candidate[name] - baseline[name], 12)
        for name in METRIC_NAMES
    }


def is_hit(rank, k=5):
    return rank is not None and rank <= k


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bm25-weight", type=float, default=0.8)
    parser.add_argument("--dense-weight", type=float, default=1.2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    baseline_artifact = json.loads(args.baseline.read_text(encoding="utf-8"))
    original_comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    baseline_config = baseline_artifact["retrieval_config"]
    if baseline_artifact["evaluation_set"]["questions"] != 40:
        raise ValueError("Day 20 experiment must use the fixed 40-question set")

    candidate_run = evaluate_all(
        model_name=args.model,
        batch_size=args.batch_size,
        candidate_k=baseline_config["candidate_k"],
        rrf_k=baseline_config["rrf_k"],
        top_k=baseline_config["evaluation_top_k"],
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )
    baseline_metrics = baseline_artifact["metrics"]["hybrid_rrf"]
    candidate_metrics = metric_slice(candidate_run["metrics"]["hybrid_rrf"])
    baseline_rows = {row["question_id"]: row for row in original_comparison["results"]["hybrid_rrf"]}
    candidate_rows = {row["question_id"]: row for row in candidate_run["results"]["hybrid_rrf"]}
    if set(baseline_rows) != set(candidate_rows) or len(candidate_rows) != 40:
        raise ValueError("Baseline and candidate do not contain the same 40 questions")

    rank_changes = []
    recovered_at_5 = []
    lost_at_5 = []
    for question_id in sorted(baseline_rows):
        before = baseline_rows[question_id]
        after = candidate_rows[question_id]
        if before["gold_rank"] != after["gold_rank"]:
            rank_changes.append({
                "question_id": question_id,
                "paper_id": before["paper_id"],
                "language": before["language"],
                "gold_page": before["gold_page"],
                "baseline_gold_rank": before["gold_rank"],
                "candidate_gold_rank": after["gold_rank"],
            })
        before_hit = is_hit(before["gold_rank"])
        after_hit = is_hit(after["gold_rank"])
        if not before_hit and after_hit:
            recovered_at_5.append(question_id)
        elif before_hit and not after_hit:
            lost_at_5.append(question_id)

    metric_delta = delta(candidate_metrics, baseline_metrics)
    language_delta = {
        language: delta(candidate_metrics["by_language"][language], baseline_metrics["by_language"][language])
        for language in ("en", "ko")
    }
    improved = metric_delta["recall_at_5"] > 0
    selected = "candidate_weighted_rrf" if improved else "baseline_equal_weight_rrf"
    artifact = {
        "schema_version": 1,
        "roadmap_day": 20,
        "experiment": "weighted RRF single-variable comparison",
        "evaluation_set": baseline_artifact["evaluation_set"],
        "controlled_change": {
            "parameter": "fusion_weights",
            "baseline": baseline_config["fusion_weights"],
            "candidate": {"bm25": args.bm25_weight, "dense": args.dense_weight},
            "fixed": {
                "corpus": baseline_artifact["corpus"],
                "questions": 40,
                "model": baseline_config["model"],
                "candidate_k": baseline_config["candidate_k"],
                "rrf_k": baseline_config["rrf_k"],
                "evaluation_top_k": baseline_config["evaluation_top_k"],
            },
        },
        "baseline": {
            "name": "hybrid_equal_weight_rrf",
            "metrics": baseline_metrics,
        },
        "candidate": {
            "name": "hybrid_weighted_rrf",
            "metrics": candidate_metrics,
            "embedding_cache": candidate_run["config"]["embedding_cache"],
        },
        "delta_candidate_minus_baseline": {
            "all": metric_delta,
            "by_language": language_delta,
        },
        "question_level_changes": {
            "recovered_at_5": recovered_at_5,
            "lost_at_5": lost_at_5,
            "rank_changes": rank_changes,
        },
        "decision": {
            "primary_metric": "recall_at_5",
            "candidate_improved_primary_metric": improved,
            "selected_configuration": selected,
            "reason": (
                "가중 RRF가 동일 평가 세트에서 Recall@5를 높여 후보 설정을 채택했다."
                if improved else
                "가중 RRF가 동일 평가 세트에서 Recall@5를 높이지 못해 기존 동일 가중치를 유지했다."
            ),
        },
        "reproduction_command": (
            f"python -B scripts/run_day20_experiment.py --bm25-weight {args.bm25_weight} "
            f"--dense-weight {args.dense_weight}"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": {name: baseline_metrics[name] for name in METRIC_NAMES},
        "candidate": {name: candidate_metrics[name] for name in METRIC_NAMES},
        "delta": metric_delta,
        "recovered_at_5": recovered_at_5,
        "lost_at_5": lost_at_5,
        "selected": selected,
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
