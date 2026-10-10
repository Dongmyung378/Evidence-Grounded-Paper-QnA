"""Fit and report question-aspect relevance thresholds on declared dev outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from answerability_policy import CONFIG_PATH, score_answerability
from model_lock import locked_model
from quality_benchmark import load_dataset
from reranker import load_reranker
from retrieval_common import PROJECT_ROOT


def select_threshold(rows, minimum_retention=0.95, retention_rows=None):
    retention_rows = retention_rows or []
    if not 0 < minimum_retention <= 1 or not rows:
        raise ValueError("Require development rows and a positive retention constraint")
    if any(row.get("split") != "dev" for row in rows + retention_rows):
        raise ValueError("Threshold fitting may use dev rows only")
    if any(row.get("answerability") != "answerable" for row in retention_rows):
        raise ValueError("Additional retention constraints must be declared answerable dev cases")
    if not any(row["answerability"] == "answerable" for row in rows) or not any(row["answerability"] != "answerable" for row in rows):
        raise ValueError("Calibration requires both answerable and unsupported examples")
    values = sorted(set(row["signals"]["max_aspect_score"] for row in rows + retention_rows
                        if row["signals"]["max_aspect_score"] is not None))
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("No specific-aspect scores were available for calibration")
    thresholds = [values[0] - 0.000001] + [(left + right) / 2 for left, right in zip(values, values[1:])] + [values[-1] + 0.000001]
    groups = {}
    for row in rows:
        groups.setdefault(row["meaning_id"], []).append(row)
    positive = [pair for pair in groups.values() if pair[0]["answerability"] == "answerable"]
    negative = [pair for pair in groups.values() if pair[0]["answerability"] != "answerable"]

    def accepted(row, threshold, include_coverage=True):
        if include_coverage and not row["signals"].get("evidence_count", 1):
            return False
        if include_coverage and row["signals"].get("missing_requested_attributes"):
            return False
        score = row["signals"]["max_aspect_score"]
        return score is None or score >= threshold

    candidates = []
    for threshold in thresholds:
        retention = sum(all(accepted(row, threshold) for row in pair) for pair in positive) / len(positive)
        refusal = sum(all(not accepted(row, threshold) for row in pair) for pair in negative) / len(negative)
        candidates.append({"threshold": threshold, "answerable_group_retention": retention,
                           "unsupported_group_refusal_recall": refusal,
                           "additional_dev_retention": all(accepted(row, threshold) for row in retention_rows)})
    eligible = [candidate for candidate in candidates if candidate["answerable_group_retention"] >= minimum_retention
                and candidate["additional_dev_retention"]]
    if not eligible:
        raise ValueError("No threshold satisfies declared dev retention constraints")
    # Retention is an explicit constraint. Within it, prefer the stricter
    # threshold when two thresholds have the same grouped refusal recall.
    best = max(eligible, key=lambda candidate: (candidate["unsupported_group_refusal_recall"], candidate["threshold"]))
    pure_refusal = sum(all(not accepted(row, best["threshold"], False) for row in pair)
                       for pair in negative) / len(negative)
    return {"selected": best, "minimum_answerable_group_retention": minimum_retention,
            "selection_priority": "maximum_grouped_refusal_then_strictest_threshold_within_declared_dev_retention",
            "fully_separable_by_relevance_threshold": pure_refusal == 1,
            "fully_separable_by_combined_policy": best["unsupported_group_refusal_recall"] == 1,
            "additional_answerable_dev_constraints": len(retention_rows),
            "candidate_thresholds": candidates,
            "answerable_groups": len(positive), "unsupported_or_false_premise_groups": len(negative)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--write-policy", action="store_true")
    parser.add_argument("--minimum-retention", type=float, default=.95)
    parser.add_argument("--query-transform", choices=["source_derived_topic_debias_v1", "source_derived_topic_debias_v2"],
                        default="source_derived_topic_debias_v2")
    parser.add_argument("--retention-predictions", type=Path,
                        help="Explicitly declared answerable dev records to preserve in addition to fitting cases")
    args = parser.parse_args()
    protected = (PROJECT_ROOT / "data").resolve()
    output = args.output.resolve()
    if output.is_relative_to(protected) or output == args.predictions.resolve() or (args.retention_predictions and output == args.retention_predictions.resolve()) or output.is_relative_to((PROJECT_ROOT / "config").resolve()) or output.is_relative_to((PROJECT_ROOT / "scripts").resolve()):
        parser.error("Write calibration reports outside inputs, frozen data, config and scripts")
    config, manifest, questions, rubric = load_dataset()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if predictions.get("benchmark_id") != config["benchmark_id"] or predictions.get("execution", {}).get("split") != "dev":
        parser.error("Calibration input must explicitly declare this benchmark's dev split")
    development = {case["case_id"]: case for case in questions if case["split"] == "dev"}
    records = predictions.get("records", [])
    if len(records) != len(development) or {row.get("case_id") for row in records} != set(development):
        parser.error("Calibration predictions must contain every dev case exactly once and no holdout case")
    lock = locked_model("reranker")
    selector = load_reranker(lock["name"], revision=lock["revision"], local_files_only=args.offline, device="cpu")
    rows = []
    for record in records:
        case = development[record["case_id"]]
        evidence = (record.get("output") or {}).get("retrieved_evidence")
        if not evidence:
            parser.error(f"Saved original candidate evidence is required: {case['case_id']}")
        signals = score_answerability(case["question"], evidence, selector, args.query_transform)
        rows.append({"case_id": case["case_id"], "meaning_id": case["meaning_id"],
                     "split": "dev", "language": case["language"], "answerability": case["answerability"],
                     "question": case["question"], "signals": signals})
        print(f"{case['case_id']}: {signals['max_aspect_score']} aspect={signals['query']}", flush=True)
    retention_rows = []
    retention_sha256 = None
    if args.retention_predictions:
        retention = json.loads(args.retention_predictions.read_text(encoding="utf-8"))
        if retention.get("execution", {}).get("split") != "dev":
            parser.error("Additional retention inputs must explicitly declare dev")
        records = retention.get("records")
        if not isinstance(records, list) or not records:
            parser.error("Additional retention inputs require nonempty records")
        ids = [record.get("case_id") for record in records]
        if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(set(ids)) != len(ids):
            parser.error("Retention cases require unique nonempty case IDs")
        for record in records:
            if record.get("split") != "dev" or record.get("answerability") != "answerable":
                parser.error("Retention records must be labelled answerable dev cases")
            evidence = record.get("retrieved_evidence")
            if not evidence or not isinstance(record.get("question"), str):
                parser.error("Retention inputs require saved original evidence and a question")
            signals = score_answerability(record["question"], evidence, selector, args.query_transform)
            retention_rows.append({"case_id": record["case_id"], "split": "dev", "answerability": "answerable",
                                   "question": record["question"], "language": record.get("language"),
                                   "signals": signals})
        retention_sha256 = hashlib.sha256(args.retention_predictions.read_bytes()).hexdigest()
    fitted = select_threshold(rows, args.minimum_retention, retention_rows)
    report = {"schema_version": 1, "execution_mode": "real_models", "split": "dev",
              "benchmark_id": config["benchmark_id"], "purpose": manifest.get("purpose"),
              "gold_used_in_runtime": False, "labels_used_for_threshold_fitting_only": True,
              "independent_human_review": False, "real_paper_accuracy_claim": False,
              "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256((PROJECT_ROOT / "scripts/answerability_policy.py").read_bytes()).hexdigest(),
              "query_transform": args.query_transform,
              "retention_predictions_sha256": retention_sha256,
              "additional_retention_rows": retention_rows,
              "reranker": lock, "threshold_fit": fitted, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_policy:
        policy = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        policy["enabled"] = True
        policy["query_transform"] = args.query_transform
        policy["minimum_aspect_score"] = fitted["selected"]["threshold"]
        policy["calibration"] = {"status": "dev_calibrated", "split": "dev", "execution_mode": "real_models",
                                 "query_transform": args.query_transform,
                                 "purpose": report["purpose"],
                                 "independent_human_review": False, "real_paper_accuracy_claim": False,
                                 "benchmark_id": config["benchmark_id"], "predictions_sha256": report["predictions_sha256"],
                                 "implementation_sha256": report["implementation_sha256"],
                                 "retention_predictions_sha256": retention_sha256,
                                 "additional_answerable_dev_constraints": len(retention_rows),
                                 "reranker": lock, "observed": fitted["selected"],
                                 "fully_separable_by_relevance_threshold": fitted["fully_separable_by_relevance_threshold"],
                                 "fully_separable_by_combined_policy": fitted["fully_separable_by_combined_policy"]}
        CONFIG_PATH.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(fitted["selected"], ensure_ascii=False))


if __name__ == "__main__":
    main()
