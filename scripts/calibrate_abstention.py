"""Measure Day 31 retrieval signals on answerable and unsupported questions."""

import argparse
import json
import os
import statistics
import time

from production_retrieval import ProductionRetrieval
from retrieval_common import (
    PROJECT_ROOT,
    configure_utf8_stdout,
    load_evaluation_data,
    load_jsonl,
)


NEGATIVE_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "abstention_questions.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day31_abstention_calibration.json"
)


def retrieval_signals(output):
    scores = [float(item["reranker_score"]) for item in output["candidates"]]
    top_five = scores[:5]
    return {
        "top_reranker_score": scores[0],
        "second_reranker_score": scores[1],
        "top_score_margin": scores[0] - scores[1],
        "top5_median_score": statistics.median(top_five),
        "positive_score_count": sum(score >= 0.0 for score in scores),
        "candidate_count": len(scores),
        "evidence_count": len(output["evidence"]),
    }


def summarize(rows):
    top_scores = [row["signals"]["top_reranker_score"] for row in rows]
    medians = [row["signals"]["top5_median_score"] for row in rows]
    return {
        "cases": len(rows),
        "top_reranker_score": {
            "minimum": min(top_scores),
            "median": statistics.median(top_scores),
            "maximum": max(top_scores),
        },
        "top5_median_score": {
            "minimum": min(medians),
            "median": statistics.median(medians),
            "maximum": max(medians),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load Hugging Face models from the local cache only.",
    )
    return parser.parse_args()


def main():
    configure_utf8_stdout()
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    retrieval = ProductionRetrieval()
    answerable, _ = load_evaluation_data()
    unsupported = [
        item
        for item in load_jsonl(NEGATIVE_PATH)
        if item["split"] == "calibration"
    ]
    rows = []

    groups = (
        ("answerable", answerable),
        ("unsupported", unsupported),
    )
    total = sum(len(items) for _, items in groups)
    for group, items in groups:
        for item in items:
            case_id = item.get("question_id", item.get("case_id"))
            output = retrieval.run(item["question"], item["paper_id"])
            row = {
                "case_id": case_id,
                "group": group,
                "paper_id": item["paper_id"],
                "question_language": item["question_language"],
                "question": item["question"],
                "signals": retrieval_signals(output),
            }
            rows.append(row)
            print(
                f"[{len(rows):02d}/{total}] {case_id} group={group} "
                f"top={row['signals']['top_reranker_score']:.4f} "
                f"top5_median={row['signals']['top5_median_score']:.4f}"
            )

    answerable_rows = [row for row in rows if row["group"] == "answerable"]
    unsupported_rows = [row for row in rows if row["group"] == "unsupported"]
    artifact = {
        "schema_version": 1,
        "roadmap_day": 31,
        "purpose": "abstention threshold calibration without LLM labels",
        "retrieval_config_fingerprint": retrieval.fingerprint,
        "dataset": {
            "answerable": "verified-40",
            "unsupported": (
                str(NEGATIVE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")
                + ":calibration"
            ),
        },
        "summary": {
            "answerable": summarize(answerable_rows),
            "unsupported": summarize(unsupported_rows),
        },
        "rows": rows,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
