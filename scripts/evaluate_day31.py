"""Evaluate Day 31 refusal behavior on a held-out unsupported-question set."""

import argparse
import json
import os
import time
from collections import Counter

from abstention_policy import AbstentionPolicy
from grounded_answer_contract import ABSTENTION_ANSWERS
from local_llm import generation_config_fingerprint
from qna_pipeline import GroundedQAPipeline
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "abstention_questions.jsonl"
CALIBRATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day31_abstention_calibration.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day31_abstention_results.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load retrieval models from the local cache only.",
    )
    return parser.parse_args()


def compact_result(case, result):
    pipeline = result["pipeline"]
    return {
        "case_id": case["case_id"],
        "paper_id": case["paper_id"],
        "question_language": case["question_language"],
        "question": case["question"],
        "expected_sufficiency": case["expected_sufficiency"],
        "response": result["response"],
        "abstention_source": pipeline["abstention_source"],
        "llm_skipped": pipeline["llm_skipped"],
        "generation_attempts": pipeline["generation_attempts"],
        "decision": pipeline["abstention"]["decision"],
        "runtime_seconds": pipeline["runtime_seconds"],
    }


def main():
    configure_utf8_stdout()
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cases = load_jsonl(CASES_PATH)
    holdout = [item for item in cases if item["split"] == "holdout"]
    assert len(holdout) == 10
    assert Counter(item["question_language"] for item in holdout) == {
        "en": 5,
        "ko": 5,
    }

    started = time.perf_counter()
    pipeline = GroundedQAPipeline(local_files_only=args.offline)
    results = []
    for index, case in enumerate(holdout, start=1):
        output = pipeline.ask(
            case["question"],
            case["paper_id"],
            question_language=case["question_language"],
        )
        row = compact_result(case, output)
        results.append(row)
        print(
            f"[{index:02d}/{len(holdout)}] {case['case_id']} "
            f"refused={row['response']['sufficiency'] == 'insufficient'} "
            f"top={row['decision']['signals']['top_reranker_score']:.4f} "
            f"llm_skipped={row['llm_skipped']}"
        )

    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    policy = AbstentionPolicy()
    answerable_rows = [
        row for row in calibration["rows"] if row["group"] == "answerable"
    ]
    calibration_unsupported = [
        row for row in calibration["rows"] if row["group"] == "unsupported"
    ]
    answerable_decisions = [
        (row["case_id"], policy.decide_from_signals(row["signals"]))
        for row in answerable_rows
    ]
    unsupported_decisions = [
        (row["case_id"], policy.decide_from_signals(row["signals"]))
        for row in calibration_unsupported
    ]
    false_abstentions = [
        case_id for case_id, decision in answerable_decisions if decision["abstain"]
    ]
    holdout_refused = sum(
        row["response"]["sufficiency"] == "insufficient"
        and row["response"]["answer"]
        == ABSTENTION_ANSWERS[row["question_language"]]
        and row["response"]["evidence_ids"] == []
        for row in results
    )
    calibration_refused = sum(
        decision["abstain"] for _, decision in unsupported_decisions
    )
    answerable_retained = len(answerable_decisions) - len(false_abstentions)
    summary = {
        "holdout_questions": len(results),
        "holdout_papers": len({row["paper_id"] for row in results}),
        "holdout_languages": dict(
            Counter(row["question_language"] for row in results)
        ),
        "holdout_refused": holdout_refused,
        "holdout_refusal_recall": holdout_refused / len(results),
        "calibration_unsupported_refused": calibration_refused,
        "calibration_unsupported_questions": len(unsupported_decisions),
        "answerable_retained": answerable_retained,
        "answerable_questions": len(answerable_decisions),
        "answerable_retention": answerable_retained / len(answerable_decisions),
        "pre_generation_false_abstentions": false_abstentions,
        "pre_generation_refused": sum(row["llm_skipped"] for row in results),
        "model_refused": sum(
            row["abstention_source"] == "model" for row in results
        ),
        "validation_fallbacks": sum(
            row["abstention_source"] == "validation_fallback"
            for row in results
        ),
        "llm_invocations": sum(row["generation_attempts"] for row in results),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    artifact = {
        "schema_version": 1,
        "roadmap_day": 31,
        "purpose": "held-out unsupported-question refusal evaluation",
        "policy": policy.metadata(),
        "retrieval_config_fingerprint": pipeline.retrieval.fingerprint,
        "generation": {
            "provider": pipeline.generation_config["provider"],
            "model": pipeline.generation_config["model"],
            "revision": pipeline.generation_config["revision"],
            "config_fingerprint": generation_config_fingerprint(
                pipeline.generation_config
            ),
        },
        "dataset": {
            "source": str(CASES_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "evaluation_split": "holdout",
            "calibration_split_excluded": True,
        },
        "summary": summary,
        "results": results,
    }
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_PATH}")

    gates = policy.config["calibration"]
    if summary["holdout_refusal_recall"] < gates[
        "minimum_unsupported_refusal_recall"
    ]:
        raise SystemExit("Day 31 holdout refusal gate failed")
    if summary["answerable_retention"] < gates["minimum_answerable_retention"]:
        raise SystemExit("Day 31 answerable-retention gate failed")
    if summary["validation_fallbacks"] != 0:
        raise SystemExit("Day 31 validation-fallback gate failed")


if __name__ == "__main__":
    main()
