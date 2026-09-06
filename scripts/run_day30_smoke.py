"""Run ten real local-LLM questions through the complete Day 30 pipeline."""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from qna_pipeline import GroundedQAPipeline
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "day30_e2e_results.json"
QUESTION_IDS = [
    "q-001-en",
    "q-005-ko",
    "q-007-en",
    "q-010-ko",
    "q-014-en",
    "q-017-ko",
    "q-021-en",
    "q-041-ko",
    "q-042-en",
    "q-029-ko",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
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
    question_by_id = {
        item["question_id"]: item
        for item in load_jsonl(
            PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
        )
    }
    selected = [question_by_id[question_id] for question_id in QUESTION_IDS]
    assert len(selected) == 10
    assert len({item["paper_id"] for item in selected}) == 10
    assert Counter(item["question_language"] for item in selected) == {
        "en": 5,
        "ko": 5,
    }

    started = time.perf_counter()
    pipeline = GroundedQAPipeline(
        local_files_only=args.offline,
        enable_abstention=False,
    )
    results = []
    for index, question in enumerate(selected, start=1):
        output = pipeline.ask(
            question["question"],
            question["paper_id"],
            question_language=question["question_language"],
            include_debug=True,
        )
        output["question_id"] = question["question_id"]
        results.append(output)
        print(
            f"[{index:02d}/10] {question['question_id']} "
            f"status={output['response']['sufficiency']} "
            f"fallback={output['pipeline']['fallback_used']} "
            f"seconds={output['pipeline']['runtime_seconds']:.3f}"
        )

    fallback_count = sum(item["pipeline"]["fallback_used"] for item in results)
    model_validated = len(results) - fallback_count
    total_invocations = sum(
        item["pipeline"]["generation_attempts"] for item in results
    )
    summary = {
        "questions": len(results),
        "papers": len({item["paper_id"] for item in results}),
        "languages": dict(Counter(item["question_language"] for item in results)),
        "completed": len(results),
        "valid_model_responses": model_validated,
        "safe_fallbacks": fallback_count,
        "llm_invocations": total_invocations,
        "sufficiency": dict(
            Counter(item["response"]["sufficiency"] for item in results)
        ),
        "responses_with_citations": sum(
            bool(item["response"]["evidence_ids"]) for item in results
        ),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "schema_version": 1,
        "roadmap_day": 30,
        "purpose": "ten-question local end-to-end Q&A smoke test",
        "quality_evaluation_deferred_to_day": 32,
        "question_ids": QUESTION_IDS,
        "summary": summary,
        "results": results,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
