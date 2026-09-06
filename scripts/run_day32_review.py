"""Run the balanced Day 32 manual-review sample without exposing Gold data."""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from abstention_policy import AbstentionPolicy
from local_llm import generation_config_fingerprint, load_generation_config
from production_retrieval import config_fingerprint, load_production_config
from qna_pipeline import GroundedQAPipeline
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day32_review_manifest.json"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "day32_qna_outputs.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"


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
    requested_output = Path(args.output)
    if not requested_output.is_absolute():
        requested_output = PROJECT_ROOT / requested_output
    if requested_output.resolve() == OUTPUT_PATH.resolve() and OUTPUT_PATH.exists():
        raise FileExistsError("The reviewed Day 32 output is frozen. Use --output with a new filename, then review the new run separately.")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    question_by_id = {
        item["question_id"]: item for item in load_jsonl(QUESTIONS_PATH)
    }
    selected = []
    for selection in manifest["questions"]:
        question = question_by_id[selection["question_id"]]
        selected.append({**question, "review_category": selection["category"]})

    assert len(selected) == 20
    assert len({item["question_id"] for item in selected}) == 20
    assert Counter(item["review_category"] for item in selected) == {
        "factual": 5,
        "numeric": 5,
        "comparison": 5,
        "limitation": 5,
    }
    assert Counter(item["question_language"] for item in selected) == {
        "en": 10,
        "ko": 10,
    }
    assert Counter(item["paper_id"] for item in selected) == {
        f"paper-{index:03d}": 2 for index in range(1, 11)
    }

    started = time.perf_counter()
    # This script reproduces the original Day 31 policy; current experiments use
    # evaluate_answer_improvement.py and evaluate_abstention_improvement.py.
    pipeline = GroundedQAPipeline(local_files_only=args.offline,
                                 abstention_config_path=PROJECT_ROOT / "config/abstention.json")
    results = []
    for index, question in enumerate(selected, start=1):
        output = pipeline.ask(
            question["question"],
            question["paper_id"],
            question_language=question["question_language"],
            include_debug=True,
        )
        output["question_id"] = question["question_id"]
        output["review_category"] = question["review_category"]
        results.append(output)
        print(
            f"[{index:02d}/20] {question['question_id']} "
            f"category={question['review_category']} "
            f"status={output['response']['sufficiency']} "
            f"source={output['pipeline']['abstention_source'] or 'model_answer'} "
            f"seconds={output['pipeline']['runtime_seconds']:.3f}"
        )

    production = load_production_config()
    generation = load_generation_config()
    policy = AbstentionPolicy()
    summary = {
        "questions": len(results),
        "papers": len({item["paper_id"] for item in results}),
        "categories": dict(Counter(item["review_category"] for item in results)),
        "languages": dict(Counter(item["question_language"] for item in results)),
        "sufficiency": dict(
            Counter(item["response"]["sufficiency"] for item in results)
        ),
        "pre_generation_refusals": sum(
            item["pipeline"]["abstention_source"] == "pre_generation_policy"
            for item in results
        ),
        "model_refusals": sum(
            item["pipeline"]["abstention_source"] == "model"
            for item in results
        ),
        "validation_fallbacks": sum(
            item["pipeline"]["fallback_used"] for item in results
        ),
        "llm_invocations": sum(
            item["pipeline"]["generation_attempts"] for item in results
        ),
        "responses_with_citations": sum(
            bool(item["response"]["evidence_ids"]) for item in results
        ),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "schema_version": 1,
        "roadmap_day": 32,
        "purpose": "blind Q&A execution for subsequent manual Gold review",
        "gold_data_loaded_by_runner": False,
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "configuration": {
            "retrieval_config_fingerprint": config_fingerprint(production),
            "generation_config_fingerprint": generation_config_fingerprint(generation),
            "abstention_config_fingerprint": policy.fingerprint,
            "model": generation["model"],
            "model_revision": generation["revision"],
            "offline": args.offline,
        },
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
