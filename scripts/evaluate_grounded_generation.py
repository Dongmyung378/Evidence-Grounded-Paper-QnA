"""Run the evidence-first answer candidate on the fixed 20-question review set."""

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from abstention_policy import AbstentionPolicy
from grounded_generation import (
    CONFIG_PATH,
    generate_evidence_first_answer,
    load_answer_generation_config,
)
from local_translation import CONFIG_PATH as TRANSLATION_CONFIG_PATH
from local_translation import LocalEnglishKoreanTranslator
from reranker import load_reranker
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


BASELINE_PATH = PROJECT_ROOT / "data" / "evaluation" / "day32_qna_outputs.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "grounded_generation_outputs.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_signals(item):
    return item["pipeline"]["abstention"]["decision"]["signals"]


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--translation-config",
        type=Path,
        default=TRANSLATION_CONFIG_PATH,
    )
    parser.add_argument("--answer-config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    args.translation_config = args.translation_config.resolve()
    args.answer_config = args.answer_config.resolve()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import set_seed

    set_seed(378)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    answer_config = load_answer_generation_config(args.answer_config)
    abstention = AbstentionPolicy(PROJECT_ROOT / "config" / "abstention_candidate.json")
    selector = load_reranker(
        answer_config["sentence_selector"]["model"],
        max_length=answer_config["sentence_selector"]["max_length"],
        local_files_only=True,
        device="cpu",
    )
    generator = LocalEnglishKoreanTranslator(
        args.translation_config,
        local_files_only=True,
    )

    results = []
    started = time.perf_counter()
    for item in baseline["results"]:
        row_started = time.perf_counter()
        decision = abstention.decide_from_signals(saved_signals(item))
        if decision["abstain"]:
            response = abstention.refusal_payload(
                item["question_language"], decision["reason_code"]
            )
            generated = {
                "selected_sentences": [],
                "attempts": [],
                "fallback_used": False,
            }
            abstention_source = "pre_generation_policy"
        else:
            generated = generate_evidence_first_answer(
                generator,
                selector,
                item["query"],
                item["retrieved_evidence"],
                item["question_language"],
                answer_config,
            )
            response = generated["response"]
            abstention_source = (
                "validation_fallback"
                if generated["fallback_used"]
                else ("model" if response["sufficiency"] == "insufficient" else None)
            )
        result = {
            "question_id": item["question_id"],
            "paper_id": item["paper_id"],
            "question": item["query"],
            "question_language": item["question_language"],
            "response": response,
            "selected_sentences": generated["selected_sentences"],
            "attempts": generated["attempts"],
            "fallback_used": generated["fallback_used"],
            "abstention_source": abstention_source,
            "runtime_seconds": round(time.perf_counter() - row_started, 3),
        }
        results.append(result)
        print(
            f"{result['question_id']} | {response['sufficiency']} | "
            f"{result['runtime_seconds']:.3f}s | {response['answer']}",
            flush=True,
        )

    sufficiency = Counter(item["response"]["sufficiency"] for item in results)
    payload = {
        "schema_version": 1,
        "evaluation": "fixed-20-answer-quality",
        "seed": 378,
        "gold_loaded_during_generation": False,
        "method": "Current abstention threshold, saved Day 32 retrieval evidence, Cross-Encoder sentence compression, plain answer generation, deterministic citation assembly and numeric claim validation.",
        "provenance": {
            "baseline_outputs": str(BASELINE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "baseline_sha256": sha256(BASELINE_PATH),
            "answer_config": str(args.answer_config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "answer_config_sha256": sha256(args.answer_config),
            "implementation_sha256": sha256(Path(__file__).with_name("grounded_generation.py")),
            "translation_config": str(args.translation_config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "translation_config_sha256": sha256(args.translation_config),
            "abstention_config": "config/abstention_candidate.json",
        },
        "models": {
            "translator": generator.metadata(),
            "sentence_selector": answer_config["sentence_selector"]["model"],
        },
        "summary": {
            "questions": len(results),
            "sufficient": sufficiency["sufficient"],
            "insufficient": sufficiency["insufficient"],
            "pre_generation_refusals": sum(
                item["abstention_source"] == "pre_generation_policy" for item in results
            ),
            "model_refusals": sum(item["abstention_source"] == "model" for item in results),
            "validation_fallbacks": sum(item["fallback_used"] for item in results),
            "generation_attempts": sum(len(item["attempts"]) for item in results),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
