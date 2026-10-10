"""Evaluate the refusal policy on answerable and unsupported questions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from abstention_policy import AbstentionPolicy
from production_retrieval import ProductionRetrieval
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


EVALUATION = PROJECT_ROOT / "data" / "evaluation"
QUESTIONS = EVALUATION / "questions.jsonl"
UNSUPPORTED = EVALUATION / "abstention_questions.jsonl"
OUTPUT = EVALUATION / "abstention_evaluation.json"
CACHE = PROJECT_ROOT / "data" / "processed" / "frozen_dense_embeddings.npz"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    configure_utf8_stdout()
    started = perf_counter()
    retrieval = ProductionRetrieval(
        local_files_only=True,
        embedding_cache_path=CACHE,
        embedding_device="cpu",
        reranker_device="cpu",
    )
    policy = AbstentionPolicy()
    cases = [
        {
            "case_id": row["question_id"],
            "paper_id": row["paper_id"],
            "question_language": row["question_language"],
            "question": row["question"],
            "expected_sufficiency": "sufficient",
            "split": "answerable",
        }
        for row in load_jsonl(QUESTIONS)
    ] + load_jsonl(UNSUPPORTED)

    records = []
    for index, case in enumerate(cases, start=1):
        output = retrieval.run(case["question"], case["paper_id"])
        decision = policy.evaluate_retrieval(output)
        predicted = "insufficient" if decision["abstain"] else "sufficient"
        records.append(
            {
                "case_id": case["case_id"],
                "paper_id": case["paper_id"],
                "question_language": case["question_language"],
                "split": case["split"],
                "expected_sufficiency": case["expected_sufficiency"],
                "predicted_sufficiency": predicted,
                "correct": predicted == case["expected_sufficiency"],
                "decision": decision,
            }
        )
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} expected="
            f"{case['expected_sufficiency']} predicted={predicted}",
            flush=True,
        )

    split_counts = Counter(row["split"] for row in records)
    split_correct = Counter(row["split"] for row in records if row["correct"])
    summary = {
        "answerable_retained": split_correct["answerable"],
        "answerable_total": split_counts["answerable"],
        "answerable_retention": round(
            split_correct["answerable"] / split_counts["answerable"], 4
        ),
        "calibration_refused": split_correct["calibration"],
        "calibration_total": split_counts["calibration"],
        "calibration_refusal_rate": round(
            split_correct["calibration"] / split_counts["calibration"], 4
        ),
        "holdout_refused": split_correct["holdout"],
        "holdout_total": split_counts["holdout"],
        "holdout_refusal_rate": round(
            split_correct["holdout"] / split_counts["holdout"], 4
        ),
    }
    payload = {
        "schema_version": 1,
        "evaluation": "abstention-policy",
        "seed": 378,
        "summary": summary,
        "provenance": {
            "questions": "data/evaluation/questions.jsonl",
            "questions_sha256": sha256(QUESTIONS),
            "unsupported_questions": "data/evaluation/abstention_questions.jsonl",
            "unsupported_questions_sha256": sha256(UNSUPPORTED),
            "policy": "config/abstention.json",
            "policy_sha256": sha256(PROJECT_ROOT / "config" / "abstention.json"),
            "production_config_fingerprint": retrieval.fingerprint,
        },
        "runtime_seconds": round(perf_counter() - started, 3),
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
