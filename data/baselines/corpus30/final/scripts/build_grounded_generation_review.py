"""Build the reviewed answer-quality artifact for the fixed evaluation set."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


EVALUATION = PROJECT_ROOT / "data" / "evaluation"
OUTPUTS = EVALUATION / "grounded_generation_outputs.json"
LABELS = EVALUATION / "grounded_generation_review_labels.jsonl"
MANIFEST = EVALUATION / "answer_review_manifest.json"
GOLD = EVALUATION / "gold_evidence.jsonl"
REVIEW = EVALUATION / "grounded_generation_review.json"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    configure_utf8_stdout()
    outputs = json.loads(OUTPUTS.read_text(encoding="utf-8"))
    labels = {row["question_id"]: row for row in load_jsonl(LABELS)}
    gold = {row["question_id"]: row for row in load_jsonl(GOLD)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    categories = {row["question_id"]: row["category"] for row in manifest["questions"]}
    results = {row["question_id"]: row for row in outputs["results"]}
    ordered_ids = [row["question_id"] for row in manifest["questions"]]
    expected = set(ordered_ids)
    if len(expected) != 20 or set(results) != expected or set(labels) != expected:
        raise ValueError("Review inputs must contain the same fixed 20 questions")

    records = []
    for question_id in ordered_ids:
        result = results[question_id]
        reference = gold[question_id]
        label = labels[question_id]
        records.append(
            {
                "question_id": question_id,
                "paper_id": result["paper_id"],
                "question_language": result["question_language"],
                "review_category": categories[question_id],
                "question": result["question"],
                "expected": {
                    "gold_answer": reference["gold_answer"],
                    "gold_page": reference["gold_page"],
                },
                "actual": {
                    "answer": result["response"]["answer"],
                    "sufficiency": result["response"]["sufficiency"],
                    "evidence_ids": result["response"]["evidence_ids"],
                    "selected_pages": list(
                        dict.fromkeys(
                            sentence["page"] for sentence in result["selected_sentences"]
                        )
                    ),
                    "runtime_seconds": result["runtime_seconds"],
                },
                "review": label,
            }
        )

    verdict_counts = Counter(record["review"]["verdict"] for record in records)
    verdicts = {name: verdict_counts[name] for name in ("pass", "partial", "fail")}
    citation_counts = Counter(record["review"]["citation_support"] for record in records)
    summary = {
        "questions_reviewed": len(records),
        "languages": dict(Counter(row["question_language"] for row in records)),
        "categories": dict(Counter(row["review_category"] for row in records)),
        "verdicts": verdicts,
        "strict_pass_rate": round(verdicts["pass"] / len(records), 4),
        "pass_or_partial_rate": round(
            (verdicts["pass"] + verdicts["partial"]) / len(records), 4
        ),
        "citation_support": {
            name: citation_counts[name] for name in ("pass", "fail", "not_applicable")
        },
        "language_match": sum(row["review"]["language_match"] for row in records),
        "false_abstentions": sum(
            "false_abstention" in row["review"]["failure_types"] for row in records
        ),
        "fixed_evidence_runtime_seconds": outputs["summary"]["runtime_seconds"],
        "fixed_evidence_generation_attempts": outputs["summary"]["generation_attempts"],
    }
    payload = {
        "schema_version": 1,
        "evaluation": "evidence-first-answer-quality",
        "review_date": "2026-09-12",
        "review_method": "Assistant-led semantic review against verified Gold; no independent human review was performed.",
        "gold_loaded_during_generation": False,
        "provenance": {
            "outputs": str(OUTPUTS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "outputs_sha256": sha256(OUTPUTS),
            "manual_labels": str(LABELS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "manual_labels_sha256": sha256(LABELS),
            "question_manifest": str(MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "question_manifest_sha256": sha256(MANIFEST),
            "verified_gold": str(GOLD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "verified_gold_sha256": sha256(GOLD),
        },
        "rubric": {
            "pass": "The answer is materially correct and complete enough, uses the question language, and its citations support the claims.",
            "partial": "The core answer is supported but a material detail, qualification, number, or comparison is missing.",
            "fail": "The response falsely abstains, is incorrect or nonresponsive, omits the requested result, or makes an unsupported material claim.",
            "citation_support": "Pass means the cited text supports the generated claims; not_applicable is used only when no answer or citation was returned.",
        },
        "summary": summary,
        "known_limits": [
            "The 20 records represent 13 distinct question meanings because some questions are bilingual pairs.",
            "The review was not performed by an independent human evaluator.",
            "Two answerable page-1 questions are rejected by the pre-generation relevance gate.",
            "Some answers remain partial because the fixed Top-5 evidence omits verified details.",
            "The NLLB translation model license limits this demo to noncommercial use.",
        ],
        "records": records,
    }
    REVIEW.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Grounded generation review built")
    print(
        f"pass={verdicts['pass']} partial={verdicts['partial']} fail={verdicts['fail']} "
        f"pass_or_partial={summary['pass_or_partial_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
