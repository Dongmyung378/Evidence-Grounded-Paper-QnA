"""Build the human-readable comparison for the evidence-first answer experiment."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


EVALUATION = PROJECT_ROOT / "data" / "evaluation"
OUTPUTS = EVALUATION / "grounded_generation_outputs.json"
LABELS = EVALUATION / "grounded_generation_review_labels.jsonl"
BASELINE = EVALUATION / "day32_manual_review.json"
BASELINE_RUNTIME = EVALUATION / "answer_runtime_outputs.json"
MANIFEST = EVALUATION / "day32_review_manifest.json"
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


def verdict_counts(records):
    counts = Counter(record["review"]["verdict"] for record in records)
    return {name: counts[name] for name in ("pass", "partial", "fail")}


def main():
    configure_utf8_stdout()
    outputs = json.loads(OUTPUTS.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_runtime = json.loads(BASELINE_RUNTIME.read_text(encoding="utf-8"))
    labels = {row["question_id"]: row for row in load_jsonl(LABELS)}
    gold = {row["question_id"]: row for row in load_jsonl(GOLD)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    categories = {
        row["question_id"]: row["category"] for row in manifest["questions"]
    }
    results = {row["question_id"]: row for row in outputs["results"]}
    expected = set(categories)
    if len(expected) != 20 or set(results) != expected or set(labels) != expected:
        raise ValueError("review inputs must contain the same fixed 20 questions")

    records = []
    for question_id in [row["question_id"] for row in manifest["questions"]]:
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
                            sentence["page"]
                            for sentence in result["selected_sentences"]
                        )
                    ),
                    "runtime_seconds": result["runtime_seconds"],
                },
                "review": label,
            }
        )

    verdicts = verdict_counts(records)
    citation_counts = Counter(
        record["review"]["citation_support"] for record in records
    )
    baseline_verdicts = baseline["summary"]["verdicts"]
    candidate_seconds = outputs["summary"]["runtime_seconds"]
    baseline_seconds = round(
        sum(row.get("seconds", 0.0) for row in baseline_runtime["results"]),
        3,
    )
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
            name: citation_counts[name]
            for name in ("pass", "fail", "not_applicable")
        },
        "language_match": sum(row["review"]["language_match"] for row in records),
        "false_abstentions": sum(
            "false_abstention" in row["review"]["failure_types"]
            for row in records
        ),
        "fixed_evidence_runtime_seconds": candidate_seconds,
        "fixed_evidence_generation_attempts": outputs["summary"][
            "generation_attempts"
        ],
    }
    comparison = {
        "baseline": {
            "verdicts": baseline_verdicts,
            "strict_pass_rate": baseline["summary"]["strict_pass_rate"],
            "pass_or_partial_rate": baseline["summary"]["pass_or_partial_rate"],
            "false_abstentions": baseline["summary"]["false_abstentions"],
            "fixed_evidence_runtime_seconds": baseline_seconds,
        },
        "candidate": summary,
        "delta": {
            "pass": verdicts["pass"] - baseline_verdicts["pass"],
            "partial": verdicts["partial"] - baseline_verdicts["partial"],
            "fail": verdicts["fail"] - baseline_verdicts["fail"],
            "strict_pass_rate": round(
                summary["strict_pass_rate"]
                - baseline["summary"]["strict_pass_rate"],
                4,
            ),
            "pass_or_partial_rate": round(
                summary["pass_or_partial_rate"]
                - baseline["summary"]["pass_or_partial_rate"],
                4,
            ),
            "false_abstentions": summary["false_abstentions"]
            - baseline["summary"]["false_abstentions"],
            "fixed_evidence_runtime_seconds": round(
                candidate_seconds - baseline_seconds,
                3,
            ),
        },
    }
    payload = {
        "schema_version": 1,
        "evaluation": "evidence-first-answer-quality",
        "review_date": "2026-09-12",
        "review_method": "Assistant-led semantic comparison against verified Gold; no independent human review was performed.",
        "gold_loaded_during_generation": False,
        "provenance": {
            "candidate_outputs": str(OUTPUTS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "candidate_outputs_sha256": sha256(OUTPUTS),
            "manual_labels": str(LABELS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "manual_labels_sha256": sha256(LABELS),
            "baseline_review": str(BASELINE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "baseline_review_sha256": sha256(BASELINE),
            "fixed_question_manifest": str(MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "verified_gold": str(GOLD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "rubric": baseline["rubric"],
        "summary": summary,
        "comparison": comparison,
        "adoption": {
            "decision": "adopt",
            "reason": "The candidate improves strict and partial-or-better review rates, reduces false abstentions, preserves citation support, and reduces fixed-evidence answer time without accepting unsupported numeric claims.",
            "known_limits": [
                "The 20 records contain English/Korean pairs and represent 13 distinct question meanings.",
                "The review was not performed by an independent human evaluator.",
                "Two page-1 questions are still rejected by the pre-generation relevance gate.",
                "Some answers remain partial because the fixed Top-5 evidence omits verified details.",
                "The NLLB translation model is CC-BY-NC-4.0 and is used only in this local non-commercial portfolio demo.",
            ],
        },
        "records": records,
    }
    REVIEW.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Grounded generation review built")
    print(
        f"pass={verdicts['pass']} partial={verdicts['partial']} "
        f"fail={verdicts['fail']} pass_or_partial={summary['pass_or_partial_rate']:.2%}"
    )
    print(
        f"false_abstentions={summary['false_abstentions']} "
        f"fixed_evidence_seconds={candidate_seconds}"
    )


if __name__ == "__main__":
    main()
