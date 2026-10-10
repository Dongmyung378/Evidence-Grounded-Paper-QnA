"""Build the reviewed Docker Compose Q&A benchmark artifact."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


EVALUATION = PROJECT_ROOT / "data" / "evaluation"
CONFIG = PROJECT_ROOT / "config" / "container_qna_benchmark.json"
RESULTS = EVALUATION / "container_qna_results.json"
LABELS = EVALUATION / "container_qna_review_labels.jsonl"
GOLD = EVALUATION / "gold_evidence.jsonl"
OUTPUT = EVALUATION / "container_qna_review.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verdict_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["review"]["verdict"] for row in rows)
    return {name: counts[name] for name in ("pass", "partial", "fail")}


def language_review(rows: list[dict], language: str) -> dict:
    selected = [row for row in rows if row["language"] == language]
    verdicts = verdict_counts(selected)
    return {
        "questions": len(selected),
        "verdicts": verdicts,
        "strict_pass_rate": round(verdicts["pass"] / len(selected), 4),
        "pass_or_partial_rate": round(
            (verdicts["pass"] + verdicts["partial"]) / len(selected), 4
        ),
        "citation_support": sum(
            row["review"]["citation_support"] == "pass" for row in selected
        ),
        "language_match": sum(row["review"]["language_match"] for row in selected),
    }


def main() -> None:
    configure_utf8_stdout()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    labels = {row["question_id"]: row for row in load_jsonl(LABELS)}
    gold = {row["question_id"]: row for row in load_jsonl(GOLD)}
    selected_questions = [
        question for paper in config["papers"] for question in paper["questions"]
    ]
    expected_ids = [question["question_id"] for question in selected_questions]
    actual = {row["question_id"]: row for row in results["results"]}
    if len(expected_ids) != 10 or set(actual) != set(expected_ids):
        raise ValueError("Container results must match the fixed ten questions")
    if set(labels) != set(expected_ids):
        raise ValueError("Review labels must match the fixed ten questions")

    records = []
    for question in selected_questions:
        question_id = question["question_id"]
        result = actual[question_id]
        reference = gold[question_id]
        pages = list(dict.fromkeys(item["page"] for item in result["evidence"]))
        records.append(
            {
                "question_id": question_id,
                "paper_id": result["benchmark_paper_id"],
                "language": result["language"],
                "question": result["question"],
                "expected": {
                    "gold_answer": reference["gold_answer"],
                    "gold_page": reference["gold_page"],
                },
                "actual": {
                    "answer": result["answer"],
                    "sufficiency": result["sufficiency"],
                    "evidence_pages": pages,
                    "wall_seconds": result["wall_seconds"],
                    "api_runtime": result["api_runtime"],
                },
                "exact_gold_page_hit": reference["gold_page"] in pages,
                "review": labels[question_id],
            }
        )

    verdicts = verdict_counts(records)
    exact_page_hits = sum(row["exact_gold_page_hit"] for row in records)
    timing = results["summary"]["by_language"]
    english_api_mean = timing["en"]["api_total_seconds"]["mean"]
    korean_api_mean = timing["ko"]["api_total_seconds"]["mean"]
    first_analysis = results["summary"]["first_analysis_seconds"]
    later_analysis_mean = results["summary"]["later_analysis_seconds"]["mean"]
    summary = {
        "papers": len(results["papers"]),
        "questions": len(records),
        "languages": dict(Counter(row["language"] for row in records)),
        "verdicts": verdicts,
        "strict_pass_rate": round(verdicts["pass"] / len(records), 4),
        "pass_or_partial_rate": round(
            (verdicts["pass"] + verdicts["partial"]) / len(records), 4
        ),
        "citation_support": {
            "pass": sum(
                row["review"]["citation_support"] == "pass" for row in records
            ),
            "fail": sum(
                row["review"]["citation_support"] == "fail" for row in records
            ),
        },
        "language_match": sum(row["review"]["language_match"] for row in records),
        "exact_gold_page_hits": exact_page_hits,
        "exact_gold_page_hit_rate": round(exact_page_hits / len(records), 4),
        "by_language": {
            language: language_review(records, language) for language in ("en", "ko")
        },
        "container_contract": {
            "sufficient": results["summary"]["sufficient"],
            "insufficient": results["summary"]["insufficient"],
            "fallbacks": results["summary"]["fallbacks"],
            "evidence_responses": results["summary"]["evidence_responses"],
            "ui_http_status": results["ui_http_status"],
        },
        "runtime": {
            "build_seconds": results["build_seconds"],
            "first_analysis_seconds": first_analysis,
            "later_analysis_mean_seconds": later_analysis_mean,
            "first_to_later_analysis_ratio": round(
                first_analysis / later_analysis_mean, 4
            ),
            "english_api_mean_seconds": english_api_mean,
            "korean_api_mean_seconds": korean_api_mean,
            "korean_to_english_api_mean_ratio": round(
                korean_api_mean / english_api_mean, 4
            ),
            "paper_warning_count": sum(
                paper["warning_count"] for paper in results["papers"]
            ),
            "model_cache_bytes": results["offline_restart"][
                "cache_after_restart"
            ]["bytes"],
        },
    }
    payload = {
        "schema_version": 1,
        "evaluation": "docker-compose-qna-review",
        "review_date": results["evaluated_at"][:10],
        "review_method": (
            "Assistant-led semantic comparison against verified Gold and live returned "
            "evidence; no independent human review was performed."
        ),
        "gold_loaded_during_container_execution": results[
            "gold_loaded_during_execution"
        ],
        "provenance": {
            "benchmark_config": str(CONFIG.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "benchmark_config_sha256": sha256(CONFIG),
            "container_results": str(RESULTS.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "container_results_sha256": sha256(RESULTS),
            "manual_labels": str(LABELS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "manual_labels_sha256": sha256(LABELS),
            "verified_gold": str(GOLD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "verified_gold_sha256": sha256(GOLD),
        },
        "rubric": {
            "pass": "Correct, responsive, and supported by the returned evidence.",
            "partial": "Supported and useful, but missing detail or reduced by source or translation quality.",
            "fail": "Incorrect, unsupported, or not responsive to the question.",
        },
        "summary": summary,
        "language_decision": {
            "decision": "retain_korean_with_documented_limits",
            "reason": (
                "All five Korean responses matched the requested language and cited supporting "
                "source text, while all were at least partially correct. Korean remains useful "
                "for the local portfolio demo, but CPU latency and translation fluency are "
                "documented limitations rather than production guarantees."
            ),
        },
        "acceptance": {
            "docker_compose_flow": "pass",
            "answer_and_evidence_contract": "pass",
            "offline_cached_restart": "pass",
            "deterministic_restart_response": "pass",
            "benchmark_cleanup": "pass",
        },
        "known_limits": [
            "The review contains five bilingual question pairs from three papers, not ten independent meanings.",
            "The semantic labels were not assigned by an independent human evaluator.",
            "All ten benchmark questions are answerable, so abstention quality is covered only by the separate unsupported holdout evaluation.",
            "Exact verified Gold page overlap is four of ten even though manual inspection found all returned evidence passages supported their answers.",
            "Korean CPU answer time is materially slower than English and some translations are awkward or incomplete.",
            "The first analysis includes an empty-cache model download and initialization cost.",
        ],
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Container Q&A review built")
    print(
        f"pass={verdicts['pass']} partial={verdicts['partial']} "
        f"fail={verdicts['fail']} citation_support={summary['citation_support']['pass']}/10"
    )
    print(
        f"en_api_mean={english_api_mean:.3f}s "
        f"ko_api_mean={korean_api_mean:.3f}s decision=retain_korean_with_documented_limits"
    )


if __name__ == "__main__":
    main()
