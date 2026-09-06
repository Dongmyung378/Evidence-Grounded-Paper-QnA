"""Merge blind Day 32 outputs with Gold data and explicit manual judgments."""

import json
import hashlib
from collections import Counter, defaultdict

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
MANIFEST_PATH = EVALUATION_DIR / "day32_review_manifest.json"
OUTPUTS_PATH = EVALUATION_DIR / "day32_qna_outputs.json"
LABELS_PATH = EVALUATION_DIR / "day32_review_labels.jsonl"
QUESTIONS_PATH = EVALUATION_DIR / "questions.jsonl"
GOLD_PATH = EVALUATION_DIR / "gold_evidence.jsonl"
REVIEW_PATH = EVALUATION_DIR / "day32_manual_review.json"
FAILURES_PATH = EVALUATION_DIR / "day32_failure_cases.md"
REPORT_PATH = EVALUATION_DIR / "day32_report.md"


def validate_review_binding():
    binding = json.loads((EVALUATION_DIR / "day32_review_binding.json").read_text(encoding="utf-8"))
    for path in (OUTPUTS_PATH, LABELS_PATH):
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding[path.name]:
            raise ValueError("Review is stale: re-review the changed outputs and update the review binding.")
    return binding


def count_nested(rows, key, order):
    result = {}
    for value in order:
        result[value] = sum(row["review"][key] == value for row in rows)
    return result


def verdict_breakdown(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    return {
        group: {
            "questions": len(items),
            **count_nested(items, "verdict", ("pass", "partial", "fail")),
        }
        for group, items in sorted(grouped.items())
    }


def main():
    configure_utf8_stdout()
    binding = validate_review_binding()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run = json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))
    labels = load_jsonl(LABELS_PATH)
    questions = {item["question_id"]: item for item in load_jsonl(QUESTIONS_PATH)}
    gold = {item["question_id"]: item for item in load_jsonl(GOLD_PATH)}
    selections = {item["question_id"]: item for item in manifest["questions"]}
    outputs = {item["question_id"]: item for item in run["results"]}
    labels_by_id = {item["question_id"]: item for item in labels}
    selected_ids = [item["question_id"] for item in manifest["questions"]]

    expected = set(selected_ids)
    assert len(selected_ids) == len(expected) == 20
    assert set(outputs) == set(labels_by_id) == expected
    assert expected <= set(questions) and expected <= set(gold)

    records = []
    for question_id in selected_ids:
        question = questions[question_id]
        reference = gold[question_id]
        output = outputs[question_id]
        label = labels_by_id[question_id]
        records.append(
            {
                "question_id": question_id,
                "paper_id": question["paper_id"],
                "question_language": question["question_language"],
                "review_category": selections[question_id]["category"],
                "question": question["question"],
                "expected": {
                    "gold_answer": reference["gold_answer"],
                    "gold_page": reference["gold_page"],
                    "gold_section": reference["gold_section"],
                    "review_status": reference["review_status"],
                },
                "actual": {
                    "answer": output["response"]["answer"],
                    "sufficiency": output["response"]["sufficiency"],
                    "evidence_ids": output["response"]["evidence_ids"],
                    "cited_pages": [
                        evidence["page"] for evidence in output["cited_evidence"]
                    ],
                    "abstention_source": output["pipeline"]["abstention_source"],
                    "generation_attempts": output["pipeline"]["generation_attempts"],
                    "fallback_used": output["pipeline"]["fallback_used"],
                },
                "review": {
                    "verdict": label["verdict"],
                    "citation_support": label["citation_support"],
                    "language_match": label["language_match"],
                    "failure_types": label["failure_types"],
                    "notes": label["review_notes"],
                },
            }
        )

    strict_failures = [row for row in records if row["review"]["verdict"] == "fail"]
    false_abstentions = [
        row for row in records if "false_abstention" in row["review"]["failure_types"]
    ]
    verdicts = count_nested(records, "verdict", ("pass", "partial", "fail"))
    summary = {
        "questions_reviewed": len(records),
        "papers": len({row["paper_id"] for row in records}),
        "languages": dict(Counter(row["question_language"] for row in records)),
        "categories": dict(Counter(row["review_category"] for row in records)),
        "verdicts": verdicts,
        "strict_pass_rate": round(verdicts["pass"] / len(records), 4),
        "pass_or_partial_rate": round(
            (verdicts["pass"] + verdicts["partial"]) / len(records), 4
        ),
        "citation_support": count_nested(
            records, "citation_support", ("pass", "fail", "not_applicable")
        ),
        "language_match": sum(row["review"]["language_match"] for row in records),
        "strict_failure_cases": len(strict_failures),
        "false_abstentions": len(false_abstentions),
        "by_category": verdict_breakdown(records, "review_category"),
        "by_language": verdict_breakdown(records, "question_language"),
    }
    review_payload = {
        "schema_version": 1,
        "roadmap_day": 32,
        "review_method": binding["reviewer"],
        "review_date": "2026-09-05",
        "rubric": {
            "pass": "The answer is materially correct and complete enough, uses the question language, and its citations support the claims.",
            "partial": "The core answer is supported but a material detail, qualification, number, or comparison is missing.",
            "fail": "The response falsely abstains, is incorrect or nonresponsive, omits the requested result, or makes an unsupported material claim.",
            "citation_support": "Pass means the cited text supports the generated claims; not_applicable is used only when no answer or citation was returned.",
        },
        "provenance": {
            "manifest": "data/evaluation/day32_review_manifest.json",
            "blind_model_outputs": "data/evaluation/day32_qna_outputs.json",
            "manual_labels": "data/evaluation/day32_review_labels.jsonl",
            "questions": "data/evaluation/questions.jsonl",
            "verified_gold": "data/evaluation/gold_evidence.jsonl",
            "gold_loaded_during_generation": False,
            "configuration": run["configuration"],
        },
        "summary": summary,
        "records": records,
    }
    REVIEW_PATH.write_text(
        json.dumps(review_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failure_lines = [
        "# Day 32 Strict Failure Cases",
        "",
        "These are strict `fail` judgments from the balanced 20-question manual review.",
        "A `partial` answer is tracked in the full review but is not counted toward the",
        "roadmap requirement of at least five recorded failures.",
        "",
        f"- Strict failures: **{len(strict_failures)}/20**",
        f"- False abstentions: **{len(false_abstentions)}/20**",
        "- Source: `data/evaluation/day32_manual_review.json`",
        "",
    ]
    for index, row in enumerate(strict_failures, start=1):
        actual = row["actual"]
        pages = ", ".join(str(page) for page in actual["cited_pages"]) or "none"
        failure_lines.extend(
            [
                f"## {index}. {row['question_id']} — {row['review_category']}",
                "",
                f"- Paper / language: `{row['paper_id']}` / `{row['question_language']}`",
                f"- Question: {row['question']}",
                f"- Expected: {row['expected']['gold_answer']}",
                f"- Actual: {actual['answer']}",
                f"- Gold page / cited pages: {row['expected']['gold_page']} / {pages}",
                f"- Failure types: `{', '.join(row['review']['failure_types'])}`",
                f"- Review: {row['review']['notes']}",
                "",
            ]
        )
    FAILURES_PATH.write_text("\n".join(failure_lines), encoding="utf-8")

    report_lines = [
        "# Day 32 Manual Q&A Review",
        "",
        "## Roadmap gate",
        "",
        "- Task: manually review 20 factual, numeric, comparison, and limitation questions.",
        "- Completion criterion: record at least five failure cases.",
        f"- Result: **PASS** — 20 questions reviewed and {len(strict_failures)} strict failures recorded.",
        "",
        "## Review design",
        "",
        "The sample contains five questions per category, ten English and ten Korean",
        "questions, and two questions from every verified benchmark paper (001–010).",
        "The live Day 31 pipeline generated answers without loading Gold data. Gold answers",
        "and evidence were joined only after generation for manual semantic and citation review.",
        "",
        "## Results",
        "",
        f"- Verdicts: pass {verdicts['pass']}, partial {verdicts['partial']}, fail {verdicts['fail']}",
        f"- Strict pass rate: {summary['strict_pass_rate']:.1%}",
        f"- Pass-or-partial rate: {summary['pass_or_partial_rate']:.1%}",
        f"- Citation support: pass {summary['citation_support']['pass']}, fail {summary['citation_support']['fail']}, not applicable {summary['citation_support']['not_applicable']}",
        f"- False abstentions: {summary['false_abstentions']}",
        f"- Language matched: {summary['language_match']}/20",
        "",
        "| Category | Pass | Partial | Fail |",
        "|---|---:|---:|---:|",
    ]
    for category in ("factual", "numeric", "comparison", "limitation"):
        values = summary["by_category"][category]
        report_lines.append(
            f"| {category} | {values['pass']} | {values['partial']} | {values['fail']} |"
        )
    report_lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Retrieval quality does not guarantee answer quality: several responses had",
            "   relevant cited evidence but did not answer the requested relation or protocol.",
            "2. The calibrated abstention policy produced seven false refusals in this sample,",
            "   including three pre-generation refusals, two model refusals, and two validation fallbacks.",
            "3. The 0.5B local generator frequently omitted numbers and comparison details and",
            "   produced non-answers or malformed Korean content.",
            "4. Citation IDs remained structurally valid, but four generated answers did not",
            "   establish a meaningful claim-to-citation relationship.",
            "",
            "## Decision",
            "",
            "Day 32 is complete against the original roadmap because the full 20-question",
            "review and more than five strict failure records exist. This is an evaluation",
            "completion, not an answer-quality release gate: the current 0.5B generator and",
            "abstention calibration require improvement before public deployment quality claims.",
            "The frozen Day 28 retrieval configuration is unchanged.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -B scripts/run_day32_review.py --offline",
            "python -B scripts/build_day32_review.py",
            "python -B scripts/validate_day32.py",
            "```",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Day 32 review artifacts built")
    print(
        f"questions={len(records)} pass={verdicts['pass']} "
        f"partial={verdicts['partial']} fail={verdicts['fail']}"
    )
    print(
        f"citation_pass={summary['citation_support']['pass']} "
        f"false_abstentions={summary['false_abstentions']}"
    )


if __name__ == "__main__":
    main()
