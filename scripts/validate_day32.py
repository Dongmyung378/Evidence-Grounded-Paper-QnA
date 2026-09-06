"""Validate the original-roadmap Day 32 manual Q&A review gate."""

import json
from build_day32_review import validate_review_binding
from collections import Counter

from abstention_policy import AbstentionPolicy
from grounded_answer_contract import validate_answer_payload
from local_llm import generation_config_fingerprint, load_generation_config
from production_retrieval import config_fingerprint, load_production_config
from retrieval_common import PROJECT_ROOT, load_jsonl


EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
MANIFEST_PATH = EVALUATION_DIR / "day32_review_manifest.json"
OUTPUTS_PATH = EVALUATION_DIR / "day32_qna_outputs.json"
LABELS_PATH = EVALUATION_DIR / "day32_review_labels.jsonl"
REVIEW_PATH = EVALUATION_DIR / "day32_manual_review.json"
FAILURES_PATH = EVALUATION_DIR / "day32_failure_cases.md"
REPORT_PATH = EVALUATION_DIR / "day32_report.md"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_day32_review.py"


def main():
    validate_review_binding()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run = json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))
    labels = load_jsonl(LABELS_PATH)
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    questions = {
        item["question_id"]: item
        for item in load_jsonl(EVALUATION_DIR / "questions.jsonl")
    }
    gold = {
        item["question_id"]: item
        for item in load_jsonl(EVALUATION_DIR / "gold_evidence.jsonl")
    }

    assert manifest["roadmap_day"] == 32
    selections = manifest["questions"]
    selected_ids = [item["question_id"] for item in selections]
    assert len(selected_ids) == len(set(selected_ids)) == 20
    assert Counter(item["category"] for item in selections) == {
        "factual": 5,
        "numeric": 5,
        "comparison": 5,
        "limitation": 5,
    }
    selected_questions = [questions[question_id] for question_id in selected_ids]
    assert Counter(item["question_language"] for item in selected_questions) == {
        "en": 10,
        "ko": 10,
    }
    assert Counter(item["paper_id"] for item in selected_questions) == {
        f"paper-{index:03d}": 2 for index in range(1, 11)
    }
    assert all(gold[question_id]["review_status"] == "verified" for question_id in selected_ids)

    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in ("gold_evidence", "gold_answer", "GOLD_PATH", "load_evaluation_data"):
        assert forbidden not in runner_source
    assert run["roadmap_day"] == 32
    assert run["gold_data_loaded_by_runner"] is False
    outputs = run["results"]
    assert len(outputs) == 20
    assert {item["question_id"] for item in outputs} == set(selected_ids)
    assert run["summary"]["questions"] == 20
    assert run["summary"]["papers"] == 10
    assert run["summary"]["categories"] == {
        "limitation": 5,
        "factual": 5,
        "comparison": 5,
        "numeric": 5,
    }
    assert run["summary"]["languages"] == {"en": 10, "ko": 10}
    assert sum(run["summary"]["sufficiency"].values()) == 20
    assert run["summary"]["runtime_seconds"] > 0

    production = load_production_config()
    generation = load_generation_config()
    policy = AbstentionPolicy()
    configuration = run["configuration"]
    assert configuration["retrieval_config_fingerprint"] == config_fingerprint(production)
    assert configuration["generation_config_fingerprint"] == generation_config_fingerprint(generation)
    assert configuration["abstention_config_fingerprint"] == policy.fingerprint
    assert configuration["model"] == generation["model"]
    assert configuration["model_revision"] == generation["revision"]
    for item in outputs:
        assert item["roadmap_day"] == 31
        assert item["roadmap_days"] == [30, 31]
        assert item["pipeline"]["abstention"]["enabled"] is True
        assert item["pipeline"]["retrieval_config_fingerprint"] == config_fingerprint(production)
        validate_answer_payload(
            item["response"], item["retrieved_evidence"], item["question_language"]
        )
        assert all(
            evidence["paper_id"] == item["paper_id"]
            for evidence in item["retrieved_evidence"]
        )

    assert len(labels) == len({item["question_id"] for item in labels}) == 20
    assert {item["question_id"] for item in labels} == set(selected_ids)
    assert all(item["verdict"] in {"pass", "partial", "fail"} for item in labels)
    assert all(
        item["citation_support"] in {"pass", "fail", "not_applicable"}
        for item in labels
    )
    assert all(isinstance(item["language_match"], bool) for item in labels)
    assert all(item["review_notes"] for item in labels)
    strict_fail_ids = {
        item["question_id"] for item in labels if item["verdict"] == "fail"
    }
    assert len(strict_fail_ids) >= 5

    assert review["roadmap_day"] == 32
    records = review["records"]
    assert len(records) == 20
    assert [item["question_id"] for item in records] == selected_ids
    summary = review["summary"]
    assert summary["questions_reviewed"] == 20
    assert summary["papers"] == 10
    assert summary["categories"] == {
        "limitation": 5,
        "factual": 5,
        "comparison": 5,
        "numeric": 5,
    }
    assert summary["languages"] == {"en": 10, "ko": 10}
    assert sum(summary["verdicts"].values()) == 20
    assert summary["verdicts"]["fail"] == len(strict_fail_ids)
    assert summary["strict_failure_cases"] == len(strict_fail_ids)
    assert summary["strict_failure_cases"] >= 5
    assert sum(summary["citation_support"].values()) == 20
    assert summary["language_match"] == 20
    for category in ("factual", "numeric", "comparison", "limitation"):
        values = summary["by_category"][category]
        assert values["questions"] == 5
        assert values["pass"] + values["partial"] + values["fail"] == 5
    for language in ("en", "ko"):
        values = summary["by_language"][language]
        assert values["questions"] == 10
        assert values["pass"] + values["partial"] + values["fail"] == 10

    failures_text = FAILURES_PATH.read_text(encoding="utf-8")
    assert "Strict failures:" in failures_text
    assert all(question_id in failures_text for question_id in strict_fail_ids)
    assert REPORT_PATH.exists() and "Result: **PASS**" in REPORT_PATH.read_text(encoding="utf-8")

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.md").read_text(encoding="utf-8")
    assert "Day 32: manual answer-quality review" in readme
    assert "Day 32 Manual Q&A Review" in requirements

    print("Day 32 manual Q&A review gate passed")
    print("reviewed=20 categories=5x4 languages=en:10,ko:10 papers=10")
    print(
        f"pass={summary['verdicts']['pass']} partial={summary['verdicts']['partial']} "
        f"fail={summary['verdicts']['fail']} false_abstentions={summary['false_abstentions']}"
    )
    print(
        f"citation_support=pass:{summary['citation_support']['pass']},"
        f"fail:{summary['citation_support']['fail']},"
        f"n/a:{summary['citation_support']['not_applicable']}"
    )


if __name__ == "__main__":
    main()
