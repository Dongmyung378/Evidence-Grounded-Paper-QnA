"""Validate the original-roadmap Day 31 abstention completion gate."""

import json
from collections import Counter

from abstention_policy import (
    CONFIG_PATH,
    AbstentionPolicy,
    abstention_config_fingerprint,
    load_abstention_config,
)
from grounded_answer_contract import ABSTENTION_ANSWERS
from local_llm import generation_config_fingerprint, load_generation_config
from production_retrieval import config_fingerprint, load_production_config
from retrieval_common import PROJECT_ROOT, load_jsonl


CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "abstention_questions.jsonl"
CALIBRATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day31_abstention_calibration.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day31_abstention_results.json"
)
REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "day31_report.md"


def main():
    config = load_abstention_config()
    policy = AbstentionPolicy()
    production = load_production_config()
    generation = load_generation_config()
    cases = load_jsonl(CASES_PATH)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    summary = evaluation["summary"]

    assert CONFIG_PATH.exists()
    assert config["roadmap_day"] == 31
    assert config["status"] == "calibrated"
    assert config["policy"]["minimum_top_reranker_score"] == -3.0
    assert config["policy"]["skip_llm_on_pre_generation_rejection"] is True
    assert policy.fingerprint == abstention_config_fingerprint(config)

    assert len(cases) == 20
    assert len({item["case_id"] for item in cases}) == 20
    assert Counter(item["split"] for item in cases) == {
        "calibration": 10,
        "holdout": 10,
    }
    for split in ("calibration", "holdout"):
        selected = [item for item in cases if item["split"] == split]
        assert Counter(item["question_language"] for item in selected) == {
            "en": 5,
            "ko": 5,
        }
        assert all(item["expected_sufficiency"] == "insufficient" for item in selected)

    retrieval_fingerprint = config_fingerprint(production)
    assert calibration["roadmap_day"] == 31
    assert calibration["retrieval_config_fingerprint"] == retrieval_fingerprint
    assert calibration["summary"]["answerable"]["cases"] == 40
    assert calibration["summary"]["unsupported"]["cases"] == 10
    assert len(calibration["rows"]) == 50

    answerable = [row for row in calibration["rows"] if row["group"] == "answerable"]
    unsupported = [row for row in calibration["rows"] if row["group"] == "unsupported"]
    answerable_decisions = [policy.decide_from_signals(row["signals"]) for row in answerable]
    unsupported_decisions = [policy.decide_from_signals(row["signals"]) for row in unsupported]
    assert sum(not item["abstain"] for item in answerable_decisions) == 37
    assert all(item["abstain"] for item in unsupported_decisions)

    assert evaluation["roadmap_day"] == 31
    assert evaluation["dataset"]["evaluation_split"] == "holdout"
    assert evaluation["dataset"]["calibration_split_excluded"] is True
    assert evaluation["policy"]["config_fingerprint"] == policy.fingerprint
    assert evaluation["retrieval_config_fingerprint"] == retrieval_fingerprint
    assert evaluation["generation"] == {
        "provider": generation["provider"],
        "model": generation["model"],
        "revision": generation["revision"],
        "config_fingerprint": generation_config_fingerprint(generation),
    }
    assert summary["holdout_questions"] == summary["holdout_papers"] == 10
    assert summary["holdout_languages"] == {"en": 5, "ko": 5}
    assert summary["holdout_refused"] == 10
    assert summary["holdout_refusal_recall"] == 1.0
    assert summary["calibration_unsupported_refused"] == 10
    assert summary["answerable_retained"] == 37
    assert summary["answerable_questions"] == 40
    assert summary["answerable_retention"] == 0.925
    assert summary["pre_generation_false_abstentions"] == [
        "q-001-en",
        "q-001-ko",
        "q-031-en",
    ]
    assert summary["pre_generation_refused"] == 9
    assert summary["model_refused"] == 1
    assert summary["validation_fallbacks"] == 0
    assert summary["llm_invocations"] == 1

    assert len(evaluation["results"]) == 10
    assert {row["paper_id"] for row in evaluation["results"]} == {
        f"paper-{index:03d}" for index in range(11, 21)
    }
    for row in evaluation["results"]:
        response = row["response"]
        assert response["sufficiency"] == "insufficient"
        assert response["answer"] == ABSTENTION_ANSWERS[row["question_language"]]
        assert response["evidence_ids"] == []
        assert response["abstention_reason"]
        assert row["abstention_source"] in {"pre_generation_policy", "model"}
        assert row["generation_attempts"] == (0 if row["llm_skipped"] else 1)

    pipeline_source = (
        PROJECT_ROOT / "scripts" / "qna_pipeline.py"
    ).read_text(encoding="utf-8")
    day30_source = (
        PROJECT_ROOT / "scripts" / "run_day30_smoke.py"
    ).read_text(encoding="utf-8")
    assert "AbstentionPolicy" in pipeline_source
    assert "enable_abstention=False" in day30_source
    assert REPORT_PATH.exists() and REPORT_PATH.stat().st_size > 0

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.md").read_text(encoding="utf-8")
    assert "Day 31: calibrated abstention" in readme
    assert "Day 31 Abstention Policy" in requirements

    print("Day 31 abstention gate passed")
    print("holdout_refusal=10/10 en=5 ko=5")
    print("pre_generation=9 model=1 fallback=0")
    print("verified_answerable_retention=37/40 (0.925)")


if __name__ == "__main__":
    main()
