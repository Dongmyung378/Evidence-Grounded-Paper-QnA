"""Validate the original-roadmap Day 30 end-to-end local Q&A gate."""

import json

from grounded_answer_contract import validate_answer_payload
from local_llm import CONFIG_PATH, generation_config_fingerprint, load_generation_config
from production_retrieval import config_fingerprint, load_production_config
from retrieval_common import PROJECT_ROOT
from run_day30_smoke import QUESTION_IDS


RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "day30_e2e_results.json"
DAY28_METRICS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day28_production_metrics.json"
)
PIPELINE_PATH = PROJECT_ROOT / "scripts" / "qna_pipeline.py"
CLI_PATH = PROJECT_ROOT / "scripts" / "ask_paper.py"


def main():
    generation_config = load_generation_config()
    production_config = load_production_config()
    run = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    summary = run["summary"]

    assert CONFIG_PATH.exists()
    assert generation_config["roadmap_day"] == 30
    assert generation_config["provider"] == "local_transformers"
    assert generation_config["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert generation_config["revision"] != "main"
    assert generation_config["do_sample"] is False
    assert generation_config["validation_attempts"] == 3

    assert run["roadmap_day"] == 30
    assert run["quality_evaluation_deferred_to_day"] == 32
    assert run["question_ids"] == QUESTION_IDS
    assert summary["questions"] == summary["papers"] == 10
    assert summary["languages"] == {"en": 5, "ko": 5}
    assert summary["completed"] == 10
    assert summary["valid_model_responses"] == 10
    assert summary["safe_fallbacks"] == 0
    assert summary["llm_invocations"] >= 10
    assert summary["runtime_seconds"] > 0

    expected_stages = [
        "query",
        "production_retrieval",
        "reranker",
        "evidence_selection",
        "local_llm",
        "answer_validation",
    ]
    expected_retrieval_fingerprint = config_fingerprint(production_config)
    expected_generation_fingerprint = generation_config_fingerprint(
        generation_config
    )
    assert len(run["results"]) == 10
    assert len({item["paper_id"] for item in run["results"]}) == 10
    for item in run["results"]:
        assert item["question_id"] in QUESTION_IDS
        assert item["roadmap_day"] == 30
        assert item["pipeline"]["stages"] == expected_stages
        assert item["pipeline"]["fallback_used"] is False
        assert item["pipeline"]["retrieval_config_fingerprint"] == (
            expected_retrieval_fingerprint
        )
        assert item["pipeline"]["llm"]["provider"] == "local_transformers"
        assert item["pipeline"]["llm"]["model"] == generation_config["model"]
        assert item["pipeline"]["llm"]["revision"] == generation_config["revision"]
        assert item["pipeline"]["llm"]["config_fingerprint"] == (
            expected_generation_fingerprint
        )
        assert item["pipeline"]["generation_attempts"] >= 1
        assert item["pipeline"]["runtime_seconds"] > 0
        assert item["debug"]["attempts"][-1]["status"] == "valid"
        assert item["debug"]["attempts"][-1]["raw_response"]

        response = validate_answer_payload(
            item["response"],
            item["retrieved_evidence"],
            item["question_language"],
        )
        assert response["evidence_ids"] == [
            evidence["evidence_id"] for evidence in item["cited_evidence"]
        ]
        assert all(
            evidence["paper_id"] == item["paper_id"]
            for evidence in item["retrieved_evidence"]
        )

    day28_metrics = json.loads(DAY28_METRICS_PATH.read_text(encoding="utf-8"))
    assert production_config["status"] == "frozen"
    assert day28_metrics["production_config_fingerprint"] == (
        expected_retrieval_fingerprint
    )
    assert PIPELINE_PATH.exists() and CLI_PATH.exists()
    pipeline_source = PIPELINE_PATH.read_text(encoding="utf-8")
    smoke_source = (
        PROJECT_ROOT / "scripts" / "run_day30_smoke.py"
    ).read_text(encoding="utf-8")
    assert "ProductionRetrieval" in pipeline_source
    assert "LocalTransformersLLM" in pipeline_source
    assert "gold_evidence" not in smoke_source
    requirements = (
        PROJECT_ROOT / "docs" / "project" / "requirements.md"
    ).read_text(encoding="utf-8")
    assert "## 로컬 Q&A 연결" in requirements

    print("Day 30 local end-to-end Q&A gate passed")
    print("questions=10 papers=10 languages=en:5,ko:5")
    print(
        f"valid_model_responses={summary['valid_model_responses']} "
        f"safe_fallbacks={summary['safe_fallbacks']} "
        f"runtime_seconds={summary['runtime_seconds']:.3f}"
    )
    print("flow=query->retrieval->reranker->evidence->local_llm->validation")


if __name__ == "__main__":
    main()
