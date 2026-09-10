"""Validate the original-roadmap Day 29 prompt/output-schema gate."""

import json

from grounded_answer_contract import (
    REQUIRED_FIELDS,
    SCHEMA_PATH,
    build_grounded_answer_request,
    resolve_answer_evidence,
    validate_answer_payload,
)
from production_retrieval import config_fingerprint, load_production_config
from retrieval_common import PROJECT_ROOT, load_chunks


EXAMPLES_PATH = PROJECT_ROOT / "data" / "evaluation" / "day29_prompt_examples.json"
DAY28_METRICS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "day28_production_metrics.json"
)
CONTRACT_PATH = PROJECT_ROOT / "scripts" / "grounded_answer_contract.py"
TEST_PATH = PROJECT_ROOT / "scripts" / "test_grounded_answer_contract.py"


def main():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    evidence = examples["evidence_fixture"]
    chunks_by_id = {item["chunk_id"]: item for item in load_chunks()}

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(REQUIRED_FIELDS)
    assert schema["properties"]["sufficiency"]["enum"] == [
        "sufficient",
        "insufficient",
    ]
    assert schema["properties"]["evidence_ids"]["maxItems"] == 5
    assert len(schema["allOf"]) == 2

    assert examples["roadmap_day"] == 29
    assert examples["model_call_performed"] is False
    assert examples["response_schema"] == "config/answer_output_schema.json"
    assert examples["summary"] == {
        "examples": 4,
        "languages": {"en": 2, "ko": 2},
        "sufficiency": {"sufficient": 2, "insufficient": 2},
        "validated": 4,
    }
    for item in evidence:
        chunk = chunks_by_id[item["chunk_id"]]
        assert chunk["page"] == item["page"]
        assert chunk.get("section") == item.get("section")
        assert chunk["text"] == item["text"]

    seen_cases = set()
    for case in examples["examples"]:
        assert case["case_id"] not in seen_cases
        seen_cases.add(case["case_id"])
        request = build_grounded_answer_request(
            case["question"],
            evidence,
            question_language=case["question_language"],
        )
        assert request["available_evidence_ids"] == case["available_evidence_ids"]
        assert request["messages"] == case["messages"]
        response = validate_answer_payload(
            case["validated_response"],
            evidence,
            case["question_language"],
        )
        resolved = resolve_answer_evidence(response, evidence)
        assert [item["evidence_id"] for item in resolved] == [
            item["evidence_id"] for item in case["resolved_evidence"]
        ]
        if response["sufficiency"] == "sufficient":
            assert resolved and response["abstention_reason"] is None
        else:
            assert not resolved and response["abstention_reason"]

    production_config = load_production_config()
    day28_metrics = json.loads(DAY28_METRICS_PATH.read_text(encoding="utf-8"))
    assert production_config["status"] == "frozen"
    assert day28_metrics["production_config_fingerprint"] == config_fingerprint(
        production_config
    )
    assert CONTRACT_PATH.exists() and TEST_PATH.exists()
    requirements = (
        PROJECT_ROOT / "docs" / "project" / "requirements.md"
    ).read_text(encoding="utf-8")
    assert "## 근거 기반 답변 계약" in requirements

    print("Day 29 prompt/output schema gate passed")
    print("fields=answer,evidence_ids,sufficiency,abstention_reason")
    print("examples=4 en=2 ko=2 sufficient=2 insufficient=2")
    print("day28_retrieval_config=frozen model_call=deferred_to_day30")


if __name__ == "__main__":
    main()
