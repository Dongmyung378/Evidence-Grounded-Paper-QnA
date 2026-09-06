"""Build bilingual sufficient/insufficient examples for the Day 29 contract."""

import json

from grounded_answer_contract import (
    build_grounded_answer_request,
    canonical_abstention_payload,
    resolve_answer_evidence,
    validate_answer_payload,
)
from retrieval_common import PROJECT_ROOT, load_jsonl


DEMO_PATH = PROJECT_ROOT / "data" / "evaluation" / "candidate_evidence_demo.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "day29_prompt_examples.json"


def compact_evidence(evidence):
    return [
        {
            "evidence_id": item["evidence_id"],
            "page": item["page"],
            "section": item.get("section"),
            "chunk_id": item["chunk_id"],
            "text": item["text"],
        }
        for item in evidence
    ]


def main():
    demo = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
    source_case = next(
        item for item in demo["examples"] if item["question_id"] == "q-001-ko"
    )
    evidence = compact_evidence(
        [item for item in source_case["evidence"] if item["page"] == 2]
    )
    assert 1 <= len(evidence) <= 5

    questions = {
        row["question_id"]: row
        for row in load_jsonl(
            PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
        )
    }
    gold = {
        row["question_id"]: row
        for row in load_jsonl(
            PROJECT_ROOT / "data" / "evaluation" / "gold_evidence.jsonl"
        )
    }
    citation_ids = [item["evidence_id"] for item in evidence]

    case_specs = [
        {
            "case_id": "day29-en-sufficient",
            "question": questions["q-001-en"]["question"],
            "question_language": "en",
            "response": {
                "answer": gold["q-001-en"]["gold_answer"],
                "evidence_ids": citation_ids,
                "sufficiency": "sufficient",
                "abstention_reason": None,
            },
        },
        {
            "case_id": "day29-ko-sufficient",
            "question": questions["q-001-ko"]["question"],
            "question_language": "ko",
            "response": {
                "answer": gold["q-001-ko"]["gold_answer"],
                "evidence_ids": citation_ids,
                "sufficiency": "sufficient",
                "abstention_reason": None,
            },
        },
        {
            "case_id": "day29-en-insufficient",
            "question": "Did the authors receive a 2026 Nobel Prize for this work?",
            "question_language": "en",
            "response": canonical_abstention_payload(
                "en",
                "The supplied evidence does not discuss awards and the question requires external knowledge.",
            ),
        },
        {
            "case_id": "day29-ko-insufficient",
            "question": "이 연구의 저자들이 2026년 노벨상을 받았는가?",
            "question_language": "ko",
            "response": canonical_abstention_payload(
                "ko",
                "제공된 근거는 수상 여부를 다루지 않으며 질문에는 외부 지식이 필요합니다.",
            ),
        },
    ]

    examples = []
    for spec in case_specs:
        request = build_grounded_answer_request(
            spec["question"],
            evidence,
            question_language=spec["question_language"],
        )
        response = validate_answer_payload(
            spec["response"],
            evidence,
            spec["question_language"],
        )
        resolved = resolve_answer_evidence(response, evidence)
        examples.append(
            {
                "case_id": spec["case_id"],
                "question": spec["question"],
                "question_language": spec["question_language"],
                "available_evidence_ids": request["available_evidence_ids"],
                "messages": request["messages"],
                "validated_response": response,
                "resolved_evidence": [
                    {
                        "evidence_id": item["evidence_id"],
                        "page": item["page"],
                        "section": item.get("section"),
                        "chunk_id": item["chunk_id"],
                    }
                    for item in resolved
                ],
            }
        )

    payload = {
        "schema_version": 1,
        "roadmap_day": 29,
        "purpose": "provider-neutral bilingual prompt and output-contract fixtures",
        "model_call_performed": False,
        "response_schema": "config/answer_output_schema.json",
        "evidence_fixture": evidence,
        "examples": examples,
        "summary": {
            "examples": len(examples),
            "languages": {"en": 2, "ko": 2},
            "sufficiency": {"sufficient": 2, "insufficient": 2},
            "validated": len(examples),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Day 29 prompt examples built")
    print("examples=4 en=2 ko=2 sufficient=2 insufficient=2")
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
