"""Shared deterministic fixtures for Q&A pipeline tests."""

import json


EVIDENCE = [
    {
        "evidence_id": "ev-01-paper-001-p002-c001",
        "rank": 1,
        "paper_id": "paper-001",
        "page": 2,
        "section": "Introduction",
        "chunk_id": "paper-001-p002-c001",
        "source_page_id": "paper-001-p002",
        "text": "The paper studies evidence-grounded question answering.",
        "char_count": 56,
        "locator": {
            "page_label": "p. 2",
            "section": "Introduction",
            "chunk_id": "paper-001-p002-c001",
        },
        "scores": {
            "reranker": 1.0,
            "hybrid": 0.1,
            "pre_rerank_rank": 1,
            "source_ranks": {"bm25": 1, "dense": 1},
        },
        "source": {"section_index": 1, "chunk_index": 1},
    }
]


class SequenceLLM:
    provider_name = "test_sequence"
    model_name = "deterministic-test-double"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, response_schema=None):
        if response_schema is not None:
            assert response_schema["title"] == "GroundedAnswerOutput"
        assert messages[0]["role"] == "system"
        self.calls += 1
        return self.responses.pop(0)

    def metadata(self):
        return {"provider": self.provider_name, "model": self.model_name}


def valid_korean_response():
    return json.dumps(
        {
            "answer": "이 논문은 근거 기반 질의응답을 연구한다.",
            "evidence_ids": [EVIDENCE[0]["evidence_id"]],
            "sufficiency": "sufficient",
            "abstention_reason": None,
        },
        ensure_ascii=False,
    )
