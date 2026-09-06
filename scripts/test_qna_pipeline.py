"""Model-free integration tests for the Day 30 Q&A pipeline."""

import json

from grounded_answer_contract import ABSTENTION_ANSWERS
from qna_pipeline import GroundedQAPipeline


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


class FakeRetrieval:
    def run(self, query, paper_id):
        return {
            "query": query,
            "paper_id": paper_id,
            "evidence": EVIDENCE,
            "production": {"config_fingerprint": "test-retrieval-fingerprint"},
        }


class SequenceLLM:
    provider_name = "test_sequence"
    model_name = "deterministic-test-double"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, response_schema=None):
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


def main():
    direct_llm = SequenceLLM([valid_korean_response()])
    direct = GroundedQAPipeline(
        retrieval=FakeRetrieval(),
        llm=direct_llm,
        enable_abstention=False,
    ).ask("이 논문은 무엇을 연구하는가?", "paper-001", include_debug=True)
    assert direct_llm.calls == 1
    assert direct["response"]["sufficiency"] == "sufficient"
    assert direct["cited_evidence"][0]["page"] == 2
    assert direct["pipeline"]["fallback_used"] is False
    assert direct["pipeline"]["generation_attempts"] == 1
    assert direct["pipeline"]["stages"] == [
        "query",
        "production_retrieval",
        "reranker",
        "evidence_selection",
        "local_llm",
        "answer_validation",
    ]

    retry_llm = SequenceLLM(["not json", valid_korean_response()])
    retry = GroundedQAPipeline(
        retrieval=FakeRetrieval(),
        llm=retry_llm,
        enable_abstention=False,
    ).ask("이 논문은 무엇을 연구하는가?", "paper-001", include_debug=True)
    assert retry_llm.calls == 2
    assert retry["pipeline"]["fallback_used"] is False
    assert [item["status"] for item in retry["debug"]["attempts"]] == [
        "invalid",
        "valid",
    ]

    fallback_llm = SequenceLLM(["not json", "still not json", "invalid again"])
    fallback = GroundedQAPipeline(
        retrieval=FakeRetrieval(),
        llm=fallback_llm,
        enable_abstention=False,
    ).ask("이 논문은 무엇을 연구하는가?", "paper-001", include_debug=True)
    assert fallback_llm.calls == 3
    assert fallback["pipeline"]["fallback_used"] is True
    assert fallback["response"]["answer"] == ABSTENTION_ANSWERS["ko"]
    assert fallback["response"]["evidence_ids"] == []
    assert fallback["cited_evidence"] == []

    print("Day 30 Q&A pipeline tests passed")
    print("paths=valid_first_try+valid_retry+safe_fallback")
    print("stages=query->retrieval->reranker->evidence->llm->validation")


if __name__ == "__main__":
    main()
