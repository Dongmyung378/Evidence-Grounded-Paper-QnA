"""Model-free unit and integration tests for the Day 31 refusal policy."""

import json

from abstention_policy import AbstentionPolicy
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
            "reranker": 6.0,
            "hybrid": 0.1,
            "pre_rerank_rank": 1,
            "source_ranks": {"bm25": 1, "dense": 1},
        },
        "source": {"section_index": 1, "chunk_index": 1},
    }
]


class FakeRetrieval:
    def __init__(self, score):
        self.score = score

    def run(self, query, paper_id):
        return {
            "query": query,
            "paper_id": paper_id,
            "candidates": [
                {
                    "candidate_rank": 1,
                    "reranker_score": self.score,
                }
            ],
            "evidence": EVIDENCE,
            "production": {"config_fingerprint": "test-retrieval-fingerprint"},
        }


class NeverCalledLLM:
    provider_name = "never_called"
    model_name = "test-double"

    def generate(self, messages, response_schema=None):
        raise AssertionError("LLM must be skipped for a pre-generation refusal")

    def metadata(self):
        return {"provider": self.provider_name, "model": self.model_name}


class OneResponseLLM:
    provider_name = "one_response"
    model_name = "test-double"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self, messages, response_schema=None):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)

    def metadata(self):
        return {"provider": self.provider_name, "model": self.model_name}


def main():
    policy = AbstentionPolicy()
    threshold = policy.config["policy"]["minimum_top_reranker_score"]

    low = policy.decide_from_signals(
        {
            "candidate_count": 20,
            "evidence_count": 5,
            "top_reranker_score": threshold - 0.01,
        }
    )
    assert low["abstain"] is True
    assert low["reason_code"] == "low_relevance"

    boundary = policy.decide_from_signals(
        {
            "candidate_count": 20,
            "evidence_count": 5,
            "top_reranker_score": threshold,
        }
    )
    assert boundary["abstain"] is False
    assert policy.decide_from_signals({})["reason_code"] == "no_candidates"
    assert policy.decide_from_signals(
        {
            "candidate_count": 20,
            "evidence_count": 0,
            "top_reranker_score": 9.0,
        }
    )["reason_code"] == "no_evidence"

    refused = GroundedQAPipeline(
        retrieval=FakeRetrieval(threshold - 1.0),
        llm=NeverCalledLLM(),
    ).ask("이 논문은 고혈압 치료제를 권장하는가?", "paper-001")
    assert refused["roadmap_day"] == 31
    assert refused["response"]["answer"] == ABSTENTION_ANSWERS["ko"]
    assert refused["response"]["evidence_ids"] == []
    assert refused["pipeline"]["llm_skipped"] is True
    assert refused["pipeline"]["generation_attempts"] == 0
    assert refused["pipeline"]["abstention_source"] == "pre_generation_policy"
    assert "local_llm" not in refused["pipeline"]["stages"]

    sufficient_payload = {
        "answer": "이 논문은 근거 기반 질의응답을 연구한다.",
        "evidence_ids": [EVIDENCE[0]["evidence_id"]],
        "sufficiency": "sufficient",
        "abstention_reason": None,
    }
    sufficient_llm = OneResponseLLM(sufficient_payload)
    sufficient = GroundedQAPipeline(
        retrieval=FakeRetrieval(threshold + 1.0),
        llm=sufficient_llm,
    ).ask("이 논문은 무엇을 연구하는가?", "paper-001")
    assert sufficient_llm.calls == 1
    assert sufficient["response"] == sufficient_payload
    assert sufficient["pipeline"]["llm_skipped"] is False
    assert sufficient["pipeline"]["abstention_source"] is None

    model_abstention = OneResponseLLM(
        {
            "answer": "근거가 부족합니다.",
            "evidence_ids": [],
            "sufficiency": "insufficient",
            "abstention_reason": "관련 문장이 없습니다.",
        }
    )
    normalized = GroundedQAPipeline(
        retrieval=FakeRetrieval(threshold + 1.0),
        llm=model_abstention,
    ).ask("이 논문은 무엇을 연구하는가?", "paper-001")
    assert normalized["response"]["answer"] == ABSTENTION_ANSWERS["ko"]
    assert normalized["pipeline"]["abstention_source"] == "model"

    print("Day 31 abstention policy tests passed")
    print("guards=empty_candidates+empty_evidence+low_relevance")
    print("paths=pre_generation_skip+model_abstention_normalization+sufficient")


if __name__ == "__main__":
    main()
