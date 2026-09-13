"""Dependency-light checks for the Docker Q&A benchmark contract."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_container_qna_benchmark import (
    CONFIG_PATH,
    PROJECT_NAME,
    validate_answer,
    validate_manifest,
)


class ContainerQNABenchmarkTests(unittest.TestCase):
    def test_manifest_is_three_papers_and_five_bilingual_pairs(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_manifest(config)
        questions = [
            question
            for paper in config["papers"]
            for question in paper["questions"]
        ]
        self.assertEqual(len(config["papers"]), 3)
        self.assertEqual(len(questions), 10)
        self.assertEqual(sum(q["language"] == "en" for q in questions), 5)
        self.assertEqual(sum(q["language"] == "ko" for q in questions), 5)
        self.assertEqual(config["seed"], 378)

    def test_runner_cannot_load_gold_during_execution(self):
        source = (ROOT / "scripts" / "run_container_qna_benchmark.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("gold_evidence.jsonl", source)
        self.assertNotIn("gold_answer", source)
        self.assertEqual(PROJECT_NAME, "paper-qna-container-benchmark")

    def test_sufficient_answer_requires_traceable_evidence(self):
        response = {
            "question_language": "en",
            "answer": "The paper compares transformer summarization models.",
            "sufficiency": "sufficient",
            "abstention_reason": None,
            "evidence": [
                {
                    "evidence_id": "ev-01",
                    "page": 3,
                    "text": "The paper compares BERT, RoBERTa, and BART.",
                    "locator": {"chunk_id": "paper-a-p003-c001", "page_label": "p. 3"},
                }
            ],
            "runtime": {
                "total_seconds": 1.0,
                "retrieval_seconds": 0.5,
                "generation_seconds": 0.5,
                "fallback_used": False,
            },
        }
        validate_answer(response, "en")
        response["evidence"] = []
        with self.assertRaisesRegex(AssertionError, "no evidence"):
            validate_answer(response, "en")

    def test_insufficient_answer_cannot_expose_evidence(self):
        response = {
            "question_language": "ko",
            "answer": "이 질문에 답할 근거가 충분하지 않습니다.",
            "sufficiency": "insufficient",
            "abstention_reason": "검색 근거가 충분하지 않습니다.",
            "evidence": [],
            "runtime": {
                "total_seconds": 0.2,
                "retrieval_seconds": 0.2,
                "generation_seconds": 0.0,
                "fallback_used": False,
            },
        }
        validate_answer(response, "ko")
        response["evidence"] = [
            {
                "evidence_id": "ev-01",
                "page": 1,
                "text": "unsupported",
                "locator": {"chunk_id": "chunk", "page_label": "p. 1"},
            }
        ]
        with self.assertRaisesRegex(AssertionError, "must not include evidence"):
            validate_answer(response, "ko")


if __name__ == "__main__":
    unittest.main()
