"""Unit tests for evidence-first answer generation."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grounded_answer_contract import AnswerValidationError
from grounded_generation import (
    build_evidence_first_request,
    load_answer_generation_config,
    select_grounded_sentences,
    split_evidence_sentences,
    validate_plain_answer,
)


EVIDENCE = [
    {
        "evidence_id": "ev-01-paper-001-p001-c001",
        "page": 1,
        "section": "Abstract",
        "text": (
            "The proposed method reduces memory usage by 40%. "
            "Accuracy remains unchanged on the held-out test set."
        ),
    },
    {
        "evidence_id": "ev-02-paper-001-p002-c002",
        "page": 2,
        "section": "Method",
        "text": "Training uses a two-stage schedule before final evaluation.",
    },
]


class StubSelector:
    def predict(self, pairs, **kwargs):
        import numpy as np

        return np.asarray([float("40%" in text) for _, text in pairs])


class StubTranslator:
    def generate(self, messages, response_schema=None):
        return "제안된 방법은 메모리 사용량을 40% 줄입니다."


class GroundedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_answer_generation_config()

    def test_sentence_selection_retains_traceable_sources(self):
        sentences = split_evidence_sentences(EVIDENCE)
        self.assertEqual(len(sentences), 3)
        selected = select_grounded_sentences(
            "How much memory is reduced?", EVIDENCE, StubSelector(), self.config
        )
        self.assertIn("40%", selected[0]["text"])
        self.assertEqual(selected[0]["evidence_id"], EVIDENCE[0]["evidence_id"])

    def test_equation_and_reference_fragments_are_not_selected(self):
        evidence = EVIDENCE + [
            {
                "evidence_id": "ev-03-paper-001-p010-c010",
                "page": 10,
                "section": "References",
                "text": "Smith, A. A cited result from another publication is described here.",
            },
            {
                "evidence_id": "ev-04-paper-001-p003-c003",
                "page": 3,
                "section": "Method",
                "text": "The boundary is u1 = 2u3 and u2 = beta u3.",
            },
        ]
        sentences = split_evidence_sentences(evidence)
        self.assertFalse(any(item["section"] == "References" for item in sentences))
        self.assertFalse(any("=" in item["text"] for item in sentences))

    def test_prompt_requests_plain_language_matched_answer(self):
        selected = split_evidence_sentences(EVIDENCE)[:1]
        request = build_evidence_first_request(
            "메모리는 얼마나 감소하는가?", selected, "ko"
        )
        self.assertIn("final answer in Korean", request["messages"][1]["content"])
        self.assertNotIn("JSON object", request["messages"][1]["content"])

    def test_numeric_claim_must_exist_in_selected_evidence(self):
        selected = split_evidence_sentences(EVIDENCE)[:1]
        payload = validate_plain_answer(
            "The method reduces memory by 40%.",
            "How much memory is reduced?",
            selected,
            "en",
            self.config,
        )
        self.assertEqual(payload["sufficiency"], "sufficient")
        with self.assertRaisesRegex(AnswerValidationError, "numbers absent"):
            validate_plain_answer(
                "The method reduces memory by 50%.",
                "How much memory is reduced?",
                selected,
                "en",
                self.config,
            )

    def test_json_compatibility_does_not_delegate_citations_to_model(self):
        selected = split_evidence_sentences(EVIDENCE)[:1]
        raw = json.dumps(
            {
                "answer": "The method reduces memory by 40%.",
                "evidence_ids": ["invented"],
                "sufficiency": "sufficient",
                "abstention_reason": None,
            }
        )
        payload = validate_plain_answer(
            raw,
            "How much memory is reduced?",
            selected,
            "en",
            self.config,
        )
        self.assertEqual(payload["evidence_ids"], [EVIDENCE[0]["evidence_id"]])

    def test_translation_mode_keeps_deterministic_citations(self):
        from grounded_generation import generate_evidence_first_answer

        result = generate_evidence_first_answer(
            StubTranslator(),
            StubSelector(),
            "메모리는 얼마나 감소하는가?",
            EVIDENCE,
            "ko",
            self.config,
        )
        self.assertFalse(result["fallback_used"])
        self.assertIn("40%", result["response"]["answer"])
        self.assertEqual(
            result["response"]["evidence_ids"][0],
            EVIDENCE[0]["evidence_id"],
        )

    def test_translated_magnitude_and_ordinal_are_supported(self):
        selected = [
            {
                "evidence_id": EVIDENCE[0]["evidence_id"],
                "page": 1,
                "section": "Abstract",
                "text": "We processed approximately1.5million records without third-party tools.",
            }
        ]
        payload = validate_plain_answer(
            "제3자 도구 없이 약 150만 개의 레코드를 처리했습니다.",
            "몇 개의 레코드를 처리했는가?",
            selected,
            "ko",
            self.config,
        )
        self.assertEqual(payload["sufficiency"], "sufficient")

    def test_runtime_pipeline_uses_evidence_first_strategy(self):
        from qna_pipeline import GroundedQAPipeline

        class Retrieval:
            def run(self, *args):
                return {
                    "evidence": EVIDENCE,
                    "candidates": [{"reranker_score": 1.0}],
                    "production": {"config_fingerprint": "test"},
                }

        pipeline = GroundedQAPipeline(
            retrieval=Retrieval(),
            translator=StubTranslator(),
            sentence_selector=StubSelector(),
            answer_strategy="evidence_first",
        )
        result = pipeline.ask(
            "메모리는 얼마나 감소하는가?",
            "paper-001",
            question_language="ko",
        )
        self.assertEqual(result["pipeline"]["answer_strategy"], "evidence_first")
        self.assertIn("sentence_selection", result["pipeline"]["stages"])
        self.assertIn("local_translation", result["pipeline"]["stages"])
        self.assertIn("40%", result["response"]["answer"])


if __name__ == "__main__":
    unittest.main()
