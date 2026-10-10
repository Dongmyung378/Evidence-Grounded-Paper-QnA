"""Stage accounting and benchmark sampling checks with explicit test doubles."""

import json
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from app.models import QuestionRuntime
from app.question_engine import RuntimeQuestionEngine
from benchmark_runtime import percentile, run_benchmark
from candidate_evidence_pipeline import CandidateEvidencePipeline
from qna_pipeline import GroundedQAPipeline
from qna_fixtures import EVIDENCE, SequenceLLM, valid_korean_response


class EmbeddingDouble:
    def encode(self, values, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in values], dtype=np.float32)


class RerankerDouble:
    def predict(self, pairs, **kwargs):
        return np.ones(len(pairs), dtype=np.float32)


class RetrievalDouble:
    def run(self, query, paper_id):
        return {"evidence": EVIDENCE, "production": {"config_fingerprint": "test"},
                "timings": {"bm25_seconds": 0.001, "rerank_seconds": 0.002}}


class TranslatorDouble:
    def __init__(self):
        self.calls = 0

    def translate(self, sentences):
        self.calls += 1
        time.sleep(0.002)
        return ["이 논문은 근거 기반 질의응답을 연구한다."]

    def metadata(self):
        return {"provider": "test_double", "model": "test_double"}


class StageTimingTests(unittest.TestCase):
    def test_refused_korean_question_does_not_initialize_translation_model(self):
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), sentence_selector=RerankerDouble(),
            answer_strategy="evidence_first", enable_abstention=False,
        )
        with patch.object(pipeline, "_get_llm", side_effect=AssertionError("unexpected model load")):
            result = pipeline.ask("이 방법이 99%의 정확도를 달성한 이유는 무엇인가요?", "paper-001")
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertFalse(result["pipeline"]["llm"]["loaded"])
        self.assertFalse(result["pipeline"]["generator_newly_loaded"])
        self.assertEqual(result["pipeline"]["stage_timings"]["generation"]["generator_activation_seconds"], 0)

    def test_answerable_korean_question_records_lazy_translation_activation(self):
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), sentence_selector=RerankerDouble(),
            answer_strategy="evidence_first", enable_abstention=False,
        )
        def load():
            time.sleep(0.002)
            pipeline.llm = TranslatorDouble()
            return pipeline.llm
        with patch.object(pipeline, "_get_llm", side_effect=load):
            result = pipeline.ask("이 논문은 무엇을 연구하는가?", "paper-001")
        self.assertEqual(result["response"]["sufficiency"], "sufficient")
        self.assertTrue(result["pipeline"]["generator_newly_loaded"])
        self.assertGreater(result["pipeline"]["stage_timings"]["generation"]["generator_activation_seconds"], 0.001)

    def test_deliberate_aspect_refusal_is_not_a_generation_attempt_or_fallback(self):
        class UnrelatedScorer:
            def predict(self, pairs, **kwargs):
                return np.full(len(pairs), -10.0, dtype=np.float32)

        translator = TranslatorDouble()
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), translator=translator,
            sentence_selector=UnrelatedScorer(), answer_strategy="evidence_first",
            enable_abstention=False,
        )
        result = pipeline.ask("이 연구에 자금을 지원한 기관은 어디인가요?", "paper-001")
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertEqual(result["pipeline"]["abstention_source"], "question_aspect_policy")
        self.assertEqual(result["pipeline"]["generation_attempts"], 0)
        self.assertFalse(result["pipeline"]["fallback_used"])
        self.assertTrue(result["pipeline"]["llm_skipped"])
        self.assertEqual(translator.calls, 0)

    def test_deliberate_premise_refusal_is_not_a_generation_attempt_or_fallback(self):
        translator = TranslatorDouble()
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), translator=translator,
            sentence_selector=RerankerDouble(), answer_strategy="evidence_first",
            enable_abstention=False,
        )
        result = pipeline.ask("이 방법이 99%의 정확도를 달성한 이유는 무엇인가요?", "paper-001")
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertEqual(result["pipeline"]["abstention_source"], "question_premise_policy")
        self.assertEqual(result["pipeline"]["generation_attempts"], 0)
        self.assertFalse(result["pipeline"]["fallback_used"])
        self.assertEqual(translator.calls, 0)

    def test_retrieval_reports_all_phases_and_reused_models(self):
        chunks = [{
            "chunk_id": "paper-001-p002-c001", "paper_id": "paper-001", "page": 2,
            "source_page_id": "paper-001-p002", "section": "Introduction",
            "text": "The paper studies evidence-grounded question answering.",
        }]
        retrieval = CandidateEvidencePipeline(
            chunks=chunks, embedding_cache_path=None,
            embedding_model=EmbeddingDouble(), reranker=RerankerDouble(),
        )
        initialized = retrieval.initialization_timings
        self.assertTrue(initialized["embedding_model_reused"])
        self.assertTrue(initialized["reranker_model_reused"])
        self.assertEqual(initialized["embedding_model_load_seconds"], 0.0)
        self.assertEqual(initialized["reranker_model_load_seconds"], 0.0)
        result = retrieval.run("What does the paper study?", "paper-001")
        timings = result["timings"]
        stages = {"bm25_seconds", "dense_query_encoding_seconds", "dense_search_seconds",
                  "fusion_seconds", "rerank_seconds", "evidence_selection_seconds"}
        self.assertEqual(set(timings), stages | {"total_seconds"})
        for value in timings.values():
            self.assertGreaterEqual(value, 0.0)
        self.assertGreaterEqual(timings["total_seconds"] + 0.00001,
                                sum(timings[key] for key in stages))

    def test_korean_translation_is_separate_from_answer_validation(self):
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), translator=TranslatorDouble(),
            sentence_selector=RerankerDouble(), answer_strategy="evidence_first",
            enable_abstention=False,
        )
        result = pipeline.ask("이 논문은 무엇을 연구하는가?", "paper-001")
        timings = result["pipeline"]["stage_timings"]
        self.assertEqual(timings["retrieval"]["bm25_seconds"], 0.001)
        self.assertGreater(timings["generation"]["translation_seconds"], 0.001)
        self.assertGreater(timings["generation"]["sentence_selection_seconds"], 0.0)
        self.assertGreater(timings["generation"]["answer_validation_seconds"], 0.0)
        self.assertEqual(timings["generation"]["generator_activation_seconds"], 0.0)
        self.assertFalse(result["pipeline"]["generator_newly_loaded"])
        self.assertEqual(result["response"]["sufficiency"], "sufficient")

    def test_english_extraction_does_not_load_a_translator(self):
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), sentence_selector=RerankerDouble(),
            answer_strategy="evidence_first", enable_abstention=False,
        )
        with patch.object(pipeline, "_get_llm", side_effect=AssertionError("unexpected model load")):
            result = pipeline.ask("What does the paper study?", "paper-001")
        timings = result["pipeline"]["stage_timings"]["generation"]
        self.assertEqual(timings["translation_seconds"], 0.0)
        self.assertGreater(timings["answer_construction_seconds"], 0.0)
        self.assertGreater(timings["answer_validation_seconds"], 0.0)

    def test_legacy_generation_keeps_inference_and_validation_separate(self):
        pipeline = GroundedQAPipeline(
            retrieval=RetrievalDouble(), llm=SequenceLLM([valid_korean_response()]),
            enable_abstention=False,
        )
        result = pipeline.ask("이 논문은 무엇을 연구하는가?", "paper-001")
        timings = result["pipeline"]["stage_timings"]["generation"]
        self.assertGreater(timings["generation_inference_seconds"], 0.0)
        self.assertGreater(timings["answer_validation_seconds"], 0.0)
        self.assertEqual(timings["translation_seconds"], 0.0)

    def test_engine_activation_is_included_only_in_the_initial_request(self):
        class Retrieval:
            def __init__(self, **kwargs):
                time.sleep(0.002)
                self.pipeline = type("Resources", (), {
                    "embedding_model": object(), "reranker": object(),
                })()
                self.initialization_timings = {"embedding_index_seconds": 0.001}

        class Pipeline:
            def __init__(self, retrieval, **kwargs):
                self.retrieval = retrieval

            def ask(self, *args, **kwargs):
                time.sleep(0.002)
                return {"pipeline": {"runtime_seconds": 0.002}}

        with TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "chunks.jsonl").write_text(
                json.dumps({"paper_id": "paper-001", "text": "Evidence"}) + "\n",
                encoding="utf-8",
            )
            engine = RuntimeQuestionEngine()
            with patch.object(engine, "_seed_runtime"), patch.object(
                engine, "_load_pipeline_classes", return_value=(Retrieval, Pipeline)
            ):
                cold = engine.ask("Question?", "paper-001", directory)["pipeline"]
                warm = engine.ask("Question?", "paper-001", directory)["pipeline"]
        self.assertEqual(cold["request_mode"], "cold")
        self.assertGreater(cold["activation_seconds"], 0.001)
        self.assertGreater(cold["runtime_seconds"], cold["pipeline_runtime_seconds"])
        self.assertEqual(warm["request_mode"], "warm")
        self.assertEqual(warm["activation_seconds"], 0.0)
        self.assertEqual(cold["activation"]["retrieval"]["embedding_index_seconds"], 0.001)

    def test_additive_api_fields_preserve_legacy_runtime_payloads(self):
        runtime = QuestionRuntime(
            total_seconds=1, retrieval_seconds=.4, generation_seconds=.6,
            generation_attempts=1, fallback_used=False,
        )
        self.assertEqual(runtime.stage_timings, {})
        self.assertEqual(runtime.activation_seconds, 0)
        self.assertIsNone(runtime.request_mode)


class BenchmarkSamplingTests(unittest.TestCase):
    def test_injected_benchmarks_cannot_be_labelled_real_models(self):
        with self.assertRaisesRegex(ValueError, "test_double"):
            run_benchmark(object(), "Question?", "paper-001", Path("."))

    def test_interpolated_p50_p95(self):
        self.assertEqual(percentile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], .95), 3.85)
        with self.assertRaises(ValueError):
            percentile([], .5)
        with self.assertRaises(ValueError):
            percentile([float("nan")], .5)

    def test_cold_and_warmup_samples_are_excluded_from_percentiles(self):
        class Engine:
            calls = 0

            def ask(self, *args):
                self.calls += 1
                duration = [100, 50, 1, 2, 3, 4][self.calls - 1]
                return {
                    "question_language": "en",
                    "response": {"answer": "Fixed answer", "sufficiency": "sufficient"},
                    "pipeline": {
                        "runtime_seconds": duration,
                        "request_mode": "cold" if self.calls == 1 else "warm",
                        "stage_timings": {"retrieval": {"bm25_seconds": duration / 10}},
                    },
                }

        engine = Engine()
        result = run_benchmark(engine, "Question?", "paper-001", Path("."),
                               repetitions=4, warmup=1, execution_mode="test_double")
        self.assertEqual(result["execution_mode"], "test_double")
        self.assertEqual(result["cold_start"]["runtime_seconds"], 100)
        self.assertEqual(result["warm"]["pipeline_latency"]["p50_seconds"], 2.5)
        self.assertEqual(result["warm"]["pipeline_latency"]["p95_seconds"], 3.85)
        self.assertEqual(result["warm"]["stages"]["retrieval.bm25_seconds"]["count"], 4)
        self.assertEqual(engine.calls, 6)
        self.assertTrue(result["warm"]["responses_identical"])


if __name__ == "__main__":
    unittest.main()
