"""Regression tests for the local 6 GB GPU runtime profile."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from app.question_engine import RuntimeQuestionEngine, load_runtime_profile
from candidate_evidence_pipeline import CandidateEvidencePipeline
from grounded_answer_contract import ABSTENTION_ANSWERS
from local_llm import LocalTransformersLLM, load_generation_config
from qna_pipeline import GroundedQAPipeline
from test_qna_pipeline import EVIDENCE, SequenceLLM


class FakeEmbedding:
    def encode(self, values, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in values], dtype=np.float32)


class FakeReranker:
    def predict(self, pairs, **kwargs):
        return np.ones(len(pairs), dtype=np.float32)


class FakeRetrieval:
    def run(self, query, paper_id):
        return {
            "query": query,
            "paper_id": paper_id,
            "evidence": EVIDENCE,
            "production": {"config_fingerprint": "test"},
        }


class RuntimePerformance(unittest.TestCase):
    def test_runtime_profile_reserves_gpu_for_generation(self):
        profile = load_runtime_profile()
        generation = load_generation_config(ROOT / profile["generation_config"])
        self.assertEqual(profile["devices"]["embedding"], "cpu")
        self.assertEqual(profile["devices"]["reranker"], "cpu")
        self.assertEqual(profile["devices"]["generator"], "auto")
        self.assertTrue(profile["prepare_models_during_analysis"])
        self.assertEqual(generation["device"], "auto")
        self.assertEqual(generation["max_new_tokens"], 384)
        self.assertEqual(generation["validation_attempts"], 3)
        self.assertEqual(generation["cpu_validation_attempts"], 2)
        self.assertEqual(generation["max_time_seconds"], 18.0)
        self.assertEqual(generation["cpu_threads"], 8)

    @patch("candidate_evidence_pipeline.load_reranker", return_value=FakeReranker())
    @patch("candidate_evidence_pipeline.load_model", return_value=FakeEmbedding())
    def test_retrieval_device_policy_is_forwarded(self, load_embedding, load_reranker):
        paper_id = "paper-001"
        chunks = [{
            "chunk_id": "paper-001-p001-c001",
            "paper_id": paper_id,
            "page": 1,
            "source_page_id": "paper-001-p001",
            "section": "Abstract",
            "text": "Grounded question answering uses paper evidence.",
        }]
        CandidateEvidencePipeline(
            chunks=chunks,
            embedding_cache_path=None,
            embedding_device="cpu",
            reranker_device="cpu",
        )
        self.assertEqual(load_embedding.call_args.kwargs["device"], "cpu")
        self.assertEqual(load_reranker.call_args.kwargs["device"], "cpu")

    def test_runtime_generation_stops_after_two_invalid_attempts(self):
        llm = SequenceLLM(["not json", "still not json", "unused"])
        llm.device = "cpu"
        pipeline = GroundedQAPipeline(
            retrieval=FakeRetrieval(),
            llm=llm,
            generation_config_path=ROOT / "config" / "generation_runtime.json",
            enable_abstention=False,
        )
        result = pipeline.ask(
            "이 논문은 무엇을 연구하는가?",
            "paper-001",
            include_debug=True,
        )
        self.assertEqual(llm.calls, 2)
        self.assertTrue(result["pipeline"]["fallback_used"])
        self.assertEqual(result["response"]["answer"], ABSTENTION_ANSWERS["ko"])
        self.assertGreaterEqual(result["pipeline"]["generation_seconds"], 0.0)
        self.assertGreaterEqual(result["pipeline"]["retrieval_seconds"], 0.0)

    def test_generator_forwards_per_attempt_time_limit(self):
        llm = object.__new__(LocalTransformersLLM)
        llm.device = "cpu"
        llm.device_fallback_reason = None
        llm.config = {
            "device": "auto",
            "max_input_tokens": 8192,
            "max_new_tokens": 384,
            "max_time_seconds": 18.0,
            "do_sample": False,
        }
        tensor = MagicMock()
        tensor.shape = (1, 3)
        tensor.to.return_value = tensor
        llm.tokenizer = MagicMock()
        llm.tokenizer.eos_token_id = 0
        llm.tokenizer.apply_chat_template.return_value = {"input_ids": tensor}
        llm.tokenizer.decode.return_value = "response"
        llm.model = MagicMock()
        llm.model.generate.return_value = MagicMock()
        llm._torch = MagicMock()
        llm._torch.inference_mode.return_value.__enter__.return_value = None
        llm._torch.inference_mode.return_value.__exit__.return_value = False

        self.assertEqual(llm.generate([{"role": "user", "content": "test"}]), "response")
        self.assertEqual(llm.model.generate.call_args.kwargs["max_time"], 18.0)

    def test_analysis_preparation_reuses_loaded_models(self):
        paper_id = "paper-" + "a" * 32

        class Retrieval:
            instances = []

            def __init__(self, **kwargs):
                self.pipeline = type("Pipeline", (), {
                    "embedding_model": kwargs.get("embedding_model") or object(),
                    "reranker": kwargs.get("reranker") or object(),
                })()
                self.embedding_cache_hit = False
                self.kwargs = kwargs
                self.__class__.instances.append(self)

        class Pipeline:
            instances = []

            def __init__(self, retrieval, generation_config_path, **kwargs):
                self.retrieval = retrieval
                self.path = Path(generation_config_path)
                self.__class__.instances.append(self)

            def prepare_generator(self):
                return {"device": "cuda", "device_fallback_reason": None}

            def ask(self, *args, **kwargs):
                return {"ok": True}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks.jsonl").write_text(
                json.dumps({
                    "chunk_id": f"{paper_id}-p001-c001",
                    "paper_id": paper_id,
                    "page": 1,
                    "source_page_id": f"{paper_id}-p001",
                    "text": "Evidence.",
                }) + "\n",
                encoding="utf-8",
            )
            engine = RuntimeQuestionEngine()
            with patch.object(
                engine,
                "_load_pipeline_classes",
                return_value=(Retrieval, Pipeline),
            ):
                prepared = engine.prepare(paper_id, root)
                self.assertEqual(engine.ask("Question?", paper_id, root), {"ok": True})

        self.assertEqual(prepared["status"], "ready")
        self.assertEqual(prepared["embedding_device"], "cpu")
        self.assertEqual(prepared["reranker_device"], "cpu")
        self.assertEqual(prepared["generator"]["device"], "cuda")
        self.assertEqual(len(Retrieval.instances), 1)
        self.assertEqual(len(Pipeline.instances), 1)
        self.assertEqual(Pipeline.instances[0].path.name, "generation_runtime.json")


if __name__ == "__main__":
    unittest.main()
