"""Regression tests for revision-pinned runtime model loading."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dense_retrieval import load_or_create_embeddings
from model_lock import load_model_lock
from production_retrieval import ProductionRetrieval


class CountingEmbedding:
    def __init__(self):
        self.calls = 0

    def encode(self, values, **kwargs):
        self.calls += 1
        return np.asarray([[1.0, 0.0] for _ in values], dtype=np.float32)


class ModelLockTests(unittest.TestCase):
    def test_lock_contains_full_revisions_for_every_role(self):
        lock = load_model_lock()
        self.assertEqual(lock["seed"], 378)
        self.assertEqual(
            set(lock["models"]),
            {"embedding", "reranker", "translation", "legacy_generator"},
        )
        self.assertTrue(
            all(len(model["revision"]) == 40 for model in lock["models"].values())
        )

    @patch("production_retrieval.CandidateEvidencePipeline")
    def test_production_retrieval_forwards_locked_revisions(self, pipeline):
        ProductionRetrieval(chunks=[{"paper_id": "paper-test"}])
        arguments = pipeline.call_args.kwargs
        self.assertEqual(
            arguments["embedding_model_revision"],
            "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        )
        self.assertEqual(
            arguments["reranker_model_revision"],
            "1427fd652930e4ba29e8149678df786c240d8825",
        )

    def test_embedding_cache_identity_includes_revision(self):
        chunks = [
            {
                "chunk_id": "paper-test-p001-c001",
                "paper_id": "paper-test",
                "text": "Evidence text.",
            }
        ]
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "embeddings.npz"
            first = CountingEmbedding()
            _, first_hit = load_or_create_embeddings(
                first,
                chunks,
                "model/name",
                cache_path=cache,
                model_revision="a" * 40,
            )
            second = CountingEmbedding()
            _, second_hit = load_or_create_embeddings(
                second,
                chunks,
                "model/name",
                cache_path=cache,
                model_revision="a" * 40,
            )
            third = CountingEmbedding()
            _, third_hit = load_or_create_embeddings(
                third,
                chunks,
                "model/name",
                cache_path=cache,
                model_revision="b" * 40,
            )
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertFalse(third_hit)
        self.assertEqual((first.calls, second.calls, third.calls), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
