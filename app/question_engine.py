"""Lazy, single-paper question engine backed by the frozen production pipeline."""

import json
import random
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class RuntimeQuestionEngine:
    """Load models lazily and keep only one uploaded paper active in memory."""

    def __init__(self, seed=378, local_files_only=False):
        self.seed = int(seed)
        self.local_files_only = bool(local_files_only)
        self._lock = threading.Lock()
        self._pipeline = None
        self._active_paper_id = None

    @property
    def status(self):
        return "ready" if self._pipeline is not None else "lazy"

    def _seed_runtime(self):
        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

    @staticmethod
    def _load_pipeline_classes():
        scripts_path = str(SCRIPTS_DIR)
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        from production_retrieval import ProductionRetrieval
        from qna_pipeline import GroundedQAPipeline

        return ProductionRetrieval, GroundedQAPipeline

    def _activate(self, paper_id, directory):
        ProductionRetrieval, GroundedQAPipeline = self._load_pipeline_classes()
        chunks = load_jsonl(directory / "chunks.jsonl")
        if not chunks or any(chunk.get("paper_id") != paper_id for chunk in chunks):
            raise RuntimeError("Uploaded paper chunks are missing or invalid")

        previous = self._pipeline.retrieval.pipeline if self._pipeline else None
        retrieval = ProductionRetrieval(
            chunks=chunks,
            embedding_cache_path=directory / "dense_embeddings.npz",
            embedding_model=(previous.embedding_model if previous else None),
            reranker=(previous.reranker if previous else None),
            local_files_only=self.local_files_only,
        )
        if self._pipeline is None:
            self._pipeline = GroundedQAPipeline(
                retrieval=retrieval,
                local_files_only=self.local_files_only,
            )
        else:
            self._pipeline.retrieval = retrieval
        self._active_paper_id = paper_id

    def ask(self, question, paper_id, directory):
        with self._lock:
            self._seed_runtime()
            if self._active_paper_id != paper_id:
                self._activate(paper_id, Path(directory))
            return self._pipeline.ask(question, paper_id, include_debug=False)
