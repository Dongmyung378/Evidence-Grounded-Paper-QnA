"""Lazy, single-paper question engine backed by the frozen production pipeline."""

import json
import random
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RUNTIME_CONFIG_PATH = ROOT / "config" / "runtime_qna.json"


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_runtime_profile(path=RUNTIME_CONFIG_PATH):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported runtime Q&A profile schema")
    devices = profile.get("devices") or {}
    for name in ("embedding", "reranker"):
        if devices.get(name) not in {"cpu", "cuda"}:
            raise ValueError(f"runtime_qna.devices.{name} must be cpu or cuda")
    if devices.get("generator") not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime_qna.devices.generator must be auto, cpu, or cuda")
    for name in ("generation_config", "abstention_config"):
        target = ROOT / profile[name]
        if not target.is_file():
            raise ValueError(f"Runtime Q&A profile points to a missing file: {profile[name]}")
    generation = json.loads(
        (ROOT / profile["generation_config"]).read_text(encoding="utf-8")
    )
    if generation.get("device") != devices["generator"]:
        raise ValueError(
            "runtime_qna.devices.generator must match generation_config.device"
        )
    return profile


class RuntimeQuestionEngine:
    """Load models lazily and keep only one uploaded paper active in memory."""

    def __init__(
        self,
        seed=378,
        local_files_only=False,
        runtime_config_path=RUNTIME_CONFIG_PATH,
    ):
        self.seed = int(seed)
        self.local_files_only = bool(local_files_only)
        self.runtime_config_path = Path(runtime_config_path)
        self.profile = load_runtime_profile(self.runtime_config_path)
        self.generation_config_path = ROOT / self.profile["generation_config"]
        self.embedding_device = self.profile["devices"]["embedding"]
        self.reranker_device = self.profile["devices"]["reranker"]
        self.prepare_models_during_analysis = bool(
            self.profile.get("prepare_models_during_analysis", False)
        )
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
            embedding_device=self.embedding_device,
            reranker_device=self.reranker_device,
        )
        if self._pipeline is None:
            self._pipeline = GroundedQAPipeline(
                retrieval=retrieval,
                generation_config_path=self.generation_config_path,
                local_files_only=self.local_files_only,
            )
        else:
            self._pipeline.retrieval = retrieval
        self._active_paper_id = paper_id

    def prepare(self, paper_id, directory):
        """Build the paper index and optionally load the generator before questions."""
        if not self.prepare_models_during_analysis:
            return {"status": "deferred", "reason": "runtime_preparation_disabled"}
        with self._lock:
            started = time.perf_counter()
            self._seed_runtime()
            if self._active_paper_id != paper_id:
                self._activate(paper_id, Path(directory))
            llm = self._pipeline.prepare_generator()
            return {
                "status": "ready",
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "embedding_device": self.embedding_device,
                "reranker_device": self.reranker_device,
                "embedding_cache_hit": self._pipeline.retrieval.embedding_cache_hit,
                "generator": llm,
            }

    def ask(self, question, paper_id, directory):
        with self._lock:
            self._seed_runtime()
            if self._active_paper_id != paper_id:
                self._activate(paper_id, Path(directory))
            return self._pipeline.ask(question, paper_id, include_debug=False)
