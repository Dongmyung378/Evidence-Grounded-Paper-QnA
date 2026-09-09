"""Frozen Day 28 production retrieval configuration and execution path."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from candidate_evidence_pipeline import CACHE_PATH, CandidateEvidencePipeline
from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "production_retrieval.json"


def load_production_config(path=CONFIG_PATH):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_production_config(config)
    return config


def validate_production_config(config):
    assert config["schema_version"] == 1
    assert config["roadmap_day"] == 28
    assert config["status"] == "frozen"
    assert config["corpus"]["processing_scope"] == "one_paper_at_a_time"
    assert config["benchmark"] == {
        "name": "verified-40",
        "papers": 10,
        "questions": 40,
        "languages": {"en": 20, "ko": 20},
    }

    retrieval = config["retrieval"]
    assert retrieval["bm25_top_k"] == retrieval["dense_top_k"] == 20
    assert retrieval["fusion"] == "equal_weight_reciprocal_rank_fusion"
    assert retrieval["fusion_weights"] == {"bm25": 1.0, "dense": 1.0}
    assert retrieval["rrf_k"] == 60
    assert retrieval["candidate_k"] == 20
    assert retrieval["front_matter_guard"] == {"enabled": True, "page": 1}

    assert config["reranking"]["top_k"] == retrieval["candidate_k"]
    assert config["reranking"]["max_length"] == 512
    assert config["evidence"] == {
        "top_k": 5,
        "max_per_page": 2,
        "near_duplicate_threshold": 0.85,
    }
    assert config["quality_gates"]["minimum_recall_at_5"] == 1.0
    assert config["quality_gates"]["maximum_lost_top5_hits"] == 0
    assert config["change_policy"]["frozen_after_day"] == 28
    return config


def config_fingerprint(config):
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProductionRetrieval:
    """The only retrieval entry point intended for API and user-facing CLI use."""

    def __init__(
        self,
        config_path=CONFIG_PATH,
        chunks=None,
        embedding_cache_path=CACHE_PATH,
        embedding_model=None,
        reranker=None,
        local_files_only=False,
        embedding_device=None,
        reranker_device=None,
    ):
        self.config_path = Path(config_path)
        self.config = load_production_config(self.config_path)
        self.fingerprint = config_fingerprint(self.config)
        models = self.config["models"]
        runtime = self.config["runtime"]
        self.pipeline = CandidateEvidencePipeline(
            embedding_model_name=models["embedding"],
            reranker_model_name=models["reranker"],
            embedding_batch_size=runtime["embedding_batch_size"],
            reranker_batch_size=runtime["reranker_batch_size"],
            max_length=self.config["reranking"]["max_length"],
            chunks=chunks,
            embedding_cache_path=embedding_cache_path,
            embedding_model=embedding_model,
            reranker=reranker,
            local_files_only=local_files_only,
            embedding_device=embedding_device,
            reranker_device=reranker_device,
        )

    @property
    def embedding_cache_hit(self):
        return self.pipeline.embedding_cache_hit

    def run(self, query, paper_id):
        retrieval = self.config["retrieval"]
        evidence = self.config["evidence"]
        result = self.pipeline.run(
            query,
            paper_id,
            source_k=retrieval["bm25_top_k"],
            candidate_k=retrieval["candidate_k"],
            evidence_k=evidence["top_k"],
            rrf_k=retrieval["rrf_k"],
            max_per_page=evidence["max_per_page"],
            near_duplicate_threshold=evidence["near_duplicate_threshold"],
            front_matter_page=retrieval["front_matter_guard"]["page"],
        )
        result["roadmap_days"] = [23, 24, 27, 28]
        result["production"] = {
            "status": self.config["status"],
            "config_path": str(self.config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "config_fingerprint": self.fingerprint,
            "config": deepcopy(self.config),
        }
        return result
