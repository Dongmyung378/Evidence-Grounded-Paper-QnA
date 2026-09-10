"""Validate the frozen production retrieval gate and its saved artifacts."""

import json

from production_retrieval import CONFIG_PATH, config_fingerprint, load_production_config
from retrieval_common import PROJECT_ROOT, load_chunks, load_jsonl


METRICS_PATH = PROJECT_ROOT / "data" / "evaluation" / "day28_production_metrics.json"
SMOKE_PATH = PROJECT_ROOT / "data" / "evaluation" / "corpus_expansion_smoke.json"
CLI_PATH = PROJECT_ROOT / "scripts" / "search_evidence.py"


def main():
    config = load_production_config()
    fingerprint = config_fingerprint(config)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    chunks = load_chunks()
    questions = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl")
    gold = load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "gold_evidence.jsonl")

    assert CONFIG_PATH.exists()
    assert config["status"] == "frozen" and config["roadmap_day"] == 28
    assert config["corpus"] == {
        "papers": 30,
        "pages": 563,
        "chunks": 2463,
        "processing_scope": "one_paper_at_a_time",
    }
    assert len(chunks) == 2463 and len({row["paper_id"] for row in chunks}) == 30
    assert len(questions) == len(gold) == 40
    assert {row["paper_id"] for row in questions} == {
        f"paper-{number:03d}" for number in range(1, 11)
    }

    assert metrics["roadmap_day"] == 28 and metrics["status"] == "passed"
    assert metrics["production_config_fingerprint"] == fingerprint
    assert metrics["metrics"]["recall_at_5"] == 1.0
    assert metrics["metrics"]["recall_at_10"] == 1.0
    assert metrics["metrics"]["mrr"] >= config["quality_gates"]["minimum_mrr"]
    assert metrics["lost_top5_hits"] == []
    assert metrics["evidence_contract_passed"] == 40
    assert all(metrics["checks"].values())

    assert smoke["production_config_fingerprint"] == fingerprint
    assert smoke["summary"]["passed"] == 20 and smoke["summary"]["failed"] == 0
    assert smoke["summary"]["english_queries"] == 10
    assert smoke["summary"]["korean_queries"] == 10
    cli_source = CLI_PATH.read_text(encoding="utf-8")
    assert "from production_retrieval import ProductionRetrieval" in cli_source
    assert "CandidateEvidencePipeline" not in cli_source

    print("Day 28 production retrieval gate passed")
    print(f"config=frozen fingerprint={fingerprint[:12]}")
    print("corpus=30 papers, 563 pages, 2463 chunks")
    print(
        f"verified_40=Recall@5:{metrics['metrics']['recall_at_5']:.3f} "
        f"Recall@10:{metrics['metrics']['recall_at_10']:.3f} "
        f"MRR:{metrics['metrics']['mrr']:.4f}"
    )
    print("expansion_smoke=20/20 lost_top5=0")


if __name__ == "__main__":
    main()
