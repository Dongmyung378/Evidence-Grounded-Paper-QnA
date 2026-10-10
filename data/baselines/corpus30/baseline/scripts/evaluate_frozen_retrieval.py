"""Evaluate verified-40 with the final retrieval configuration and model lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

from production_retrieval import ProductionRetrieval
from retrieval_common import (
    PROJECT_ROOT,
    configure_utf8_stdout,
    load_evaluation_data,
    metrics_from_results,
)


OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "frozen_retrieval_metrics.json"
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "frozen_dense_embeddings.npz"
FREEZE_PATH = PROJECT_ROOT / "config" / "evaluation_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_candidate_pages(result: dict, top_k: int = 10) -> list[int]:
    pages = []
    for candidate in result["candidates"]:
        if candidate["page"] not in pages:
            pages.append(candidate["page"])
        if len(pages) == top_k:
            break
    return pages


def main() -> None:
    configure_utf8_stdout()
    started = perf_counter()
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("seed") != 378:
        raise ValueError("Final evaluation policy is not frozen with seed 378")
    production = ProductionRetrieval(
        local_files_only=True,
        embedding_cache_path=CACHE_PATH,
        embedding_device="cpu",
        reranker_device="cpu",
    )
    questions, gold_by_question = load_evaluation_data()
    rows = []
    evidence_contract_passed = 0
    for index, question in enumerate(questions, start=1):
        output = production.run(question["question"], question["paper_id"])
        pages = unique_candidate_pages(output)
        gold_page = gold_by_question[question["question_id"]]["gold_page"]
        gold_rank = pages.index(gold_page) + 1 if gold_page in pages else None
        evidence = output["evidence"]
        contract_ok = len(output["candidates"]) == 20 and len(evidence) == 5 and all(
            {
                "evidence_id",
                "page",
                "section",
                "chunk_id",
                "text",
                "locator",
                "scores",
            }
            <= set(item)
            for item in evidence
        )
        evidence_contract_passed += contract_ok
        rows.append(
            {
                "question_id": question["question_id"],
                "paper_id": question["paper_id"],
                "language": question["question_language"],
                "gold_page": gold_page,
                "predicted_pages": pages,
                "gold_rank": gold_rank,
                "candidate_count": len(output["candidates"]),
                "evidence_count": len(evidence),
                "evidence_contract_passed": bool(contract_ok),
                "front_matter_candidate_injected": output["candidate_policy"][
                    "front_matter_guard"
                ]["candidate_injected"],
            }
        )
        print(
            f"[{index:02d}/{len(questions)}] {question['question_id']} "
            f"gold_rank={gold_rank} evidence={len(evidence)}",
            flush=True,
        )

    metrics = metrics_from_results(rows)
    gate = freeze["quality_gates"]["retrieval"]
    lost_top5 = [
        row["question_id"]
        for row in rows
        if row["gold_rank"] is None or row["gold_rank"] > 5
    ]
    checks = {
        "verified_40_complete": len(rows) == 40,
        "candidate_and_evidence_contract": evidence_contract_passed == 40,
        "recall_at_5": metrics["recall_at_5"] >= gate["minimum_recall_at_5"],
        "recall_at_10": metrics["recall_at_10"] >= gate["minimum_recall_at_10"],
        "mrr": metrics["mrr"] >= gate["minimum_mrr"],
        "lost_top5_hits": len(lost_top5) <= gate["maximum_lost_top5_hits"],
        "model_revisions_pinned": all(
            len(production.model_lock["models"][role]["revision"]) == 40
            for role in ("embedding", "reranker")
        ),
    }
    output = {
        "schema_version": 1,
        "evaluation": "final-frozen-retrieval",
        "status": "passed" if all(checks.values()) else "failed",
        "seed": 378,
        "gold_loaded_only_for_scoring": True,
        "provenance": {
            "questions": "data/evaluation/questions.jsonl",
            "questions_sha256": sha256(
                PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
            ),
            "gold_evidence": "data/evaluation/gold_evidence.jsonl",
            "gold_evidence_sha256": sha256(
                PROJECT_ROOT / "data" / "evaluation" / "gold_evidence.jsonl"
            ),
            "chunks": "data/processed/chunks.jsonl",
            "chunks_sha256": sha256(
                PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
            ),
            "production_config": "config/production_retrieval.json",
            "production_config_fingerprint": production.fingerprint,
            "model_lock": "config/model_lock.json",
            "model_lock_sha256": production.model_lock_fingerprint,
            "evaluation_freeze": "config/evaluation_freeze.json",
            "evaluation_freeze_sha256": sha256(FREEZE_PATH),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "models": {
            role: production.model_lock["models"][role]
            for role in ("embedding", "reranker")
        },
        "devices": {"embedding": "cpu", "reranker": "cpu"},
        "corpus": production.config["corpus"],
        "benchmark": production.config["benchmark"],
        "metrics": metrics,
        "checks": checks,
        "lost_top5_hits": lost_top5,
        "evidence_contract_passed": evidence_contract_passed,
        "front_matter_guard_injections": sum(
            row["front_matter_candidate_injected"] for row in rows
        ),
        "embedding_cache": "hit" if production.embedding_cache_hit else "created",
        "runtime_seconds": round(perf_counter() - started, 3),
        "results": rows,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "metrics": metrics,
                "checks": checks,
                "lost_top5_hits": lost_top5,
                "runtime_seconds": output["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved: {OUTPUT_PATH}")
    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
