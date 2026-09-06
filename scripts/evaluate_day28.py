"""Evaluate the frozen Day 28 production retrieval path on verified-40."""

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


OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "day28_production_metrics.json"


def unique_candidate_pages(result, top_k=10):
    pages = []
    for candidate in result["candidates"]:
        page = candidate["page"]
        if page not in pages:
            pages.append(page)
        if len(pages) == top_k:
            break
    return pages


def main():
    configure_utf8_stdout()
    started = perf_counter()
    production = ProductionRetrieval()
    questions, gold_by_question = load_evaluation_data()
    results = []
    evidence_contract_passed = 0

    for index, question in enumerate(questions, start=1):
        output = production.run(question["question"], question["paper_id"])
        pages = unique_candidate_pages(output)
        gold_page = gold_by_question[question["question_id"]]["gold_page"]
        gold_rank = pages.index(gold_page) + 1 if gold_page in pages else None
        evidence = output["evidence"]
        contract_ok = len(output["candidates"]) == 20 and len(evidence) == 5 and all(
            {"evidence_id", "page", "section", "chunk_id", "text", "locator", "scores"}
            <= set(item)
            for item in evidence
        )
        evidence_contract_passed += contract_ok
        results.append({
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
        })
        print(
            f"[{index:02d}/{len(questions)}] {question['question_id']} "
            f"gold_rank={gold_rank} evidence={len(evidence)}"
        )

    metrics = metrics_from_results(results)
    gate = production.config["quality_gates"]
    lost_top5 = [row["question_id"] for row in results if row["gold_rank"] is None or row["gold_rank"] > 5]
    checks = {
        "questions_are_verified_40": len(results) == 40,
        "candidate_and_evidence_contract": evidence_contract_passed == 40,
        "recall_at_5": metrics["recall_at_5"] >= gate["minimum_recall_at_5"],
        "recall_at_10": metrics["recall_at_10"] >= gate["minimum_recall_at_10"],
        "mrr": metrics["mrr"] >= gate["minimum_mrr"],
        "lost_top5_hits": len(lost_top5) <= gate["maximum_lost_top5_hits"],
    }
    artifact = {
        "schema_version": 1,
        "roadmap_day": 28,
        "gate": "production_retrieval_freeze",
        "status": "passed" if all(checks.values()) else "failed",
        "production_config_path": "config/production_retrieval.json",
        "production_config_fingerprint": production.fingerprint,
        "corpus": production.config["corpus"],
        "benchmark": production.config["benchmark"],
        "metrics": metrics,
        "checks": checks,
        "lost_top5_hits": lost_top5,
        "evidence_contract_passed": evidence_contract_passed,
        "front_matter_guard_injections": sum(
            row["front_matter_candidate_injected"] for row in results
        ),
        "embedding_cache": "hit" if production.embedding_cache_hit else "created",
        "runtime_seconds": round(perf_counter() - started, 3),
        "results": results,
    }
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"],
        "metrics": metrics,
        "checks": checks,
        "lost_top5_hits": lost_top5,
        "runtime_seconds": artifact["runtime_seconds"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_PATH}")
    if artifact["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
