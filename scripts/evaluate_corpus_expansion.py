"""Model-backed smoke evaluation for the 20-paper corpus expansion."""

import json
import time
from pathlib import Path

from production_retrieval import ProductionRetrieval
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "corpus_expansion_smoke.json"
EXPANSION_PAPERS = [f"paper-{number:03d}" for number in range(11, 31)]
QUERIES = {
    "en": "What is the main objective of this paper?",
    "ko": "이 논문의 주요 연구 목적은 무엇인가?",
}


def main():
    configure_utf8_stdout()
    started = time.perf_counter()
    production = ProductionRetrieval()
    cases = []

    for offset, paper_id in enumerate(EXPANSION_PAPERS):
        language = "en" if offset % 2 == 0 else "ko"
        result = production.run(QUERIES[language], paper_id)
        evidence = result["evidence"]
        case = {
            "paper_id": paper_id,
            "question_language": language,
            "query": QUERIES[language],
            "candidate_count": len(result["candidates"]),
            "evidence_count": len(evidence),
            "evidence_pages": [item["page"] for item in evidence],
            "evidence_chunk_ids": [item["chunk_id"] for item in evidence],
            "front_matter_injected": result["candidate_policy"]["front_matter_guard"][
                "candidate_injected"
            ],
            "status": "pass" if len(result["candidates"]) == 20 and len(evidence) == 5 else "fail",
        }
        cases.append(case)
        print(
            f"[{case['status'].upper()}] {paper_id} language={language} "
            f"candidates={case['candidate_count']} evidence={case['evidence_count']}"
        )

    payload = {
        "schema_version": 1,
        "purpose": "30-paper corpus expansion runtime smoke test",
        "benchmark_policy": {
            "verified_40_papers": [f"paper-{number:03d}" for number in range(1, 11)],
            "expansion_papers": EXPANSION_PAPERS,
            "expansion_used_for_accuracy_claims": False,
        },
        "models": {
            "embedding": production.config["models"]["embedding"],
            "reranker": production.config["models"]["reranker"],
        },
        "production_config_fingerprint": production.fingerprint,
        "embedding_cache": "hit" if production.embedding_cache_hit else "created",
        "summary": {
            "papers": len(cases),
            "passed": sum(case["status"] == "pass" for case in cases),
            "failed": sum(case["status"] != "pass" for case in cases),
            "english_queries": sum(case["question_language"] == "en" for case in cases),
            "korean_queries": sum(case["question_language"] == "ko" for case in cases),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
        "cases": cases,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT_PATH}")

    if payload["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
