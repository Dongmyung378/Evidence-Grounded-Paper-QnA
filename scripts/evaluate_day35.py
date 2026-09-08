"""Run and record the original-roadmap Day 35 complete API flow gate."""

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_acceptance import digest, run_complete_api_flow


OUTPUT = ROOT / "data" / "evaluation" / "day35_integration_results.json"
SOURCE_PDF = ROOT / "data" / "raw" / "papers" / "paper-003.pdf"
QUESTION = "What is the main focus of this paper's review of text summarization?"
EXPECTED_FLOW = [
    "health",
    "upload",
    "parse",
    "chunk",
    "paper_result",
    "dense_index",
    "question",
    "answer",
]
FROZEN_FILES = [
    "config/production_retrieval.json",
    "config/generation.json",
    "config/abstention_candidate.json",
    "config/runtime_qna.json",
    "data/processed/chunks.jsonl",
    "data/processed/pages.jsonl",
    "data/evaluation/questions.jsonl",
    "data/evaluation/gold_evidence.jsonl",
]
IMPLEMENTATION_FILES = [
    "app/main.py",
    "app/models.py",
    "app/question_engine.py",
    "app/service.py",
    "app/storage.py",
    "scripts/abstention_policy.py",
    "scripts/answer_quality.py",
    "scripts/api_acceptance.py",
    "scripts/bm25_retrieval.py",
    "scripts/candidate_evidence_pipeline.py",
    "scripts/dense_retrieval.py",
    "scripts/evaluate_day35.py",
    "scripts/evidence_selector.py",
    "scripts/grounded_answer_contract.py",
    "scripts/hybrid_retrieval.py",
    "scripts/ingest_papers.py",
    "scripts/local_llm.py",
    "scripts/production_retrieval.py",
    "scripts/qna_pipeline.py",
    "scripts/reranker.py",
    "scripts/validate_day35.py",
    "tests/test_day33_api.py",
    "tests/test_day34_api.py",
    "tests/test_improvements.py",
]


def implementation_hashes():
    return {name: digest(ROOT / name) for name in IMPLEMENTATION_FILES}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen_before = {name: digest(ROOT / name) for name in FROZEN_FILES}
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"),
        pattern="test_*.py",
    )
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit(1)

    with TemporaryDirectory(prefix="paper-qna-day35-") as directory:
        complete_flow = run_complete_api_flow(
            Path(directory),
            SOURCE_PDF,
            QUESTION,
            seed=378,
        )

    assert complete_flow["flow_steps"] == EXPECTED_FLOW
    frozen_after = {name: digest(ROOT / name) for name in FROZEN_FILES}
    assert frozen_before == frozen_after, "Frozen inputs changed during Day 35"

    result = {
        "roadmap_day": 35,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 378,
        "roadmap_gate": "PDF -> parse -> index -> question -> answer",
        "tests_run": test_result.testsRun,
        "test_failures": len(test_result.failures),
        "test_errors": len(test_result.errors),
        "complete_flow": complete_flow,
        "frozen_inputs_unchanged": True,
        "frozen_input_sha256": frozen_after,
        "implementation_sha256": implementation_hashes(),
        "temporary_runtime_removed": True,
        "quality_claim": "functional integration only",
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tests": test_result.testsRun,
                "flow": complete_flow["flow_steps"],
                "pages": complete_flow["traceability"]["pages"],
                "chunks": complete_flow["traceability"]["chunks"],
                "question_status": complete_flow["question_status_code"],
                "sufficiency": complete_flow["question_result"]["sufficiency"],
                "seed": 378,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
