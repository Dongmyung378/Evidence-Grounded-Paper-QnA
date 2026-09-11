"""Run Day 34 tests and save a real curl acceptance result."""

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_acceptance import digest, run_complete_api_flow


OUTPUT = ROOT / "data" / "evaluation" / "day34_api_results.json"
SOURCE_PDF = ROOT / "data" / "raw" / "papers" / "paper-003.pdf"
QUESTION = "What is the main focus of this paper's review of text summarization?"
FROZEN_FILES = [
    "config/production_retrieval.json",
    "config/generation.json",
    "config/abstention_candidate.json",
    "config/runtime_qna.json",
    "config/generation_runtime.json",
    "data/processed/chunks.jsonl",
    "data/processed/pages.jsonl",
    "data/evaluation/questions.jsonl",
    "data/evaluation/gold_evidence.jsonl",
]
IMPLEMENTATION_FILES = [
    "app/config.py",
    "app/main.py",
    "app/models.py",
    "app/question_engine.py",
    "app/service.py",
    "app/storage.py",
    "scripts/api_acceptance.py",
    "scripts/bm25_retrieval.py",
    "scripts/candidate_evidence_pipeline.py",
    "scripts/dense_retrieval.py",
    "scripts/evaluate_day34.py",
    "scripts/evidence_selector.py",
    "scripts/hybrid_retrieval.py",
    "scripts/production_retrieval.py",
    "scripts/qna_pipeline.py",
    "scripts/reranker.py",
    "scripts/validate_day34.py",
    "tests/test_day33_api.py",
    "tests/test_day34_api.py",
    "tests/test_improvements.py",
    "tests/test_runtime_performance.py",
]


def implementation_hashes():
    return {name: digest(ROOT / name) for name in IMPLEMENTATION_FILES}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    before = {name: digest(ROOT / name) for name in FROZEN_FILES}
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"),
        pattern="test_*.py",
    )
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit(1)
    with TemporaryDirectory(prefix="paper-qna-day34-http-") as directory:
        smoke = run_complete_api_flow(
            Path(directory),
            SOURCE_PDF,
            QUESTION,
            seed=378,
        )
    after = {name: digest(ROOT / name) for name in FROZEN_FILES}
    assert before == after, "Frozen corpus or evaluation inputs changed"
    payload = {
        "roadmap_day": 34,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 378,
        "tests_run": test_result.testsRun,
        "test_failures": len(test_result.failures),
        "test_errors": len(test_result.errors),
        "http_smoke": smoke,
        "frozen_inputs_unchanged": before == after,
        "frozen_input_sha256": after,
        "implementation_sha256": implementation_hashes(),
        "temporary_runtime_removed": True,
        "scope": "health, paper result, and one-paper grounded question API",
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tests": test_result.testsRun,
        "http_transport": smoke["transport"],
        "question_status": smoke["question_status_code"],
        "sufficiency": smoke["question_result"]["sufficiency"],
        "evidence_pages": [
            item["page"] for item in smoke["question_result"]["evidence"]
        ],
        "seed": 378,
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
