"""Run the repository's lightweight tests and saved-artifact validators."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STANDALONE_TESTS = [
    "test_abstention_policy.py",
    "test_bm25_retrieval.py",
    "test_evidence_selector.py",
    "test_grounded_answer_contract.py",
    "test_ingest_exceptions.py",
    "test_production_config.py",
    "test_qna_pipeline.py",
    "test_rerank_pool.py",
    "test_reranker_core.py",
    "test_retrieval_core.py",
]
VALIDATORS = [
    "validate_questions.py",
    "validate_gold_evidence.py",
    "validate_papers.py",
    "validate_ingestion.py",
    "validate_pipeline.py",
    "validate_corpus_expansion.py",
    "validate_day7.py",
    "validate_day15_day17.py",
    "validate_day18_day20.py",
    "validate_day21_day22.py",
    "validate_day23_day24.py",
    "validate_day25_day27.py",
    "validate_day28.py",
    "validate_day29.py",
    "validate_day30.py",
    "validate_day31.py",
    "validate_day32.py",
    "validate_day33.py",
    "validate_day34.py",
    "validate_day35.py",
    "validate_day36.py",
    "validate_evidence_ui.py",
    "validate_improvements.py",
]


def run(arguments):
    command = [sys.executable, "-B", *arguments]
    print(f"\n[실행] {' '.join(arguments)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-api",
        action="store_true",
        help="저장 결과 검증 전에 실제 35일차 HTTP 흐름을 다시 실행합니다.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    run(["-m", "unittest", "discover", "-s", "tests", "-v"])
    for name in STANDALONE_TESTS:
        run([f"scripts/{name}"])
    if args.full_api:
        run(["scripts/evaluate_day35.py"])
    for name in VALIDATORS:
        run([f"scripts/{name}"])

    print("\n전체 프로젝트 검증을 통과했습니다.")


if __name__ == "__main__":
    main()
