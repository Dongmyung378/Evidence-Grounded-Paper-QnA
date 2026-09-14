"""Run lightweight regression tests and saved-artifact validators."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATORS = [
    "validate_questions.py",
    "validate_gold_evidence.py",
    "validate_papers.py",
    "validate_ingestion.py",
    "validate_pipeline.py",
    "validate_corpus_expansion.py",
    "validate_grounded_generation.py",
    "validate_container.py",
    "validate_container_qna.py",
    "validate_evaluation_freeze.py",
    "validate_portfolio_scope.py",
]


def run(arguments):
    print(f"\n[run] {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, check=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(["-m", "unittest", "discover", "-s", "tests", "-v"])
    for name in VALIDATORS:
        run([f"scripts/{name}"])
    print("\nProject verification passed.")


if __name__ == "__main__":
    main()
