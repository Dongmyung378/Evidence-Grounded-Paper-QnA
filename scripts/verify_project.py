"""Run lightweight regression tests and saved-artifact validators."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_VALIDATORS = [
    "validate_questions.py",
    "validate_ingestion.py",
    "validate_pipeline.py",
    "validate_grounded_generation.py",
    "validate_container.py",
    "validate_container_qna.py",
    "validate_evaluation_freeze.py",
    "validate_portfolio_scope.py",
]
LOCAL_CORPUS_VALIDATORS = [
    "validate_gold_evidence.py",
    "validate_papers.py",
    "validate_corpus_expansion.py",
]


def run(arguments):
    print(f"\n[run] {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, check=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(["-m", "unittest", "discover", "-s", "tests", "-v"])
    for name in PUBLIC_VALIDATORS:
        run([f"scripts/{name}"])

    raw_papers = list((ROOT / "data" / "raw" / "papers").glob("paper-*.pdf"))
    dense_cache = ROOT / "data" / "processed" / "frozen_dense_embeddings.npz"
    local_corpus_ready = len(raw_papers) == 30 and dense_cache.is_file()
    if local_corpus_ready:
        for name in LOCAL_CORPUS_VALIDATORS:
            run([f"scripts/{name}"])
        scope = "public artifacts and local 30-paper corpus"
    else:
        print(
            "\n[skip] Local corpus validation requires 30 private PDFs and the "
            "generated dense cache.",
            flush=True,
        )
        scope = "public clone artifacts"
    print(f"\nProject verification passed: {scope}.")


if __name__ == "__main__":
    main()
