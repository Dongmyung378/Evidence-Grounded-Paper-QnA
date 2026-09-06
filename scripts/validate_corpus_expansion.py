"""Validate the 10-to-30 paper expansion without changing verified-40 semantics."""

import csv
import json
from pathlib import Path

import numpy as np

from dense_retrieval import CACHE_PATH, DEFAULT_MODEL, chunks_fingerprint
from retrieval_common import PROJECT_ROOT, load_chunks, load_jsonl


MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "paper_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "ingestion_report.json"
SMOKE_PATH = PROJECT_ROOT / "data" / "evaluation" / "corpus_expansion_smoke.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"


def main():
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    expected = {f"paper-{number:03d}" for number in range(1, 31)}
    benchmark = {f"paper-{number:03d}" for number in range(1, 11)}
    expansion = expected - benchmark

    assert len(manifest) == 30
    assert {row["paper_id"] for row in manifest} == expected
    assert all(row["language"] == "en" and row["text_extractable"] == "yes" for row in manifest)
    assert all(int(row["page_count"]) > 0 for row in manifest)
    assert all((PROJECT_ROOT / "data" / "raw" / "papers" / row["pdf_filename"]).exists() for row in manifest)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["summary"]["papers_found"] == 30
    assert report["summary"]["papers_succeeded"] == 30
    assert report["summary"]["papers_failed"] == 0
    assert {row["paper_id"] for row in report["papers"]} == expected

    chunks = load_chunks()
    assert {chunk["paper_id"] for chunk in chunks} == expected
    assert all(chunk["text"].strip() for chunk in chunks)
    assert len(chunks) == report["summary"]["chunks_created"]

    questions = load_jsonl(QUESTIONS_PATH)
    assert len(questions) == 40
    assert {row["paper_id"] for row in questions} == benchmark
    assert not ({row["paper_id"] for row in questions} & expansion)

    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        assert str(cache["model"].item()) == DEFAULT_MODEL
        assert str(cache["fingerprint"].item()) == chunks_fingerprint(chunks)
        assert len(cache["embeddings"]) == len(chunks)

    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    assert set(smoke["benchmark_policy"]["expansion_papers"]) == expansion
    assert smoke["benchmark_policy"]["expansion_used_for_accuracy_claims"] is False
    assert smoke["summary"]["papers"] == 20
    assert smoke["summary"]["passed"] == 20 and smoke["summary"]["failed"] == 0
    assert smoke["summary"]["english_queries"] == 10
    assert smoke["summary"]["korean_queries"] == 10
    assert all(case["candidate_count"] == 20 and case["evidence_count"] == 5 for case in smoke["cases"])

    print("30-paper corpus expansion validation passed")
    print(
        f"papers=30 pages={report['summary']['pages_extracted']} "
        f"chunks={report['summary']['chunks_created']}"
    )
    print("benchmark=verified-40 on paper-001..010 (frozen)")
    print("expansion_smoke=20/20 papers, en=10 ko=10")


if __name__ == "__main__":
    main()
