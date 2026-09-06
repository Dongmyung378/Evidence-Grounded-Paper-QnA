"""Validate chunk invariants produced by the Day 11/12 ingestion pipeline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
REPORT_PATH = ROOT / "data" / "processed" / "ingestion_report.json"


def main():
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line]
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    config = report["config"]

    assert chunks, "No chunks were created"
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks), "Duplicate chunk IDs"
    assert all(chunk["text"].strip() for chunk in chunks), "Empty chunk text"
    assert all(chunk["section"].strip() for chunk in chunks), "Missing section fallback"
    assert all(chunk["char_count"] == len(chunk["text"]) for chunk in chunks), "Incorrect char_count"
    assert max(chunk["char_count"] for chunk in chunks) <= config["chunk_size"], "Chunk exceeds size limit"
    summary = report["summary"]
    expected_papers = summary["papers_found"]
    assert summary["papers_succeeded"] == expected_papers, "All discovered PDFs must be processed"
    assert summary["papers_failed"] == 0, "Unexpected PDF parsing failure"
    assert len({chunk["paper_id"] for chunk in chunks}) == expected_papers, "Paper coverage mismatch"
    assert summary["chunks_created"] == len(chunks), "Report/chunk count mismatch"

    print("Ingestion validation passed")
    print(f"papers={expected_papers} succeeded={summary['papers_succeeded']}")
    print(f"chunks={len(chunks)} max_chunk_chars={max(chunk['char_count'] for chunk in chunks)}")
    print(f"config={config}")
    print(f"warning_counts={report['summary']['warning_counts']}")


if __name__ == "__main__":
    main()
