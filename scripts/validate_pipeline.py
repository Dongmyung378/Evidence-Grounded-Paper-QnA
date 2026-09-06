"""Validate the complete Day 13/14 pipeline outputs and traceability."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES_PATH = ROOT / "data" / "processed" / "pages.jsonl"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
REPORT_PATH = ROOT / "data" / "processed" / "ingestion_report.json"
MANIFEST_PATH = ROOT / "data" / "processed" / "pipeline_manifest.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    pages = load_jsonl(PAGES_PATH)
    chunks = load_jsonl(CHUNKS_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    page_by_id = {page["page_id"]: page for page in pages}
    assert len(page_by_id) == len(pages), "Duplicate page_id"
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks), "Duplicate chunk_id"
    assert report["summary"]["pages_extracted"] == len(pages), "Page count mismatch"
    assert report["summary"]["chunks_created"] == len(chunks), "Chunk count mismatch"
    assert manifest["summary"] == report["summary"], "Manifest/report summary mismatch"
    assert manifest["pipeline"] == "PDF → pages.jsonl → chunks.jsonl", "Pipeline manifest is invalid"

    for page in pages:
        assert page["text"].strip(), f"Empty page text: {page['page_id']}"
        assert page["char_count"] == len(page["text"]), f"Bad page char_count: {page['page_id']}"
        assert page["sections"], f"Missing sections: {page['page_id']}"

    for chunk in chunks:
        page = page_by_id.get(chunk["source_page_id"])
        assert page is not None, f"Unknown source page: {chunk['chunk_id']}"
        assert chunk["paper_id"] == page["paper_id"] and chunk["page"] == page["page"], (
            f"Broken page trace: {chunk['chunk_id']}"
        )
        assert chunk["text"].strip() and chunk["char_count"] == len(chunk["text"]), (
            f"Invalid chunk: {chunk['chunk_id']}"
        )
        assert any(section["section"] == chunk["section"] for section in page["sections"]), (
            f"Broken section trace: {chunk['chunk_id']}"
        )

    print("Pipeline validation passed")
    print(f"papers={report['summary']['papers_succeeded']} pages={len(pages)} chunks={len(chunks)}")
    print("traceability=PDF page → page_id → chunk_id")


if __name__ == "__main__":
    main()
