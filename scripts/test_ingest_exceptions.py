"""Exercise the five Day 12 exception-handling scenarios without altering data."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ingest_papers import (
    DEFAULT_SECTION,
    detect_two_column_layout,
    footnote_line_count,
    process_paper,
    split_page_into_sections,
    split_text,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "processed" / "exception_test_report.json"


def main():
    cases = []

    segments, section, _ = split_page_into_sections("", DEFAULT_SECTION)
    assert segments == [] and section == DEFAULT_SECTION
    cases.append({"case": "empty_page", "result": "passed", "handling": "skipped without stopping ingestion"})

    two_column_lines = ["left text    right text"] * 6
    assert detect_two_column_layout(two_column_lines)
    cases.append({"case": "two_column", "result": "passed", "handling": "flagged for review"})

    assert footnote_line_count(["1 This note remains in the source text."]) == 1
    cases.append({"case": "footnote", "result": "passed", "handling": "retained and reported"})

    chunks = split_text("A" * 2600, chunk_size=1200, overlap=200)
    assert len(chunks) == 3 and max(len(chunk) for chunk in chunks) <= 1200
    assert chunks[0][-200:] == chunks[1][:200]
    cases.append({"case": "long_segment", "result": "passed", "handling": "split with configured overlap"})

    with TemporaryDirectory() as directory:
        broken_pdf = Path(directory) / "paper-broken.pdf"
        broken_pdf.write_bytes(b"not a PDF")
        records, pages, report = process_paper(broken_pdf, chunk_size=1200, overlap=200)
        assert records == [] and pages == [] and report["status"] == "failed"
    cases.append({"case": "malformed_pdf", "result": "passed", "handling": "failure recorded; batch can continue"})

    REPORT_PATH.write_text(
        json.dumps({"cases": cases, "passed": len(cases)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Day 12 exception tests passed: empty_page, two_column, footnote, long_segment, malformed_pdf")


if __name__ == "__main__":
    main()
