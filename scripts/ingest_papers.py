"""Convert text-extractable paper PDFs into section-aware search chunks."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
PAGES_PATH = PROJECT_ROOT / "data" / "processed" / "pages.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "ingestion_report.json"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SECTION = "Front Matter"

KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "method",
    "methods", "approach", "data", "dataset", "experiments", "experiment",
    "results", "discussion", "analysis", "limitations", "conclusion",
    "conclusions", "references", "acknowledgments", "appendix",
}
MONTH_NAMES = (
    "Jan ", "Feb ", "Mar ", "Apr ", "May ", "Jun ",
    "Jul ", "Aug ", "Sep ", "Oct ", "Nov ", "Dec ",
)
AFFILIATION_MARKERS = (
    "university", "department", "corporation", "institute", "school of",
    "email", "@",
)


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text, chunk_size, overlap):
    """Split text while keeping a deterministic character overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def detect_heading(line):
    """Return a plausible section heading, or ``None`` for ordinary text."""
    line = re.sub(r"\s+", " ", line).strip(" :\t")
    if not line or len(line) < 3 or len(line) > 100:
        return None

    normalized = line.lower().rstrip(".")
    numbered_match = re.match(r"^(\d{1,2}(?:\.\d{1,2})*)[.)]?\s+(.+)$", line)
    if not numbered_match:
        # Accept compact headings such as ``1.INTRODUCTION`` but not numeric
        # table values or prose such as ``1.0Edge F-score``.
        numbered_match = re.match(
            r"^(\d{1,2}(?:\.\d{1,2})*)\.([A-Z][A-Z\s-]{2,80})$",
            line,
        )
    roman_match = re.match(r"^[IVXLCDM]+\.\s+([A-Z][A-Z\s-]{2,80})$", line)
    abstract_match = re.match(
        r"^abstract\s*(?:\u2014|:|-)",
        line,
        flags=re.IGNORECASE,
    )
    known = normalized in KNOWN_HEADINGS
    appendix = re.match(r"^appendix(?:\s+[a-z])?$", normalized)
    punctuation_target = numbered_match.group(2) if numbered_match else line
    has_sentence_punctuation = any(mark in punctuation_target for mark in ".,;?!")
    letters = sum(char.isalpha() for char in line)
    compact_heading = letters / max(len(line), 1) >= 0.55

    numbered_body = numbered_match.group(2) if numbered_match else ""
    numbered_is_heading = (
        numbered_match is not None
        and numbered_body[:1].isupper()
        and ":" not in numbered_body
        and "=" not in numbered_body
        and "/" not in numbered_body
        and not any(char.isdigit() for char in numbered_body)
        and not numbered_body.startswith(("How ", "What ", "Why ", "When ", "Which "))
        and not numbered_body.startswith(MONTH_NAMES)
        and "cutoff" not in numbered_body.lower()
        and not any(marker in numbered_body.lower() for marker in AFFILIATION_MARKERS)
        and (
            numbered_body.lower() in KNOWN_HEADINGS
            or len(numbered_body.split()) >= 2
        )
        and len(line) <= 90
    )
    if abstract_match:
        return "Abstract"
    if roman_match:
        return line
    if (numbered_is_heading or known or appendix) and not has_sentence_punctuation and compact_heading:
        return line
    return None


def detect_two_column_layout(lines):
    """Flag likely two-column extraction so it is visible in the report."""
    candidates = [line for line in lines if line.strip()]
    if len(candidates) < 6:
        return False
    multi_gap_lines = sum(bool(re.search(r"\S {3,}\S", line)) for line in candidates)
    return multi_gap_lines / len(candidates) >= 0.25


def footnote_line_count(lines):
    """Count likely footnote lines without removing source text evidence."""
    count = 0
    for line in lines:
        stripped = line.strip()
        if detect_heading(stripped):
            continue
        if len(stripped) <= 240 and re.match(r"^(?:\d+|[*†‡])\s+", stripped):
            count += 1
    return count


def extract_page_text(page):
    """Use pypdf's normal extraction for stable heading and word spacing."""
    return page.extract_text() or ""


def extract_layout_text(page):
    """Use layout extraction only as a two-column detection signal."""
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except (TypeError, ValueError):
        return ""


def split_page_into_sections(raw_text, current_section):
    """Keep section boundaries intact when a page has multiple headings."""
    lines = raw_text.splitlines()
    section = current_section or DEFAULT_SECTION
    segments = []
    buffer = []

    def flush():
        text = clean_text("\n".join(buffer))
        if text:
            segments.append((section, text))

    for raw_line in lines:
        heading = detect_heading(raw_line)
        if heading:
            flush()
            section = heading
            buffer = [heading]
        else:
            buffer.append(raw_line)
    flush()
    return segments, section, lines


def process_paper(pdf_path, chunk_size, overlap):
    """Return chunks and a per-paper report without stopping the batch on error."""
    paper_id = pdf_path.stem
    try:
        display_path = str(pdf_path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(pdf_path)
    paper_report = {
        "paper_id": paper_id,
        "pdf_path": display_path,
        "status": "success",
        "page_count": 0,
        "chunk_count": 0,
        "warnings": [],
    }
    records = []
    page_records = []

    try:
        reader = PdfReader(str(pdf_path))
        paper_report["page_count"] = len(reader.pages)
    except Exception as exc:
        paper_report["status"] = "failed"
        paper_report["error"] = f"{type(exc).__name__}: {exc}"
        return records, page_records, paper_report

    current_section = DEFAULT_SECTION
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = extract_page_text(page)
        except Exception as exc:
            paper_report["warnings"].append({
                "page": page_number,
                "code": "page_text_extraction_failed",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue

        if not raw_text.strip():
            paper_report["warnings"].append({"page": page_number, "code": "empty_page"})
            continue

        segments, current_section, lines = split_page_into_sections(raw_text, current_section)
        page_id = f"{paper_id}-p{page_number:03d}"
        page_records.append({
            "page_id": page_id,
            "paper_id": paper_id,
            "page": page_number,
            "text": clean_text(raw_text),
            "char_count": len(clean_text(raw_text)),
            "sections": [
                {
                    "position": section_index,
                    "section": section,
                    "text": segment_text,
                    "char_count": len(segment_text),
                }
                for section_index, (section, segment_text) in enumerate(segments, start=1)
            ],
        })
        layout_text = extract_layout_text(page)
        if layout_text and detect_two_column_layout(layout_text.splitlines()):
            paper_report["warnings"].append({"page": page_number, "code": "two_column_suspected"})

        footnotes = footnote_line_count(lines)
        if footnotes:
            paper_report["warnings"].append({
                "page": page_number,
                "code": "footnote_like_lines_retained",
                "count": footnotes,
            })

        page_chunk_number = 0
        for section_index, (section, segment_text) in enumerate(segments, start=1):
            if len(segment_text) > chunk_size:
                paper_report["warnings"].append({
                    "page": page_number,
                    "code": "long_segment_split",
                    "section": section,
                    "char_count": len(segment_text),
                })
            for chunk_index, chunk in enumerate(split_text(segment_text, chunk_size, overlap), start=1):
                page_chunk_number += 1
                records.append({
                    "chunk_id": f"{paper_id}-p{page_number:03d}-c{page_chunk_number:03d}",
                    "paper_id": paper_id,
                    "page": page_number,
                    "source_page_id": page_id,
                    "section": section,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "char_count": len(chunk),
                })

    paper_report["chunk_count"] = len(records)
    paper_report["page_record_count"] = len(page_records)
    return records, page_records, paper_report


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def display_path(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_pipeline(papers_dir, chunks_output, pages_output, report_path, chunk_size, overlap):
    """Run the complete PDF → pages JSONL → chunks JSONL batch pipeline."""
    chunks = []
    pages = []
    papers = []
    for pdf_path in sorted(papers_dir.glob("paper-*.pdf")):
        paper_chunks, paper_pages, paper_report = process_paper(pdf_path, chunk_size, overlap)
        chunks.extend(paper_chunks)
        pages.extend(paper_pages)
        papers.append(paper_report)

    chunk_ids = [record["chunk_id"] for record in chunks]
    page_ids = [record["page_id"] for record in pages]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id generated; outputs were not written.")
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("Duplicate page_id generated; outputs were not written.")

    write_jsonl(pages_output, pages)
    write_jsonl(chunks_output, chunks)

    warning_counts = Counter(
        warning["code"]
        for paper in papers
        for warning in paper["warnings"]
    )
    report = {
        "config": {"chunk_size": chunk_size, "overlap": overlap},
        "outputs": {
            "pages": display_path(pages_output),
            "chunks": display_path(chunks_output),
        },
        "summary": {
            "papers_found": len(papers),
            "papers_succeeded": sum(paper["status"] == "success" for paper in papers),
            "papers_failed": sum(paper["status"] == "failed" for paper in papers),
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "papers": papers,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--pages-output", type=Path, default=PAGES_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        parser.error("chunk-size must be positive and overlap must be smaller than chunk-size")

    report = run_pipeline(
        args.papers_dir,
        args.output,
        args.pages_output,
        args.report,
        args.chunk_size,
        args.overlap,
    )

    print(f"Created: {args.output}")
    print(f"Created: {args.pages_output}")
    print(f"Report: {args.report}")
    print(f"Papers: {report['summary']['papers_succeeded']}/{report['summary']['papers_found']} succeeded")
    print(f"Pages: {report['summary']['pages_extracted']}")
    print(f"Total chunks: {report['summary']['chunks_created']}")


if __name__ == "__main__":
    main()
