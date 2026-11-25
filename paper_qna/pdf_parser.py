"""PDF ingestion and corpus30 chunk loader with deterministic text repair."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from paper_qna.verifier import repair_pdf_chunk_text

SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(Abstract|Introduction|Background|Related Work|Methodology|Methods?|Approach|"
    r"Experiments?|Evaluation|Results?|Discussion|Limitations?|Conclusion|Conclusions|"
    r"References|Appendix)\b",
    flags=re.I,
)


def chunk_page_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    cleaned = repair_pdf_chunk_text(re.sub(r"-\s*\n", "", str(text or "")))
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            cut = cleaned.rfind(". ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut + 1
        piece = cleaned[start:end].strip()
        if len(piece) >= 80 or not chunks:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _detect_section(page_text: str, current_section: str) -> str:
    for line in str(page_text or "").splitlines()[:8]:
        m = SECTION_HEADING_RE.match(line.strip())
        if m:
            return m.group(1).title()
    return current_section


def parse_pdf_bytes(
    pdf_bytes: bytes,
    filename: str = "uploaded.pdf",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> dict[str, Any]:
    """Extract page- and section-aware overlapping chunks from a text-based PDF."""
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Uploaded file is not a valid PDF document.")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to parse uploaded PDF files.") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    if not reader.pages:
        raise ValueError("The uploaded PDF contains zero pages.")

    digest = hashlib.sha256(pdf_bytes).hexdigest()[:10]
    paper_id = f"upload-{digest}"
    chunks: list[dict[str, Any]] = []
    current_section = "Abstract"
    title_guess = Path(filename).stem.replace("_", " ").replace("-", " ").strip()

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if page_idx == 1 and raw_text.strip():
            first_lines = [ln.strip() for ln in raw_text.splitlines() if len(ln.strip()) >= 12]
            if first_lines:
                title_guess = repair_pdf_chunk_text(first_lines[0])[:140]
        current_section = _detect_section(raw_text, current_section)
        page_chunks = chunk_page_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        for ci, ctext in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}-p{page_idx:03d}-c{ci:03d}",
                    "page": page_idx,
                    "section": current_section,
                    "text": ctext,
                }
            )

    if not chunks:
        raise ValueError(
            "No extractable text was found in this PDF. Scanned image-only PDFs require OCR and are not supported."
        )

    return {
        "paper_id": paper_id,
        "title": title_guess,
        "filename": filename,
        "page_count": len(reader.pages),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "source": "uploaded",
    }


def load_corpus30_papers(root: Path) -> dict[str, dict[str, Any]]:
    """Load the 30 benchmark papers and their repaired chunks."""
    manifest_path = root / "data" / "benchmarks" / "corpus30" / "manifest.json"
    chunks_path = root / "data" / "processed" / "chunks.jsonl"
    if not manifest_path.is_file() or not chunks_path.is_file():
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ch = json.loads(line)
        ch["text"] = repair_pdf_chunk_text(ch.get("text", ""))
        by_paper[ch["paper_id"]].append(ch)

    papers: dict[str, dict[str, Any]] = {}
    for meta in manifest.get("papers", []):
        pid = meta["paper_id"]
        chs = sorted(by_paper.get(pid, []), key=lambda c: (c["page"], c["chunk_id"]))
        if not chs:
            continue
        papers[pid] = {
            "paper_id": pid,
            "title": meta.get("title", pid),
            "split": meta.get("split", "holdout"),
            "domain": meta.get("domain", "scientific"),
            "source_url": meta.get("source_url", ""),
            "page_count": max(c["page"] for c in chs),
            "chunk_count": len(chs),
            "chunks": chs,
            "source": "corpus30",
        }
    return papers
