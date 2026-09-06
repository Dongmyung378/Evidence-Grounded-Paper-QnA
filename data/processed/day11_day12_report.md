# Day 11–12 Ingestion Report

## Day 11 — Section-aware chunking

- Chunk size is configurable; the default is 1,200 characters.
- Overlap is configurable; the default is 200 characters.
- A page is split into section segments before chunking, so chunks do not cross
  a detected section boundary within a page.
- Every chunk carries a deterministic `chunk_id`, `paper_id`, `page`,
  `section`, source text, and character count.
- `validate_ingestion.py` checks unique IDs, non-empty text, section fallback,
  length limits, and character-count consistency.

## Day 12 — Exception handling

- Empty pages are skipped and recorded in `ingestion_report.json`.
- Layout-like two-column pages are detected with a pypdf layout probe and
  flagged for review; normal text extraction remains the evidence source.
- Footnote-like lines are retained to avoid losing possible evidence, then
  flagged in the report.
- Long section segments are split with the configured overlap and recorded.
- A malformed PDF is reported as failed without stopping the remaining batch.

The report records the actual processing result for every project PDF. The
exception test script covers the five supported handling scenarios without
claiming that the current ten valid PDFs are failures.

The five scenarios are reproducibly checked by `scripts/test_ingest_exceptions.py`.

## Latest verified run

- Papers found/succeeded/failed: 10 / 10 / 0
- Chunks created: 906
- Chunk configuration: 1,200 characters with 200-character overlap
- Reported warnings: 151 two-column candidates, 90 footnote-like cases, and
  217 long section segments split into chunks
