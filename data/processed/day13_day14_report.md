# Day 13–14 Pipeline Report

## Day 13 — Batch pipeline

`python scripts/run_ingestion_pipeline.py` processes every `paper-*.pdf` in
`data/raw/papers/` in one batch. A single failed PDF is reported and does not
stop the remaining papers.

## Day 14 — Pipeline gate

The same command produces the complete traceable pipeline:

```text
PDF → pages.jsonl → chunks.jsonl → ingestion_report.json → pipeline_manifest.json
```

- `pages.jsonl` stores page-level source text and detected section segments.
- `chunks.jsonl` stores retrieval chunks with `source_page_id`, section index,
  and chunk index for reverse tracing.
- `validate_pipeline.py` verifies IDs, output counts, page/chunk links, and
  section links.

## Portfolio evidence

The project can demonstrate a reproducible batch ingestion system in which a
folder of academic PDFs is converted into structured, page-traceable JSONL and
search chunks through one command. This supports the later evidence display of
original text, page number, section, and chunk ID.
