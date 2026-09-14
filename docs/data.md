# Data and Licensing

[English](data.md) | [한국어](data_KO.md)

## Corpus

The local corpus contains 30 English research-paper PDFs. All 30 passed text extraction and ingestion checks. The public repository keeps metadata and processed text artifacts needed for reproducibility; raw PDFs remain local because redistribution terms differ by paper.

`data/metadata/paper_manifest.csv` records each paper's title, source URL, domain, extraction status, page count, and reviewed license status. A source URL is provenance, not proof that a PDF may be redistributed.

## Traceability

The ingestion pipeline preserves this chain:

```text
paper PDF -> paper_id -> page -> page_id -> section -> chunk_id -> evidence_id
```

Processed pages are stored in `data/processed/pages.jsonl`, searchable chunks in `data/processed/chunks.jsonl`, and pipeline counts in `data/processed/pipeline_manifest.json`. Evidence returned by the API includes the page, section, original text, and chunk identifier.

## Evaluation data

`questions.jsonl` and `gold_evidence.jsonl` contain 40 verified retrieval cases for papers 001-010. The answer review uses a fixed 20-question subset defined in `answer_review_manifest.json`. Additional unsupported questions evaluate refusal behavior. See [Evaluation](evaluation.md) for claim boundaries.

## Licensing boundary

- Raw PDFs are ignored by Git and are not copied into Docker images.
- User uploads remain in the local runtime volume and are ignored by Git.
- The translation component uses `facebook/nllb-200-distilled-600M` under CC-BY-NC-4.0, so this packaged demo is documented as noncommercial.
- Before distributing any raw source, verify the license for that exact paper version.
