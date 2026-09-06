# 30-Paper Corpus Expansion Report

## Decision

The local corpus was expanded from 10 to 30 English academic papers. The
original 10-paper, 40-question verified benchmark remains frozen so that
retrieval metrics stay directly comparable. Papers 011-030 are a robustness
set and are not used for accuracy claims without manually verified gold
evidence.

## PDF validation

- Expected files: 30
- Present files: 30
- Missing or duplicate IDs: 0
- PDFs passing basic parsing and text extraction: 30/30
- New papers passing full-page text extraction: 20/20
- New-paper page range: 10-27 pages
- Files over the 20 MB MVP limit: 0

## Ingestion result

| Measure | Original corpus | Expanded corpus | Added |
|---|---:|---:|---:|
| Papers | 10 | 30 | 20 |
| Pages | 188 | 563 | 375 |
| Chunks | 906 | 2,463 | 1,557 |

All 30 papers succeeded. Page-to-chunk traceability remains intact through
`paper_id`, `page_id`, `source_page_id`, and `chunk_id`.

## Retrieval regression

The dense cache was rebuilt for all 2,463 chunks. Accuracy was re-evaluated on
the unchanged verified-40 benchmark:

- Dense Recall@5: 0.775
- Dense Recall@10: 0.975
- Day 27 production candidate fix Recall@5: 1.000
- Day 27 production candidate fix Recall@10: 1.000
- Day 27 production candidate fix MRR: 0.6317
- Existing Top-5 hits lost after expansion: 0

Because retrieval is scoped to the single uploaded paper, adding unrelated
papers does not compete with the active paper's chunks.

## Expansion runtime smoke test

Each new paper was passed through the model-backed production retrieval path:

`BM25 -> multilingual E5 -> equal-weight RRF -> Cross-Encoder -> evidence selection`

- Papers tested: 20
- English queries: 10
- Korean queries: 10
- Successful candidate/evidence outputs: 20/20
- Expected output per paper: 20 reranked candidates and 5 evidence objects

Machine-readable results are stored in
`data/evaluation/corpus_expansion_smoke.json`.

## Known constraints

- `paper-020` is verified as arXiv:2608.26855v1 under the arXiv non-exclusive
  distribution license and remains local-only.
- `paper-026` is verified as arXiv:2607.03214v1 under CC BY-NC-ND 4.0.
- All former `check` license values were resolved against official arXiv or
  journal pages. Raw PDFs under the arXiv non-exclusive distribution license
  remain local-only.
- Equation-heavy papers test text extraction only; formula interpretation is
  excluded from the MVP.
- The expansion set demonstrates runtime and format robustness, not retrieval
  accuracy, until additional gold evidence is manually verified.

## Reproduction

```text
python -B scripts/validate_papers.py
python -B scripts/run_ingestion_pipeline.py
python -B scripts/validate_pipeline.py
python -B scripts/evaluate_dense.py --output data/evaluation/dense_metrics.json
python -B scripts/evaluate_rerank_pool_fix.py
python -B scripts/evaluate_corpus_expansion.py
python -B scripts/validate_corpus_expansion.py
```
