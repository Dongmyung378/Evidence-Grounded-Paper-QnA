# Day 7 Gate Report

Date: 2026-08-28

## Decision

**PASS WITH LOCAL-DATA RESTRICTION**

The project data definition, evaluation set, split policy, and retrieval
results are fixed for the next implementation stage. Raw PDFs and QASPER data
remain local because redistribution rights are not verified for every paper.

## Checks

| Area | Result | Evidence |
|---|---|---|
| Project questions | Pass | 40 records, 20 English and 20 Korean |
| Question IDs | Pass | 40 unique IDs; no missing IDs in Gold evidence |
| Gold evidence | Pass | 40 records; all `review_status=verified` |
| Paper coverage | Pass | 10/10 papers represented in chunks and evaluation |
| QASPER split | Pass | Local source split counts: train 888, validation 281, test 416 |
| Evaluation policy | Pass | `verified-40` is the fixed project test set; QASPER is not mixed into it |
| PDF extraction | Pass | 10/10 PDFs text-extractable; 834 chunks at the Day 7 gate; 0 empty chunk texts |
| Section fallback | Pass | Every chunk has a section label; unrecognized front matter uses `Front Matter` |
| Redistribution safety | Restricted | Four manifest rows still require individual license verification |
| Git data policy | Pass | `data/raw/qasper/` and raw PDFs are ignored by `.gitignore` |

## Fixed evaluation results

| Retriever | Recall@5 | Notes |
|---|---:|---|
| BM25 | 0.850 | English 1.000; Korean 0.700 |
| Multilingual E5 dense | 0.700 | English 0.600; Korean 0.800 |

Dense metrics are stored in `dense_metrics.json`. The next comparison must use
the same 40 questions, gold pages, and page-level metric definition.

## License handling

The manifest keeps the original arXiv URLs and marks unresolved licenses as
`check`. The four unresolved papers are paper-001, paper-003, paper-007, and
paper-010. Until each license is verified, the PDFs must stay local and must
not be committed to a public repository.

## Gate outcome

Do not change the project evaluation questions, Gold evidence, paper IDs, or
split policy during the next retrieval stage. Proceed to BM25 + Dense Hybrid
Retrieval using the fixed evaluation set.

After this gate, Day 11 section-aware rechunking produced 892 chunks. Retrieval
metrics from before rechunking are historical baselines and must be rerun
before the next comparison.
