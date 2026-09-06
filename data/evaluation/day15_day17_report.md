# Days 15–17 Retrieval Baseline Report

## Scope and fixed inputs

- Corpus: 10 local English paper PDFs, 188 pages, 906 chunks
- Evaluation set: 40 verified questions (English 20, Korean 20)
- Relevance target: gold evidence page
- Ranking unit: unique pages, using the highest-scoring chunk per page

The same processed chunks, questions, gold pages, page-deduplication rule, and
top-k cutoff are used for BM25, dense, and hybrid retrieval. This makes the
comparison interpretable; it is not a comparison of differently prepared data.

## Day 15 — Dense baseline

- Model: `intfloat/multilingual-e5-small`
- Passage format: `passage: {chunk text}`
- Query format: `query: {user question}`
- Similarity: cosine similarity of normalized embeddings
- Cache: `data/processed/dense_embeddings.npz`, keyed by model name and a
  deterministic chunk-id/text fingerprint
- Top-k output: `scripts/search_dense.py --output ...` writes rank, cosine
  score, paper/page/section/chunk ID, and original text.

## Day 16 — BM25

- Each paper builds an independent BM25 index, matching the service's
  one-paper-at-a-time scope.
- One shared preprocessing function is used by search and evaluation.
- Korean syllables, English terms, numeric values, and hyphenated/dotted model
  identifiers are tokenized. A full model name and its components are both
  retained (for example, `bert-base-uncased`, `bert`, `base`, `uncased`).
- `scripts/test_bm25_retrieval.py` confirms that numeric and model-name
  queries retrieve the intended chunk.

## Day 17 — RRF hybrid

- Candidate pool: top 20 chunks from each of BM25 and dense retrieval
- Fusion: reciprocal-rank fusion, `1 / (60 + rank)`
- Output: hybrid results retain the rank contributed by each source, so a
  future reranker can inspect why a chunk was selected.
- `scripts/evaluate_retrievers.py` produces the 40-question comparison in JSON
  and CSV. Each individual search command can save scored Top-K evidence with
  its `--output` option.

## Measured results

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.325 | 0.725 | 0.825 | 0.900 | 0.545 |
| Dense | 0.300 | 0.575 | 0.775 | 0.975 | 0.489 |
| Hybrid RRF | **0.350** | **0.775** | **0.925** | **0.975** | **0.569** |

| Language | BM25 Recall@5 | Dense Recall@5 | Hybrid RRF Recall@5 |
| --- | ---: | ---: | ---: |
| English | 0.950 | 0.800 | **1.000** |
| Korean | 0.700 | 0.750 | **0.850** |

Hybrid RRF is the current selected baseline. It gives the best Recall@1,
Recall@3, Recall@5, and MRR; its Recall@10 ties dense retrieval. The result
supports moving to the planned Day 18 metric artifact and Day 19 error analysis,
not skipping them: top-10 recall remains below 1.0 and the 40-question set is
still intentionally small.

## Verification commands

```bash
python scripts/test_retrieval_core.py
python scripts/test_bm25_retrieval.py
python scripts/evaluate_retrievers.py
python scripts/validate_day15_day17.py
```
