# Days 21–22 Retrieval Gate and Cross-Encoder Reranker Report

## Fixed conditions

- Corpus: 10 English paper PDFs, 188 pages, 906 chunks
- Evaluation set: 40 verified questions (English 20, Korean 20)
- Relevance target: one verified gold evidence page per question
- Hybrid candidates: equal-weight RRF over BM25 Top-20 and dense Top-20 chunks
- Reranker input: Hybrid Top-20 chunks
- Evaluation unit: unique PDF page, using the highest-scoring chunk per page

The question set, gold pages, chunks, embedding model, fusion constant, and
metric implementation are unchanged from the Day 18 baseline.

## Day 21 — Baseline table and explainable demo

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.325 | 0.825 | 0.900 | 0.5452 |
| Dense | 0.300 | 0.775 | 0.975 | 0.4886 |
| Hybrid RRF | **0.350** | **0.925** | **0.975** | **0.5692** |

`retrieval_demo.md` contains three reproducible cases:

1. `q-001-ko`: multilingual success where BM25 and dense signals complement each other.
2. `q-042-en`: a number-heavy question where BM25 rescues a weak dense rank.
3. `q-049-ko`: a known Top-5 failure with its chunking diagnosis and next action.

This satisfies the gate requirement that low retrieval performance can be
explained rather than only reported as a score.

## Day 22 — Pretrained Cross-Encoder

- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Base architecture: multilingual MiniLMv2, approximately 0.1B parameters
- Training source: multilingual MS MARCO
- Maximum pair length: 512 tokens
- Batch size: 8 query/passage pairs
- Local evaluation runtime: 64.254 seconds for 40 questions and 800 candidate pairs

The model receives each question together with each Hybrid candidate chunk and
returns a relevance logit. The raw logit is used only for sorting; no threshold
is applied. Every saved result retains the pre-rerank position, Hybrid score,
Cross-Encoder score, source ranks, page, section, chunk ID, and original text.

Model reference: [Hugging Face model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).

## Measured comparison

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Hybrid Top-20 | 0.350 | 0.775 | 0.925 | 0.975 | 0.5692 |
| Hybrid + Reranker | **0.375** | **0.800** | **0.975** | **0.975** | **0.6233** |
| Delta | +0.025 | +0.025 | +0.050 | 0.000 | +0.0542 |

| Language | Hybrid Recall@5 | Reranker Recall@5 | Hybrid MRR | Reranker MRR |
|---|---:|---:|---:|---:|
| English | 1.000 | 1.000 | 0.5992 | **0.6892** |
| Korean | 0.850 | **0.950** | 0.5392 | **0.5575** |

The reranker recovered both `q-043-ko` and `q-049-ko` from gold rank 8 to rank
1 without losing any existing Recall@5 hit. `q-005-ko` remains outside the
evaluated Top-10, so candidate generation and terminology mismatch remain a
known limitation. Korean Recall@3 declined from 0.800 to 0.700 even though
Korean Recall@5 and MRR improved; this trade-off must remain visible in later
ablation work.

## Completion decision

- Day 21: complete — baseline table and an explainable success/failure demo exist.
- Day 22: complete — a pretrained Cross-Encoder reranks Hybrid Top-20 candidates and the same 40 questions have measured metrics.
- Current production candidate: Hybrid + pretrained reranker, pending the Day 23 candidate/evidence policy and Day 28 final retrieval gate.

## Reproduction

```bash
python -B scripts/build_retrieval_demo.py
python -B scripts/evaluate_reranker.py
python -B scripts/search_reranker.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?" --top-k 5
python -B scripts/validate_day21_day22.py
```
