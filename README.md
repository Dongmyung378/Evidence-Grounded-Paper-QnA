# Evidence-Grounded Paper Q&A

An evidence-grounded multilingual Q&A service for one English academic paper
at a time.

## Problem

Reading an academic paper and finding the exact evidence needed to answer a
question can take significant time.

Many paper summarization tools provide a general summary but do not clearly
show which parts of the paper support each answer.

## Solution

This project accepts one English academic paper PDF and provides:

- Text-based paper overview
- Korean and English questions
- Korean and English answers
- Multilingual evidence retrieval
- BM25 and dense retrieval
- Page and section references

Planned next steps:

- Grounded answer generation
- Abstention when the paper does not contain enough evidence
- FastAPI and a simple web UI

## Core Pipeline

```text
PDF Upload
    ↓
Text Parsing
    ↓
Page and Section Metadata
    ↓
Chunking
    ↓
BM25 + Multilingual Dense Retrieval
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Evidence Selection
    ↓
Grounded Answer
```

## Example

### User Question

```text
이 논문의 주요 연구 목적은 무엇인가?
```

### Answer

```text
이 논문의 주요 연구 목적은 제안된 방법의 성능을 평가하는 것입니다.
```

### Evidence

```text
Page: 3
Section: Introduction
Chunk ID: paper-001-page-003-chunk-002

Original text:
The main objective of this study is to evaluate the proposed method.
```

## Supported Scope

- Input: one English academic paper PDF
- Processing: one paper per session
- Questions: Korean or English
- Answers: Korean or English
- Evidence: original English text with page and section metadata
- File size limit: 20MB
- PDF type: text-extractable PDFs

## Corpus and benchmark

- Local robustness corpus: **30 English papers** (`paper-001` through `paper-030`)
- Processed corpus: **563 pages and 2,463 traceable chunks**
- Frozen accuracy benchmark: **40 verified bilingual questions** on the original
  10 papers (`paper-001` through `paper-010`)
- Expansion set: 20 additional papers used for ingestion and runtime smoke tests

The expansion papers are deliberately not included in retrieval accuracy claims
until manually verified questions and gold evidence are created for them. This
keeps the original benchmark comparable while still testing broader PDF and
domain coverage.

## Limitations

The MVP does not fully support:

- Scanned PDFs
- OCR
- Image or graph understanding
- Complete mathematical formula understanding
- Multiple papers at the same time
- Cross-paper comparison
- Korean, Chinese, or Japanese paper input
- User login and account management
- Large-scale paper search

## Evaluation Plan

The system will compare:

- BM25 retrieval
- Dense retrieval
- Hybrid retrieval
- Hybrid retrieval with reranking

Planned metrics:

- Recall@1
- Recall@5
- Recall@10
- MRR

Validate the 30-paper expansion with:

```text
python -B scripts/validate_papers.py
python -B scripts/validate_pipeline.py
python -B scripts/evaluate_corpus_expansion.py
python -B scripts/validate_corpus_expansion.py
```
- MRR or nDCG
- Evidence correctness
- Abstention accuracy

## Implemented retrieval baseline (Days 15–17)

The official local corpus contains 10 PDFs, 188 pages, and 906 searchable
chunks. The fixed evaluation set has 40 verified questions (20 English, 20
Korean) and evaluates page-level evidence retrieval.

- **Dense:** `intfloat/multilingual-e5-small`, `passage:` / `query:` prefixes,
  normalized cosine similarity, and a fingerprinted local embedding cache.
- **BM25:** shared Korean/English tokenization that retains numbers and full
  hyphenated model names (for example, `bert-base-uncased`) as well as their
  components.
- **Hybrid:** RRF fuses the top 20 BM25 and dense chunk candidates with
  `rrf_k=60`, preserving the source rank from each retriever.

All three methods use the same questions, gold pages, page-deduplication rule,
and metrics. The latest comparison is stored in
`data/evaluation/retrieval_comparison.json` and
`data/evaluation/retrieval_comparison.csv`.

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.325 | 0.725 | 0.825 | 0.900 | 0.545 |
| Dense | 0.300 | 0.575 | 0.775 | 0.975 | 0.489 |
| Hybrid RRF | **0.350** | **0.775** | **0.925** | **0.975** | **0.569** |

Hybrid RRF is the selected retrieval baseline: it improves Recall@5 by 0.100
over BM25 and 0.150 over dense retrieval, while retaining dense retrieval's
Recall@10 of 0.975. Korean Recall@5 improves from 0.700 (BM25) and 0.750
(dense) to 0.850.

The Day 7 data/evaluation gate is documented in
`data/evaluation/day7_gate_report.md`. The verified-40 set is fixed as the
project test set; QASPER remains a separate source dataset. Raw PDFs and
QASPER files are local-only until each paper's redistribution license is
verified.

### Retrieval commands

```bash
pip install -r requirements.txt

# Individual Top-K searches. --output saves score, page, section, chunk ID, and text.
python scripts/search_dense.py --paper-id paper-001 --query "What is the main problem?" --output data/evaluation/dense_search.json
python scripts/search_bm25.py --paper-id paper-001 --query "What is the main problem?" --output data/evaluation/bm25_search.json
python scripts/search_hybrid.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?" --output data/evaluation/hybrid_search.json

# Same 40 questions, same page-level metrics, all three retrievers.
python scripts/evaluate_retrievers.py

# Lightweight acceptance checks.
python scripts/test_retrieval_core.py
python scripts/test_bm25_retrieval.py
python scripts/validate_day15_day17.py
```

The first dense run downloads the embedding model if it is not already present.
Passage embeddings are then saved locally in `data/processed/dense_embeddings.npz`
and automatically rebuilt only if the model name or chunk corpus changes.

## Days 18–20: metrics and first retrieval experiment

The Day 18 baseline is frozen in `data/evaluation/baseline_metrics.json` with
Recall@1, Recall@5, Recall@10, and MRR. Equal-weight hybrid RRF is the selected
baseline at Recall@5 `0.925` and MRR `0.5692`.

Day 19 manually reviews 10 Recall@5 failures and records four planned primary
causes: chunking (2), terminology mismatch (3), number/identifier matching (3),
and multiple evidence (2). The case-level analysis is available in
`data/evaluation/failure_analysis.json` and
`data/evaluation/failure_analysis.md`.

Day 20 changes only the fusion weights while fixing the same 40 questions,
corpus, model, Top-20 candidate pool, and `rrf_k=60`:

| Configuration | BM25 weight | Dense weight | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected baseline | 1.0 | 1.0 | **0.350** | **0.925** | 0.975 | **0.5692** |
| Day 20 candidate | 0.8 | 1.2 | 0.300 | 0.900 | 0.975 | 0.5245 |

The candidate is not adopted because Recall@5 and MRR declined. This negative
result and the question-level rank changes are retained in
`data/evaluation/day20_improvement.json`; the full interpretation is in
`data/evaluation/day18_day20_report.md`.

```bash
python -B scripts/build_baseline_metrics.py
python -B scripts/analyze_retrieval_failures.py
python -B scripts/run_day20_experiment.py
python -B scripts/validate_day18_day20.py
```

## Days 21–22: retrieval gate and pretrained reranker

Day 21 freezes the baseline table and provides an explainable three-case demo
covering multilingual success, lexical rescue for a number-heavy question, and
a known Korean Top-5 failure. See `data/evaluation/retrieval_demo.md`.

Day 22 applies `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` to the equal-weight
Hybrid Top-20 chunks. The Cross-Encoder scores each question/chunk pair and
retains the Hybrid rank and score together with page, section, chunk ID, and
original evidence text.

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Hybrid Top-20 | 0.350 | 0.775 | 0.925 | 0.975 | 0.5692 |
| Hybrid + Reranker | **0.375** | **0.800** | **0.975** | **0.975** | **0.6233** |

The reranker improves Korean Recall@5 from `0.850` to `0.950` and English MRR
from `0.5992` to `0.6892`. It recovers `q-043-ko` and `q-049-ko` at Top-5
without losing an existing Top-5 hit. Full results and language slices are in
`data/evaluation/reranker_metrics.json` and
`data/evaluation/day21_day22_report.md`.

```bash
# Evaluate Hybrid Top-20 before and after Cross-Encoder reranking.
python -B scripts/evaluate_reranker.py

# Inspect scored evidence for one question.
python -B scripts/search_reranker.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?" --top-k 5

# Lightweight and completion-gate checks.
python -B scripts/test_reranker_core.py
python -B scripts/validate_day21_day22.py
```

## Days 23–24: candidate policy and UI evidence objects

Day 23 keeps the measured Day 22 retrieval configuration and formalizes its
output boundary: equal-weight BM25+dense Hybrid produces 20 unique candidates,
the multilingual Cross-Encoder reranks all 20, and a deterministic selector
returns five evidence chunks. It removes duplicate chunk IDs, normalized exact
text, and token 3-shingle near duplicates (`Jaccard >= 0.85`), while limiting
one PDF page to two selected chunks.

Day 24 converts every selected chunk into a UI-ready evidence object containing
the original text, `page`, `section`, `chunk_id`, `source_page_id`, display
locator, source indices, reranker score, Hybrid score, and BM25/dense ranks. The
contract is defined in `data/evaluation/evidence_object_schema.json`.

The reproducible three-case demo contains 20 candidates and five evidence
objects per case. It includes two verified-page hits and the known `q-005-ko`
candidate miss, so a reviewer can judge both answerable and insufficient search
results without an uncalibrated automatic threshold. See
`data/evaluation/candidate_evidence_demo.md` and
`data/evaluation/day23_day24_report.md`.

```bash
# Lightweight policy test and actual model-backed examples.
python -B scripts/test_evidence_selector.py
python -B scripts/build_candidate_evidence_demo.py

# Inspect or save one UI-ready evidence response.
python -B scripts/search_evidence.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?" --output data/evaluation/search_evidence_example.json

# Original-roadmap completion gate.
python -B scripts/validate_day23_day24.py
```

## Days 25–27: fine-tuning readiness, ablation, and error correction

Day 25 builds leakage-safe reranker data from QASPER train/validation only:
4,000 train pairs and 800 validation pairs with one text-evidence positive and
three same-paper BM25 hard negatives per question. A CPU smoke run completes a
real backward pass and optimizer step, but does not save or adopt a one-step
checkpoint. The measured pretrained Cross-Encoder remains the runtime fallback,
and `verified-40` remains test-only.

Day 26 freezes `data/evaluation/ablation.csv`, comparing BM25, dense, Hybrid
RRF, and Hybrid + Reranker on all/en/ko scopes. The reranker is strongest at
Recall@5 `0.975` and MRR `0.6233` before Day 27 correction.

Day 27 fixes the sole Top-5 miss without question-specific rules. If Hybrid
Top-20 lacks page 1, the best available page-1/front-matter chunk replaces only
the lowest candidate before reranking. On the same 40 questions, Recall@5 and
Recall@10 improve from `0.975` to `1.000`, MRR improves from `0.6233` to
`0.6317`, `q-005-ko` is recovered, and no Top-5 hit is lost. A broader source-
union attempt is retained as a rejected experiment because it introduced one
Top-5 regression. Full reasoning is in
`data/evaluation/day25_day27_report.md`.

```bash
python -B scripts/build_reranker_training_pairs.py
python -B scripts/smoke_train_reranker.py
python -B scripts/build_ablation.py
python -B scripts/test_rerank_pool.py
python -B scripts/evaluate_rerank_pool_fix.py
python -B scripts/validate_day25_day27.py
```

## Day 28: frozen production retrieval

Day 28 freezes one production search path so the API and UI cannot drift away
from the evaluated configuration. The source of truth is
`config/production_retrieval.json`, and user-facing callers use
`scripts/production_retrieval.py` instead of selecting models, weights, or
Top-K values independently.

The fixed path is BM25 Top-20 + multilingual E5 Top-20, equal-weight RRF,
20-candidate Cross-Encoder reranking, the Day 27 page-1 guard, and five
deduplicated evidence objects. On the frozen verified-40 benchmark it reaches
Recall@5 `1.000`, Recall@10 `1.000`, and MRR `0.6317`, with no lost Top-5 hit.
The full 30-paper corpus contains 563 pages and 2,463 chunks; all 20 expansion
papers pass the production-path bilingual smoke check. Expansion papers are not
used for accuracy claims until manually verified gold evidence is added.

```bash
# User-facing retrieval: all production parameters come from the frozen config.
python -B scripts/search_evidence.py --paper-id paper-001 --query "What is the main contribution?"

# Reproduce and validate the Day 28 gate.
python -B scripts/evaluate_day28.py
python -B scripts/evaluate_corpus_expansion.py
python -B scripts/test_production_config.py
python -B scripts/validate_day28.py
```

Any later retrieval change must be recorded as a separate experiment, rerun
the same verified-40 regression, and introduce no Top-5 loss. See
`data/evaluation/day28_gate_report.md` for the decision record.

## Day 29: grounded answer contract

Day 29 defines a provider-neutral JSON contract before connecting an LLM. Every
response contains exactly `answer`, `evidence_ids`, `sufficiency`, and
`abstention_reason`. Sufficient answers must cite one to five IDs from the
supplied Day 24 evidence objects; insufficient answers cite no supporting IDs
and return a localized Korean or English abstention message and reason.

The prompt builder treats paper text as untrusted quoted source data, requires
the question language, excludes external knowledge and non-text interpretation,
and returns the JSON Schema alongside the messages for a future structured-
output call. The strict parser rejects invalid JSON, unknown or duplicate
evidence IDs, extra fields, and inconsistent sufficiency/abstention values.
Cited IDs resolve back to page, section, chunk ID, and source text in the
existing evidence object.

```bash
python -B scripts/test_grounded_answer_contract.py
python -B scripts/build_day29_prompt_examples.py
python -B scripts/validate_day29.py
```

Four deterministic fixtures cover English/Korean and
sufficient/insufficient responses. No LLM call is made on Day 29; model
integration remains the Day 30 task. See `data/evaluation/day29_report.md`.

## Day 30: local end-to-end grounded Q&A

Day 30 connects the frozen production retriever to a real local answer model:

`query → BM25+dense hybrid → Cross-Encoder → evidence Top-5 → local LLM → JSON validation`

The default generator is
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
with an exact model revision recorded in `config/generation.json`. It runs with
Transformers on CUDA float16 when available and CPU float32 otherwise. No API
key is needed, and cached models can run fully offline.

The fixed smoke run covers all ten benchmark papers with five English and five
Korean questions. All 10/10 completed with model-generated JSON that passed the
Day 29 contract, zero safe fallbacks, and evidence IDs resolvable to page,
section, chunk ID, and text. The run took 60.184 seconds on an RTX 3060 Laptop
GPU. These are integration results; manual answer correctness is intentionally
deferred to Day 32.

```bash
# First run downloads the pinned local model if needed.
python -B scripts/run_day30_smoke.py

# Repeat without network access after models are cached.
python -B scripts/run_day30_smoke.py --offline

# Ask one question through the full pipeline.
python -B scripts/ask_paper.py --offline --paper-id paper-001 --query "What is the main problem addressed in this paper?"

python -B scripts/test_qna_pipeline.py
python -B scripts/validate_day30.py
```

Saved model outputs and early quality observations are documented in
`data/evaluation/day30_e2e_results.json` and
`data/evaluation/day30_report.md`.

## Day 31: calibrated abstention

Day 31 adds a calibrated refusal policy before and after generation. Empty
retrieval results and low-relevance evidence are rejected before the LLM is
called. Borderline cases may reach the grounded answer model, but every
insufficient result is normalized to a language-matched refusal with no
evidence IDs. The calibrated threshold and policy are recorded in
`config/abstention.json`.

Calibration uses the verified 40 answerable questions plus ten separate
unsupported questions. A held-out set contains ten different unsupported
questions across `paper-011` through `paper-020`, balanced between English and
Korean. The final local run refused 10/10 holdout questions: nine before LLM
generation and one by the grounded model, with zero validation fallbacks. The
pre-generation policy retained 37/40 verified answerable questions (92.5%), so
the three known false refusals are explicitly carried into Day 32 review.

```bash
python -B scripts/calibrate_abstention.py --offline
python -B scripts/evaluate_day31.py --offline
python -B scripts/test_abstention_policy.py
python -B scripts/validate_day31.py
```

Detailed results are in `data/evaluation/day31_abstention_results.json` and
`data/evaluation/day31_report.md`.

## Day 32: manual answer-quality review

Day 32 runs the current Day 31 user path on a balanced 20-question sample and
then performs a separate manual comparison against verified Gold records. The
sample contains five factual, five numeric, five comparison, and five limitation
questions; ten are English and ten are Korean; every verified benchmark paper
from `paper-001` through `paper-010` contributes two questions. The execution
script does not load Gold answers or Gold evidence.

The strict review produced 1 pass, 4 partial answers, and 15 failures. All 20
responses used the question language, but seven answerable questions were falsely
refused. Of the 13 responses that attempted an answer, nine had citations that
supported the generated claim and four did not establish a meaningful
claim-to-citation relationship. Numeric questions were the weakest category at
0 pass, 0 partial, and 5 failures. These results complete the original review
gate, but they also show that the 0.5B generator and abstention calibration are
not yet ready for public answer-quality claims.

```bash
# Regenerate the blind model outputs (uses the local cached model).
python -B scripts/run_day32_review.py --offline

# Rebuild the review artifacts from the saved outputs and manual labels.
python -B scripts/build_day32_review.py
python -B scripts/validate_day32.py
```

The full audit is in `data/evaluation/day32_manual_review.json`; all 15 strict
failures are documented in `data/evaluation/day32_failure_cases.md`, and the
decision summary is in `data/evaluation/day32_report.md`.

## Day 33: PDF upload and analysis API

The FastAPI service registers one text-based PDF through `POST /upload` and starts
an asynchronous parsing/chunking job through `POST /analyze`. Poll
`GET /jobs/{job_id}` for progress. SQLite persists jobs, duplicate requests reuse
the existing job, failed jobs can be retried, and restart recovery resumes queued
or interrupted work. One background worker processes papers serially.

```bash
python -m pip install -r requirements.txt
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Use `http://127.0.0.1:8000/docs` to upload a PDF and start analysis. Uploaded files
and results live in the Git-ignored `data/runtime/`, separately from the benchmark.
The PDF limit is 20 MiB; malformed, encrypted, and textless files are rejected.
The default seed for new stochastic work is **378**, recorded in
`config/reproducibility.json`; parsing is deterministic and UUIDs remain unseeded.

The real HTTP acceptance run produced 10 pages and 40 chunks from `paper-003.pdf`.
Ten API tests cover normal/error inputs, file limits, isolation, concurrency,
restart recovery, and retry behavior. This completes the original Day 33 gate.
The job stage is `parsing_and_chunking`; indexing and Q&A API integration follow
on subsequent days. Day 32 answer-quality failures remain open.

```bash
python -B scripts/evaluate_day33.py
python -B scripts/validate_day33.py
```

See [API usage and behavior](docs/day33_api.md) and
[Day 33 review](data/evaluation/day33_report.md).

## Batch ingestion

Run the full PDF-to-search-data pipeline with one command:

```bash
python scripts/run_ingestion_pipeline.py
python scripts/validate_pipeline.py
```

It produces `pages.jsonl`, `chunks.jsonl`, an ingestion report, and a pipeline
manifest. Each chunk can be traced back to its original paper page and section.

## Planned Technologies

- Python
- FastAPI
- Streamlit
- BM25
- Multilingual embedding model
- Cross-Encoder reranker
- FAISS
- SQLite
- Docker

## Planned next steps

- Day 34 FastAPI question, paper result, and health endpoints
- Day 35 full API flow: upload, parse, index, question, and answer
- FastAPI, Streamlit, and Docker service integration
