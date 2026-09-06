# Days 18–20 Metric, Failure Analysis, and First Improvement Report

## Fixed evaluation conditions

- Corpus: 10 English paper PDFs, 188 pages, 906 chunks
- Evaluation set: 40 verified questions (English 20, Korean 20)
- Relevance target: one manually verified gold evidence page per question
- Ranking unit: unique PDF page
- Hybrid candidate pool: BM25 Top-20 + dense Top-20 chunks
- Dense model: `intfloat/multilingual-e5-small`
- RRF constant: `k=60`

All before/after values use the same corpus, questions, gold pages, model,
candidate pool, and evaluation code. The only Day 20 variable is the BM25/dense
fusion weight.

## Day 18 — Baseline metrics

`baseline_metrics.json` records the required Recall@1, Recall@5, Recall@10,
and MRR values for all three retrievers.

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.325 | 0.825 | 0.900 | 0.5452 |
| Dense | 0.300 | 0.775 | 0.975 | 0.4886 |
| Hybrid equal-weight RRF | **0.350** | **0.925** | **0.975** | **0.5692** |

Equal-weight hybrid RRF remains the baseline because it has the best Recall@1,
Recall@5, and MRR while tying dense retrieval at Recall@10.

## Day 19 — Ten-case failure analysis

Ten questions that at least one retriever missed at Recall@5 were manually
reviewed against the question, gold evidence text, and all three page rankings.

| Primary cause | Cases | Main observation |
| --- | ---: | --- |
| Chunking | 2 | Abstract/conclusion content was split or mixed with extraction noise. |
| Terminology | 3 | Korean wording did not directly match English technical terms or acronyms. |
| Numbers | 3 | Numeric values, model names, and hardware identifiers dominated relevance. |
| Multiple evidence | 2 | Supporting evidence was distributed across chunks or pages. |

The case-level evidence, rankings, rationale, and recommended follow-up are in
`failure_analysis.json` and `failure_analysis.md`. These labels are diagnostic
hypotheses, not population-level error-rate estimates.

## Day 20 — One-variable weighted-RRF experiment

The candidate changed only the fusion weights:

- Baseline: BM25 `1.0`, dense `1.0`
- Candidate: BM25 `0.8`, dense `1.2`

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.350 | 0.300 | -0.050 |
| Recall@5 | 0.925 | 0.900 | -0.025 |
| Recall@10 | 0.975 | 0.975 | 0.000 |
| MRR | 0.5692 | 0.5245 | -0.0447 |

English Recall@5 remained 1.000, while Korean Recall@5 declined from 0.850 to
0.800. The candidate moved `q-043-ko` and `q-049-ko` from rank 8 to rank 7 but
did not recover them at Top-5; it also moved `q-017-ko` from rank 2 outside the
evaluated Top-10. Therefore the candidate is rejected and the equal-weight RRF
configuration remains selected.

This is a valid negative experiment: the implementation supports weighted RRF,
but the measured result prevents an unsupported configuration change. Because
the same 40-question set was used for diagnosis and the experiment, a future
larger held-out set is still required before production tuning.

## Roadmap completion gates

- Day 18: complete — `baseline_metrics.json` exists with Recall@1/5/10 and MRR.
- Day 19: complete — exactly 10 failures are recorded across all four planned causes.
- Day 20: complete — one fusion parameter was changed and compared on the identical evaluation set; the worse candidate was not adopted.

## Reproduction

```bash
python -B scripts/build_baseline_metrics.py
python -B scripts/analyze_retrieval_failures.py
python -B scripts/run_day20_experiment.py
python -B scripts/validate_day18_day20.py
```
