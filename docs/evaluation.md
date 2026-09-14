# Evaluation

[English](evaluation.md) | [한국어](evaluation_KO.md)

## Evaluation boundary

The corpus contains 30 text-extractable English papers. Accuracy labels cover 40 verified questions on the first 10 papers: 20 English and 20 Korean. The additional 20 papers test parsing and retrieval runtime robustness only.

| Area | Scope | Result |
|---|---|---:|
| Corpus | 30 papers | 563 pages, 2,463 chunks |
| Expansion smoke | papers 011-030 | 20/20 passed |
| Retrieval | verified 40 | Recall@1 0.300 |
| Retrieval | verified 40 | Recall@3 0.850 |
| Retrieval | verified 40 | Recall@5 and @10 1.000 |
| Retrieval | verified 40 | MRR 0.5838 |
| Retrieval by language | 20 EN / 20 KO | MRR 0.6308 / 0.5367 |
| Refusal policy | 40 answerable / 10 holdout | 95% retained / 90% refused |
| Answer review | fixed 20 | 3 pass, 14 partial, 3 fail |
| Citation review | fixed 20 | 0 support failures |
| Docker Q&A | 10 questions | 4 pass, 6 partial, 0 fail |
| Docker API latency | CPU laptop | EN 1.47 s, KO 11.95 s mean |

Retrieval succeeds at placing every verified Gold page within the first five unique candidate pages. MRR shows that the exact page is not always ranked first, especially for Korean questions. The answer review has high citation support but only a 15% strict pass rate; most usable responses are partial because selected evidence can omit a requested detail. This distinction is intentionally visible instead of being summarized as answer accuracy.

The refusal threshold retains 38 of 40 answerable questions, rejects all 10 unsupported calibration questions, and rejects 9 of 10 unsupported holdout questions. One holdout miss is reported rather than hidden behind the earlier calibration result.

## Artifacts

- [`frozen_retrieval_metrics.json`](../data/evaluation/frozen_retrieval_metrics.json) - per-question ranks and aggregate retrieval metrics
- [`abstention_evaluation.json`](../data/evaluation/abstention_evaluation.json) - answerable retention and unsupported-question refusal
- [`grounded_generation_review.json`](../data/evaluation/grounded_generation_review.json) - fixed answer review with rubric and labels
- [`container_qna_review.json`](../data/evaluation/container_qna_review.json) - reviewed live Docker responses
- [`final_evaluation.json`](../data/evaluation/final_evaluation.json) - frozen metric and provenance manifest
- [`final_performance.csv`](../data/evaluation/final_performance.csv) - compact machine-readable portfolio table

## Limitations

- Only 10 papers have manually verified retrieval labels.
- Answer labels were assistant-led and were not independently human-reviewed.
- Bilingual pairs reduce the number of independent meanings in the answer samples.
- CPU Korean translation is materially slower than English extraction.
- Docker latency is a measurement from one laptop, not a service-level guarantee.
- Gold evidence is loaded only for scoring and post-generation review, never for user-answer generation.
