# Day 28 Production Retrieval Gate

## Roadmap requirement

Day 28 freezes the production retrieval combination. After this gate, the
user-facing search path must use one versioned configuration, and retrieval
changes require a separate experiment and regression evidence.

## Frozen production combination

| Stage | Frozen choice |
|---|---|
| Scope | Search only the currently selected paper |
| Sparse retrieval | BM25 Top-20 |
| Dense retrieval | `intfloat/multilingual-e5-small` Top-20 |
| Fusion | Equal-weight reciprocal rank fusion (`1.0:1.0`, `rrf_k=60`) |
| Candidate policy | 20 unique chunks; add the best page-1 candidate only when page 1 is absent |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, all 20 candidates |
| Evidence selector | Top-5; at most two chunks per page; near-duplicate threshold `0.85` |

The machine-readable source of truth is
`config/production_retrieval.json`. `scripts/production_retrieval.py` is the
single production entry point used by the CLI and expansion smoke evaluation.
The frozen configuration fingerprint is
`186d2f3c8b0b858eae1a309c2444ddd1eca09ebfa3613bb9214b86f40770703e`.

## Gate results

The accuracy benchmark remains the original manually verified set: 40
questions over papers 001–010, split evenly between English and Korean. The 20
new papers are intentionally excluded from accuracy claims until they receive
their own manually verified gold evidence.

| Scope | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| All | 40 | 0.375 | 0.825 | **1.000** | **1.000** | **0.6317** |
| English | 20 | 0.450 | 0.900 | **1.000** | **1.000** | 0.6892 |
| Korean | 20 | 0.300 | 0.750 | **1.000** | **1.000** | 0.5742 |

- Verified questions satisfying the 20-candidate/5-evidence contract: 40/40
- Existing Top-5 hits lost by the Day 27 correction: 0
- New papers passing bilingual production-path smoke checks: 20/20
- Processed corpus: 30 papers, 563 pages, 2,463 chunks
- Runtime scope: one paper at a time

The full per-question output is saved in
`data/evaluation/day28_production_metrics.json`; the expansion cases are in
`data/evaluation/corpus_expansion_smoke.json`.

## Selection rationale

- Equal-weight RRF is retained because it is the measured stable baseline.
  The earlier `BM25 0.8 / dense 1.2` trial reduced performance and is not used.
- The pretrained multilingual Cross-Encoder remains the production reranker.
  The Day 25 training run proved the fine-tuning pipeline only; its one-step
  smoke checkpoint is not a production model.
- Broadly widening the candidate union is rejected because the measured trial
  introduced a Top-5 regression.
- The deterministic page-1 guard is retained because it recovered the only
  remaining Top-5 miss without losing an existing Top-5 hit.

## Change-control rule

The retrieval configuration is frozen after Day 28. A future change is allowed
only when it is developed as an explicit experiment, saves before/after
metrics, reruns the same verified-40 benchmark, and introduces no Top-5 loss.
The API and web UI must call `ProductionRetrieval`; they must not duplicate
model names, weights, or Top-K values.

## Reproduction

```bash
HF_HUB_OFFLINE=1 python -B scripts/evaluate_day28.py
HF_HUB_OFFLINE=1 python -B scripts/evaluate_corpus_expansion.py
python -B scripts/test_production_config.py
python -B scripts/validate_day28.py
```

On PowerShell, set offline mode with
`$env:HF_HUB_OFFLINE='1'` before running the model-backed commands.

## Decision

**Passed.** The original Day 28 completion criterion is met: one production
retrieval combination is fixed, its user-facing entry point is centralized,
and regression gates protect it from unmeasured changes.
