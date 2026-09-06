# Days 25-27 Fine-tuning Readiness, Ablation, and Error Fix Report

## Original roadmap contract

- Day 25: optionally start reranker training when time permits; the pretrained
  reranker must remain a valid fallback even if fine-tuning fails.
- Day 26: compare Dense, BM25, Hybrid, and Reranker; `ablation.csv` must exist.
- Day 27: fix only the largest one or two errors and save before/after cases.

## Day 25 - leakage-safe optional fine-tuning

Fine-tuning was started only as a non-production smoke test. Training data comes
from QASPER train/validation, while the project `verified-40` remains test-only.
QASPER test was not used.

| Split | Questions | Positive pairs | Hard negatives | Total pairs |
|---|---:|---:|---:|---:|
| Train | 1,000 | 1,000 | 3,000 | 4,000 |
| Validation | 200 | 200 | 600 | 800 |

Only paragraph text evidence is used. `FLOAT SELECTED` table/figure placeholders
are excluded, matching the MVP text-only scope. For each positive, three BM25
hard negatives are selected from another paragraph in the same paper.

The CPU smoke run loaded the pretrained multilingual Cross-Encoder, processed a
balanced batch of two positive and two negative pairs, completed one backward
pass and optimizer step, and produced finite loss values. No checkpoint was
saved because one step is not a valid model. The measured pretrained reranker
therefore remains the selected runtime fallback, exactly as the roadmap allows.

## Day 26 - four-system ablation

All systems use the same 40 verified questions and the same 906 chunks.

| System | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.325 | 0.725 | 0.825 | 0.900 | 0.5452 |
| Dense | 0.300 | 0.575 | 0.775 | 0.975 | 0.4886 |
| Hybrid RRF | 0.350 | 0.775 | 0.925 | 0.975 | 0.5692 |
| Hybrid + Reranker | **0.375** | **0.800** | **0.975** | 0.975 | **0.6233** |

`ablation.csv` also includes English and Korean slices, resulting in 12 rows:
four systems multiplied by three scopes (`all`, `en`, `ko`). The ablation shows
that sparse and dense retrieval contribute complementary candidates, Hybrid
improves coverage, and the Cross-Encoder improves the final ranking.

## Day 27 - largest-error correction

Before correction, `q-005-ko` was the only Recall@5 failure. Its verified page 1
appeared in dense page Top-10 but was removed when chunk-level RRF was truncated
to 20 before reranking. A broad first attempt reranked the entire BM25/Dense
union. It recovered `q-005-ko` but lost `q-045-ko` at Top-5, so it was rejected
and preserved as `day27_pool_widening_rejected.json`.

The accepted fix is deliberately narrower: keep reranker input at 20 chunks,
but when page 1 is absent, replace only the lowest Hybrid candidate with the
best available page-1 chunk. English papers normally place the title, abstract,
or contribution summary in this front matter. No question ID, answer text, gold
page, or Korean-to-English term dictionary is used at runtime.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.375 | 0.375 | 0.000 |
| Recall@3 | 0.800 | 0.825 | +0.025 |
| Recall@5 | 0.975 | **1.000** | **+0.025** |
| Recall@10 | 0.975 | **1.000** | **+0.025** |
| MRR | 0.6233 | **0.6317** | **+0.0083** |

The saved before/after case shows `q-005-ko` moving from no gold page in Top-10
to gold rank 3. No existing Top-5 hit was lost. The guard is now connected to
the candidate/evidence pipeline, while Day 28 remains responsible for freezing
the production retrieval combination.

## Completion decision

- Day 25: complete. Training pairs and a real optimizer step exist; verified-40
  leakage is prevented and pretrained fallback is explicit.
- Day 26: complete. `ablation.csv` compares all four required systems under
  fixed conditions and includes language slices.
- Day 27: complete. One largest error was corrected, the failed broad attempt
  and accepted before/after case are both saved, and the full 40-question metric
  improved without a Top-5 regression.

## Reproduction

```bash
python -B scripts/build_reranker_training_pairs.py
python -B scripts/smoke_train_reranker.py
python -B scripts/build_ablation.py
python -B scripts/test_rerank_pool.py
python -B scripts/evaluate_rerank_pool_fix.py
python -B scripts/validate_day25_day27.py
```
