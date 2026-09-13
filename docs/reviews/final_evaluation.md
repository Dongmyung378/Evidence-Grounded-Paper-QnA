# Final Evaluation Freeze

The gates below validate saved baseline artifacts, not a new end-to-end run. The subsequent [opt-in experiment](answer_coverage.md) reproduced 9/10 baseline refusals on the former holdout, not the historical 10/10. The candidate is not the default API and Docker was not remeasured. Experiment results are preserved separately in the manifest.

[English](final_evaluation.md) | [한국어](final_evaluation_KO.md)

## What is frozen

The final portfolio evaluation binds the evaluation inputs, runtime settings, model revisions, measured artifacts, critical implementation files, and Docker packaging with SHA-256 hashes. Seed 378 is fixed for current stochastic evaluation paths.

The freeze does not turn the 20 expansion papers into an accuracy benchmark. Accuracy claims remain limited to 40 verified questions from the original 10 papers. The expansion papers support parsing and runtime robustness claims only.

## Model lock

| Runtime role | Model | Revision |
|---|---|---|
| Multilingual embedding | `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Candidate and sentence reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` |
| English to Korean translation | `facebook/nllb-200-distilled-600M` | `38bd1c6a0f58097b26ef7c7cd73a0dcac034350a` |
| Historical generator baseline | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |

The retrieval revisions were read from both the host cache and the cache used by the successful Docker benchmark. Runtime loaders now pass these exact revisions to Sentence Transformers. The embedding cache identity also includes the revision so a changed checkpoint cannot silently reuse old vectors.

NLLB remains limited to the local noncommercial portfolio scope under its recorded CC-BY-NC-4.0 license. Qwen is retained only for historical regression comparison and is not the current browser answer path.

## Final performance table

| Area | Metric | Result |
|---|---|---:|
| Corpus robustness | Papers, pages, chunks | 30, 563, 2,463 |
| Expansion smoke | Passed | 20/20 |
| Current retrieval | Recall@1 | 0.300 |
| Current retrieval | Recall@3 | 0.850 |
| Current retrieval | Recall@5 | 1.000 |
| Current retrieval | Recall@10 | 1.000 |
| Current retrieval | MRR | 0.5838 |
| Current retrieval | English MRR | 0.6308 |
| Current retrieval | Korean MRR | 0.5367 |
| Abstention | Answerable retention | 38/40, 0.950 |
| Abstention | Unsupported holdout refusal | 10/10, 1.000 |
| Fixed answer review | Pass, partial, fail | 3, 14, 3 |
| Fixed answer review | Pass or partial | 0.850 |
| Fixed answer review | Citation failures | 0 |
| Docker Q&A review | Pass, partial, fail | 4, 6, 0 |
| Docker Q&A review | English API mean | 1.470 s |
| Docker Q&A review | Korean API mean | 11.950 s |
| Docker analysis | Empty-cache first paper | 106.471 s |
| Docker analysis | Later-paper mean | 6.027 s |

The machine-readable table contains 30 rows with scope, unit, source artifact, and qualification for each value. Latency values are hardware-specific observations and are not service-level guarantees.

## MRR correction

The historical Day 28 artifact reports MRR 0.6317. A new run of the current code with CPU retrieval and exact model revisions produced MRR 0.5838 while preserving Recall@5 and Recall@10 at 1.000 with zero Top-5 losses.

The old page order does not match later saved answer evidence or the current pipeline for several questions. Its embedding cache recorded only the model name, not a revision, so the old execution state cannot be reconstructed exactly. The repository therefore keeps 0.6317 as a labeled historical result and uses 0.5838 as the current headline metric. This avoids presenting a stale value as current performance.

## Acceptance gates

The frozen state passes all five final groups:

- Retrieval: Recall@5 and Recall@10 are 1.000, MRR is at least 0.58, and no verified Gold page is lost from Top-5.
- Corpus: all 20 expansion papers pass the runtime smoke test.
- Abstention: answerable retention is 0.95, unsupported holdout refusal is 1.00, and no holdout fallback is accepted.
- Answer quality: pass-or-partial rate is 0.85, citation failures are zero, and false abstentions do not exceed two.
- Container: 10/10 questions match the requested language and citation contract, with offline restart and cleanup verified.

## Integrity and change control

`final_evaluation.json` records the SHA-256 digest of every listed source and a combined source-bundle digest. `final_performance.csv` is also linked by its own digest. The validator rebuilds both representations in memory and requires exact equality with the saved files.

Any change to a frozen input, configuration, critical runtime file, evaluation artifact, dependency file, Dockerfile, or Compose file requires rebuilding the freeze, rerunning the validators, and reviewing changed claims.

```bash
python -B scripts/evaluate_frozen_retrieval.py
python -B scripts/build_evaluation_freeze.py
python -B scripts/validate_evaluation_freeze.py
python -B scripts/verify_project.py
```

## Known limits

- Only 10 of 30 papers have manually verified questions and Gold evidence.
- Answer labels are assistant-led rather than independently human-reviewed.
- The 20-question answer review represents 13 independent meanings.
- The container review represents five bilingual meanings from three papers.
- Korean answers are slower and can be less fluent or less complete than English answers.
- Exact Gold-page overlap in the container review is 4/10 even though all returned passages support their answers.
- Public hosting, accounts, OCR, visual figure interpretation, formula interpretation, and multi-paper comparison remain outside the portfolio MVP.
