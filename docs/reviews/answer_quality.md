# Answer quality review

## Adopted answer path

The browser runtime now uses an evidence-first answer path:

1. The existing production retriever returns five evidence chunks.
2. The multilingual Cross-Encoder reranks complete sentences within those chunks.
3. References, extraction fragments, table-heavy text, and equations are excluded.
4. English questions receive an extractive answer from up to three selected sentences.
5. Korean questions receive translations of up to two selected English sentences from the revision-pinned NLLB model.
6. The application assembles evidence IDs itself and rejects unsupported numeric claims, formulas, untranslated CJK text, and validation-instruction leakage.

This design removes free-form citation generation and avoids asking a small causal language model to solve content selection, translation, scientific synthesis, and JSON formatting in one step.

## Fixed 20-question comparison

The candidate was run on the exact saved retrieval evidence used by the earlier balanced review. Gold answers were not loaded during answer construction. Gold was opened only after the output artifact had been saved.

| Metric | Previous 0.5B path | Evidence-first path | Change |
|---|---:|---:|---:|
| Pass | 1 | 3 | +2 |
| Partial | 4 | 14 | +10 |
| Fail | 15 | 3 | -12 |
| Strict pass rate | 0.050 | 0.150 | +0.100 |
| Pass-or-partial rate | 0.250 | 0.850 | +0.600 |
| False abstentions | 7 | 2 | -5 |
| Citation support | 9 pass | 18 pass | +9 |
| Fixed-evidence answer time | 118.863 s | 12.480 s | -106.383 s |

The timing comparison excludes model loading in both runs and uses saved retrieval evidence. It measures the answer stage, not first-start latency or the full upload-to-answer API flow.

## Adoption decision

The evidence-first path is adopted because it materially improves reviewed answer usefulness, keeps every returned claim tied to selected source text, removes validation fallbacks in the 20-question run, and reduces answer-stage latency. The old Qwen 0.5B JSON path and its artifacts remain as a historical baseline.

This is not a production-quality claim. The review was assistant-led rather than independently human-reviewed, and the English/Korean pairs represent 13 distinct question meanings. Two page-1 questions are still rejected by the relevance gate, and several partial answers are caused by details missing from the fixed Top-5 evidence. The NLLB checkpoint uses CC-BY-NC-4.0 and is limited here to a local non-commercial portfolio demo.

## Reproduce

```bash
python -B scripts/evaluate_grounded_generation.py
python -B scripts/build_grounded_generation_review.py
python -B scripts/validate_grounded_generation.py
```

The saved outputs, review labels, and hash-bound review are in `data/evaluation/grounded_generation_*.json*`.
