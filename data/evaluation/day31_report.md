# Day 31 Abstention Policy

## Original-roadmap requirement

Day 31 handles questions outside the selected paper and questions with
insufficient text evidence. The completion gate is that the system refuses an
answer that is not grounded in the paper.

## Policy design

The policy is intentionally separate from the answer prompt and is configured
in `config/abstention.json`.

1. Refuse before generation when retrieval returns no candidates or no usable
   evidence.
2. Refuse before generation when the highest Cross-Encoder relevance score is
   below `-3.0`.
3. Let the grounded answer model assess borderline cases that pass the
   deterministic retrieval gate.
4. Normalize every model-declared insufficient response to the localized
   canonical refusal with no evidence IDs.
5. Keep the existing validation fallback as the final protection against
   malformed model output.

The threshold was calibrated on the 40 verified answerable questions and ten
synthetic unsupported questions. The unsupported calibration set was not used
as the final holdout evaluation set.

## Calibration result

| Check | Result |
|---|---:|
| Verified answerable questions | 40 |
| Unsupported calibration questions | 10 |
| Answerable top-score median | 3.126 |
| Unsupported top-score median | -6.964 |
| Unsupported calibration refusals | 10/10 |
| Answerable questions retained before generation | 37/40 |
| Answerable retention | 92.5% |

The conservative threshold pre-refuses `q-001-en`, `q-001-ko`, and `q-031-en`.
This is an explicit safety-versus-coverage trade-off, not hidden as a perfect
classifier result.

## Held-out evaluation

The holdout contains ten different unsupported questions for `paper-011`
through `paper-020`, balanced between English and Korean. The real production
retriever and local answer model were used.

| Check | Result |
|---|---:|
| Holdout questions | 10 |
| Holdout papers | 10 |
| English / Korean | 5 / 5 |
| Correct refusals | 10/10 |
| Refusal recall | 100% |
| Refused before LLM generation | 9 |
| Refused by grounded model | 1 |
| Validation fallbacks | 0 |
| LLM invocations | 1 |
| Runtime | 25.976 seconds |

Every refusal uses the question language, returns `sufficiency=insufficient`,
contains an empty `evidence_ids` array, and includes a non-empty reason. The one
borderline question that reached `Qwen/Qwen2.5-0.5B-Instruct` was rejected on
the first valid generation attempt.

## Scope and limitations

This is an MVP safety gate, not a universal out-of-distribution detector. The
negative questions are synthetic, the holdout has ten cases, and the threshold
is tied to the current Cross-Encoder and corpus. Changes to the reranker or
corpus require recalibration. Day 32 must manually review answer correctness,
false refusals, numbers, comparisons, and limitation questions.

## Reproduction

```bash
python -B scripts/calibrate_abstention.py --offline
python -B scripts/evaluate_day31.py --offline
python -B scripts/test_abstention_policy.py
python -B scripts/validate_day31.py
```

## Decision

**Passed.** All ten held-out unsupported questions were refused without a
malformed-output fallback, while the pre-generation gate retained 92.5% of the
verified answerable benchmark. The known coverage trade-off is carried forward
to the Day 32 manual review.
