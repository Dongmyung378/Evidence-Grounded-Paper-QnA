# Day 32 Manual Q&A Review

## Roadmap gate

- Task: manually review 20 factual, numeric, comparison, and limitation questions.
- Completion criterion: record at least five failure cases.
- Result: **PASS** - 20 questions reviewed and 15 strict failures recorded.

## Review design

The sample contains five questions per category, ten English and ten Korean
questions, and two questions from every verified benchmark paper (001-010).
Judgments were made by the assistant, not an independent human reviewer.
Bilingual counterparts are correlated (13 distinct question stems in 20 records).
The sample is diagnostic and does not estimate accuracy across all 30 papers.
The live Day 31 pipeline generated answers without loading Gold data. Gold answers
and evidence were joined only after generation for manual semantic and citation review.

## Results

- Verdicts: pass 1, partial 4, fail 15
- Strict pass rate: 5.0%
- Pass-or-partial rate: 25.0%
- Citation support: pass 9, fail 4, not applicable 7
- False abstentions: 7
- Language matched: 20/20

| Category | Pass | Partial | Fail |
|---|---:|---:|---:|
| factual | 1 | 2 | 2 |
| numeric | 0 | 0 | 5 |
| comparison | 0 | 1 | 4 |
| limitation | 0 | 1 | 4 |

## Findings

1. Retrieval quality does not guarantee answer quality: several responses had
   relevant cited evidence but did not answer the requested relation or protocol.
2. The calibrated abstention policy produced seven false refusals in this sample,
   including three pre-generation refusals, two model refusals, and two validation fallbacks.
3. The 0.5B local generator frequently omitted numbers and comparison details and
   produced non-answers or malformed Korean content.
4. Citation IDs remained structurally valid, but four generated answers did not
   establish a meaningful claim-to-citation relationship.

## Decision

Day 32 is complete against the original roadmap because the full 20-question
review and more than five strict failure records exist. This is an evaluation
completion, not an answer-quality release gate: the current 0.5B generator and
abstention calibration require improvement before public deployment quality claims.
The frozen Day 28 retrieval configuration is unchanged.

## Review provenance

Judgments are bound to the SHA-256 hashes of the reviewed outputs and labels.
After any model rerun, review the new answers and refresh the binding explicitly.
Never reuse old labels automatically for changed model outputs.

## Reproduce

```bash
# Optional new run; review its outputs separately before assigning labels.
python -B scripts/run_day32_review.py --offline --output data/evaluation/day32_rerun.json
# Rebuild and validate the frozen original review.
python -B scripts/build_day32_review.py
python -B scripts/validate_day32.py
```
