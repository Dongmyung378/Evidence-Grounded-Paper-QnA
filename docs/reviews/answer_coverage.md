# Answer coverage experiment

[English](answer_coverage.md) | [한국어](answer_coverage_KO.md)

The candidate recovered missing details but is not the default API. Some details regressed, Korean terminology errors remain, and sentence ranking adds work. It is retained as opt-in; further tuning has stopped.

| Paired comparison | Baseline | Candidate |
|---|---:|---:|
| Fixed-20 pass | 3 | 10 |
| Partial | 14 | 10 |
| Fail | 3 | 0 |
| Answerable questions answered | 38/40 | 40/40 |
| Unsupported questions refused | 19/20 | 20/20 |
| Mean host answer-stage time | 0.761 s | 2.618 s |

Seed was 378. Generation did not load Gold; the fixed 20 were reviewed afterward. Pass means key requested content is present, not flawless prose. These are assistant-led judgments on reused development data, not independent human evaluation or unseen-test accuracy. The fixed 20 represent 13 distinct meanings. Answering is not evidence of correctness.

Candidate time includes separately recorded sentence ranking. Retrieval, initialization and HTTP are excluded. Retrieval ran on CPU and translation on CUDA; execution order and translation caching can affect timings. These are not Docker CPU latency measurements or a speed improvement claim.

## Changes and limitations

The candidate reconstructs overlapping fragments using same-page neighbors from the selected paper, ranks sentences with local context, and cites at most five original chunks. A narrowly defined paper-problem route uses the abstract; other accepted answers also require dense similarity. Source character spans are verified. Translations with unsupported numbers or unexpected scripts are dropped, but this does not establish semantic fidelity.

Recovered details include ContCore's 11 schedules and datasets, the two Voronoi constructions, and ABIDE I with seven MS-WDRO baselines. However, `q-041-en` lost the three-gripper and success-rate statements. Korean translations of simplex and singular point remain unreliable.

The 0.78 dense threshold was calibrated on reused answerable questions. Former holdout errors have now been inspected, so 20/20 refusal is not an independent test result. The historical baseline holdout result was 10/10; this fresh paired run reproduced 9/10. Historical artifacts are not a current guarantee.

## Execution and evidence

Use `GroundedQAPipeline(..., enable_answer_coverage=True)` for opt-in. The default is `False`; UI/API behavior is unchanged. `config/answer_coverage.json` describes the candidate, not default API activation. Real-model evaluation used the standalone runner; runtime wiring was regression-tested. Docker was not remeasured.

`python -B scripts/review_answer_coverage.py` validates saved results without models. A deliberate rerun uses `python -B scripts/evaluate_answer_coverage.py --all-questions`.

`data/evaluation/answer_coverage_outputs.json` stores paired responses; `answer_coverage_review_labels.jsonl` stores judgments; `answer_coverage_review.json` stores aggregates and hashes. Source traceability was checked for 40 candidate answers; semantic review covered only the fixed 20.
