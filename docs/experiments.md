# Bounded answer-model experiment

[English](experiments.md) | [한국어](experiments_KO.md)

The cached Qwen2.5-1.5B-Instruct candidate was not adopted. In a predeclared eight-output development trial, it added no complete visible answers and introduced two materially wrong claims. Model expansion stopped, and the default extraction/translation runtime stayed unchanged.

## Comparison design

Four disclosed DEV question groups cover augmentation lists, projection-head comparisons, scalar temperature parameters and a calibration exception. Each has an English and Korean question. Both systems receive identical production chunks from the gold source page. This oracle-page selection deliberately removes retrieval failures from the feasibility check; it does not estimate full-pipeline accuracy.

The reference is the answer builder developed for this project, including its refusal guards, sentence selector and Korean translator. The candidate is prompted directly with the same evidence, a requested language and a four-field output contract. This compares candidate builders; it is not an isolated decoder ablation. No reference answers or grading labels enter the candidate prompt.

The [protocol](../data/quality_benchmark/runs/answer-builder-pilot/protocol.json), [review rubric](../data/quality_benchmark/runs/answer-builder-pilot/review-rubric.json), [input evidence](../data/quality_benchmark/runs/answer-builder-pilot/inputs.json) and [prompt receipt](../data/quality_benchmark/runs/answer-builder-pilot/prompt-receipt.json) were frozen before inference. Continuation required at least two additional complete outputs, no additional materially unsupported outputs and mean candidate time at most 60 seconds. One attempt was allowed per question, with no JSON repair, retry or prompt tuning.

## Observed result

| Measure | Project answer builder | Qwen candidate |
| --- | ---: | ---: |
| Complete outputs under the declared rubric and contract | 2/8 | 0/8 |
| Valid output contract | 8/8 | 0/8 |
| Complete visible answer content, separately ignoring wrapper/schema faults | 2/8 | 2/8 |
| Materially wrong or unsupported claims | 0/8 | 2/8 |
| Mean per-question wall time, excluding initial model load | 2.978 s | 30.861 s |
| Initial model load | 14.608 s | 12.040 s |

The candidate wrapped every output in Markdown code fences. One omitted `abstention_reason`, and another used invalid JSON escapes around a formula. These failures count against the declared contract. A separate diagnostic inspection of the visible text still found no completeness gain, so the decision does not rest on formatting alone.

| Question | Project answer builder | Candidate's visible content |
| --- | --- | --- |
| Augmentation, EN | Refuses despite answerable page content. | Correct list with crop, resize and random flip; invalid output contract. |
| Augmentation, KO | Discusses cropped views, omitting the requested families. | Announces a list without supplying its items. |
| Projection, EN | Refuses because the page lacks the explicit SimCLR identifier. | Omits +3%, >10% and the before-head representation qualifier. |
| Projection, KO | Retains quantities but loses the before-head qualifier in its later explanation. | Answers in English and omits the quantities. |
| Temperature, EN | Retains one scalar for all classes and T > 0. | Retains the same facts; invalid output contract. |
| Temperature, KO | Retains one scalar for all classes and T > 0. | Claims four scalar parameters instead of one. |
| Calibration exception, EN | Reuters and low ECE appear; the improvement explanation is missing. | Gives only Reuters; omits low initial error and limited improvement. |
| Calibration exception, KO | Reuters and low ECE appear; the improvement explanation is missing. | Names Stanford Cars instead of Reuters and answers in English. |

Sources: [baseline outputs](../data/quality_benchmark/runs/answer-builder-pilot/baseline-outputs.json), [candidate raw outputs](../data/quality_benchmark/runs/answer-builder-pilot/candidate-outputs.json), [source-based curator review](../data/quality_benchmark/runs/answer-builder-pilot/review.json), [decision summary](../data/quality_benchmark/runs/answer-builder-pilot/summary.json). The labels are AI-assisted curator judgments, not independent human evaluation.

## Reproduction and limits

The candidate uses [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32, eight threads, seed 378 and greedy decoding. Limits are 4,096 input tokens, 512 new tokens and 60 seconds per generation. Recorded package versions and hashes are included with the outputs.

With `requirements.txt` installed and pinned models cached, run from the repository root:

```bash
python -B scripts/experiment_answer_builder.py baseline --output-dir outputs/answer-builder-replay
python -B scripts/experiment_answer_builder.py candidate --output-dir outputs/answer-builder-replay
python -B scripts/validate_answer_experiment.py
```

The portable replay entry point writes new outputs under `outputs/`. The [measured runner snapshot](../data/quality_benchmark/runs/answer-builder-pilot/measured_runner.py) preserves the original source bytes for provenance. Its recorded `input_tokens=2` field mistakenly counted `BatchEncoding` keys. [Corrected token counts](../data/quality_benchmark/runs/answer-builder-pilot/token-counts.json) are computed from the frozen prompts without rerunning inference. The generation adapter itself checked actual token length, so the error is in telemetry. Raw measured outputs are preserved.

The baseline's first Korean request includes deferred translator loading; both means exclude initial process model load. These single-run timings do not establish throughput or a service SLA. Eight outputs on two disclosed DEV papers do not establish unseen accuracy, model training disjointness or the performance of larger models. No larger model was downloaded or adopted as part of this trial. Further model selection would require a new bounded study and new evaluation papers.

## Thirty paper development strategies

The four engineering work areas were implemented: request tracking, adjacent source restoration with one bounded retry, source/translation alignment and an evaluation covering all thirty papers. Only DEV questions influenced strategy selection. The protocol required at least two additional paired groups with all fact aliases, no additional negative answers, at most one previously answered group lost and a runtime ratio no greater than 2.5. Alias coverage is not semantic accuracy.

| DEV strategy | Complete alias groups / 40 | Previously answered groups lost | Negative answers / 20 | Adoption |
| --- | ---: | ---: | ---: | --- |
| Pre-study implementation | 8 | 0 | 10 | Reference |
| `planner-context-v1` | 10 | 4 | 10 | Rejected |
| `planner-context-v2` | 10 | 3 | 11 | Rejected |
| `context-alignment-v3` | 9 | 3 | 11 | Rejected |

The first variant gained two complete groups but lost four previously answered groups. Later strategies still lost three and additionally answered an unsupported request. None was admissible. The final policy disables experimental context expansion, ordering, answer expansion and retry. Alignment remains active. These are combined strategy trials rather than causal ablations of each component. The recorded DEV means include a deferred first translation-model activation and do not establish a warm service latency claim.

The [original selection baseline](../data/benchmarks/corpus30/results/selection-baseline-dev-report.json), [trial outputs and reports](../data/benchmarks/corpus30/results/) and [fixed criteria](../data/benchmarks/corpus30/protocol.json) are public. The final all-paper comparison and execution deviations are in [results](results.md).

## Premise verifier pilot

A separate frozen DEV pilot tested a pinned English SNLI/MultiNLI classifier on ten true and ten false premises in both languages. At an entailment threshold of 0.8, it refused all ten false-premise pairs but accepted only two of ten true-premise pairs. Adoption required at least nine true pairs accepted and eight false pairs refused. It was rejected because its refusal improvement came with a severe true-premise regression. Korean reverse translation and mismatch between short NLI training inputs and paper excerpts limit interpretation.

[Protocol, labels, outputs and decision](../data/benchmarks/corpus30/experiments/premise-verifier/README.md) and the pinned-model replay script are public. The classifier is not part of default inference. Human factual review and a new reserved evaluation set would be required before trying a new strategy after this disclosed comparison.
