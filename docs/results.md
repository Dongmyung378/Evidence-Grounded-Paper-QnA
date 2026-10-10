# Results and observed limits

[English](results.md) | [한국어](results_KO.md)

The application and public verification workflow are reproducible. Reliable scientific answers across new papers are not established. The current frozen evaluation exposes incomplete answers, translation errors and failures to refuse unsupported requests.

## Current real-paper evaluation

Each split contains two papers, 16 bilingual meaning groups and 32 language outputs. Ten groups are answerable, four unsupported and two have false premises. A group passes only when both English and Korean pass. DEV informed development; the evaluation papers were executed after freezing the final runtime. Their results are now disclosed and must be treated as regression evidence during future tuning.

| Automatic check | DEV | Frozen evaluation papers |
| --- | ---: | ---: |
| Valid response contract | 32/32 outputs | 32/32 outputs |
| Answerable request receives an answer | 8/10 groups | 7/10 groups |
| Unsupported request refused | 4/4 groups | 1/4 groups |
| False-premise request refused | 2/2 groups | 0/2 groups |
| Answer returned for a negative request | 0/6 groups | 5/6 groups |
| All required pages cited | 7/10 answerable groups | 5/10 answerable groups |
| All required lexical markers present | 4/10 answerable groups | 1/10 answerable groups |
| Semantic accuracy | Not independently measured | Not independently measured |

Sources: [DEV report](../data/quality_benchmark/runs/2026-10-10-final/real-dev-completion-report.json), [evaluation report](../data/quality_benchmark/runs/2026-10-10-final/real-holdout-report.json), [DEV outputs](../data/quality_benchmark/runs/2026-10-10-final/real-dev-completion-predictions.json), [evaluation outputs](../data/quality_benchmark/runs/2026-10-10-final/real-holdout-predictions.json).

English retained 10/10 answerable outputs and Korean 7/10 on the evaluation papers. Unsupported refusal was 3/4 in English and 2/4 in Korean; false-premise refusal was 1/2 and 0/2 respectively. Status, lexical coverage and page coverage are distinct from factual correctness. Source-backed facts can still be irrelevant to the question, and an expected page can be cited while its requested facts are omitted.

The [freeze receipt](../data/quality_benchmark/runs/2026-10-10-final/real-paper-runtime-freeze.json) and [measured source snapshot](../data/quality_benchmark/runs/2026-10-10-final/real-paper-runtime-freeze-sources/) preserve 20 implementation files and 10 configurations. Hash checks run before and after evaluation. An earlier candidate run existed but was excluded from the primary comparison; its outputs were not used for subsequent DEV changes according to the operator's record. That non-use is not independently attested. A new accuracy estimate requires new evaluation papers and independent factual review.

## Failures observed in source review

The following observations come from agent inspection of saved answers and source text. They are not independently scored semantic accuracy.

| Area | Observed failure | Practical implication |
| --- | --- | --- |
| Numerical results | A cited chunk contains CNN correlation R = 0.86, damping 0.01 and non-convergence, but the answer omits all three. | Correct page retrieval does not establish answer completeness. |
| Lists | Korean augmentation omits horizontal flipping; baseline lists can lose an item at chunk boundaries. | Review every requested item and its qualifiers. |
| Comparisons | Retraining answers omit the approximation/cost comparison; operator answers omit the fixed-policy condition. | Retain both comparison targets and scope. |
| Technical translation | Hessian-vector products becomes “제품”; metric and differentiability terminology is mistranslated. | Protected terms and numerical checks do not establish semantic fidelity. |
| Unsupported requests | A clinical-trial question receives Atari results; banking and credit requests receive unrelated optimization facts. | Valid source references do not establish question relevance. |
| False premises | Universal contraction is explicitly denied, but the answer is still marked sufficient without the requested universal rate. | A negative answer return is not necessarily a fabricated affirmative claim; refusal/status also needs review. |

The default path remains extraction plus translation. The [bounded generative model trial](experiments.md) records its adoption decision separately.

## Runtime measurements

Windows CPU measurements use the same [49-chunk calibration paper](../data/quality_benchmark/runs/2026-10-10-final/runtime-calibration-chunks.jsonl), one excluded warm-up and ten repetitions of the same temperature question. The models use eight Torch threads. Initial activation and deferred Korean model loading are included in cold wall time.

| Condition | Cold wall | Warm p50 | Warm p95 | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| [English](../data/quality_benchmark/runs/2026-10-10-final/runtime-en-enabled.json) | 19.912 s | 2.134 s | 2.201 s | 1.31 GiB |
| [Korean, repeated sentence cache](../data/quality_benchmark/runs/2026-10-10-final/runtime-ko-enabled.json) | 25.861 s | 2.301 s | 2.379 s | 4.69 GiB |
| [Korean, sentence cache cleared per request](../data/quality_benchmark/runs/2026-10-10-final/runtime-ko-disabled.json) | 26.756 s | 4.337 s | 4.730 s | 4.69 GiB |

These are repeated-question measurements, not diverse-query latency, throughput, concurrency or an SLA. Clearing sentence outputs retains loaded models. A fresh process does not clear OS file caches. Nested stage timings must not be added twice. The environment is Python 3.12.7, CPU Torch 2.14.1, Transformers 5.16.1, Sentence Transformers 6.0.0 and NumPy 1.26.4.

## Environment and HTTP reproduction

A fresh Linux container with Python 3.12.15 and CPU Torch 2.5.1 reproduced all 32 saved evaluation answers, cited content and report metrics. Dense embeddings were regenerated. Reranker scores differed by at most 8.94e-6. Seven HTTP cases passed: five authored PDFs and two English/Korean questions on the bundled original PDF. The original paper produced 11 text pages and 63 chunks. The UI health endpoint passed.

See the [environment summary](../data/quality_benchmark/reproduction/separate_environment_summary.json), [answer comparison](../data/quality_benchmark/reproduction/holdout-comparison.json) and [HTTP comparison](../data/quality_benchmark/reproduction/api-comparison.json). The reproduction ran 367 tests and nine public validators. This describes the measured snapshot, not every future source change.

The Linux environment shared the physical host through WSL2 and pinned model files through a read-only cache. First model downloading and interactive browser behavior were not tested. Private corpus checks were skipped. The runs were agent-operated and do not establish independent human evaluation. Host-specific execution logs remain local; public summaries and comparisons retain the measured conclusions.

## Reproduce

For public integrity and regression checks:

```bash
python -m pip install -r requirements-test.txt
python -B scripts/verify_project.py
```

For actual models, install `requirements.txt`. `--offline` requires all pinned models to be cached; omit it for first download. Generate current DEV outputs and a report without overwriting published artifacts:

```bash
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json validate
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json run --split dev --offline --device cpu --embedding-cache outputs/real-dev-embeddings.npz --output outputs/real-dev-predictions.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json evaluate --predictions outputs/real-dev-predictions.json --output outputs/real-dev-report.json
```

The disclosed evaluation set can be replayed for regression using the freeze/run commands in [scripts](../scripts/README.md). Replay does not provide a new unseen estimate. The [quality benchmark](quality_benchmark.md) explains split and scoring rules.

For the upload and HTTP path, run Compose, then execute on the host:

```bash
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/deployment/reproduction.json
```

The [licensed PDF and cases](../data/quality_benchmark/deployment/README.md) are included in the public source. Follow the [external review procedure](third_party_review.md) for another person's factual review. Historical retrieval scores and historical answer reviews remain in [historical evaluation](evaluation.md), attached to their original implementation snapshots.
