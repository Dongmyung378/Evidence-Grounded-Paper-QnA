# Scripts

[English](README.md) | [한국어](README_KO.md)

This directory contains only current runtime, evaluation, and validation entry points. Historical milestone scripts were removed from the portfolio branch.

## Main commands

From the repository root, install the public test dependencies and verify the saved artifacts first:

```bash
python -m pip install -r requirements-test.txt
python -B scripts/verify_project.py
python -B scripts/quality_benchmark.py validate
python -B scripts/quality_benchmark.py smoke --output outputs/quality-smoke.json
python -B scripts/ask_paper.py --help
```

`verify_project.py` works on a public clone with tracked artifacts alone. If the private 30-paper PDF corpus and generated dense cache are available locally, it detects them and runs the additional source checks automatically.

## Ingest your own PDFs

Replace `YOUR_PDF_DIRECTORY` with a directory containing text-based English PDFs named `paper-*.pdf`; only that filename pattern is selected. Use separate output paths to preserve the bundled corpus:

```bash
python -B scripts/run_ingestion_pipeline.py --papers-dir YOUR_PDF_DIRECTORY --pages-output outputs/ingestion/pages.jsonl --chunks-output outputs/ingestion/chunks.jsonl --report outputs/ingestion/ingestion_report.json
```

## Historical tools

Preserve the original bytes under `data/evaluation`; do not regenerate those results during current verification. These filenames describe historical workflows, not public quick-start commands:

- `evaluate_frozen_retrieval.py` and `evaluate_abstention.py` write fixed historical destinations and have no `--output` option.
- `evaluate_grounded_generation.py` and `run_container_qna_benchmark.py` accept `--output`, but their defaults point to preserved historical artifacts.

Builders convert saved outputs and review labels into final reviewed artifacts. Validators check those artifacts without downloading models.

## Current model checks

Direct model runs require the full runtime dependencies. `--offline` requires every pinned model revision to be cached already; remove it for a first download. Runtime benchmarking uses `--local-files-only` for the same restriction.

```bash
python -m pip install -r requirements.txt
```

Current model-backed checks and development comparisons:

```bash
python -B scripts/quality_benchmark.py run --split dev --offline --output outputs/quality-dev-predictions.json
python -B scripts/quality_benchmark.py evaluate --predictions outputs/quality-dev-predictions.json --output outputs/quality-dev-report.json
python -B scripts/compare_answer_versions.py --question-ids q-041-en q-041-ko q-042-en q-043-ko --offline --output outputs/development-answer-comparison.json
python -B scripts/check_local_demo.py --offline --output outputs/local-demo-check.json
python -B scripts/calibrate_answerability.py --predictions outputs/quality-dev-predictions.json --offline --output outputs/answerability-calibration.json
python -B scripts/benchmark_runtime.py --help
```

The comparison reuses saved development evidence and compares answer construction only. Both builders use the current translator, so it does not reproduce the full historical runtime, measure controlled latency changes or establish independent holdout accuracy. Runtime reports separate initial activation from warm p50 and p95 and retain environment and configuration hashes. Nested stage timings overlap with parent timings and must not be summed twice.

For the real-paper benchmark, finalize settings on DEV and freeze the runtime before opening HOLDOUT.

```bash
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json validate
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json run --split dev --offline --device cpu --embedding-cache outputs/real-dev-embeddings.npz --output outputs/real-dev-predictions.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json freeze-run --output outputs/real-runtime-freeze.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json run --split holdout --runtime-freeze outputs/real-runtime-freeze.json --offline --device cpu --embedding-cache outputs/real-holdout-embeddings.npz --output outputs/real-holdout-predictions.json
python -B scripts/quality_benchmark.py --config data/quality_benchmark/real_paper_config.json evaluate --predictions outputs/real-holdout-predictions.json --output outputs/real-holdout-report.json
```

`check_local_demo.py` uses an in-process test client. With the API and UI running, execute the network check on the host from the public clone:

```bash
python -B scripts/check_deployment.py --base-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --real-cases data/quality_benchmark/deployment/real_pdf_cases.json --output outputs/deployment/current-check.json
```

The [bundled licensed PDF and cases](../data/quality_benchmark/deployment/README.md) provide the original PDF, questions and required lexical groups with an expected byte hash. Fixture data is not copied into the Docker image. The check exercises PDF upload, parsing and HTTP Q&A; the UI check covers its health endpoint, not browser interaction. Client hashes and responses do not identify server source or package versions; record those separately. Use the [external review procedure](../docs/third_party_review.md) to record another person's verification.

The runtime benchmark's `--translation-cache-mode disabled` clears sentence outputs before every request while retaining loaded models. Compare it with `enabled` using the same question and repetition count. Both modes repeat one question and do not measure throughput on new questions.
