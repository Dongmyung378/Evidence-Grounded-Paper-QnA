# Scripts

[English](README.md) | [한국어](README_KO.md)

This directory contains only current runtime, evaluation, and validation entry points. Historical milestone scripts were removed from the portfolio branch.

## Main commands

```bash
python -B scripts/run_ingestion_pipeline.py
python -B scripts/ask_paper.py --help
python -B scripts/verify_project.py
```

Long model-backed evaluations are separate from the lightweight verifier:

```bash
python -B scripts/evaluate_frozen_retrieval.py
python -B scripts/evaluate_abstention.py
python -B scripts/evaluate_grounded_generation.py
python -B scripts/run_container_qna_benchmark.py
```

Builders convert saved outputs and review labels into final reviewed artifacts. Validators check those artifacts without downloading models.
