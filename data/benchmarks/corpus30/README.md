# Thirty paper bilingual benchmark

[English](README.md) | [한국어](README_KO.md)

This benchmark evaluates final answers across all 30 papers in the existing text corpus. It contains 180 meaning groups and 360 English/Korean outputs. Each paper has four source-grounded questions, one unsupported request and one question with a premise that contradicts the source. Translations stay in the same split and are not counted as independent scientific questions.

Papers 001–010 are development data because they were already used for labelled retrieval and implementation work. Papers 011–030 previously supported ingestion/search smoke checks; project records do not show labelled answer tuning on them. This usage history is an operator attestation, not an independently audited contamination guarantee. Model pretraining exposure is unknown.

The comparison labels "Initial development version" and "Final development version" refer to two versions of this project's own system using pretrained models. The initial version is the implementation at the start of this expansion study, and the final version is the implementation selected for publication after reviewing the experiments. Their saved artifacts and replay commands retain the identifiers `baseline` and `final`.

## Frozen inputs

- `manifest.json` records paper attribution, source links, licences, corpus hashes and split assignments.
- `questions.jsonl` records paired questions, acceptable source pages and required fact aliases.
- `protocol.json` records selection criteria and the bounded development experiment.
- `freeze.json` protects the exact input bytes.

The pipeline receives only the question and requested paper ID. Reference locations, answerability labels and fact aliases are available to scoring only. `gold_pages` lists acceptable source locations; citing any of those pages is a location diagnostic, not proof that a claim is supported.

## Measurement scope

Fact aliases measure observable lexical coverage. They can miss correct synonyms and accept a fact in an incorrect sentence. Answer retention, refusal status, output validity and citation locations are reported separately. Semantic accuracy and independent human review remain separate from these automatic checks. Labels were authored by an AI-assisted curator after reading source text; they are not independent human ground truth.

The main paired score requires both languages to pass. Reports also expose language-specific and paper-specific failures. Source text evaluation bypasses PDF parsing and does not evaluate visual tables, figures or mathematical derivations.

## Reproduction

Python 3.12, `requirements.txt` and pinned model files are required for inference. Use `--offline` only with all models already cached.

```bash
python -B scripts/corpus_benchmark.py validate
python -B scripts/corpus_benchmark.py run --split dev --variant reproduction --offline --output outputs/reproduction-dev.json
python -B scripts/corpus_benchmark.py evaluate --predictions outputs/reproduction-dev.json --output outputs/reproduction-dev-report.json
```

Experimental writes are restricted to `outputs/` and do not overwrite public evidence. A completed evaluation split becomes regression data after publication. Use newly reserved papers for further generalization estimates.

## Published comparison

[Results and failure analysis](../../../docs/results.md) separates DEV selection, the final twenty-paper comparison, a known scalar regression and execution deviations. [Saved outputs and reports](results/) contain both whole 360-output executions and explicitly derived DEV/HOLDOUT views. The original DEV selection baseline is retained separately from the formal repeated baseline. Both views use a common scorer.

Recalculate published metrics without inference using `python -B scripts/validate_corpus_benchmark.py`. To regenerate answers, use `scripts/replay_corpus_benchmark.py --variant baseline|final --split dev|holdout --offline --output outputs/replay.json` from the repository root; choose one value for each option. It restores only the files named in the measured snapshot and rebuilds embeddings. Optional experimental selection is disabled in the final default because all three DEV variants failed the fixed criteria. Source alignment and the scalar prose correction remain active.
