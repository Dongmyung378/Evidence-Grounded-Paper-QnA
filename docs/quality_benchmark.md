# Complete final-answer quality benchmark

[English](quality_benchmark.md) | [한국어](quality_benchmark_KO.md)

This harness evaluates the final response returned after retrieval, refusal
policy, answer construction, translation, validation and fallback. A retrieval
threshold decision is not counted as a successful final refusal.

The bundled inputs are four **authored fictional research texts**, not newly
collected research papers. Their 40 English and Korean outputs represent
**20 meaning groups**: ten in dev and ten in holdout. Synthetic fixture scores
test evaluation plumbing and model integration. They establish neither
independent real-paper accuracy nor third-party verification.

Meaning-group counts aggregate translations. The fictional texts share
controlled structures; distinct IDs do not establish statistical independence
between text templates or broader scientific generalization.

## Offline checks with no models or private PDFs

Run from the repository root with Python 3.12:

```bash
python scripts/validate_quality_benchmark.py
python -m unittest discover -s tests -p test_quality_benchmark.py -v
python scripts/quality_benchmark.py smoke --output data/quality_benchmark/smoke_report.local.json
```

`smoke` constructs deliberately correct outputs using fixture reference answers.
It sets `execution.kind` to `stub_fixture` and
`gold_used_to_construct_outputs` to `true`. A perfect smoke score verifies the
scorer, never the answer pipeline. Output paths are explicit and existing frozen
evaluation artifacts in `data/evaluation` are not overwritten by these commands.
The CLI rejects output targets inside frozen evaluation data, baselines,
configuration and source directories, as well as fixture inputs or the prediction
file being scored. These checks run before model initialization.

## Execute the actual local pipeline

Install the repository's runtime dependencies and prepare the exact cached model
revisions from `config/model_lock.json` first. See the repository setup guide for
model downloads. `--offline` forbids downloading missing model files.

```bash
python scripts/quality_benchmark.py run --split dev --offline --output data/quality_benchmark/dev_predictions.local.json
python scripts/quality_benchmark.py evaluate --predictions data/quality_benchmark/dev_predictions.local.json --output data/quality_benchmark/dev_report.local.json
```

This mode constructs chunks from source pages only and invokes the unchanged
public production retrieval entry point and `GroundedQAPipeline`. It uses the
embedding model, reranker, refusal policy, answer construction and Korean
translation model. Gold pages, reference answers, category labels and
answerability labels are never supplied to those components. It bypasses PDF
parsing, so it does not verify ingestion. The embedding cache is separate from
historical evaluation caches.

Select retrieval CPU execution with `--device cpu` if needed. Translation uses
the device policy in `config/translation.json`. Missing runtime dependencies or
models fail initialization visibly. Failed individual calls remain present as
invalid final outputs and remain in every denominator.

Develop with dev only. Once settings are fixed, run the synthetic holdout:

```bash
python scripts/quality_benchmark.py run --split holdout --offline --output data/quality_benchmark/holdout_predictions.local.json
python scripts/quality_benchmark.py evaluate --predictions data/quality_benchmark/holdout_predictions.local.json --output data/quality_benchmark/holdout_report.local.json
```

The saved run records exact runtime configuration and implementation SHA256
values, package versions, platform and model-pinned configuration. Supply
`--pipeline-revision YOUR_EXACT_COMMIT` for a committed release. The working-tree
default describes a local development run, not a reproducible release tag.

## Score saved outputs or use an already running API

```bash
python scripts/quality_benchmark.py template --output data/quality_benchmark/predictions_template.local.json
python scripts/quality_benchmark.py evaluate --predictions YOUR_PREDICTIONS.json --output YOUR_REPORT.json
```

The prediction template is marked `not_run` with null outputs. Replace each
output with either the entire internal pipeline result or the public `/question`
response. Preserve case IDs and declare the execution kind truthfully. Evaluation
requires every case exactly once for the declared split (`all` by default).
Missing, unknown or duplicate cases are rejected. Execution errors may remain as
null outputs; they are invalid results, not successful refusals.

For an already prepared API, upload and analyze the exact benchmark source
documents and create a JSON map from benchmark paper IDs to the returned
`paper-<32 hexadecimal characters>` IDs. Distinct source papers must map to
distinct runtime papers. The API runner posts only the runtime paper ID and the
question to `/question`:

```bash
python scripts/quality_benchmark.py run-api --base-url http://localhost:8000 --paper-map YOUR_PAPER_MAP.json --pipeline-revision YOUR_EXACT_COMMIT --output YOUR_API_PREDICTIONS.json
python scripts/quality_benchmark.py evaluate --predictions YOUR_API_PREDICTIONS.json --output YOUR_API_REPORT.json
```

The bundled JSON source pages are not themselves uploadable PDFs. `run` is the
direct supported route for these fixtures. Use `run-api` only after preparing
matching supported documents or after versioning a real-paper benchmark.

## Isolation, counting and scoring

- A source paper belongs to one split. An identical source SHA256 cannot appear
  in both splits even under different paper IDs.
- Repeating the same source bytes under different IDs within one split is also
  rejected so duplicate papers cannot inflate the sample count.
- Each meaning group contains exactly one English and one Korean question, with
  the same paper, split, type, answerability and gold pages.
- Both splits cover fact, numeric, list, comparison and limitation questions,
  with answerable, unsupported and false-premise cases.
- Main metrics use meaning groups. Both language outputs must pass for a group
  to count as retained or correctly refused. Per-language results are reported
  separately; translated questions do not increase the independent sample count.
- A successful refusal requires final `insufficient`, canonical refusal text,
  no cited evidence and a non-empty reason. A fabricated substantive answer
  labelled `insufficient` is invalid. Canonical refusal text labelled
  `sufficient` is also invalid.
- `answerable_retention` measures a valid final answered response. It does not
  measure scientific correctness. False refusals, unsupported answers, invalid
  outputs and validation fallbacks remain visible.
- Required-page coverage checks citation locators only. A correct page number
  does not prove that the cited text supports a claim.
- Zero-denominator metrics are `null`. They are never reported as perfect scores.

`config/quality_benchmark.json` identifies the input files. `freeze.json` binds
the exact manifest, questions and rubric bytes. The manifest separately binds
each source text. Altered inputs are rejected; introduce a reviewed benchmark
version for intentional changes. These inputs are separate from the historical
frozen evaluation set.

## Prepare independent real-paper evaluation

`data/quality_benchmark/real_holdout_template.json` is explicitly
`not_prepared_not_run`. Select papers absent from previous corpora and tuning
experiments, record source URL and license, freeze source hashes and questions,
and keep every paper and its translations in one split. Fix settings using dev
only before opening holdout outputs.

The frozen rubric defines manual factual correctness, completeness and citation
support labels. A review record should contain:

```json
{
  "case_id": "YOUR_CASE_ID",
  "predictions_sha256": "EXACT_FILE_HASH",
  "benchmark_freeze_sha256": "EXACT_FILE_HASH",
  "reviewer_id": "STABLE_NAME_OR_PSEUDONYM",
  "relationship_to_author": "SELF_OR_DOCUMENTED_RELATIONSHIP",
  "reviewed_at": "ISO_8601_TIMESTAMP",
  "factual_correctness": "pass",
  "completeness": "partial",
  "citation_support": "pass",
  "notes": "Specific supported claims and any missing qualifications."
}
```

Publish raw final outputs, failures and review provenance. Self-authored fixtures
and self-review cannot be described as independent review. This harness always
sets automatic semantic accuracy to `null`, independent human review to `false`
and third-party validation to `not_attested`; separate attributable reviewer
evidence is required before making those claims.
