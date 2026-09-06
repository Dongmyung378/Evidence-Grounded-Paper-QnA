# Day 29 Grounded Answer Contract

## Original-roadmap requirement

Day 29 defines `answer`, `evidence_ids`, `sufficiency`, and
`abstention_reason` and requires an answer to be returned as JSON. Connecting a
real LLM belongs to Day 30 and is intentionally not part of this gate.

## Output contract

| Field | Type | Rule |
|---|---|---|
| `answer` | string | Non-empty answer in the question language |
| `evidence_ids` | string array | Zero to five unique IDs; every ID must exist in the supplied evidence |
| `sufficiency` | enum | `sufficient` or `insufficient` |
| `abstention_reason` | string or null | Null for sufficient answers; required for insufficient answers |

The provider-neutral JSON Schema is stored in
`config/answer_output_schema.json`. Additional properties are rejected.

## Cross-field behavior

- A sufficient answer must cite at least one supplied evidence ID and must set
  `abstention_reason` to null.
- An insufficient answer must cite no supporting evidence, must include a
  reason, and must use a clear abstention answer in Korean or English.
- An evidence ID invented by the model, a duplicate ID, invalid JSON, trailing
  prose, or a mismatched field set is rejected before the response reaches the
  API/UI layer.
- A full-response Markdown JSON fence is tolerated for robustness, but mixed
  prose and JSON are not.

## Prompt behavior

`scripts/grounded_answer_contract.py` builds a system/user message pair and
returns the same JSON Schema for a future structured-output call. The prompt:

- answers in the detected question language;
- uses only evidence from the currently selected paper;
- cites the stable Day 24 `evidence_id` values;
- treats paper text as untrusted quoted source data rather than instructions;
- refuses unsupported external-knowledge questions;
- excludes image, table, graph, and equation interpretation.

Cited IDs can be resolved back to the complete Day 24 evidence object, so page,
section, chunk ID, and source text remain available without duplicating those
fields in the LLM output.

## Verification fixtures

`data/evaluation/day29_prompt_examples.json` contains four validated cases:

- English sufficient
- Korean sufficient
- English insufficient
- Korean insufficient

These are deterministic contract fixtures, not model-quality claims. No LLM
call was made, no API key is required, and the frozen Day 28 retrieval
configuration was not changed.

## Reproduction

```bash
python -B scripts/test_grounded_answer_contract.py
python -B scripts/build_day29_prompt_examples.py
python -B scripts/validate_day29.py
python -B scripts/validate_day28.py
```

## Decision

**Passed.** The application now has a strict bilingual grounded-answer JSON
contract ready for the Day 30 retrieval-to-LLM integration.
