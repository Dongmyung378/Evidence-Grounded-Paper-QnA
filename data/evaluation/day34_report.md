# Day 34 implementation and review

## Roadmap decision

Task: `question, paper result, health endpoint 구현`.

Gate: `curl에서 질문을 보내고 답을 받는다.`

Status: **Complete**. A real local Uvicorn server accepted a question through the
installed `curl` executable and returned HTTP 200 with an English answer and
page-linked paper evidence.

## Implemented

- `GET /health` reports API version, seed, SQLite availability, queue counts, and
  whether the question models are lazy or ready.
- `GET /papers/{paper_id}` reports upload metadata, analysis state and counts,
  extracted abstract, and the ordered section list without exposing local paths.
- `POST /question` accepts one English or Korean question only after analysis is
  complete and returns the same-language answer, sufficiency decision, abstention
  reason, and cited evidence text with page and section locators.
- Uploaded-paper chunks now enter the frozen production retrieval configuration
  without altering the 30-paper corpus or verified-40 benchmark artifacts.
- Dense embeddings are cached per uploaded paper. Model objects are loaded lazily,
  serialized for the local single-process MVP, and reused when the active paper
  changes.
- Public errors are stable and sanitized for missing papers, paper state conflicts,
  invalid questions, and unavailable local models.

## Actual verification

- Automated tests: 24 passed, 0 failures, 0 errors.
- Real transport: `curl` over TCP loopback to Uvicorn.
- Source: `paper-003.pdf`, 10 pages, 40 runtime chunks.
- HTTP flow: health 200, upload 201, analyze 202, paper result 200, question 200.
- Answer result: `sufficient`, English, with four cited evidence objects on pages
  2, 1, 2, and 8.
- The saved answer correctly identifies the review's transformer and LLM focus and
  specifically names BERT, RoBERTa, and BART.
- Page 1 and page 2 text directly support the answer. The page 8 reference entry
  is unnecessary, so citation minimality remains an answer-quality issue.
- A private per-paper dense embedding cache was created during the question and the
  entire temporary runtime directory was removed after the test.
- Seed 378 was recorded and used. Frozen corpus, evaluation, and model configuration
  hashes were unchanged before and after the run.
- The evaluator fixes its temporary paper ID to a deterministic UUID so repeated
  runs use the same model prompt. Production uploads remain random and unique, as
  covered by the upload isolation tests.

Machine-readable evidence is saved in `data/evaluation/day34_api_results.json`.
It contains current implementation hashes, so `validate_day34.py` rejects stale
evidence after relevant code changes.

## Review boundary

Day 34 is complete against the original functional criterion. Day 35 remains the
formal whole-flow integration day and should test upload, parse, index, question,
and answer as one supported scenario across the API and UI boundary.

This result does not claim deployment-grade answer accuracy. The Day 32 manual
review result of 1 pass, 4 partial, and 15 fail remains the controlling aggregate
quality assessment. Day 34 proves that the selected pipeline is connected to the
API, returns grounded fields, and can complete a real request on this laptop.
