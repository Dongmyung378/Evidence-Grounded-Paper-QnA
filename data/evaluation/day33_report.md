# Day 33 implementation and review

## Original roadmap decision

Task: `POST upload와 analyze endpoint 구현`.
Gate: `API로 PDF를 등록하고 분석 job을 시작한다.`

**Completed.** A real local Uvicorn HTTP run registered a PDF (`201`), accepted an
analysis request (`202`), and reached `completed` with page/chunk artifacts.

## Implemented

- `POST /upload`: one text-based PDF, unique paper ID, readable-text preflight,
  20 MiB limit, type/signature checks, and failed-upload cleanup.
- `POST /analyze`: durable SQLite job, serial background parsing/chunking,
  duplicate request reuse, failure retry, and restart recovery.
- `GET /jobs/{job_id}`: status, page/chunk counts, warnings, and public errors.
- Separate per-upload runtime storage, exclusive worker ownership, and page-to-chunk
  traceability using the existing parser.
- Default stochastic seed 378 in `config/reproducibility.json`, read by API jobs
  and applied to randomized test ordering. Deterministic parsing needs no RNG.
- Dependency declarations, interactive `/docs`, and shell-specific run examples.

## Actual verification

- Ten automated tests passed, with multiple negative-input subcases.
- Tests cover corrupt/empty/encrypted/textless files, two-file rejection,
  exact/over-limit sizes, missing Content-Length, unknown IDs, path-like filenames,
  separate-paper outputs, concurrent duplicate requests, error sanitization,
  failure retry, incomplete extraction, exclusive ownership, and restart recovery.
- Real TCP HTTP smoke: `paper-003.pdf`, 341,371 bytes, 10 PDF pages, 10 text pages,
  **40 chunks**, 23 nonfatal warnings, successful traceability and repeated job reuse.
- Corpus/evaluation/configuration SHA-256 hashes match before and after the run.
- Machine-readable results and source hashes: `day33_api_results.json`.
- A Windows lock error discovered by the second-service test was fixed and retested.
- Seven existing checks also passed: pipeline validation, ingestion exception tests,
  and Day 28/29/30/31/32 validators. These validate saved historical evaluations;
  they are not new retrieval or LLM performance measurements.
- Final diff whitespace check passed. The evaluation's temporary uploaded PDF copy
  and job database were removed by the test harness; original corpus files remain.

## Review boundaries

Day 33 completion covers registration and starting/completing ingestion jobs.
Index building and upload-specific Q&A API integration still require the following
roadmap work. No generated-answer quality improvement is claimed. The Day 32
20-question result (1 pass, 4 partial, 15 fail) remains a separate unresolved
quality issue.

The single-process worker supports this laptop/local MVP. Analysis deadlines and
production worker isolation are future deployment concerns. Original PDF parsing
warnings remain visible, and request/response tests are distinct from PDF layout
or linguistic quality verification.

See `docs/day33_api.md` for execution, routes, errors, storage, and reproducibility.
