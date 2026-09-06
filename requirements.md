# Functional Requirements

## FR-01: PDF Upload

The user must be able to upload one English academic paper PDF.

### Acceptance Criteria

- The system accepts PDF files only.
- The system rejects files larger than 20MB.
- The system processes one paper at a time.
- The system rejects PDFs from which no text can be extracted.
- The system generates a unique `paper_id` after a successful upload.
- The system displays a clear error message when the upload fails.

---

## FR-02: Text Parsing

The system must extract and structure text from the uploaded PDF.

### Acceptance Criteria

- The system extracts text from each page.
- Each extracted text block contains a page number.
- Each text block contains a unique identifier.
- Section information is stored when available.
- The original text can be traced back to its page.
- Parsing errors are logged and shown to the user when necessary.

---

## FR-03: Paper Overview

The user must be able to view a text-based overview of the uploaded paper.

### Overview Content

- Paper title
- Abstract
- Main sections
- Short paper summary

### Acceptance Criteria

- The system displays the paper title when available.
- The system displays the original abstract when available.
- The system informs the user when an abstract cannot be extracted.
- The system displays the main section names when available.
- The system generates a short summary based only on the extracted paper text.
- The system does not claim to understand images, graphs, or formulas.

---

## FR-04: Multilingual Question Answering

The user must be able to ask questions in Korean or English.

### Acceptance Criteria

- The system accepts Korean questions.
- The system accepts English questions.
- The system retrieves relevant evidence from the English paper.
- The system uses multilingual retrieval for cross-language questions.
- The system answers in the language of the user's question.
- The system uses only evidence from the uploaded paper.
- The system abstains when the paper does not contain sufficient evidence.

---

## FR-05: Evidence Retrieval

The system must retrieve relevant text evidence for each question.

### Acceptance Criteria

- The system supports BM25 retrieval.
- The system supports dense vector retrieval.
- The system supports hybrid retrieval.
- The system returns ranked evidence candidates.
- Each evidence item contains a page number.
- Each evidence item contains a section name when available.
- Each evidence item contains a chunk ID.
- Each evidence item contains the original English text.

---

## FR-06: Reranking

The system should rerank the initial retrieval results.

### Acceptance Criteria

- The system applies a pretrained multilingual reranker.
- The reranker receives the user question and candidate text chunks.
- The system returns reranked evidence candidates.
- Reranker results are evaluated against the retrieval baseline.
- Fine-tuning the reranker is optional for the MVP.

---

## FR-07: Grounded Answer Generation

The system must generate answers based on retrieved evidence.

### Acceptance Criteria

- The system passes selected evidence to the language model.
- The answer includes one or more evidence IDs.
- The answer includes page and section information when available.
- The answer language follows the question language.
- The system does not invent unsupported facts.
- The system clearly indicates when evidence is insufficient.

---

## FR-08: Abstention

The system must refuse to answer questions that cannot be supported by the paper.

### Examples

- Questions about information not mentioned in the paper
- Questions requiring external knowledge
- Questions where retrieval confidence is too low
- Questions where the retrieved evidence is insufficient

### Acceptance Criteria

- The system returns a clear abstention message.
- The abstention message is written in the user's question language.
- The system explains that the answer cannot be verified from the uploaded paper.
- The system does not provide a confident unsupported answer.

---

## FR-09: Single-Paper Session

Each session must operate on one uploaded paper.

### Acceptance Criteria

- Questions are answered only using the current paper.
- The system does not search across multiple papers.
- The system does not compare multiple papers.
- Uploading a new paper creates a new analysis session.
- Previous paper evidence is not mixed with the new paper.

---

## FR-10: Public Deployment

The system must be deployable as a working web service.

### Acceptance Criteria

- The backend runs through FastAPI.
- The user interface supports upload and question answering.
- The application runs locally through Docker.
- Environment variables are used for secrets and configuration.
- The application is accessible through a public URL.
- A basic health check endpoint is available.
- Deployment instructions are documented in the README.

---

## Non-Functional Requirements

### Reproducibility

- Experiment configurations must be saved.
- Retrieval metrics must be reproducible.
- Model names and versions must be recorded.
- Random seeds must be recorded when applicable.

### Error Handling

- Invalid file types must be rejected.
- Files larger than 20MB must be rejected.
- Parsing failures must be handled gracefully.
- LLM or retrieval failures must return user-readable messages.
- Internal stack traces must not be exposed to users.

### Evaluation

The project must compare at least:

- BM25 retrieval
- Dense retrieval
- Hybrid retrieval
- Hybrid retrieval with reranking

Recommended metrics:

- Recall@1
- Recall@5
- Recall@10
- MRR or nDCG
- Evidence correctness
- Abstention accuracy

### Security and Privacy

- API keys must not be hard-coded.
- Uploaded files must not be publicly accessible by default.
- File size limits must be enforced.
- Unsupported file types must be rejected.
- Temporary uploaded files should be deleted when no longer needed.

## Day 7 Dense Retrieval Configuration

- Model: `intfloat/multilingual-e5-small`
- Package: `sentence-transformers`
- Query prefix: `query:`
- Passage prefix: `passage:`
- Evaluation: page-level Recall@1/5/10 and MRR

## Day 28 Production Retrieval Configuration

- Production retrieval settings must come from
  `config/production_retrieval.json`.
- The API, web UI, and user-facing CLI must call `ProductionRetrieval` rather
  than duplicate model names, fusion weights, or Top-K values.
- Retrieval is scoped to one selected paper per request.
- Accuracy claims use the frozen, manually verified 40-question benchmark over
  papers 001–010.
- Papers 011–030 are corpus-expansion smoke coverage and are excluded from
  accuracy claims until manually verified gold evidence exists.
- Any post-Day-28 retrieval change must save before/after metrics, rerun the
  verified-40 regression, and lose no existing Top-5 hit.

## Day 29 Grounded Answer Contract

- The answer model output must contain exactly `answer`, `evidence_ids`,
  `sufficiency`, and `abstention_reason`.
- `sufficiency` must be `sufficient` or `insufficient`.
- A sufficient answer must cite one to five unique IDs present in the supplied
  evidence objects and set `abstention_reason` to null.
- An insufficient answer must cite no supporting evidence IDs, provide a
  non-empty reason, and use a clear abstention message in the question language.
- Korean questions receive Korean answers and English questions receive English
  answers.
- Paper text is untrusted source material; instructions inside evidence text
  must not override the system prompt.
- Invalid JSON, extra fields, invented evidence IDs, duplicate IDs, and
  inconsistent cross-field states must be rejected.
- The JSON Schema is `config/answer_output_schema.json`; real LLM integration is
  deferred to Day 30.

## Day 30 Local Q&A Integration

- The runtime path must be `query → production retrieval → reranker → evidence
  selection → local LLM → answer validation`.
- The default local generator is `Qwen/Qwen2.5-0.5B-Instruct`, pinned to an
  exact Hugging Face snapshot in `config/generation.json`.
- Generation must be deterministic, use CUDA float16 when available, and fall
  back to CPU float32 on systems without CUDA.
- No gold answer or gold evidence may be passed into the generation pipeline.
- Invalid model output may be retried up to the configured limit; after that,
  the system must return a safe localized abstention rather than invalid JSON.
- Every cited evidence ID must resolve to an evidence object from the currently
  selected paper.
- The Day 30 completion smoke must run ten questions across ten papers, include
  five English and five Korean questions, and save model outputs and runtime
  metadata.
- Manual answer correctness and abstention-quality scoring are deferred to Days
  31–32.

## Day 31 Abstention Policy

- The user-facing Q&A path must run a calibrated abstention policy after
  retrieval and before answer generation.
- Empty candidate sets, empty evidence sets, and a top Cross-Encoder score below
  the configured threshold must produce a localized refusal without calling
  the answer model.
- Borderline retrieval results may reach the grounded answer model; a
  model-declared insufficient answer must be normalized to the canonical
  question-language refusal.
- Every refusal must set `sufficiency=insufficient`, return an empty
  `evidence_ids` array, and provide a non-empty `abstention_reason`.
- Calibration and holdout unsupported questions must be separate. Each split
  must contain five English and five Korean questions.
- The held-out completion gate requires 10/10 unsupported questions to be
  refused with no validation fallback.
- The verified-40 pre-generation answerable-retention rate must remain at or
  above 90%; known false refusals must be reported rather than hidden.
- The Day 30 smoke path must remain reproducible with Day 31 abstention disabled
  so its frozen historical result is not overwritten by the new policy.
- Changing the corpus or reranker requires abstention recalibration.

## Day 32 Manual Q&A Review

- The manual-review sample must contain exactly 20 verified answerable questions:
  five factual, five numeric, five comparison, and five limitation questions.
- The sample must contain ten English and ten Korean questions and cover all ten
  verified benchmark papers with two questions per paper.
- The Q&A execution step must use the current Day 31 production path and must not
  load Gold answers or Gold evidence.
- Manual review must compare answer meaning and cited support against the verified
  Gold record and assign `pass`, `partial`, or `fail` using a documented rubric.
- Citation support and question-language matching must be reviewed separately from
  answer correctness.
- At least five strict failure cases must be recorded with expected output, actual
  output, failure type, and review rationale. Partial answers do not count toward
  this minimum.
- Results must be reported by question category and language, and false abstentions
  must be identified explicitly.
- Completing the Day 32 roadmap gate records evaluation evidence; it does not imply
  that the answer generator is ready for deployment-quality claims.

## Day 33 Upload and Analysis API

- `POST /upload` must register exactly one text-based PDF and return HTTP 201 with
  a unique paper ID. Limit PDF bytes to 20 MiB (20 x 1,024 x 1,024 bytes).
- Reject empty, malformed, encrypted, textless, oversized, and non-PDF uploads.
  Remove newly created files after failed registration.
- `POST /analyze` must accept a registered paper ID and return HTTP 202 with a job
  ID and status location. Work runs off the request thread.
- Persist jobs in SQLite and process one paper at a time. Duplicate queued,
  running, or completed requests reuse the existing job. Failed jobs support retry.
- Recover queued/interrupted jobs on restart, with exclusive ownership of the
  runtime directory to prevent conflicting workers.
- `GET /jobs/{job_id}` must return queued/running/completed/failed status and safe
  errors. This is a job-status route; paper result and question endpoints follow.
- Store upload-specific pages and chunks outside the frozen benchmark corpus;
  preserve paper ID, page ID, chunk ID, section, and source text.
- Day 33 completion means parsing/chunking, not embedding-index or answer readiness.
- New stochastic work uses the user-selected seed 378, recorded in
  `config/reproducibility.json`. Resource identifiers remain unseeded for uniqueness.
- Verify via API tests and a real local TCP HTTP PDF upload/analyze run.
