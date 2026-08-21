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