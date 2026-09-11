# Evidence-Grounded Paper Q&A

[English](README.md) | [한국어](README_KO.md)

Evidence-Grounded Paper Q&A is a local-first service for asking questions about one English research paper at a time. It accepts questions in English or Korean and returns an answer in the same language, together with the original supporting text, page number, section, and chunk ID.

The project focuses on a simple requirement: an answer should be easy to trace back to the paper it came from.

## What it does

- Uploads one text-extractable English PDF up to 20 MiB
- Extracts a paper overview, page text, and section-aware chunks
- Accepts English and Korean questions
- Combines BM25 and multilingual dense retrieval with reciprocal rank fusion
- Reranks candidates with a multilingual Cross-Encoder
- Generates answers locally with a pinned Qwen model
- Returns page-linked source evidence for supported answers
- Refuses questions when the paper does not provide enough evidence
- Rejects oversized or invalid PDFs before analysis and keeps the upload screen recoverable after parsing failures
- Exposes the complete workflow through FastAPI and a bilingual Streamlit interface
- Runs the backend and UI as separate, health-checked Docker Compose services

## How it works

```text
PDF upload
  -> page-level text extraction
  -> section-aware chunking
  -> BM25 and dense indexes
  -> reciprocal rank fusion
  -> Cross-Encoder reranking
  -> evidence selection
  -> local answer generation
  -> answer and citation validation
```

The runtime only searches chunks that belong to the selected paper. Uploaded papers and their indexes are stored separately from the frozen evaluation corpus.

## Current evaluation

| Check | Result |
|---|---:|
| Local robustness corpus | 30 papers, 563 pages, 2,463 chunks |
| Verified benchmark | 40 questions across 10 papers |
| Benchmark languages | 20 English, 20 Korean |
| Production retrieval Recall@5 | 1.000 |
| Production retrieval MRR | 0.6317 |
| Unsupported holdout questions refused | 10/10 |
| Real HTTP integration flow | Passed |
| Real browser PDF upload flow | Passed |
| Real browser answer and evidence flow | Passed |
| Real browser error handling and retry flow | Passed |
| Docker Compose backend and UI health gate | Passed |

The retrieval figures use one manually verified gold page per question. The 20 expansion papers are used for parsing and runtime robustness checks, not accuracy claims, because they do not yet have manually verified questions and gold evidence.

The answer generator is still the weakest part of the system. In a balanced 20-question manual review, 1 answer passed, 4 were partial, and 15 failed. The API and evidence traceability are working, but the current 0.5B local model is not presented as production-quality answer generation.

## Quick start

### Docker Compose

Docker Desktop or Docker Engine with Compose v2 is required. The default configuration starts both services, stores uploaded-paper runtime data and model downloads in named volumes, and keeps the UI connected to the backend through the private Compose network.

```bash
docker compose up --build
```

Open `http://127.0.0.1:8501` for the web interface or `http://127.0.0.1:8000/docs` for the interactive API documentation. Stop the services without deleting uploaded-paper data or the model cache:

```bash
docker compose down
```

The first backend build installs a CPU-only PyTorch runtime. The first analysis can download the pinned embedding, reranker, and generation models. Copy `.env.example` to `.env` only when ports, queue size, model preparation, offline mode, logging, or a Hugging Face token must be changed. The real `.env` file is excluded from Git and the Docker build context.

### Python

Python 3.11 or later is recommended for running without containers.

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Start Streamlit in a second terminal:

```bash
python -m streamlit run ui/app.py
```

Open `http://127.0.0.1:8501` for the web interface or `http://127.0.0.1:8000/docs` for the interactive API documentation.

During analysis, the service prepares the pinned embedding, reranker, and generation models. The first run may download them from Hugging Face. The 6 GB GPU profile keeps retrieval models on CPU and reserves GPU memory for answer generation. Restart FastAPI after changing runtime settings.

## API workflow

Register a PDF:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@data/raw/papers/paper-003.pdf"
```

Start analysis with the returned `paper_id`:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-..."}'
```

Poll the returned job and inspect the paper result:

```bash
curl http://127.0.0.1:8000/jobs/job-...
curl http://127.0.0.1:8000/papers/paper-...
```

Ask a question after analysis is complete:

```bash
curl -X POST http://127.0.0.1:8000/question \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-...","question":"What is the main contribution of this paper?"}'
```

A supported response contains the answer and only the evidence objects cited by the model:

```json
{
  "paper_id": "paper-...",
  "question": "What is the main contribution of this paper?",
  "question_language": "en",
  "answer": "...",
  "sufficiency": "sufficient",
  "abstention_reason": null,
  "runtime_seconds": 9.7,
  "runtime": {
    "total_seconds": 9.7,
    "retrieval_seconds": 0.97,
    "generation_seconds": 8.729,
    "generation_attempts": 1,
    "llm_device": "cuda",
    "device_fallback_reason": null,
    "fallback_used": false,
    "abstention_source": null
  },
  "evidence": [
    {
      "evidence_id": "ev-...",
      "page": 3,
      "section": "Introduction",
      "text": "Original text from the paper...",
      "locator": {
        "page_label": "p. 3",
        "section": "Introduction",
        "chunk_id": "paper-...-p003-c001"
      }
    }
  ]
}
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check storage, queue, and model readiness |
| `POST` | `/upload` | Register one PDF |
| `POST` | `/analyze` | Queue parsing and chunking |
| `GET` | `/jobs/{job_id}` | Read analysis status |
| `GET` | `/papers/{paper_id}` | Read the paper overview and processing result |
| `POST` | `/question` | Ask an English or Korean question |

## Models and retrieval

- Embedding: `intfloat/multilingual-e5-small`
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Sparse retrieval: BM25 Top-20
- Dense retrieval: Top-20 normalized cosine similarity
- Fusion: equal-weight reciprocal rank fusion with `rrf_k=60`
- Final evidence: Top-5, at most two chunks per page

Model revisions and runtime parameters are pinned in `config/` so results can be checked against the saved evaluation artifacts.

The browser runtime uses `config/generation_runtime.json`. It keeps the reviewed 384-token budget, allows up to three validation attempts on GPU, and limits CPU fallback to two time-bounded attempts. See [local runtime performance](docs/reviews/runtime_performance.md) for the measured acceptance result and limitations.

## Repository layout

```text
app/                 FastAPI routes and runtime services
config/              retrieval, generation, and abstention settings
data/
  metadata/          paper sources and license records
  processed/         page and chunk artifacts
  evaluation/        benchmark inputs and evaluation results
docs/
  project/           scope, requirements, and data policy
  api/               API behavior and acceptance checks
  ui/                interface behavior and evidence presentation
  reviews/           quality and improvement reviews
scripts/             ingestion, retrieval, evaluation, and validation tools
tests/               API and regression tests
ui/                  bilingual Streamlit interface and API client
Dockerfile           split backend and UI image targets
compose.yaml         local two-service demo and persistent volumes
```

## Verification

Run the saved-artifact validators and regression tests:

```bash
python -B scripts/verify_project.py
```

Re-run the real HTTP flow from PDF upload through answer generation:

```bash
python -B scripts/evaluate_day35.py
python -B scripts/validate_day35.py
```

Validate the saved real-browser Streamlit acceptance result:

```bash
python -B scripts/validate_day36.py
python -B scripts/validate_evidence_ui.py
python -B scripts/validate_error_recovery_ui.py
```

Re-run or validate the Docker Compose acceptance gate:

```bash
python -B scripts/evaluate_container.py
python -B scripts/validate_container.py
```

The integration evaluator starts Uvicorn on a temporary loopback port, calls the question endpoint with curl, verifies page and chunk traceability, and removes its temporary runtime directory afterward.

## Scope and limitations

The current MVP does not support scanned PDFs, OCR, image or graph interpretation, reliable formula interpretation, multi-paper comparison, user accounts, or a public deployment. Docker packaging has been verified locally, but deployment to an internet-facing server has not. Table and figure text may be extracted as plain text, but the system does not interpret their visual structure.

Raw PDFs, QASPER source files, and user uploads remain local unless their redistribution terms explicitly allow publication. See [data sources and handling](docs/project/data_sources.md) for details.

## Documentation

- [Project scope](docs/project/scope.md)
- [Functional requirements](docs/project/requirements.md)
- [API integration gate](docs/api/integration.md)
- [Streamlit interface and evidence panel](docs/ui/interface.md)
- [Docker Compose local run](docs/deployment/docker.md)
- [Quality and improvement review](docs/reviews/improvements.md)
- [Local runtime performance](docs/reviews/runtime_performance.md)

The browser flow covers PDF upload, analysis, paper overview, question entry, answer display, traceable source evidence, localized upload errors, and recovery after parsing failure. The backend and UI also run as verified Docker Compose services. Public deployment remains outside the current verified implementation.
