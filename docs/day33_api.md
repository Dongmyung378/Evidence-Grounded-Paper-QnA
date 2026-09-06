# Day 33 — upload and analysis API

Original roadmap: **POST upload와 analyze endpoint 구현**.
Completion criterion: **API로 PDF를 등록하고 분석 job을 시작한다.**

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs` for the interactive request forms.
The service uses one background analysis worker and SQLite. A runtime-directory
lock rejects a second service using the same directory, so startup recovery
cannot reset another live worker's jobs.

Optional storage override:

```bash
# Git Bash
export PAPER_QNA_DATA_DIR='C:/Users/leona/Desktop/Evidence-Grounded-Paper-QnA/data/runtime'
```

```powershell
$env:PAPER_QNA_DATA_DIR = 'C:\Users\leona\Desktop\Evidence-Grounded-Paper-QnA\data\runtime'
```

The default `data/runtime/` is ignored by Git. Choose a private location if you
override it. No static route exposes PDFs, SQLite, or extracted text.

## Request flow

| Request | Purpose | Success |
|---|---|---|
| `POST /upload` | Register exactly one `file` multipart PDF | `201`, paper ID and page count |
| `POST /analyze` | Queue the uploaded paper for parsing/chunking | `202`, job ID and `Location` |
| `GET /jobs/{job_id}` | Poll the job | `200`, status, counts, or safe error |

Git Bash examples (replace IDs with the preceding response):

```bash
curl -sS -F 'file=@data/raw/papers/paper-003.pdf;type=application/pdf' http://127.0.0.1:8000/upload
curl -sS -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{"paper_id":"<paper_id from upload>"}'
curl -sS 'http://127.0.0.1:8000/jobs/<job_id from analyze>'
```

PowerShell users can use `/docs`, or use `curl.exe` for the file upload and:

```powershell
$job = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/analyze' -ContentType 'application/json' -Body (@{paper_id='<paper_id from upload>'} | ConvertTo-Json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/jobs/$($job.job_id)"
```

Jobs progress from `queued` to `running` to `completed` or `failed`. The response
to `POST /analyze` is an acceptance snapshot; it need not reflect the latest worker
state. Repeated requests for a queued, running, or completed paper return the same
job. Requesting analysis after failure queues the same job ID for retry. A server
restart recovers queued/interrupted jobs; completed jobs retain their status.

## Input and output behavior

- File cap: 20 MiB (20 × 1,024 × 1,024 bytes). A multipart request has an additional
  64 KiB envelope allowance; actual PDF bytes are checked independently.
- Limits also apply when `Content-Length` is absent. The request body is bounded
  before multipart parsing and spools to disk above 1 MiB.
- Non-PDF extension/type: `415`. Empty, corrupt, encrypted, or textless PDF:
  `422`. Oversized file/request: `413`. Unknown paper/job: `404`.
- Text-based English PDFs are the intended input. The preflight checks readable
  text, not English-language classification. Scanned PDFs require OCR, which is
  outside the MVP.
- Uploaded filenames are display metadata only. Server-generated IDs determine
  storage locations; uploads with the same filename remain separate papers.
- Each paper has its own PDF, `pages.jsonl`, `chunks.jsonl`, and ingestion report.
  The existing section-aware parser uses 1,200 characters with 200-character overlap.
- Page extraction failures yield a failed job, not a misleading completed state.
  Layout/footnote warnings and skipped empty pages are counted in the report.
- Generated JSONL files are replaced atomically. Consumers must require the job's
  completed status before using the set of outputs.
- Client errors contain public codes/messages. Internal parser exceptions remain
  in server logs.

## Scope and reproducibility

Day 33 `completed` means **parsing and chunking completed** (`stage=parsing_and_chunking`).
It does not mean an embedding index or a Q&A answer is ready. Upload-specific
retrieval/index integration, question/results/health endpoints, and the whole API
flow belong to the following roadmap days. The service does not load the LLM or
require CUDA for these Day 33 operations.

`config/reproducibility.json` records the user's default seed **378** for new
stochastic work. The API reads it and records it on jobs; the randomized negative
test order uses `random.Random(378)`. PDF parsing itself is deterministic and
needs no RNG. UUID paper/job identifiers remain unique across restarts. Historical
evaluation artifacts keep their original configuration.

This is the local, single-process MVP. Shutdown waits for active/queued analysis
work. Hard per-PDF execution deadlines and a separate production worker system
are not implemented here. The answer-quality failures identified on Day 32 remain
open; these API tests do not measure generated-answer quality.

## Verification

```bash
python -B -m unittest discover -s tests -v
python -B scripts/evaluate_day33.py
python -B scripts/validate_day33.py
```

The evaluator runs the tests and a real Uvicorn TCP server on a temporary local
port. It uploads the existing `paper-003.pdf`, starts the analysis, polls to
completion, checks source-page/chunk IDs, verifies duplicate-request reuse, and
shuts down the server. Temporary uploaded copies and job databases are removed;
the benchmark corpus and configuration hashes are checked before/after.

Evidence is saved in `data/evaluation/day33_api_results.json`, including
implementation hashes so the validator detects stale test evidence after edits.

Implementation references: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/),
[lifespan](https://fastapi.tiangolo.com/advanced/events/),
[testing](https://fastapi.tiangolo.com/tutorial/testing/).
