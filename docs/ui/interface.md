# Streamlit Interface

[English](interface.md) | [한국어](interface_KO.md)

## User flow

The Streamlit interface in `ui/app.py` connects to FastAPI through `ui/api_client.py`. Parsing, retrieval, reranking, generation, and citation validation remain in the backend.

The interface supports:

- English and Korean display modes
- one text-based PDF upload with a 20 MiB limit
- queued, running, completed, and failed analysis states
- page, text-page, chunk, and warning counts
- extracted abstract and detected section headings
- a question field enabled only after analysis succeeds
- answer, abstention reason, and runtime diagnostics
- a source-evidence panel beside the answer
- localized upload validation and recoverable parsing-failure states

## Evidence presentation

Each cited evidence item is displayed in a bordered card with its citation number, page, section, exact paper text, and chunk ID. The answer panel reports the number of cited evidence items. When the backend returns an insufficient result, the interface shows the reason without creating evidence cards.

This layout makes the answer traceable to the original paper while preserving the backend's validated evidence contract. It does not claim that the local generation model has production-level answer quality.

## Error handling and recovery

The interface applies the same 20 MiB product limit as FastAPI. File size, extension, content type, and PDF header are checked before upload, while full PDF readability and text extraction remain backend checks. English and Korean messages cover oversized files, invalid or encrypted PDFs, missing text, queue pressure, timeouts, parsing failure, and unavailable services. Unknown backend errors use a safe generic message instead of exposing internal details.

When parsing fails, the progress indicator is cleared, the analysis status closes as an error, and the question field stays locked. The file selector and upload button remain available so the user can choose another PDF without refreshing the page.

## Local run

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Start Streamlit in a second terminal:

```bash
python -m streamlit run ui/app.py
```

Open `http://127.0.0.1:8501`. If FastAPI uses another address, set it in the sidebar or with `PAPER_QNA_API_URL`.

## Verification

The browser gates use a real loopback FastAPI server, Streamlit server, and headless Chromium browser. One gate verifies PDF upload, analysis, overview, and question readiness. The other submits a known answerable question and verifies that the answer and source evidence appear side by side with page, section, original text, and chunk ID.

| Check | Result |
|---|---:|
| Regression and UI tests | 44 passed |
| Browser PDF upload and analysis | Passed |
| Browser question submission | Passed |
| Answer and evidence side by side | Passed |
| Evidence cards displayed | 4 |
| Page, section, original text, and chunk ID | Passed |
| Oversized and invalid PDF blocked before API | Passed |
| Parsing failure message and closed progress state | Passed |
| Successful upload after parsing failure | Passed |
| Seed | 378 |
| Frozen evaluation inputs changed | No |
| Temporary runtime retained | No |

```bash
python -B scripts/validate_day36.py
python -B scripts/validate_evidence_ui.py
python -B scripts/validate_error_recovery_ui.py
```

Machine-readable records are stored in `data/evaluation/day36_ui_results.json`, `data/evaluation/evidence_ui_results.json`, and `data/evaluation/error_recovery_ui_results.json`. Rerunning the browser evaluators requires Node.js, Playwright, and a local Chromium browser in addition to the Python dependencies.
