# Day 36 Streamlit UI

[English](day36.md) | [한국어](day36_KO.md)

## Roadmap requirement

- Work: implement PDF upload, analysis status, paper overview, and question input screens.
- Completion criterion: a user can upload a PDF from a browser.

## Implementation

The Streamlit interface in `ui/app.py` connects to the existing FastAPI service through `ui/api_client.py`. It keeps browser concerns separate from parsing, retrieval, and generation code.

The Day 36 screen provides:

- English and Korean display modes
- one-PDF browser upload with a 20 MiB limit
- queued, running, completed, and failed analysis states
- page, text-page, chunk, and warning counts
- extracted abstract and detected section headings
- a question field that stays disabled until analysis succeeds
- question submission, answer display, abstention reason, and runtime breakdown through the existing API
- stable, user-readable messages for connection, file, and analysis errors

Detailed evidence text, page, and section presentation remains Day 37 work. The Day 36 UI does not alter the API evidence contract or the frozen retrieval evaluation.

## Local run

Install dependencies once:

```bash
python -m pip install -r requirements.txt
```

Start FastAPI in the first terminal:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Start Streamlit in the second terminal:

```bash
python -m streamlit run ui/app.py
```

Open `http://127.0.0.1:8501`. If FastAPI uses another address, set it in the UI sidebar or through `PAPER_QNA_API_URL`.

## Verification

The saved acceptance run used a real loopback FastAPI server, a real Streamlit server, and headless Microsoft Edge. The browser selected `paper-003.pdf`, clicked the upload and analysis control, and waited for the overview and question input.

| Check | Result |
|---|---:|
| Regression and UI tests | 33 passed |
| Browser PDF upload | Passed |
| Analysis status completion | Passed |
| Parsed paper | 10 pages, 40 chunks |
| Overview visible | Passed |
| Question input enabled after analysis | Passed |
| Question runtime prepared during analysis | Passed |
| Seed | 378 |
| Frozen evaluation inputs changed | No |
| Temporary runtime retained | No |

Run the saved-result validator:

```bash
python -B scripts/validate_day36.py
```

The complete machine-readable record is stored in `data/evaluation/day36_ui_results.json`. The automated browser evaluator is `scripts/evaluate_day36.py`; rerunning it requires Node.js, Playwright, and a local Chromium browser in addition to the application dependencies.

## Result

Day 36 meets the original completion criterion. A user can upload a text-based English PDF from the browser, follow its analysis state, inspect the generated overview, and reach an enabled question field. This result is a functional UI integration claim and does not revise the Day 32 answer-quality assessment.
