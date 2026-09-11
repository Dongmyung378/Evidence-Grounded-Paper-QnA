"""Run the browser gate for upload errors, parsing failure, and recovery."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_day36 import (
    digest,
    free_port,
    resolve_browser,
    run_browser,
    stop_process,
    wait_for_url,
)
from ui.error_messages import MAX_UPLOAD_BYTES


OUTPUT = ROOT / "data" / "evaluation" / "error_recovery_ui_results.json"
FROZEN_FILES = [
    "config/production_retrieval.json",
    "config/generation.json",
    "config/abstention_candidate.json",
    "config/runtime_qna.json",
    "config/generation_runtime.json",
    "data/processed/chunks.jsonl",
    "data/processed/pages.jsonl",
    "data/evaluation/questions.jsonl",
    "data/evaluation/gold_evidence.jsonl",
]
IMPLEMENTATION_FILES = [
    ".streamlit/config.toml",
    "app/config.py",
    "app/main.py",
    "app/service.py",
    "ui/api_client.py",
    "ui/error_messages.py",
    "ui/app.py",
    "tests/test_error_handling.py",
    "scripts/evaluate_day36.py",
    "scripts/evaluate_error_recovery_ui.py",
    "scripts/validate_error_recovery_ui.py",
    "scripts/verify_project.py",
]

PARSE_PAPER_ID = "paper-" + "a" * 32
VALID_PAPER_ID = "paper-" + "b" * 32
PARSE_JOB_ID = "job-" + "c" * 32
VALID_JOB_ID = "job-" + "d" * 32

BROWSER_SCRIPT = r"""
const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const uiUrl = process.argv[1];
  const oversizedPath = process.argv[2];
  const browserPath = process.argv[3];
  const invalidPath = process.argv[4];
  const parseFailurePath = process.argv[5];
  const validPath = process.argv[6];
  const browser = await chromium.launch({
    headless: true,
    executablePath: browserPath || undefined,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    page.setDefaultTimeout(90000);
    await page.goto(uiUrl, { waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { name: "Evidence-Grounded Paper Q&A" }).waitFor();

    async function chooseFile(filePath) {
      await page.locator('input[type="file"]').setInputFiles(filePath);
      await page.getByText(path.basename(filePath), { exact: false }).first().waitFor();
    }

    await chooseFile(oversizedPath);
    const oversizedMessage = page.getByText(
      "The PDF must be no larger than 20 MiB. Choose a smaller file.",
      { exact: true },
    );
    await oversizedMessage.waitFor();
    const oversizedVisible = await oversizedMessage.isVisible();
    const oversizedButton = page.getByRole("button", {
      name: "Upload and analyze",
      exact: true,
    });
    const oversizedDisabled = await oversizedButton.isDisabled();

    await chooseFile(invalidPath);
    const invalidMessage = page.getByText(
      "The selected file is not a readable PDF. Choose another PDF.",
      { exact: true },
    );
    await invalidMessage.waitFor();
    const invalidVisible = await invalidMessage.isVisible();
    const invalidDisabled = await page.getByRole("button", {
      name: "Upload and analyze",
      exact: true,
    }).isDisabled();

    await chooseFile(parseFailurePath);
    const parseButton = page.getByRole("button", {
      name: "Upload and analyze",
      exact: true,
    });
    await parseButton.click();
    const parseButtonReady = true;
    const parseMessage = page.getByText(
      "Some PDF pages could not be read. Export the paper as a text-based PDF and try again.",
      { exact: true },
    );
    await parseMessage.waitFor();
    const parseMessageVisible = await parseMessage.isVisible();
    const parsingStatus = page.getByText("Analysis failed.", { exact: true }).first();
    await parsingStatus.waitFor();
    const parsingStatusVisible = await parsingStatus.isVisible();
    const controlsAfterFailure = await page.locator('input[type="file"]').isEnabled();
    const retryButtonReady = await page.getByRole("button", {
      name: "Upload and analyze",
      exact: true,
    }).isEnabled();
    const questionAfterFailure = page.getByLabel(
      "Ask a question about this paper",
      { exact: true },
    );
    const questionLockedAfterFailure = await questionAfterFailure.isDisabled();
    const noSpinnerAfterFailure = await page.locator('[data-testid="stSpinner"]').count() === 0;

    await chooseFile(validPath);
    const validButton = page.getByRole("button", {
      name: "Upload and analyze",
      exact: true,
    });
    await validButton.click();
    await page.getByText("Analysis complete.", { exact: true }).first().waitFor();
    await page.getByRole("heading", { name: "2. Paper overview", exact: true }).waitFor();
    const questionAfterRetry = page.getByLabel(
      "Ask a question about this paper",
      { exact: true },
    );

    process.stdout.write(JSON.stringify({
      browser_loaded: true,
      oversized_message_visible: oversizedVisible,
      oversized_upload_disabled: oversizedDisabled,
      invalid_pdf_message_visible: invalidVisible,
      invalid_upload_disabled: invalidDisabled,
      parse_button_initially_ready: parseButtonReady,
      parsing_failure_message_visible: parseMessageVisible,
      parsing_status_closed_as_error: parsingStatusVisible,
      upload_control_ready_after_failure: controlsAfterFailure,
      retry_button_ready_after_failure: retryButtonReady,
      question_locked_after_failure: questionLockedAfterFailure,
      no_spinner_after_failure: noSpinnerAfterFailure,
      successful_retry_completed: true,
      question_enabled_after_retry: await questionAfterRetry.isEnabled(),
    }));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  process.stderr.write(String(error.stack || error));
  process.exit(1);
});
"""


def implementation_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in IMPLEMENTATION_FILES}


def create_scenario_api(request_log: dict[str, list[str]]) -> FastAPI:
    """Create a deterministic API that exposes recoverable UI failure states."""
    api = FastAPI()

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "seed": 378}

    @api.post("/upload", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict:
        filename = file.filename or ""
        request_log["uploads"].append(filename)
        await file.read()
        if filename == "parse-failure.pdf":
            return {
                "paper_id": PARSE_PAPER_ID,
                "filename": filename,
                "size_bytes": 16,
                "page_count": 2,
                "status": "uploaded",
            }
        if filename == "valid.pdf":
            return {
                "paper_id": VALID_PAPER_ID,
                "filename": filename,
                "size_bytes": 16,
                "page_count": 1,
                "status": "uploaded",
            }
        raise HTTPException(
            422,
            detail={"code": "invalid_pdf", "message": "Invalid test PDF."},
        )

    @api.post("/analyze", status_code=202)
    def analyze(body: dict) -> dict:
        paper_id = str(body.get("paper_id") or "")
        request_log["analyses"].append(paper_id)
        if paper_id == PARSE_PAPER_ID:
            return {
                "job_id": PARSE_JOB_ID,
                "paper_id": paper_id,
                "status": "queued",
            }
        if paper_id == VALID_PAPER_ID:
            return {
                "job_id": VALID_JOB_ID,
                "paper_id": paper_id,
                "status": "queued",
            }
        raise HTTPException(404)

    @api.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        request_log["jobs"].append(job_id)
        if job_id == PARSE_JOB_ID:
            return {
                "job_id": job_id,
                "paper_id": PARSE_PAPER_ID,
                "status": "failed",
                "seed": 378,
                "error_code": "partial_extraction_failed",
                "error_message": "Internal parser detail is not shown by the UI.",
            }
        if job_id == VALID_JOB_ID:
            return {
                "job_id": job_id,
                "paper_id": VALID_PAPER_ID,
                "status": "completed",
                "seed": 378,
                "page_count": 1,
                "text_page_count": 1,
                "chunk_count": 1,
                "warning_count": 0,
            }
        raise HTTPException(404)

    @api.get("/papers/{paper_id}")
    def paper(paper_id: str) -> dict:
        request_log["papers"].append(paper_id)
        if paper_id != VALID_PAPER_ID:
            raise HTTPException(404)
        return {
            "paper_id": paper_id,
            "filename": "valid.pdf",
            "status": "ready",
            "page_count": 1,
            "text_page_count": 1,
            "chunk_count": 1,
            "warning_count": 0,
            "overview": {
                "abstract": "A valid paper recovered after the parsing failure.",
                "sections": ["Introduction"],
            },
        }

    return api


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-modules")
    parser.add_argument("--browser-path")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen_before = {name: digest(ROOT / name) for name in FROZEN_FILES}
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)

    browser_path = resolve_browser(args.browser_path)
    request_log = {"uploads": [], "analyses": [], "jobs": [], "papers": []}
    with TemporaryDirectory(prefix="paper-qna-error-recovery-") as directory:
        temporary_root = Path(directory)
        oversized = temporary_root / "oversized.pdf"
        invalid = temporary_root / "invalid.pdf"
        parse_failure = temporary_root / "parse-failure.pdf"
        valid = temporary_root / "valid.pdf"
        oversized.write_bytes(
            b"%PDF-" + b"0" * (MAX_UPLOAD_BYTES + 1 - len(b"%PDF-"))
        )
        invalid.write_bytes(b"This is not a PDF.")
        parse_failure.write_bytes(b"%PDF-1.4\nparse failure fixture\n")
        valid.write_bytes(b"%PDF-1.4\nvalid recovery fixture\n")

        ui_log_path = temporary_root / "ui.log"
        ui_port = free_port()
        api_socket = socket.socket()
        api_socket.bind(("127.0.0.1", 0))
        api_port = int(api_socket.getsockname()[1])
        api_url = f"http://127.0.0.1:{api_port}"
        ui_url = f"http://127.0.0.1:{ui_port}"
        environment = os.environ.copy()
        environment.update({
            "PAPER_QNA_API_URL": api_url,
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        api_server = uvicorn.Server(
            uvicorn.Config(create_scenario_api(request_log), log_level="warning")
        )
        api_thread = threading.Thread(
            target=api_server.run,
            kwargs={"sockets": [api_socket]},
            daemon=True,
        )
        api_thread.start()
        with ui_log_path.open("w", encoding="utf-8") as ui_log:
            ui = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "streamlit",
                    "run",
                    "ui/app.py",
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(ui_port),
                    "--server.fileWatcherType",
                    "none",
                ],
                cwd=ROOT,
                env=environment,
                stdout=ui_log,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_for_url(f"{api_url}/health")
                wait_for_url(f"{ui_url}/_stcore/health")
                browser = run_browser(
                    ui_url,
                    browser_path,
                    args.node_modules,
                    browser_script=BROWSER_SCRIPT,
                    source_pdf=oversized,
                    extra_arguments=[
                        str(invalid),
                        str(parse_failure),
                        str(valid),
                    ],
                )
            except Exception as error:
                ui_log.flush()
                ui_tail = ui_log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-3000:]
                raise RuntimeError(
                    f"Error recovery acceptance failed: {error}\nUI log:\n{ui_tail}"
                ) from error
            finally:
                stop_process(ui)
                api_server.should_exit = True
                api_thread.join(timeout=30)
                api_socket.close()
                if api_thread.is_alive():
                    raise RuntimeError("Scenario API server did not shut down")

    frozen_after = {name: digest(ROOT / name) for name in FROZEN_FILES}
    assert frozen_before == frozen_after, "Frozen evaluation inputs changed"
    assert all(value is True for value in browser.values()), browser
    assert request_log["uploads"] == ["parse-failure.pdf", "valid.pdf"]
    assert request_log["analyses"] == [PARSE_PAPER_ID, VALID_PAPER_ID]
    assert request_log["papers"] == [VALID_PAPER_ID]

    result = {
        "schema_version": 1,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 378,
        "requirement": "file size, invalid PDF, parsing failure, and successful retry",
        "tests_run": tests.testsRun,
        "test_failures": len(tests.failures),
        "test_errors": len(tests.errors),
        "transport": "headless Chromium against Streamlit and deterministic loopback API",
        "browser_executable": browser_path.name,
        "checks": browser,
        "request_log": request_log,
        "client_rejected_before_api": ["oversized.pdf", "invalid.pdf"],
        "recovery_sequence": [
            "oversized rejected",
            "invalid PDF rejected",
            "parsing failure displayed",
            "valid PDF completed",
        ],
        "frozen_inputs_unchanged": True,
        "frozen_input_sha256": frozen_after,
        "implementation_sha256": implementation_hashes(),
        "temporary_files_removed": True,
        "quality_claim": "UI failure handling and recovery only",
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tests": result["tests_run"],
        "browser_gate": True,
        "recovered_after_failure": browser["successful_retry_completed"],
        "api_uploads": request_log["uploads"],
        "seed": result["seed"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
