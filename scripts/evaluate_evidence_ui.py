"""Run and record the real-browser answer and source-evidence UI gate."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app
from app.service import Settings
from evaluate_day36 import (
    digest,
    free_port,
    resolve_browser,
    run_browser,
    runtime_summary,
    stop_process,
    wait_for_url,
)


OUTPUT = ROOT / "data" / "evaluation" / "evidence_ui_results.json"
SOURCE_PDF = ROOT / "data" / "raw" / "papers" / "paper-003.pdf"
QUESTION = "What is the main focus of this paper's review of text summarization?"
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
    "ui/__init__.py",
    "ui/api_client.py",
    "ui/error_messages.py",
    "ui/app.py",
    "tests/test_day36_ui.py",
    "tests/test_error_handling.py",
    "scripts/evaluate_day36.py",
    "scripts/evaluate_evidence_ui.py",
    "scripts/validate_evidence_ui.py",
    "scripts/verify_project.py",
]

BROWSER_SCRIPT = r"""
const { chromium } = require("playwright");

(async () => {
  const uiUrl = process.argv[1];
  const pdfPath = process.argv[2];
  const browserPath = process.argv[3];
  const questionText = process.argv[4];
  const browser = await chromium.launch({
    headless: true,
    executablePath: browserPath || undefined,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
    page.setDefaultTimeout(240000);
    await page.goto(uiUrl, { waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { name: "Evidence-Grounded Paper Q&A" }).waitFor();
    await page.locator('input[type="file"]').setInputFiles(pdfPath);
    await page.getByText("Selected file:", { exact: false }).waitFor();
    await page.getByRole("button", { name: "Upload and analyze", exact: true }).click();
    await page.getByText("Analysis complete.", { exact: true }).first().waitFor();

    const question = page.getByLabel("Ask a question about this paper", { exact: true });
    await question.fill(questionText);
    await page.getByRole("button", { name: "Ask", exact: true }).click();

    const answerHeading = page.getByRole("heading", { name: "Answer", exact: true });
    const evidenceHeading = page.getByRole("heading", { name: "Source evidence", exact: true });
    await answerHeading.waitFor();
    await evidenceHeading.waitFor();
    const evidenceItems = page.getByRole("heading", { name: /^Evidence [0-9]+$/ });
    const evidenceCount = await evidenceItems.count();
    const body = await page.locator("body").innerText();
    const answerBox = await answerHeading.boundingBox();
    const evidenceBox = await evidenceHeading.boundingBox();
    const sideBySide = Boolean(
      answerBox && evidenceBox &&
      evidenceBox.x > answerBox.x &&
      Math.abs(evidenceBox.y - answerBox.y) < 120
    );
    process.stdout.write(JSON.stringify({
      browser_loaded: true,
      question_submitted: true,
      answer_visible: await answerHeading.isVisible(),
      evidence_panel_visible: await evidenceHeading.isVisible(),
      answer_and_evidence_side_by_side: sideBySide,
      evidence_count: evidenceCount,
      page_visible: /Page [0-9]+ \| Section:/.test(body),
      section_visible: body.includes("Section:"),
      original_text_visible: body.includes("Original paper text"),
      chunk_id_visible: body.includes("Chunk ID:"),
      cited_evidence_count_visible: body.includes("Cited evidence:"),
      no_insufficient_warning: !body.includes("does not provide enough evidence"),
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
    with TemporaryDirectory(prefix="paper-qna-evidence-ui-") as directory:
        temporary_root = Path(directory)
        runtime = temporary_root / "runtime"
        ui_log_path = temporary_root / "ui.log"
        ui_port = free_port()
        api_socket = socket.socket()
        api_socket.bind(("127.0.0.1", 0))
        api_port = int(api_socket.getsockname()[1])
        api_url = f"http://127.0.0.1:{api_port}"
        ui_url = f"http://127.0.0.1:{ui_port}"
        environment = os.environ.copy()
        environment.update({
            "PAPER_QNA_DATA_DIR": str(runtime),
            "PAPER_QNA_API_URL": api_url,
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        api_server = uvicorn.Server(uvicorn.Config(
            create_app(Settings(
                data_dir=runtime,
                seed=378,
                model_local_files_only=True,
                prepare_question_runtime=True,
            )),
            log_level="warning",
        ))
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
                    source_pdf=SOURCE_PDF,
                    extra_arguments=[QUESTION],
                )
                with httpx.Client(base_url=api_url, timeout=10, trust_env=False) as client:
                    health_after = client.get("/health").json()
            except Exception as error:
                ui_log.flush()
                ui_tail = ui_log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise RuntimeError(
                    f"Evidence UI acceptance failed: {error}\nUI log:\n{ui_tail}"
                ) from error
            finally:
                stop_process(ui)
                api_server.should_exit = True
                api_thread.join(timeout=30)
                api_socket.close()
                if api_thread.is_alive():
                    raise RuntimeError("Local API server did not shut down")
        runtime_result = runtime_summary(runtime / "service.sqlite3")

    frozen_after = {name: digest(ROOT / name) for name in FROZEN_FILES}
    assert frozen_before == frozen_after, "Frozen evaluation inputs changed"
    boolean_checks = {
        name: value
        for name, value in browser.items()
        if name != "evidence_count"
    }
    assert all(value is True for value in boolean_checks.values()), boolean_checks
    assert browser["evidence_count"] >= 1
    result = {
        "schema_version": 1,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 378,
        "requirement": "answer beside cited evidence text, page, and section",
        "tests_run": tests.testsRun,
        "test_failures": len(tests.failures),
        "test_errors": len(tests.errors),
        "transport": "headless Chromium against Streamlit and real loopback FastAPI",
        "browser_executable": browser_path.name,
        "question": QUESTION,
        "checks": browser,
        "runtime": runtime_result,
        "api_health_after": health_after,
        "frozen_inputs_unchanged": True,
        "frozen_input_sha256": frozen_after,
        "implementation_sha256": implementation_hashes(),
        "temporary_runtime_removed": True,
        "quality_claim": "evidence presentation and traceability only",
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tests": result["tests_run"],
        "browser_gate": True,
        "evidence_count": browser["evidence_count"],
        "side_by_side": browser["answer_and_evidence_side_by_side"],
        "seed": result["seed"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
