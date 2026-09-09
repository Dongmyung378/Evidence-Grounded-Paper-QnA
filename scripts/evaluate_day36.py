"""Run and record the original-roadmap Day 36 Streamlit browser gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import unittest
from contextlib import closing
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


OUTPUT = ROOT / "data" / "evaluation" / "day36_ui_results.json"
SOURCE_PDF = ROOT / "data" / "raw" / "papers" / "paper-003.pdf"
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
    "requirements.txt",
    "ui/__init__.py",
    "ui/api_client.py",
    "ui/app.py",
    "tests/test_day36_ui.py",
    "scripts/evaluate_day36.py",
    "scripts/validate_day36.py",
    "scripts/verify_project.py",
]

BROWSER_SCRIPT = r"""
const { chromium } = require("playwright");

(async () => {
  const uiUrl = process.argv[1];
  const pdfPath = process.argv[2];
  const browserPath = process.argv[3];
  const browser = await chromium.launch({
    headless: true,
    executablePath: browserPath || undefined,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    page.setDefaultTimeout(180000);
    await page.goto(uiUrl, { waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { name: "Evidence-Grounded Paper Q&A" }).waitFor();
    const input = page.locator('input[type="file"]');
    const uploader = page.locator('[data-testid="stFileUploader"]');
    await input.setInputFiles(pdfPath);
    await page.getByText("Selected file:", { exact: false }).waitFor();
    await page.getByRole("button", { name: "Upload and analyze", exact: true }).click();
    await page.getByText("Analysis complete.", { exact: true }).first().waitFor();
    await page.getByRole("heading", { name: "2. Paper overview", exact: true }).waitFor();
    const question = page.getByLabel("Ask a question about this paper", { exact: true });
    await question.waitFor();
    const metricValues = await page.locator('[data-testid="stMetricValue"]').allTextContents();
    const body = await page.locator("body").innerText();
    process.stdout.write(JSON.stringify({
      browser_loaded: true,
      file_uploader_visible: await uploader.isVisible(),
      selected_filename_visible: body.includes("paper-003.pdf"),
      analysis_completed_visible: body.includes("Analysis complete."),
      overview_visible: body.includes("2. Paper overview") && body.includes("Abstract"),
      question_input_visible: await question.isVisible(),
      question_input_enabled: await question.isEnabled(),
      metric_values: metricValues,
    }));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  process.stderr.write(String(error.stack || error));
  process.exit(1);
});
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in IMPLEMENTATION_FILES}


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def wait_for_url(url: str, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    last_error = None
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return response.text
                last_error = RuntimeError(f"HTTP {response.status_code}")
            except httpx.RequestError as error:
                last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"Server did not become ready at {url}: {last_error}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def resolve_browser(requested: str | None) -> Path:
    candidates = [
        Path(requested) if requested else None,
        Path(os.environ["PAPER_QNA_BROWSER_PATH"])
        if os.environ.get("PAPER_QNA_BROWSER_PATH")
        else None,
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise RuntimeError("No supported local Chromium browser was found")


def run_browser(
    ui_url: str,
    browser_path: Path,
    node_modules: str | None,
) -> dict:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required for the automated browser gate")
    environment = os.environ.copy()
    if node_modules:
        environment["NODE_PATH"] = node_modules
    probe = subprocess.run(
        [node, "-e", "require.resolve('playwright')"],
        env=environment,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "Playwright is required for the automated browser gate. "
            "Install it or pass --node-modules with a directory that contains it."
        )
    completed = subprocess.run(
        [
            node,
            "-e",
            BROWSER_SCRIPT,
            ui_url,
            str(SOURCE_PDF),
            str(browser_path),
        ],
        env=environment,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=210,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Browser acceptance failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def runtime_summary(database: Path) -> dict:
    with closing(sqlite3.connect(database)) as connection:
        papers = connection.execute("SELECT count(*) FROM papers").fetchone()[0]
        jobs = dict(
            connection.execute(
                "SELECT status, count(*) FROM jobs GROUP BY status"
            ).fetchall()
        )
        paper = connection.execute(
            "SELECT paper_id, filename, page_count FROM papers "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        job = connection.execute(
            "SELECT seed, page_count, text_page_count, chunk_count, warning_count "
            "FROM jobs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    report = json.loads(
        (database.parent / "papers" / paper[0] / "ingestion_report.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "papers": papers,
        "jobs": jobs,
        "filename": paper[1],
        "uploaded_page_count": paper[2],
        "seed": job[0],
        "analyzed_page_count": job[1],
        "text_page_count": job[2],
        "chunk_count": job[3],
        "warning_count": job[4],
        "question_runtime_preparation": report.get("question_runtime_preparation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node-modules",
        help="Optional NODE_PATH directory containing Playwright.",
    )
    parser.add_argument(
        "--browser-path",
        help="Optional Edge or Chrome executable used for headless acceptance.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen_before = {name: digest(ROOT / name) for name in FROZEN_FILES}
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit(1)

    browser_path = resolve_browser(args.browser_path)
    with TemporaryDirectory(prefix="paper-qna-day36-") as directory:
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
        environment.update(
            {
                "PAPER_QNA_DATA_DIR": str(runtime),
                "PAPER_QNA_API_URL": api_url,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        api_server = uvicorn.Server(
            uvicorn.Config(
                create_app(Settings(
                    data_dir=runtime,
                    seed=378,
                    model_local_files_only=True,
                    prepare_question_runtime=True,
                )),
                log_level="warning",
            )
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
                streamlit_health = wait_for_url(f"{ui_url}/_stcore/health")
                browser = run_browser(ui_url, browser_path, args.node_modules)
                with httpx.Client(base_url=api_url, timeout=10, trust_env=False) as client:
                    health_after = client.get("/health").json()
            except Exception as error:
                ui_log.flush()
                ui_tail = ui_log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise RuntimeError(
                    f"Day 36 acceptance failed: {error}\nUI log:\n{ui_tail}"
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
    assert frozen_before == frozen_after, "Frozen evaluation inputs changed during Day 36"
    checks = {
        "streamlit_server_ready": streamlit_health.strip() == "ok",
        **browser,
        "one_pdf_registered": runtime_result["papers"] == 1,
        "analysis_job_completed": runtime_result["jobs"] == {"completed": 1},
        "overview_data_created": (
            runtime_result["analyzed_page_count"] > 0
            and runtime_result["text_page_count"] > 0
            and runtime_result["chunk_count"] > 0
        ),
        "question_runtime_prepared": (
            runtime_result["question_runtime_preparation"]["status"] == "ready"
        ),
    }
    failed_checks = {
        key: value
        for key, value in checks.items()
        if key != "metric_values" and value is not True
    }
    assert not failed_checks, f"Day 36 browser checks failed: {failed_checks}"
    result = {
        "roadmap_day": 36,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 378,
        "roadmap_gate": "browser PDF upload -> analysis status -> overview -> question input",
        "tests_run": test_result.testsRun,
        "test_failures": len(test_result.failures),
        "test_errors": len(test_result.errors),
        "transport": "headless Chromium against Streamlit and real loopback FastAPI",
        "browser_executable": browser_path.name,
        "checks": checks,
        "runtime": runtime_result,
        "api_health_after": health_after,
        "frozen_inputs_unchanged": True,
        "frozen_input_sha256": frozen_after,
        "implementation_sha256": implementation_hashes(),
        "temporary_runtime_removed": True,
        "quality_claim": "functional UI integration only; answer quality is unchanged",
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tests": result["tests_run"],
        "browser_gate": all(
            value is True for key, value in checks.items() if key != "metric_values"
        ),
        "filename": runtime_result["filename"],
        "pages": runtime_result["analyzed_page_count"],
        "chunks": runtime_result["chunk_count"],
        "seed": result["seed"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
