"""Run the Day 34 tests and prove a real curl question over local HTTP."""

import hashlib
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID

import httpx
import uvicorn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app
from app.service import Settings


OUTPUT = ROOT / "data" / "evaluation" / "day34_api_results.json"
SOURCE_PDF = ROOT / "data" / "raw" / "papers" / "paper-003.pdf"
QUESTION = "What is the main focus of this paper's review of text summarization?"
FROZEN_FILES = [
    "config/production_retrieval.json",
    "config/generation.json",
    "config/abstention_candidate.json",
    "config/runtime_qna.json",
    "data/processed/chunks.jsonl",
    "data/processed/pages.jsonl",
    "data/evaluation/questions.jsonl",
    "data/evaluation/gold_evidence.jsonl",
]
IMPLEMENTATION_FILES = [
    "app/main.py",
    "app/models.py",
    "app/question_engine.py",
    "app/service.py",
    "app/storage.py",
    "scripts/bm25_retrieval.py",
    "scripts/candidate_evidence_pipeline.py",
    "scripts/dense_retrieval.py",
    "scripts/evidence_selector.py",
    "scripts/evaluate_day34.py",
    "scripts/hybrid_retrieval.py",
    "scripts/production_retrieval.py",
    "scripts/qna_pipeline.py",
    "scripts/reranker.py",
    "scripts/validate_day34.py",
    "tests/test_day33_api.py",
    "tests/test_day34_api.py",
    "tests/test_improvements.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    return {name: digest(ROOT / name) for name in IMPLEMENTATION_FILES}


def wait_for_server(server, thread, timeout=15):
    deadline = time.monotonic() + timeout
    while not server.started:
        if time.monotonic() >= deadline or not thread.is_alive():
            raise RuntimeError("Local HTTP server failed to start")
        time.sleep(0.01)


def wait_for_job(client, location, timeout=90):
    deadline = time.monotonic() + timeout
    states = []
    while time.monotonic() < deadline:
        response = client.get(location)
        assert response.status_code == 200, response.text
        job = response.json()
        states.append(job["status"])
        if job["status"] in {"completed", "failed"}:
            return job, states
        time.sleep(0.05)
    raise AssertionError("Analysis job timed out")


def curl_question(base_url, payload):
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if executable is None:
        raise RuntimeError("curl is required for the Day 34 acceptance check")
    command = [
        executable,
        "-sS",
        "--max-time",
        "300",
        "-X",
        "POST",
        f"{base_url}/question",
        "-H",
        "Content-Type: application/json",
        "--data-raw",
        json.dumps(payload, ensure_ascii=False),
        "-w",
        "\n%{http_code}",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=310,
    )
    body, status = completed.stdout.rsplit("\n", 1)
    return int(status), json.loads(body)


def run_http_smoke(runtime_root):
    socket_handle = socket.socket()
    socket_handle.bind(("127.0.0.1", 0))
    port = socket_handle.getsockname()[1]
    settings = Settings(
        data_dir=runtime_root,
        seed=378,
        model_local_files_only=True,
    )
    fixed_paper_uuid = UUID(int=378)
    with patch("app.service.uuid4", return_value=fixed_paper_uuid):
        server = uvicorn.Server(
            uvicorn.Config(create_app(settings), log_level="warning")
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [socket_handle]},
            daemon=True,
        )
        thread.start()
        try:
            wait_for_server(server, thread)
            base_url = f"http://127.0.0.1:{port}"
            started = time.perf_counter()
            with httpx.Client(base_url=base_url, timeout=90, trust_env=False) as client:
                health_before = client.get("/health")
                assert health_before.status_code == 200, health_before.text
                assert health_before.json()["question_engine"] == "lazy"

                with SOURCE_PDF.open("rb") as source:
                    uploaded_response = client.post(
                        "/upload",
                        files={"file": (SOURCE_PDF.name, source, "application/pdf")},
                    )
                assert uploaded_response.status_code == 201, uploaded_response.text
                uploaded = uploaded_response.json()

                analysis_response = client.post(
                    "/analyze",
                    json={"paper_id": uploaded["paper_id"]},
                )
                assert analysis_response.status_code == 202, analysis_response.text
                location = analysis_response.headers["location"]
                job, states = wait_for_job(client, location)
                assert job["status"] == "completed", job

                paper_response = client.get(f"/papers/{uploaded['paper_id']}")
                assert paper_response.status_code == 200, paper_response.text
                paper = paper_response.json()
                assert paper["status"] == "ready"
                assert paper["overview"]["abstract"]
                assert paper["overview"]["sections"]

            question_status, answer = curl_question(
                base_url,
                {"paper_id": uploaded["paper_id"], "question": QUESTION},
            )
            assert question_status == 200, answer
            assert answer["paper_id"] == uploaded["paper_id"]
            assert answer["question_language"] == "en"
            assert answer["answer"].strip()
            assert answer["sufficiency"] == "sufficient", answer
            assert answer["evidence"], answer
            assert all(
                item["page"] >= 1 and item["text"].strip()
                for item in answer["evidence"]
            )

            with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
                health_after = client.get("/health")
                assert health_after.status_code == 200, health_after.text
                assert health_after.json()["question_engine"] == "ready"

            paper_directory = runtime_root / "papers" / uploaded["paper_id"]
            assert (paper_directory / "dense_embeddings.npz").is_file()
            return {
                "transport": "curl over real TCP HTTP on loopback with Uvicorn",
                "source_pdf": SOURCE_PDF.relative_to(ROOT).as_posix(),
                "source_sha256": digest(SOURCE_PDF),
                "health_before": health_before.json(),
                "upload_status_code": uploaded_response.status_code,
                "analyze_status_code": analysis_response.status_code,
                "paper_result_status_code": paper_response.status_code,
                "question_status_code": question_status,
                "upload": uploaded,
                "job": job,
                "observed_job_states": states,
                "paper_result": paper,
                "question_result": answer,
                "health_after": health_after.json(),
                "paper_id_fixed_for_reproducible_acceptance": True,
                "runtime_embedding_cache_created": True,
                "runtime_seconds": round(time.perf_counter() - started, 3),
            }
        finally:
            server.should_exit = True
            thread.join(timeout=30)
            socket_handle.close()
            if thread.is_alive():
                raise RuntimeError("Local HTTP server did not shut down")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    before = {name: digest(ROOT / name) for name in FROZEN_FILES}
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"),
        pattern="test_*.py",
    )
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise SystemExit(1)
    with TemporaryDirectory(prefix="paper-qna-day34-http-") as directory:
        smoke = run_http_smoke(Path(directory))
    after = {name: digest(ROOT / name) for name in FROZEN_FILES}
    assert before == after, "Frozen corpus or evaluation inputs changed"
    payload = {
        "roadmap_day": 34,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 378,
        "tests_run": test_result.testsRun,
        "test_failures": len(test_result.failures),
        "test_errors": len(test_result.errors),
        "http_smoke": smoke,
        "frozen_inputs_unchanged": before == after,
        "frozen_input_sha256": after,
        "implementation_sha256": implementation_hashes(),
        "temporary_runtime_removed": True,
        "scope": "health, paper result, and one-paper grounded question API",
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tests": test_result.testsRun,
        "http_transport": smoke["transport"],
        "question_status": smoke["question_status_code"],
        "sufficiency": smoke["question_result"]["sufficiency"],
        "evidence_pages": [
            item["page"] for item in smoke["question_result"]["evidence"]
        ],
        "seed": 378,
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
