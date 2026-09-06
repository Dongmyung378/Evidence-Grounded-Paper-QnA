"""Exercise a real local HTTP server and record the Day 33 acceptance evidence."""

import hashlib
import json
import socket
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app
from app.service import Settings


OUTPUT = ROOT / "data" / "evaluation" / "day33_api_results.json"
SOURCE_PDF = ROOT / "data" / "raw" / "papers" / "paper-003.pdf"
FROZEN_FILES = ["config/production_retrieval.json", "config/generation.json",
                "config/abstention.json", "data/processed/chunks.jsonl",
                "data/processed/pages.jsonl", "data/evaluation/questions.jsonl",
                "data/evaluation/gold_evidence.jsonl"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    paths = sorted((ROOT / "app").glob("*.py")) + sorted((ROOT / "tests").glob("test_*.py")) + [
        ROOT / "config" / "reproducibility.json",
        ROOT / "scripts" / "ingest_papers.py", Path(__file__).resolve(),
        ROOT / "scripts" / "answer_quality.py", ROOT / "scripts" / "qna_pipeline.py",
        ROOT / "scripts" / "abstention_policy.py", ROOT / "scripts" / "local_llm.py",
        ROOT / "scripts" / "build_day32_review.py",
        ROOT / "config" / "runtime_qna.json", ROOT / "config" / "abstention_candidate.json",
    ]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def run_http_smoke(root):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(Settings(data_dir=root)), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError("Local HTTP server failed to start")
            time.sleep(0.01)
        started = time.perf_counter()
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False) as client:
            with SOURCE_PDF.open("rb") as source:
                response = client.post("/upload", files={"file": (SOURCE_PDF.name, source, "application/pdf")})
            assert response.status_code == 201, response.text
            uploaded = response.json()
            response = client.post("/analyze", json={"paper_id": uploaded["paper_id"]})
            assert response.status_code == 202, response.text
            accepted = response.json()
            deadline = time.monotonic() + 60
            states = [accepted["status"]]
            while True:
                response = client.get(response.headers.get("location", f"/jobs/{accepted['job_id']}"))
                assert response.status_code == 200, response.text
                completed = response.json()
                states.append(completed["status"])
                if completed["status"] in {"completed", "failed"}:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError("Analysis timed out")
                time.sleep(0.02)
            assert completed["status"] == "completed", completed
            directory = root / "papers" / uploaded["paper_id"]
            pages = [json.loads(x) for x in (directory / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
            chunks = [json.loads(x) for x in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            page_ids = {x["page_id"] for x in pages}
            assert all(x["source_page_id"] in page_ids and x["paper_id"] == uploaded["paper_id"] for x in chunks)
            assert len(chunks) == len({x["chunk_id"] for x in chunks}) > 0
            assert all(x["text"].strip() and x["page"] > 0 for x in chunks)
            assert completed["page_count"] == uploaded["page_count"] == 10
            assert completed["text_page_count"] == len(pages)
            assert completed["chunk_count"] == len(chunks)
            repeated = client.post("/analyze", json={"paper_id": uploaded["paper_id"]}).json()
            assert repeated["job_id"] == completed["job_id"]
            assert repeated["status"] == "completed"
        return {"transport": "real TCP HTTP on loopback with Uvicorn",
                "source_pdf": SOURCE_PDF.relative_to(ROOT).as_posix(),
                "source_sha256": digest(SOURCE_PDF), "upload_status_code": 201,
                "analyze_status_code": 202, "upload": uploaded, "job": completed,
                "observed_states": states, "traceability_passed": True,
                "duplicate_analyze_reused_job": True,
                "runtime_seconds": round(time.perf_counter() - started, 3)}
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("Local HTTP server did not shut down")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    before = {name: digest(ROOT / name) for name in FROZEN_FILES}
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    with TemporaryDirectory(prefix="paper-qna-http-") as directory:
        smoke = run_http_smoke(Path(directory))
    after = {name: digest(ROOT / name) for name in FROZEN_FILES}
    assert before == after, "Frozen corpus or evaluation inputs changed"
    payload = {"roadmap_day": 33, "evaluated_at": datetime.now(timezone.utc).isoformat(),
               "seed": Settings().seed, "tests_run": result.testsRun,
               "test_failures": len(result.failures), "test_errors": len(result.errors),
               "http_smoke": smoke, "frozen_inputs_unchanged": before == after,
               "frozen_input_sha256": after, "implementation_sha256": implementation_hashes(),
               "temporary_runtime_removed": True,
               "scope": "upload registration and parsing/chunking jobs; no retrieval index or Q&A API yet"}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tests": result.testsRun, "http": smoke, "seed": Settings().seed}, ensure_ascii=False, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
