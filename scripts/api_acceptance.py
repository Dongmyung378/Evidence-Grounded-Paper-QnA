"""Reusable real-HTTP acceptance flow for the local one-paper API."""

import hashlib
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import httpx
import uvicorn

from app.main import create_app
from app.service import Settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


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
        raise RuntimeError("curl is required for the API acceptance check")
    completed = subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=310,
    )
    body, status = completed.stdout.rsplit("\n", 1)
    return int(status), json.loads(body)


def validate_traceability(directory, paper_id):
    pages = load_jsonl(directory / "pages.jsonl")
    chunks = load_jsonl(directory / "chunks.jsonl")
    page_ids = {page["page_id"] for page in pages}
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    assert len(page_ids) == len(pages) > 0
    assert len(chunk_ids) == len(chunks) > 0
    assert all(page["paper_id"] == paper_id for page in pages)
    assert all(
        chunk["paper_id"] == paper_id
        and chunk["source_page_id"] in page_ids
        and chunk["text"].strip()
        for chunk in chunks
    )
    return pages, chunks


def run_complete_api_flow(
    runtime_root,
    source_pdf,
    question,
    *,
    seed=378,
    model_local_files_only=True,
):
    runtime_root = Path(runtime_root)
    source_pdf = Path(source_pdf)
    socket_handle = socket.socket()
    socket_handle.bind(("127.0.0.1", 0))
    port = socket_handle.getsockname()[1]
    settings = Settings(
        data_dir=runtime_root,
        seed=seed,
        model_local_files_only=model_local_files_only,
        prepare_question_runtime=True,
    )
    fixed_paper_uuid = UUID(int=seed)
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

                missing_paper = client.post(
                    "/question",
                    json={
                        "paper_id": "paper-" + fixed_paper_uuid.hex,
                        "question": question,
                    },
                )
                assert missing_paper.status_code == 404

                with source_pdf.open("rb") as source:
                    uploaded_response = client.post(
                        "/upload",
                        files={"file": (source_pdf.name, source, "application/pdf")},
                    )
                assert uploaded_response.status_code == 201, uploaded_response.text
                uploaded = uploaded_response.json()

                before_analysis = client.post(
                    "/question",
                    json={"paper_id": uploaded["paper_id"], "question": question},
                )
                assert before_analysis.status_code == 409
                assert before_analysis.json()["detail"]["code"] == "analysis_required"

                analysis_response = client.post(
                    "/analyze",
                    json={"paper_id": uploaded["paper_id"]},
                )
                assert analysis_response.status_code == 202, analysis_response.text
                job, states = wait_for_job(
                    client,
                    analysis_response.headers["location"],
                )
                assert job["status"] == "completed", job

                paper_response = client.get(f"/papers/{uploaded['paper_id']}")
                assert paper_response.status_code == 200, paper_response.text
                paper = paper_response.json()
                assert paper["status"] == "ready"
                assert paper["overview"]["abstract"]
                assert paper["overview"]["sections"]

            paper_directory = runtime_root / "papers" / uploaded["paper_id"]
            ingestion_report = json.loads(
                (paper_directory / "ingestion_report.json").read_text(encoding="utf-8")
            )
            pages, chunks = validate_traceability(
                paper_directory,
                uploaded["paper_id"],
            )
            chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

            question_status, answer = curl_question(
                base_url,
                {"paper_id": uploaded["paper_id"], "question": question},
            )
            assert question_status == 200, answer
            assert answer["paper_id"] == uploaded["paper_id"]
            assert answer["answer"].strip()
            assert answer["sufficiency"] == "sufficient", answer
            assert answer["evidence"], answer
            for evidence in answer["evidence"]:
                chunk_id = evidence["locator"]["chunk_id"]
                assert chunk_id in chunk_by_id
                assert evidence["page"] == chunk_by_id[chunk_id]["page"]
                assert evidence["text"] == chunk_by_id[chunk_id]["text"]

            embedding_cache = paper_directory / "dense_embeddings.npz"
            assert embedding_cache.is_file() and embedding_cache.stat().st_size > 0

            with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
                health_after = client.get("/health")
                assert health_after.status_code == 200, health_after.text
                assert health_after.json()["question_engine"] == "ready"

            return {
                "transport": "curl over real TCP HTTP on loopback with Uvicorn",
                "source_pdf": source_pdf.name,
                "source_sha256": digest(source_pdf),
                "flow_steps": [
                    "health",
                    "upload",
                    "parse",
                    "chunk",
                    "paper_result",
                    "dense_index",
                    "question",
                    "answer",
                ],
                "health_before": health_before.json(),
                "unknown_paper_question_status_code": missing_paper.status_code,
                "question_before_analysis_status_code": before_analysis.status_code,
                "upload_status_code": uploaded_response.status_code,
                "analyze_status_code": analysis_response.status_code,
                "paper_result_status_code": paper_response.status_code,
                "question_status_code": question_status,
                "upload": uploaded,
                "job": job,
                "observed_job_states": states,
                "paper_result": paper,
                "question_runtime_preparation": ingestion_report.get(
                    "question_runtime_preparation"
                ),
                "traceability": {
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "unique_page_ids": len(pages),
                    "unique_chunk_ids": len(chunks),
                    "all_chunks_link_to_source_page": True,
                    "all_citations_link_to_runtime_chunk": True,
                },
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
