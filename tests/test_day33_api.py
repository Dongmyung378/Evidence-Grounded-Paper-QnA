"""HTTP contract, real parser, isolation, size limits, and durable queue tests."""

import io
import json
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.main import create_app
from app.service import Settings, Service


SEED = 378


def pdf_bytes(text="Abstract. This paper describes evidence grounded text retrieval.", encrypted=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("test-password")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def await_job(client, job_id, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Job did not terminate within the test timeout")


class Day33API(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="paper-qna-test-")
        self.settings = Settings(data_dir=Path(self.temp.name))
        self.api = create_app(self.settings)
        self.client = TestClient(self.api).__enter__()
        self.pdf = pdf_bytes()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self, content=None, name="test.pdf", mime="application/pdf"):
        return self.client.post("/upload", files={"file": (name, self.pdf if content is None else content, mime)})

    def test_real_parser_and_traceability(self):
        upload = self.upload()
        self.assertEqual(upload.status_code, 201)
        paper = upload.json()
        self.assertEqual(paper["page_count"], 1)
        response = self.client.post("/analyze", json={"paper_id": paper["paper_id"]})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        self.assertEqual(response.headers["location"], f"/jobs/{job_id}")
        result = await_job(self.client, job_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["seed"], SEED)
        self.assertEqual(result["chunk_count"], 1)
        directory = self.settings.data_dir / "papers" / paper["paper_id"]
        pages = [json.loads(x) for x in (directory / "pages.jsonl").read_text().splitlines()]
        chunks = [json.loads(x) for x in (directory / "chunks.jsonl").read_text().splitlines()]
        self.assertEqual(chunks[0]["source_page_id"], pages[0]["page_id"])
        self.assertEqual(chunks[0]["paper_id"], paper["paper_id"])
        self.assertIn("evidence grounded", chunks[0]["text"])
        self.assertNotIn(str(directory), response.text + json.dumps(result))

    def test_invalid_uploads_are_cleaned(self):
        cases = [(b"", "x.pdf", "application/pdf", 422, "empty_file"),
                 (b"bad", "x.pdf", "application/pdf", 422, "invalid_pdf"),
                 (b"%PDF-1.7 broken", "x.pdf", "application/pdf", 422, "invalid_pdf"),
                 (self.pdf, "x.txt", "application/pdf", 415, "invalid_file_type"),
                 (self.pdf, "x.pdf", "text/plain", 415, "invalid_file_type"),
                 (pdf_bytes(text=""), "x.pdf", "application/pdf", 422, "no_extractable_text"),
                 (pdf_bytes(encrypted=True), "x.pdf", "application/pdf", 422, "encrypted_pdf")]
        random.Random(SEED).shuffle(cases)
        for data, filename, mime, status, code in cases:
            with self.subTest(code=code, filename=filename):
                response = self.upload(data, filename, mime)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["detail"]["code"], code)
        with self.api.state.service.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM papers").fetchone()[0], 0)
        self.assertEqual(list((self.settings.data_dir / "papers").iterdir()), [])

    def test_exactly_one_file_and_schema(self):
        self.assertEqual(self.client.post("/upload").status_code, 422)
        response = self.client.post("/upload", files=[("file", ("a.pdf", self.pdf)), ("file", ("b.pdf", self.pdf))])
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/upload", files=[("file", ("a.pdf", self.pdf)), ("other", ("b.pdf", self.pdf))])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/analyze", json={"paper_id": "../outside"}).status_code, 422)
        self.assertEqual(self.client.post("/analyze", json={"paper_id": "paper-" + "a" * 32}).status_code, 404)
        self.assertEqual(self.client.get("/jobs/unknown").status_code, 404)

    def test_path_filename_and_two_paper_isolation(self):
        a = self.upload(name="../../../../outside.pdf").json()
        b = self.upload(pdf_bytes("Second paper has distinct source text.")).json()
        self.assertEqual(a["filename"], "outside.pdf")
        self.assertNotEqual(a["paper_id"], b["paper_id"])
        for paper in (a, b):
            job = self.client.post("/analyze", json={"paper_id": paper["paper_id"]}).json()
            self.assertEqual(await_job(self.client, job["job_id"])["status"], "completed")
            path = self.settings.data_dir / "papers" / paper["paper_id"] / "chunks.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertTrue(all(x["paper_id"] == paper["paper_id"] for x in rows))
        self.assertFalse((self.settings.data_dir / "outside.pdf").exists())

    def test_file_boundary_and_chunked_request_limit(self):
        with TemporaryDirectory() as tmp:
            settings = Settings(data_dir=Path(tmp), max_upload_bytes=2048)
            with TestClient(create_app(settings)) as client:
                exact = self.pdf + b" " * (2048 - len(self.pdf))
                self.assertEqual(client.post("/upload", files={"file": ("x.pdf", exact)}).status_code, 201)
                over = client.post("/upload", files={"file": ("x.pdf", exact + b" ")})
                self.assertEqual(over.status_code, 413)
                self.assertEqual(over.json()["detail"]["code"], "file_too_large")
                chunked = client.post("/upload", content=iter([b"x" * 40000, b"x" * 40000]),
                                      headers={"content-type": "application/octet-stream"})
                self.assertEqual(chunked.status_code, 413)
                declared = client.post("/upload", content=b"x", headers={"content-length": "99999999"})
                self.assertEqual(declared.status_code, 413)

    def test_concurrent_analyze_is_idempotent_and_nonblocking(self):
        paper_id = self.upload().json()["paper_id"]
        started, release = threading.Event(), threading.Event()
        from app.service import process_paper as real_process
        def blocking(*args, **kwargs):
            started.set()
            if not release.wait(10):
                raise RuntimeError("Test failed to release parser")
            return real_process(*args, **kwargs)
        with patch("app.service.process_paper", side_effect=blocking) as parser:
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    responses = list(pool.map(lambda _: self.client.post("/analyze", json={"paper_id": paper_id}), range(8)))
                self.assertTrue(started.wait(3))
                self.assertTrue(all(x.status_code == 202 for x in responses))
                ids = {x.json()["job_id"] for x in responses}
                self.assertEqual(len(ids), 1)
                job_id = ids.pop()
                self.assertEqual(self.client.get(f"/jobs/{job_id}").json()["status"], "running")
                self.assertEqual(parser.call_count, 1)
            finally:
                release.set()
            self.assertEqual(await_job(self.client, job_id)["status"], "completed")
        repeated = self.client.post("/analyze", json={"paper_id": paper_id}).json()
        self.assertEqual(repeated["job_id"], job_id)
        self.assertEqual(repeated["status"], "completed")

    def test_worker_failure_is_sanitized_and_retryable(self):
        paper_id = self.upload().json()["paper_id"]
        with patch("app.service.process_paper", side_effect=RuntimeError("secret C:/private/path")):
            job = self.client.post("/analyze", json={"paper_id": paper_id}).json()
            failed = await_job(self.client, job["job_id"])
            self.assertEqual(failed["status"], "failed")
            self.assertNotIn("secret", json.dumps(failed))
        retry = self.client.post("/analyze", json={"paper_id": paper_id}).json()
        self.assertEqual(retry["job_id"], job["job_id"])
        self.assertEqual(await_job(self.client, job["job_id"])["status"], "completed")

    def test_partial_page_failure_is_not_reported_as_complete(self):
        paper_id = self.upload().json()["paper_id"]
        result = ([{"text": "partial"}], [{"text": "partial"}],
                  {"status": "success", "warnings": [{"code": "page_text_extraction_failed"}]})
        with patch("app.service.process_paper", return_value=result):
            job = self.client.post("/analyze", json={"paper_id": paper_id}).json()
            failed = await_job(self.client, job["job_id"])
            self.assertEqual(failed["error_code"], "partial_extraction_failed")
            self.assertEqual(failed["status"], "failed")

    def test_second_service_cannot_share_runtime(self):
        with self.assertRaisesRegex(RuntimeError, "one API worker"):
            Service(self.settings)

    def test_restart_recovers_interrupted_and_queued_jobs(self):
        papers = [self.upload().json()["paper_id"] for _ in range(2)]
        self.client.__exit__(None, None, None)
        store = self.api.state.service.store
        a, _ = store.enqueue(papers[0], SEED)
        b, _ = store.enqueue(papers[1], SEED)
        store.claim(a["job_id"])
        self.client = TestClient(create_app(self.settings)).__enter__()
        for job in (a, b):
            self.assertEqual(await_job(self.client, job["job_id"])["status"], "completed")
        # Completed status also survives another service startup.
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(self.settings)).__enter__()
        self.assertEqual(self.client.get(f"/jobs/{a['job_id']}").json()["status"], "completed")


if __name__ == "__main__":
    unittest.main()
