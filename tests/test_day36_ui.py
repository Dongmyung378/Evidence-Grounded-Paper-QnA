"""Day 36 UI client contract and Streamlit shell tests."""

import json
import unittest
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from ui.api_client import ApiClientError, PaperQnaClient


ROOT = Path(__file__).resolve().parents[1]
PAPER_ID = "paper-" + "a" * 32
JOB_ID = "job-" + "b" * 32


class ApiClientContract(unittest.TestCase):
    def test_upload_analysis_overview_and_question_flow(self):
        job_reads = 0
        calls = []

        def handler(request):
            nonlocal job_reads
            calls.append((request.method, request.url.path))
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "seed": 378})
            if request.url.path == "/upload":
                self.assertIn(b"sample.pdf", request.content)
                return httpx.Response(201, json={
                    "paper_id": PAPER_ID,
                    "filename": "sample.pdf",
                    "size_bytes": 10,
                    "page_count": 2,
                    "status": "uploaded",
                })
            if request.url.path == "/analyze":
                self.assertEqual(json.loads(request.content)["paper_id"], PAPER_ID)
                return httpx.Response(202, json={
                    "job_id": JOB_ID,
                    "paper_id": PAPER_ID,
                    "status": "queued",
                })
            if request.url.path == f"/jobs/{JOB_ID}":
                job_reads += 1
                return httpx.Response(200, json={
                    "job_id": JOB_ID,
                    "paper_id": PAPER_ID,
                    "status": "running" if job_reads == 1 else "completed",
                    "seed": 378,
                })
            if request.url.path == f"/papers/{PAPER_ID}":
                return httpx.Response(200, json={
                    "paper_id": PAPER_ID,
                    "status": "ready",
                    "page_count": 2,
                    "text_page_count": 2,
                    "chunk_count": 4,
                    "warning_count": 0,
                    "overview": {"abstract": "A short abstract.", "sections": ["Introduction"]},
                })
            if request.url.path == "/question":
                body = json.loads(request.content)
                self.assertEqual(body, {"paper_id": PAPER_ID, "question": "What is new?"})
                return httpx.Response(200, json={
                    "paper_id": PAPER_ID,
                    "question": body["question"],
                    "question_language": "en",
                    "answer": "A grounded answer.",
                    "sufficiency": "sufficient",
                    "abstention_reason": None,
                    "evidence": [],
                    "runtime_seconds": 0.2,
                    "runtime": {
                        "total_seconds": 0.2,
                        "retrieval_seconds": 0.05,
                        "generation_seconds": 0.15,
                        "generation_attempts": 1,
                        "llm_device": "cuda",
                        "device_fallback_reason": None,
                        "fallback_used": False,
                        "abstention_source": None,
                    },
                })
            return httpx.Response(404, json={"detail": "not found"})

        states = []
        with PaperQnaClient(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        ) as client:
            self.assertEqual(client.health()["seed"], 378)
            uploaded = client.upload("sample.pdf", b"%PDF-test")
            queued = client.analyze(uploaded["paper_id"])
            job = client.wait_for_job(
                queued["job_id"],
                poll_interval=0,
                on_update=lambda value: states.append(value["status"]),
            )
            paper = client.paper(uploaded["paper_id"])
            answer = client.ask(uploaded["paper_id"], "What is new?")

        self.assertEqual(states, ["running", "completed"])
        self.assertEqual(job["seed"], 378)
        self.assertEqual(paper["overview"]["abstract"], "A short abstract.")
        self.assertEqual(answer["answer"], "A grounded answer.")
        self.assertEqual(calls[0], ("GET", "/health"))

    def test_structured_api_error_is_preserved(self):
        def handler(request):
            return httpx.Response(413, json={
                "detail": {"code": "file_too_large", "message": "Too large."}
            })

        with PaperQnaClient(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        ) as client:
            with self.assertRaises(ApiClientError) as caught:
                client.upload("large.pdf", b"large")
        self.assertEqual(caught.exception.code, "file_too_large")
        self.assertEqual(caught.exception.status_code, 413)


class StreamlitShell(unittest.TestCase):
    def test_initial_screen_has_day36_controls(self):
        app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=10).run()
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(app.title[0].value, "Evidence-Grounded Paper Q&A")
        self.assertEqual(len(app.get("file_uploader")), 1)
        labels = [button.label for button in app.button]
        self.assertIn("Upload and analyze", labels)
        self.assertIn("Check API", labels)
        self.assertEqual(app.text_area[0].label, "Ask a question about this paper")
        self.assertTrue(app.text_area[0].disabled)

    def test_korean_screen_copy(self):
        app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=10).run()
        app.selectbox[0].select("한국어").run()
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(app.title[0].value, "근거 기반 논문 질의응답")
        labels = [button.label for button in app.button]
        self.assertIn("업로드하고 분석", labels)
        self.assertEqual(app.text_area[0].label, "이 논문에 관해 질문하세요")


if __name__ == "__main__":
    unittest.main()
