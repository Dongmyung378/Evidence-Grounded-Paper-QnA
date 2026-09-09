"""Day 34 paper-result, question, health, and runtime retrieval tests."""

import io
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.main import create_app
from app.service import Settings


SEED = 378


def pdf_bytes(text="ABSTRACT Evidence grounded question answering uses paper text."):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): writer._add_object(font),
        }),
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def await_job(client, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get(f"/jobs/{job_id}").json()
        if result["status"] in {"completed", "failed"}:
            return result
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


class FakeQuestionEngine:
    def __init__(self):
        self.status = "lazy"
        self.calls = []

    def ask(self, question, paper_id, directory):
        self.status = "ready"
        self.calls.append((question, paper_id, Path(directory)))
        language = "ko" if any("가" <= char <= "힣" for char in question) else "en"
        answer = "논문 텍스트가 답을 뒷받침합니다." if language == "ko" else "The paper text supports the answer."
        evidence = {
            "evidence_id": f"ev-01-{paper_id}-p001-c001",
            "page": 1,
            "section": "Abstract",
            "text": "Evidence grounded question answering uses paper text.",
            "locator": {
                "page_label": "p. 1",
                "section": "Abstract",
                "chunk_id": f"{paper_id}-p001-c001",
            },
        }
        return {
            "question_language": language,
            "response": {
                "answer": answer,
                "evidence_ids": [evidence["evidence_id"]],
                "sufficiency": "sufficient",
                "abstention_reason": None,
            },
            "cited_evidence": [evidence],
            "pipeline": {
                "runtime_seconds": 0.012,
                "retrieval_seconds": 0.007,
                "generation_seconds": 0.005,
                "generation_attempts": 1,
                "fallback_used": False,
                "abstention_source": None,
                "llm": {
                    "device": "cuda",
                    "device_fallback_reason": None,
                },
            },
        }


class FailingQuestionEngine:
    status = "lazy"

    def ask(self, question, paper_id, directory):
        raise RuntimeError("secret C:/private/model/path")


class Day34API(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="paper-qna-day34-")
        self.settings = Settings(data_dir=Path(self.temp.name))
        self.engine = FakeQuestionEngine()
        self.api = create_app(self.settings, question_engine=self.engine)
        self.client = TestClient(self.api).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self):
        response = self.client.post(
            "/upload",
            files={"file": ("sample.pdf", pdf_bytes(), "application/pdf")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def analyze(self, paper_id):
        accepted = self.client.post("/analyze", json={"paper_id": paper_id})
        self.assertEqual(accepted.status_code, 202, accepted.text)
        result = await_job(self.client, accepted.json()["job_id"])
        self.assertEqual(result["status"], "completed", result)
        return result

    def test_health_paper_result_and_english_question(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["version"], "0.34.0")
        self.assertEqual(health.json()["seed"], SEED)
        self.assertEqual(health.json()["question_engine"], "lazy")
        self.assertEqual(health.json()["papers"], 0)
        self.assertEqual(health.json()["jobs"], {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        })

        uploaded = self.upload()
        before = self.client.get(f"/papers/{uploaded['paper_id']}")
        self.assertEqual(before.status_code, 200)
        self.assertEqual(before.json()["status"], "uploaded")
        self.assertIsNone(before.json()["analysis"])
        self.assertIsNone(before.json()["overview"])

        blocked = self.client.post(
            "/question",
            json={"paper_id": uploaded["paper_id"], "question": "What is supported?"},
        )
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.json()["detail"]["code"], "analysis_required")

        job = self.analyze(uploaded["paper_id"])
        paper = self.client.get(f"/papers/{uploaded['paper_id']}").json()
        self.assertEqual(paper["status"], "ready")
        self.assertEqual(paper["chunk_count"], job["chunk_count"])
        self.assertEqual(paper["analysis"]["job_id"], job["job_id"])
        self.assertIsInstance(paper["overview"]["sections"], list)
        self.assertNotIn(str(self.settings.data_dir), json.dumps(paper))

        response = self.client.post(
            "/question",
            json={"paper_id": uploaded["paper_id"], "question": "What is supported?"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertEqual(answer["question_language"], "en")
        self.assertEqual(answer["sufficiency"], "sufficient")
        self.assertEqual(answer["evidence"][0]["page"], 1)
        self.assertEqual(answer["evidence"][0]["locator"]["page_label"], "p. 1")
        self.assertEqual(answer["runtime"]["generation_attempts"], 1)
        self.assertEqual(answer["runtime"]["llm_device"], "cuda")
        self.assertEqual(answer["runtime"]["retrieval_seconds"], 0.007)
        self.assertNotIn(str(self.settings.data_dir), response.text)
        final_health = self.client.get("/health").json()
        self.assertEqual(final_health["question_engine"], "ready")
        self.assertEqual(final_health["papers"], 1)
        self.assertEqual(final_health["jobs"]["completed"], 1)

    def test_korean_question_preserves_language(self):
        uploaded = self.upload()
        self.analyze(uploaded["paper_id"])
        response = self.client.post(
            "/question",
            json={"paper_id": uploaded["paper_id"], "question": "무엇을 뒷받침하나요?"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question_language"], "ko")
        self.assertIn("논문", response.json()["answer"])

    def test_question_errors_are_stable_and_sanitized(self):
        unknown_id = "paper-" + "a" * 32
        self.assertEqual(self.client.get(f"/papers/{unknown_id}").status_code, 404)
        self.assertEqual(
            self.client.post(
                "/question",
                json={"paper_id": unknown_id, "question": "What is supported?"},
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                "/question",
                json={"paper_id": unknown_id, "question": "   "},
            ).status_code,
            422,
        )
        uploaded = self.upload()
        self.analyze(uploaded["paper_id"])
        self.api.state.service.question_engine = FailingQuestionEngine()
        failed = self.client.post(
            "/question",
            json={"paper_id": uploaded["paper_id"], "question": "What is supported?"},
        )
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(failed.json()["detail"]["code"], "question_service_unavailable")
        self.assertNotIn("secret", failed.text)
        self.assertEqual(failed.headers["retry-after"], "5")

    def test_running_and_failed_analysis_block_questions(self):
        uploaded = self.upload()
        started = threading.Event()
        release = threading.Event()
        from app.service import process_paper as real_process

        def blocking_parser(*args, **kwargs):
            started.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release parser")
            return real_process(*args, **kwargs)

        with patch("app.service.process_paper", side_effect=blocking_parser):
            accepted = self.client.post(
                "/analyze",
                json={"paper_id": uploaded["paper_id"]},
            )
            self.assertTrue(started.wait(3))
            blocked = self.client.post(
                "/question",
                json={"paper_id": uploaded["paper_id"], "question": "What is supported?"},
            )
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["detail"]["code"], "analysis_in_progress")
            self.assertEqual(blocked.headers["retry-after"], "2")
            release.set()
            await_job(self.client, accepted.json()["job_id"])

        failed_upload = self.upload()
        with patch("app.service.process_paper", side_effect=RuntimeError("parser failed")):
            accepted = self.client.post(
                "/analyze",
                json={"paper_id": failed_upload["paper_id"]},
            )
            failed_job = await_job(self.client, accepted.json()["job_id"])
        self.assertEqual(failed_job["status"], "failed")
        blocked = self.client.post(
            "/question",
            json={"paper_id": failed_upload["paper_id"], "question": "What is supported?"},
        )
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.json()["detail"]["code"], "analysis_failed")


class FakeEmbedding:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("retrieval" in lowered),
                float("unrelated" in lowered),
            ])
        values = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1.0)


class FakeReranker:
    def predict(self, pairs, **kwargs):
        return np.asarray([
            2.0 if "retrieval" in passage.lower() else -2.0
            for _, passage in pairs
        ], dtype=np.float32)


class RuntimeRetrieval(unittest.TestCase):
    def test_production_path_accepts_one_uploaded_paper(self):
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in __import__("sys").path:
            __import__("sys").path.insert(0, scripts)
        from production_retrieval import ProductionRetrieval

        paper_id = "paper-" + "b" * 32
        chunks = [
            {
                "chunk_id": f"{paper_id}-p001-c001",
                "paper_id": paper_id,
                "page": 1,
                "source_page_id": f"{paper_id}-p001",
                "section": "Abstract",
                "text": "Multilingual retrieval finds relevant paper evidence.",
            },
            {
                "chunk_id": f"{paper_id}-p002-c001",
                "paper_id": paper_id,
                "page": 2,
                "source_page_id": f"{paper_id}-p002",
                "section": "Methods",
                "text": "An unrelated control passage.",
            },
        ]
        retrieval = ProductionRetrieval(
            chunks=chunks,
            embedding_cache_path=None,
            embedding_model=FakeEmbedding(),
            reranker=FakeReranker(),
        )
        result = retrieval.run("How does retrieval find evidence?", paper_id)
        self.assertEqual(result["paper_id"], paper_id)
        self.assertEqual(result["evidence"][0]["page"], 1)
        self.assertEqual(result["candidates"][0]["paper_id"], paper_id)
        self.assertEqual(result["production"]["status"], "frozen")


if __name__ == "__main__":
    unittest.main()
