"""PDF preflight and serial, restartable ingestion jobs using the existing parser."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from scripts.ingest_papers import process_paper, write_jsonl
from .storage import Store


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = json.loads((ROOT / "config" / "reproducibility.json").read_text(encoding="utf-8"))["default_seed"]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data" / "runtime"
    max_upload_bytes: int = 20 * 1024 * 1024
    seed: int = DEFAULT_SEED
    max_pending_jobs: int = 16


class UploadError(Exception):
    def __init__(self, code, message, status=422):
        self.code, self.message, self.status = code, message, status


class Service:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "worker.lock").open("a+b")
        try:
            self.lock.seek(0)
            if self.lock.read(1) == b"":
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("Runtime directory already in use; use one API worker.") from None
        try:
            self.store = Store(self.root)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-analysis")
            for job_id in self.store.recover():
                self.executor.submit(self.run, job_id)
        except Exception:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=True)
            self.lock.close()
            raise

    def close(self):
        try:
            self.executor.shutdown(wait=True)
        finally:
            self.lock.close()

    def upload(self, file):
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename.lower().endswith(".pdf"):
            raise UploadError("invalid_file_type", "Upload one PDF file.", 415)
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise UploadError("invalid_file_type", "The upload must have a PDF content type.", 415)
        paper_id = "paper-" + uuid4().hex  # Resource identities must remain unique, not seeded.
        directory = self.root / "papers" / paper_id
        directory.mkdir(parents=True)
        pdf = directory / f"{paper_id}.pdf"
        try:
            size = 0
            with pdf.open("xb") as destination:
                while block := file.file.read(64 * 1024):
                    size += len(block)
                    if size > self.settings.max_upload_bytes:
                        raise UploadError("file_too_large", "PDF must be at most 20 MiB.", 413)
                    destination.write(block)
            if not size:
                raise UploadError("empty_file", "The uploaded file is empty.")
            with pdf.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise UploadError("invalid_pdf", "The file is not a valid PDF.")
                try:
                    reader = PdfReader(source)
                    if reader.is_encrypted:
                        raise UploadError("encrypted_pdf", "Password-protected PDFs are not supported.")
                    page_count = len(reader.pages)
                    if not page_count or not any((p.extract_text() or "").strip() for p in reader.pages):
                        raise UploadError("no_extractable_text", "No text was found. Upload a text-based PDF.")
                except UploadError:
                    raise
                except Exception:
                    raise UploadError("invalid_pdf", "The PDF could not be read.") from None
            self.store.register(paper_id, filename, size, page_count)
        except Exception:
            pdf.unlink(missing_ok=True)
            directory.rmdir()  # Only this new upload directory; never a corpus directory.
            raise
        return dict(paper_id=paper_id, filename=filename, size_bytes=size,
                    page_count=page_count, status="uploaded")

    def analyze(self, paper_id):
        job, schedule = self.store.enqueue(paper_id, self.settings.seed, self.settings.max_pending_jobs)
        if schedule:
            self.executor.submit(self.run, job["job_id"])
        return job

    def run(self, job_id):
        if not self.store.claim(job_id):
            return
        job = self.store.job(job_id)
        directory = self.root / "papers" / job["paper_id"]
        try:
            chunks, pages, report = process_paper(
                directory / f"{job['paper_id']}.pdf", chunk_size=1200, overlap=200,
            )
            if report["status"] != "success" or not chunks or not pages:
                self.store.finish(job_id, error_code="analysis_failed",
                                  error_message="No usable text could be extracted from this PDF.")
                return
            if any(w["code"] == "page_text_extraction_failed" for w in report["warnings"]):
                self.store.finish(job_id, error_code="partial_extraction_failed",
                                  error_message="Some PDF pages could not be read. Try another PDF export.")
                return
            for name, rows in (("pages", pages), ("chunks", chunks)):
                temporary = directory / f"{name}.jsonl.tmp"
                write_jsonl(temporary, rows)
                temporary.replace(directory / f"{name}.jsonl")
            report["pdf_path"] = f"{job['paper_id']}.pdf"
            report["seed"] = job["seed"]
            report["config"] = {"chunk_size": 1200, "overlap": 200}
            report_path = directory / "ingestion_report.json.tmp"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report_path.replace(directory / "ingestion_report.json")
            self.store.finish(job_id, page_count=report["page_count"],
                              text_page_count=len(pages), chunk_count=len(chunks),
                              warning_count=len(report["warnings"]))
        except Exception:
            LOGGER.exception("Analysis job %s failed", job_id)
            self.store.finish(job_id, error_code="analysis_failed",
                              error_message="Analysis failed. Retry the analysis request.")
