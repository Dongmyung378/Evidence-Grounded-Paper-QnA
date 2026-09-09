"""Day 34 API: run with python -m uvicorn app.main:app --workers 1."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .models import (
    AnalyzeRequest,
    HealthResult,
    JobResult,
    PaperResult,
    QuestionRequest,
    QuestionResult,
    UploadResult,
)
from .service import PaperStateError, Service, Settings, UploadError
from .storage import QueueFull


LOGGER = logging.getLogger(__name__)


class BodyLimit:
    """Bound multipart bytes before parsing, including chunked requests."""

    def __init__(self, app, maximum):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.maximum if scope["path"].rstrip("/") == "/upload" else 64 * 1024
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = limit + 1
        if declared < 0 or declared > limit:
            return await self.reject(scope, receive, send)
        with SpooledTemporaryFile(max_size=1024 * 1024) as body:
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                data = message.get("body", b"")
                size += len(data)
                if size > limit:
                    return await self.reject(scope, receive, send)
                await run_in_threadpool(body.write, data)
                if not message.get("more_body", False):
                    break
            body.seek(0)
            delivered = False
            async def replay():
                nonlocal delivered
                if delivered:
                    return await receive()
                block = await run_in_threadpool(body.read, 64 * 1024)
                delivered = body.tell() >= size
                return {"type": "http.request", "body": block, "more_body": not delivered}
            await self.app(scope, replay, send)

    @staticmethod
    async def reject(scope, receive, send):
        await JSONResponse(status_code=413, content={"detail": {
            "code": "request_too_large", "message": "Request exceeds the allowed size."}
        })(scope, receive, send)


def create_app(settings=None, question_engine=None):
    if settings is None:
        prepare_runtime = os.environ.get(
            "PAPER_QNA_PREPARE_MODELS", "1"
        ).strip().lower() not in {"0", "false", "no"}
        settings = Settings(
            data_dir=Path(os.environ.get("PAPER_QNA_DATA_DIR", Settings().data_dir)),
            prepare_question_runtime=prepare_runtime,
        )

    @asynccontextmanager
    async def lifespan(api):
        service = await run_in_threadpool(Service, settings, question_engine)
        api.state.service = service
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    api = FastAPI(title="Evidence Grounded Paper Q&A", version="0.34.0", lifespan=lifespan)
    api.add_middleware(BodyLimit, maximum=settings.max_upload_bytes + 64 * 1024)

    @api.post("/upload", response_model=UploadResult, status_code=201,
              summary="Register one text-based English PDF")
    async def upload(request: Request, files: Annotated[list[UploadFile], File(alias="file")]):
        form = await request.form()
        if len(files) != 1 or len(form.multi_items()) != 1:
            raise HTTPException(422, detail={"code": "one_file_required", "message": "Upload exactly one file."})
        try:
            return await run_in_threadpool(request.app.state.service.upload, files[0])
        except UploadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code, "message": exc.message}) from None
        except Exception:
            raise HTTPException(500, detail={"code": "storage_error", "message": "The PDF could not be saved."}) from None

    @api.post("/analyze", response_model=JobResult, status_code=202,
              summary="Queue parsing and chunking for an uploaded paper")
    def analyze(body: AnalyzeRequest, request: Request, response: Response):
        try:
            job = request.app.state.service.analyze(body.paper_id)
        except KeyError:
            raise HTTPException(404, detail={"code": "paper_not_found", "message": "Upload the paper first."}) from None
        except QueueFull:
            raise HTTPException(503, detail={"code": "queue_full", "message": "Analysis queue is full. Retry shortly."},
                                headers={"Retry-After": "5"}) from None
        response.headers["Location"] = f"/jobs/{job['job_id']}"
        return job

    @api.get("/jobs/{job_id}", response_model=JobResult, summary="Read analysis job status")
    def job_status(job_id: str, request: Request):
        try:
            return request.app.state.service.store.job(job_id)
        except KeyError:
            raise HTTPException(404, detail={"code": "job_not_found", "message": "Analysis job not found."}) from None

    @api.get("/papers/{paper_id}", response_model=PaperResult,
             summary="Read the uploaded paper and analysis result")
    def paper_result(paper_id: str, request: Request):
        try:
            return request.app.state.service.paper(paper_id)
        except KeyError:
            raise HTTPException(404, detail={"code": "paper_not_found", "message": "Paper not found."}) from None

    @api.post("/question", response_model=QuestionResult,
              summary="Answer one Korean or English question with paper evidence")
    async def question(body: QuestionRequest, request: Request):
        try:
            return await run_in_threadpool(
                request.app.state.service.ask,
                body.paper_id,
                body.question,
            )
        except KeyError:
            raise HTTPException(404, detail={"code": "paper_not_found", "message": "Paper not found."}) from None
        except PaperStateError as exc:
            headers = {"Retry-After": "2"} if exc.code == "analysis_in_progress" else None
            raise HTTPException(
                exc.status,
                detail={"code": exc.code, "message": exc.message},
                headers=headers,
            ) from None
        except Exception:
            LOGGER.exception("Question request failed for paper %s", body.paper_id)
            raise HTTPException(
                503,
                detail={
                    "code": "question_service_unavailable",
                    "message": "The local question service could not produce an answer.",
                },
                headers={"Retry-After": "5"},
            ) from None

    @api.get("/health", response_model=HealthResult, summary="Check API readiness")
    def health(request: Request):
        return request.app.state.service.health()

    return api


app = create_app()
