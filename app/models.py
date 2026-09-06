"""Public API schemas; no filesystem paths or parser exceptions are exposed."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str = Field(pattern=r"^paper-[0-9a-f]{32}$")


class UploadResult(BaseModel):
    paper_id: str
    filename: str
    size_bytes: int
    page_count: int
    status: Literal["uploaded"] = "uploaded"


class JobResult(BaseModel):
    job_id: str
    paper_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: Literal["parsing_and_chunking"] = "parsing_and_chunking"
    seed: int
    created_at: str
    updated_at: str
    page_count: int | None = None
    text_page_count: int | None = None
    chunk_count: int | None = None
    warning_count: int | None = None
    error_code: str | None = None
    error_message: str | None = None
