"""Public API schemas; no filesystem paths or parser exceptions are exposed."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class PaperOverview(BaseModel):
    abstract: str | None = None
    sections: list[str]


class PaperResult(BaseModel):
    paper_id: str
    filename: str
    size_bytes: int
    uploaded_at: str
    status: Literal["uploaded", "queued", "processing", "ready", "failed"]
    page_count: int
    text_page_count: int | None = None
    chunk_count: int | None = None
    warning_count: int | None = None
    analysis: JobResult | None = None
    overview: PaperOverview | None = None


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str = Field(pattern=r"^paper-[0-9a-f]{32}$")
    question: str = Field(min_length=2, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value):
        if not value.strip():
            raise ValueError("question must contain text")
        return value


class EvidenceLocator(BaseModel):
    page_label: str
    section: str | None = None
    chunk_id: str


class EvidenceResult(BaseModel):
    evidence_id: str
    page: int
    section: str | None = None
    text: str
    locator: EvidenceLocator


class QuestionRuntime(BaseModel):
    total_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    generation_attempts: int
    llm_device: Literal["cpu", "cuda"] | None = None
    device_fallback_reason: str | None = None
    fallback_used: bool
    abstention_source: str | None = None


class QuestionResult(BaseModel):
    paper_id: str
    question: str
    question_language: Literal["en", "ko"]
    answer: str
    sufficiency: Literal["sufficient", "insufficient"]
    abstention_reason: str | None = None
    evidence: list[EvidenceResult]
    runtime_seconds: float
    runtime: QuestionRuntime


class QueueHealth(BaseModel):
    queued: int
    running: int
    completed: int
    failed: int


class HealthResult(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["evidence-grounded-paper-qna"] = "evidence-grounded-paper-qna"
    version: str
    seed: int
    storage: Literal["ok"] = "ok"
    question_engine: Literal["lazy", "ready"]
    papers: int
    jobs: QueueHealth
