"""Upload preflight and localized messages for recoverable UI errors."""

from __future__ import annotations

from pathlib import PurePath


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", ""}

ERROR_MESSAGES = {
    "en": {
        "api_error": "The request could not be completed. Try again.",
        "api_unreachable": "Cannot reach FastAPI. Start the API server and check the URL.",
        "invalid_api_url": "Enter an API URL that starts with http:// or https://.",
        "invalid_api_response": "The API returned an invalid response. Restart the API and try again.",
        "one_file_required": "Upload exactly one PDF file.",
        "request_too_large": "The upload request is too large. Choose a PDF no larger than 20 MiB.",
        "invalid_file_type": "Choose a PDF file.",
        "file_too_large": "The PDF must be no larger than 20 MiB. Choose a smaller file.",
        "empty_file": "The selected file is empty. Choose another PDF.",
        "invalid_pdf": "The selected file is not a readable PDF. Choose another PDF.",
        "encrypted_pdf": "Password-protected PDFs are not supported. Remove the password and try again.",
        "no_extractable_text": "No text was found. Choose a text-based PDF instead of a scanned document.",
        "storage_error": "The PDF could not be saved. Check local storage and try again.",
        "paper_not_found": "The uploaded paper is no longer available. Upload it again.",
        "queue_full": "The analysis queue is busy. Wait a moment and try again.",
        "job_not_found": "The analysis job is no longer available. Upload the PDF again.",
        "analysis_failed": "The PDF could not be parsed. Try another PDF export.",
        "partial_extraction_failed": "Some PDF pages could not be read. Export the paper as a text-based PDF and try again.",
        "analysis_timeout": "Analysis took too long. Check the API and try again.",
        "analysis_required": "Analyze the paper before asking a question.",
        "analysis_in_progress": "Paper analysis is still in progress. Try again shortly.",
        "question_service_unavailable": "The local question service could not answer. Try again shortly.",
    },
    "ko": {
        "api_error": "요청을 처리하지 못했습니다. 다시 시도하세요.",
        "api_unreachable": "FastAPI에 연결할 수 없습니다. API 서버와 주소를 확인하세요.",
        "invalid_api_url": "http:// 또는 https://로 시작하는 API 주소를 입력하세요.",
        "invalid_api_response": "API 응답 형식이 올바르지 않습니다. API를 다시 시작한 뒤 시도하세요.",
        "one_file_required": "PDF 파일 한 편만 업로드하세요.",
        "request_too_large": "업로드 요청이 너무 큽니다. 20 MiB 이하의 PDF를 선택하세요.",
        "invalid_file_type": "PDF 파일을 선택하세요.",
        "file_too_large": "PDF 크기는 20 MiB 이하여야 합니다. 더 작은 파일을 선택하세요.",
        "empty_file": "선택한 파일이 비어 있습니다. 다른 PDF를 선택하세요.",
        "invalid_pdf": "선택한 파일은 읽을 수 있는 PDF가 아닙니다. 다른 PDF를 선택하세요.",
        "encrypted_pdf": "비밀번호로 보호된 PDF는 지원하지 않습니다. 비밀번호를 제거한 뒤 시도하세요.",
        "no_extractable_text": "텍스트를 찾지 못했습니다. 스캔 문서가 아닌 텍스트 기반 PDF를 선택하세요.",
        "storage_error": "PDF를 저장하지 못했습니다. 로컬 저장 공간을 확인한 뒤 시도하세요.",
        "paper_not_found": "업로드한 논문을 찾을 수 없습니다. 다시 업로드하세요.",
        "queue_full": "분석 대기열이 사용 중입니다. 잠시 후 다시 시도하세요.",
        "job_not_found": "분석 작업을 찾을 수 없습니다. PDF를 다시 업로드하세요.",
        "analysis_failed": "PDF를 파싱하지 못했습니다. 다른 방식으로 내보낸 PDF를 사용하세요.",
        "partial_extraction_failed": "일부 PDF 페이지를 읽지 못했습니다. 텍스트 기반 PDF로 다시 내보낸 뒤 시도하세요.",
        "analysis_timeout": "분석 시간이 너무 오래 걸립니다. API 상태를 확인하고 다시 시도하세요.",
        "analysis_required": "질문하기 전에 논문을 분석하세요.",
        "analysis_in_progress": "논문을 분석하고 있습니다. 잠시 후 다시 시도하세요.",
        "question_service_unavailable": "로컬 질문 서비스가 답변하지 못했습니다. 잠시 후 다시 시도하세요.",
    },
}


def localized_error_message(language: str, code: str) -> str:
    """Return stable copy without exposing backend exception text."""
    messages = ERROR_MESSAGES.get(language, ERROR_MESSAGES["en"])
    return messages.get(code, messages["api_error"])


def validate_upload(
    filename: str,
    content_type: str | None,
    size_bytes: int,
    header: bytes,
) -> str | None:
    """Return an error code for checks that do not require PDF parsing."""
    if size_bytes <= 0:
        return "empty_file"
    if size_bytes > MAX_UPLOAD_BYTES:
        return "file_too_large"
    if PurePath(filename).suffix.lower() != ".pdf":
        return "invalid_file_type"
    if (content_type or "").lower() not in ALLOWED_CONTENT_TYPES:
        return "invalid_file_type"
    if header != b"%PDF-":
        return "invalid_pdf"
    return None
