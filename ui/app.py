"""Streamlit interface for one-paper questions with visible source evidence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.api_client import ApiClientError, PaperQnaClient


DEFAULT_API_URL = os.environ.get("PAPER_QNA_API_URL", "http://127.0.0.1:8000")

COPY = {
    "en": {
        "language": "Language / 언어",
        "api_settings": "API connection",
        "api_url": "FastAPI URL",
        "check_api": "Check API",
        "api_ready": "API is ready.",
        "title": "Evidence-Grounded Paper Q&A",
        "intro": (
            "Upload one text-based English paper, review its overview, and ask "
            "questions in English or Korean."
        ),
        "upload_title": "1. Upload and analyze",
        "upload_label": "Research paper PDF",
        "upload_help": "One text-extractable PDF, up to 20 MiB.",
        "upload_button": "Upload and analyze",
        "uploading": "Uploading the PDF...",
        "queued": "Analysis queued.",
        "running": "Extracting pages and building searchable chunks...",
        "completed": "Analysis complete.",
        "failed": "Analysis failed.",
        "file_selected": "Selected file",
        "overview_title": "2. Paper overview",
        "pages": "Pages",
        "text_pages": "Text pages",
        "chunks": "Chunks",
        "warnings": "Warnings",
        "abstract": "Abstract",
        "abstract_missing": "No abstract was identified automatically.",
        "sections": "Detected sections",
        "sections_missing": "No section headings were identified automatically.",
        "question_title": "3. Ask a question",
        "question_label": "Ask a question about this paper",
        "question_placeholder": "What is the main contribution of this paper?",
        "question_locked": "Upload and analyze a paper before asking a question.",
        "ask_button": "Ask",
        "asking": "Searching the paper and generating an answer...",
        "first_question_note": (
            "Models are prepared during analysis. If preparation was deferred, "
            "the first question can still take longer."
        ),
        "answer": "Answer",
        "evidence_title": "Source evidence",
        "evidence_intro": "Only passages cited by the answer are shown.",
        "evidence_item": "Evidence {index}",
        "evidence_count": "Cited evidence: {count}",
        "page": "Page {page}",
        "section": "Section: {section}",
        "section_unknown": "Section not identified",
        "source_text": "Original paper text",
        "chunk_id": "Chunk ID: {chunk_id}",
        "no_evidence": "No source passage is cited for an insufficient answer.",
        "insufficient": "The paper does not provide enough evidence for this question.",
        "reason": "Reason: {reason}",
        "runtime": "Answer time: {seconds:.2f} seconds",
        "runtime_detail": (
            "Retrieval {retrieval:.2f}s | Generation {generation:.2f}s | "
            "Attempts {attempts} | Device {device}"
        ),
        "unexpected": "An unexpected UI error occurred. Please try again.",
    },
    "ko": {
        "language": "Language / 언어",
        "api_settings": "API 연결",
        "api_url": "FastAPI 주소",
        "check_api": "API 확인",
        "api_ready": "API가 준비되었습니다.",
        "title": "근거 기반 논문 질의응답",
        "intro": (
            "텍스트 기반 영어 논문 한 편을 업로드하고 개요를 확인한 뒤, "
            "한국어 또는 영어로 질문하세요."
        ),
        "upload_title": "1. 업로드와 분석",
        "upload_label": "연구 논문 PDF",
        "upload_help": "텍스트를 추출할 수 있는 PDF 한 편, 최대 20 MiB입니다.",
        "upload_button": "업로드하고 분석",
        "uploading": "PDF를 업로드하고 있습니다...",
        "queued": "분석 작업이 대기 중입니다.",
        "running": "페이지를 추출하고 검색용 청크를 만들고 있습니다...",
        "completed": "분석을 완료했습니다.",
        "failed": "분석에 실패했습니다.",
        "file_selected": "선택한 파일",
        "overview_title": "2. 논문 개요",
        "pages": "전체 페이지",
        "text_pages": "텍스트 페이지",
        "chunks": "청크",
        "warnings": "경고",
        "abstract": "초록",
        "abstract_missing": "초록을 자동으로 식별하지 못했습니다.",
        "sections": "식별한 절",
        "sections_missing": "절 제목을 자동으로 식별하지 못했습니다.",
        "question_title": "3. 질문하기",
        "question_label": "이 논문에 관해 질문하세요",
        "question_placeholder": "이 논문의 주요 기여는 무엇인가요?",
        "question_locked": "질문하려면 먼저 논문을 업로드하고 분석하세요.",
        "ask_button": "질문",
        "asking": "논문을 검색하고 답변을 생성하고 있습니다...",
        "first_question_note": (
            "모델은 논문 분석 중 미리 준비합니다. 준비가 지연된 경우 첫 질문은 "
            "조금 더 오래 걸릴 수 있습니다."
        ),
        "answer": "답변",
        "evidence_title": "원문 근거",
        "evidence_intro": "답변이 실제로 인용한 논문 원문만 표시합니다.",
        "evidence_item": "근거 {index}",
        "evidence_count": "인용 근거: {count}개",
        "page": "{page}페이지",
        "section": "절: {section}",
        "section_unknown": "절 정보 없음",
        "source_text": "논문 원문",
        "chunk_id": "청크 ID: {chunk_id}",
        "no_evidence": "근거 부족 답변에는 인용 원문이 없습니다.",
        "insufficient": "이 질문에 답할 만한 근거가 논문에 충분하지 않습니다.",
        "reason": "사유: {reason}",
        "runtime": "답변 시간: {seconds:.2f}초",
        "runtime_detail": (
            "검색 {retrieval:.2f}초 | 생성 {generation:.2f}초 | "
            "시도 {attempts}회 | 장치 {device}"
        ),
        "unexpected": "화면 처리 중 오류가 발생했습니다. 다시 시도하세요.",
    },
}

ERROR_COPY = {
    "en": {
        "api_unreachable": "Cannot reach FastAPI. Start the API server and check the URL.",
        "invalid_file_type": "Upload a PDF file.",
        "file_too_large": "The PDF must be no larger than 20 MiB.",
        "empty_file": "The selected file is empty.",
        "invalid_pdf": "The selected file is not a readable PDF.",
        "encrypted_pdf": "Password-protected PDFs are not supported.",
        "no_extractable_text": "No text was found. Upload a text-based PDF.",
        "analysis_timeout": "Analysis took too long. Check the API and try again.",
    },
    "ko": {
        "api_unreachable": "FastAPI에 연결할 수 없습니다. API 서버와 주소를 확인하세요.",
        "invalid_file_type": "PDF 파일을 업로드하세요.",
        "file_too_large": "PDF 크기는 20 MiB 이하여야 합니다.",
        "empty_file": "선택한 파일이 비어 있습니다.",
        "invalid_pdf": "읽을 수 있는 PDF 파일이 아닙니다.",
        "encrypted_pdf": "비밀번호로 보호된 PDF는 지원하지 않습니다.",
        "no_extractable_text": "텍스트가 없습니다. 텍스트 기반 PDF를 업로드하세요.",
        "analysis_timeout": "분석 시간이 너무 오래 걸립니다. API 상태를 확인하고 다시 시도하세요.",
    },
}


def reset_paper_state() -> None:
    for key in ("paper", "paper_id", "job", "answer", "workflow_error"):
        st.session_state.pop(key, None)


def display_error(error: ApiClientError, language: str) -> None:
    message = ERROR_COPY[language].get(error.code, error.message)
    st.error(message)


def analyze_uploaded_file(client: PaperQnaClient, uploaded_file, text: dict[str, str]) -> None:
    progress = st.progress(5, text=text["uploading"])
    status_box = st.status(text["uploading"], expanded=True)
    uploaded = client.upload(
        uploaded_file.name,
        uploaded_file.getvalue(),
        uploaded_file.type or "application/pdf",
    )
    st.session_state.paper_id = uploaded["paper_id"]
    progress.progress(25, text=text["queued"])
    status_box.write(text["queued"])
    job = client.analyze(uploaded["paper_id"])

    def update(current: dict) -> None:
        state = current.get("status")
        if state == "queued":
            progress.progress(35, text=text["queued"])
        elif state == "running":
            progress.progress(70, text=text["running"])
            status_box.write(text["running"])

    job = client.wait_for_job(job["job_id"], on_update=update)
    st.session_state.job = job
    if job.get("status") != "completed":
        raise ApiClientError(
            str(job.get("error_code") or "analysis_failed"),
            str(job.get("error_message") or text["failed"]),
        )
    paper = client.paper(uploaded["paper_id"])
    st.session_state.paper = paper
    st.session_state.answer = None
    progress.progress(100, text=text["completed"])
    status_box.update(label=text["completed"], state="complete", expanded=False)


def render_overview(paper: dict, text: dict[str, str]) -> None:
    st.subheader(text["overview_title"])
    columns = st.columns(4)
    columns[0].metric(text["pages"], paper.get("page_count") or 0)
    columns[1].metric(text["text_pages"], paper.get("text_page_count") or 0)
    columns[2].metric(text["chunks"], paper.get("chunk_count") or 0)
    columns[3].metric(text["warnings"], paper.get("warning_count") or 0)

    overview = paper.get("overview") or {}
    st.markdown(f"#### {text['abstract']}")
    abstract = overview.get("abstract")
    st.write(abstract if abstract else text["abstract_missing"])
    st.markdown(f"#### {text['sections']}")
    sections = overview.get("sections") or []
    if sections:
        st.write("  \n".join(f"{index}. {section}" for index, section in enumerate(sections, 1)))
    else:
        st.write(text["sections_missing"])


def render_question(
    client: PaperQnaClient | None,
    paper: dict | None,
    text: dict[str, str],
    language: str,
) -> None:
    st.subheader(text["question_title"])
    ready = bool(client is not None and paper and paper.get("status") == "ready")
    if not ready:
        st.text_area(
            text["question_label"],
            placeholder=text["question_placeholder"],
            disabled=True,
        )
        st.info(text["question_locked"])
        return

    with st.form("question_form"):
        question = st.text_area(
            text["question_label"],
            placeholder=text["question_placeholder"],
            max_chars=2000,
        )
        submitted = st.form_submit_button(text["ask_button"], type="primary")
    st.caption(text["first_question_note"])
    if submitted and question.strip():
        try:
            with st.spinner(text["asking"]):
                st.session_state.answer = client.ask(paper["paper_id"], question.strip())
        except ApiClientError as error:
            display_error(error, language)

    result = st.session_state.get("answer")
    if result:
        render_answer_with_evidence(result, text)


def render_answer_with_evidence(result: dict, text: dict[str, str]) -> None:
    """Render the answer beside every cited source passage."""
    answer_column, evidence_column = st.columns((0.9, 1.1), gap="large")
    evidence = result.get("evidence") or []

    with answer_column:
        st.markdown(f"#### {text['answer']}")
        with st.container(border=True):
            if result.get("sufficiency") == "insufficient":
                st.warning(text["insufficient"])
            st.write(result.get("answer") or text["insufficient"])
            if result.get("sufficiency") == "sufficient":
                st.caption(text["evidence_count"].format(count=len(evidence)))
            reason = result.get("abstention_reason")
            if reason:
                st.caption(text["reason"].format(reason=reason))
            runtime = float(result.get("runtime_seconds") or 0.0)
            st.caption(text["runtime"].format(seconds=runtime))
            details = result.get("runtime") or {}
            if details:
                st.caption(
                    text["runtime_detail"].format(
                        retrieval=float(details.get("retrieval_seconds") or 0.0),
                        generation=float(details.get("generation_seconds") or 0.0),
                        attempts=int(details.get("generation_attempts") or 0),
                        device=str(details.get("llm_device") or "N/A").upper(),
                    )
                )

    with evidence_column:
        st.markdown(f"#### {text['evidence_title']}")
        st.caption(text["evidence_intro"])
        if not evidence:
            st.info(text["no_evidence"])
            return
        for index, item in enumerate(evidence, start=1):
            locator = item.get("locator") or {}
            page = item.get("page")
            section = item.get("section") or locator.get("section")
            chunk_id = locator.get("chunk_id") or "N/A"
            with st.container(border=True):
                st.markdown(f"##### {text['evidence_item'].format(index=index)}")
                st.caption(
                    f"{text['page'].format(page=page)} | "
                    f"{text['section'].format(section=section or text['section_unknown'])}"
                )
                st.markdown(f"**{text['source_text']}**")
                st.write(item.get("text") or "")
                st.caption(text["chunk_id"].format(chunk_id=chunk_id))


def main() -> None:
    st.set_page_config(
        page_title="Evidence-Grounded Paper Q&A",
        page_icon="PDF",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1080px; padding-top: 2.4rem; padding-bottom: 4rem;}
        [data-testid="stMetric"] {border: 1px solid #e5e7eb; padding: 0.9rem; border-radius: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    selected = st.sidebar.selectbox(
        "Language / 언어",
        ("English", "한국어"),
        key="display_language",
    )
    language = "en" if selected == "English" else "ko"
    text = COPY[language]

    st.sidebar.subheader(text["api_settings"])
    api_url = st.sidebar.text_input(
        text["api_url"],
        value=DEFAULT_API_URL,
        key="api_url",
        on_change=reset_paper_state,
    )

    try:
        client = PaperQnaClient(api_url)
    except ValueError as error:
        st.sidebar.error(str(error))
        client = None

    if st.sidebar.button(text["check_api"], disabled=client is None):
        try:
            health = client.health()
            st.sidebar.success(f"{text['api_ready']} Seed {health.get('seed', 378)}")
        except ApiClientError as error:
            message = ERROR_COPY[language].get(error.code, error.message)
            st.sidebar.error(message)

    st.title(text["title"])
    st.write(text["intro"])
    st.subheader(text["upload_title"])
    uploaded_file = st.file_uploader(
        text["upload_label"],
        type=("pdf",),
        accept_multiple_files=False,
        help=text["upload_help"],
        on_change=reset_paper_state,
    )
    if uploaded_file is not None:
        size_mib = uploaded_file.size / (1024 * 1024)
        st.caption(f"{text['file_selected']}: {uploaded_file.name} ({size_mib:.2f} MiB)")

    if st.button(
        text["upload_button"],
        type="primary",
        disabled=uploaded_file is None or client is None,
    ):
        reset_paper_state()
        try:
            analyze_uploaded_file(client, uploaded_file, text)
            st.success(text["completed"])
        except ApiClientError as error:
            st.session_state.workflow_error = error.code
            display_error(error, language)
        except Exception:
            st.session_state.workflow_error = "unexpected_ui_error"
            st.error(text["unexpected"])

    paper = st.session_state.get("paper")
    if paper:
        render_overview(paper, text)
    render_question(client, paper, text, language)
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
