"""Interactive web demo and REST API for Evidence Grounded Paper Q&A (FastAPI + stdlib fallback)."""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from paper_qna.pipeline import PaperQnAEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    class AskRequest(BaseModel):
        paper_id: str
        question: str = Field(..., min_length=1, max_length=2000)
        nli_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
        conformal_alpha: float = Field(default=0.05, ge=0.01, le=0.5)
        prefer_live_llm: bool = True

    class NLICheckRequest(BaseModel):
        question: str = Field(..., min_length=1, max_length=2000)
        evidence_texts: list[str] = Field(..., min_length=1, max_length=10)
        threshold: float = Field(default=0.95, ge=0.0, le=1.0)
except ImportError:
    FastAPI = File = HTTPException = UploadFile = HTMLResponse = Any  # type: ignore[misc, assignment]
    AskRequest = Any  # type: ignore[misc, assignment]
    NLICheckRequest = Any  # type: ignore[misc, assignment]


def _slim_presets(engine: PaperQnAEngine) -> list[dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "category": p["category"],
            "language": p["language"],
            "paper_id": p["paper_id"],
            "paper_title": p["paper_title"],
            "question": p["question"],
        }
        for p in engine.presets
    ]


def _extract_multipart_file(raw_body: bytes, content_type: str) -> tuple[str, bytes]:
    """Minimal RFC 7578 multipart parser for single PDF upload in stdlib mode."""
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise ValueError("Missing multipart boundary.")
    sep = ("--" + boundary).encode("latin-1")
    for section in raw_body.split(sep):
        if b"Content-Disposition:" not in section or b"\r\n\r\n" not in section:
            continue
        header_bytes, payload = section.split(b"\r\n\r\n", 1)
        payload = payload.rstrip(b"\r\n")
        if payload.endswith(b"--"):
            payload = payload[:-2].rstrip(b"\r\n")
        header_txt = header_bytes.decode("utf-8", errors="replace")
        filename = "uploaded.pdf"
        for token in header_txt.split(";"):
            tok = token.strip()
            if tok.lower().startswith("filename="):
                filename = tok.split("=", 1)[1].strip().strip('"')
        return filename, payload
    raise ValueError("No uploaded file part found in multipart request.")


def create_stdlib_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    engine: PaperQnAEngine | None = None,
    **engine_kwargs: Any,
) -> ThreadingHTTPServer:
    """Zero-dependency HTTP server mirroring all FastAPI routes."""
    active_engine = engine if engine is not None else PaperQnAEngine(**engine_kwargs)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if path == "/health":
                st = active_engine.status()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "version": st["version"],
                        "nli_onnx_ready": st["nli_onnx"]["available"],
                        "papers_loaded": st["papers_loaded"],
                    },
                )
                return
            if path == "/api/status":
                self._send_json(HTTPStatus.OK, active_engine.status())
                return
            if path == "/api/papers":
                self._send_json(HTTPStatus.OK, {"papers": active_engine.list_papers()})
                return
            if path == "/api/presets":
                self._send_json(HTTPStatus.OK, {"presets": _slim_presets(active_engine)})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0"))
            if path == "/api/upload":
                if length <= 0 or length > MAX_UPLOAD_BYTES + 65536:
                    self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"detail": "PDF exceeds 20 MiB limit."})
                    return
                try:
                    raw = self.rfile.read(length)
                    ctype = self.headers.get("Content-Type", "")
                    fname, pdf_bytes = _extract_multipart_file(raw, ctype)
                    self._send_json(HTTPStatus.CREATED, active_engine.register_pdf(pdf_bytes, filename=fname))
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})
                return

            if length <= 0 or length > 250_000:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Invalid request body size."})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Invalid JSON body."})
                return

            if path == "/api/ask":
                try:
                    out = active_engine.ask(
                        paper_id=str(body.get("paper_id", "")),
                        question=str(body.get("question", "")),
                        nli_threshold=float(body.get("nli_threshold", 0.95)),
                        conformal_alpha=float(body.get("conformal_alpha", 0.05)),
                        prefer_live_llm=bool(body.get("prefer_live_llm", True)),
                    )
                    self._send_json(HTTPStatus.OK, out)
                except KeyError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})
                return

            if path == "/api/nli-check":
                try:
                    out = active_engine.check_nli(
                        str(body.get("question", "")),
                        list(body.get("evidence_texts") or []),
                        threshold=float(body.get("threshold", 0.95)),
                    )
                    self._send_json(HTTPStatus.OK, out)
                except RuntimeError as exc:
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

    return ThreadingHTTPServer((host, port), Handler)


def create_app(engine: PaperQnAEngine | None = None, **engine_kwargs: Any) -> Any:
    """Create a FastAPI ASGI application (used when fastapi is installed)."""
    if FastAPI is Any:
        raise ImportError("fastapi is not installed.")
    active_engine = engine if engine is not None else PaperQnAEngine(**engine_kwargs)
    app = FastAPI(
        title="Evidence Grounded Paper Q&A",
        version="2.0.0",
        description="Bilingual EN/KO Scientific Paper Q&A with BGE-M3 Retrieval, Fine Tuned mDeBERTa-v3 Scientific NLI, Conformal Gating, and Guided JSON Verification.",
    )
    app.state.engine = active_engine

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        st = active_engine.status()
        return {
            "status": "ok",
            "version": st["version"],
            "nli_onnx_ready": st["nli_onnx"]["available"],
            "papers_loaded": st["papers_loaded"],
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return active_engine.status()

    @app.get("/api/papers")
    def papers() -> dict[str, Any]:
        return {"papers": active_engine.list_papers()}

    @app.get("/api/presets")
    def presets() -> dict[str, Any]:
        return {"presets": _slim_presets(active_engine)}

    @app.post("/api/upload", status_code=201)
    async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF file exceeds 20 MiB limit.")
        try:
            return active_engine.register_pdf(raw, filename=file.filename or "uploaded.pdf")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        try:
            return active_engine.ask(
                paper_id=body.paper_id,
                question=body.question,
                nli_threshold=body.nli_threshold,
                conformal_alpha=body.conformal_alpha,
                prefer_live_llm=body.prefer_live_llm,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from None

    @app.post("/api/nli-check")
    def nli_check(body: NLICheckRequest) -> dict[str, Any]:
        try:
            return active_engine.check_nli(body.question, body.evidence_texts, threshold=body.threshold)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None

    return app


def serve_demo(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    llm_url: str | None = None,
    llm_model: str | None = None,
    use_neural_retrieval: bool = False,
) -> None:
    url = f"http://{host}:{port}/"
    try:
        import uvicorn

        app = create_app(llm_url=llm_url, llm_model=llm_model, use_neural_retrieval=use_neural_retrieval)
        print(f"[demo] Evidence Grounded Paper Q&A (FastAPI) listening at {url}", flush=True)
        if open_browser:
            threading.Timer(0.9, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except ImportError:
        server = create_stdlib_server(
            host=host,
            port=port,
            llm_url=llm_url,
            llm_model=llm_model,
            use_neural_retrieval=use_neural_retrieval,
        )
        print(f"[demo] Evidence Grounded Paper Q&A (stdlib HTTP) listening at {url}", flush=True)
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[demo] shutting down")
        finally:
            server.server_close()
