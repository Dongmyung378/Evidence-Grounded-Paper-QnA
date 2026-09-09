"""Small synchronous client used by the Streamlit interface."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Stable UI-facing representation of an API or transport failure."""

    def __init__(self, code: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class PaperQnaClient:
    """Call the FastAPI service without exposing HTTP details to the UI."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ):
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("API URL must start with http:// or https://")
        self.base_url = normalized
        self._client = httpx.Client(
            base_url=normalized,
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
            transport=transport,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PaperQnaClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _error(response: httpx.Response) -> ApiClientError:
        code = "api_error"
        message = f"API request failed with HTTP {response.status_code}."
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail", payload)
            if isinstance(detail, dict):
                code = str(detail.get("code") or code)
                message = str(detail.get("message") or message)
            elif isinstance(detail, str) and detail.strip():
                message = detail.strip()
        return ApiClientError(code, message, response.status_code)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise ApiClientError(
                "api_unreachable",
                "The API server could not be reached.",
            ) from exc
        if response.is_error:
            raise self._error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiClientError(
                "invalid_api_response",
                "The API returned an invalid response.",
                response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise ApiClientError(
                "invalid_api_response",
                "The API returned an unexpected response shape.",
                response.status_code,
            )
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def upload(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/upload",
            files={"file": (filename, content, content_type)},
        )

    def analyze(self, paper_id: str) -> dict[str, Any]:
        return self._request("POST", "/analyze", json={"paper_id": paper_id})

    def job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

    def paper(self, paper_id: str) -> dict[str, Any]:
        return self._request("GET", f"/papers/{paper_id}")

    def ask(self, paper_id: str, question: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/question",
            json={"paper_id": paper_id, "question": question},
        )

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 180.0,
        poll_interval: float = 0.5,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            result = self.job(job_id)
            if on_update is not None:
                on_update(result)
            if result.get("status") in {"completed", "failed"}:
                return result
            time.sleep(max(0.0, poll_interval))
        raise ApiClientError(
            "analysis_timeout",
            "Paper analysis did not finish within the allowed time.",
        )
