"""OpenAI-compatible local LLM client (llama-server / Ollama / vLLM) with Guided JSON."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleLLMClient:
    """Connects to any local OpenAI-compatible server serving Qwen3-4B-Instruct-2507."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ):
        self.base_url = (base_url or os.environ.get("PAPER_QNA_LLM_URL") or "").rstrip("/")
        self.model = model or os.environ.get("PAPER_QNA_LLM_MODEL") or "Qwen/Qwen3-4B-Instruct-2507"
        self.timeout = int(timeout)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def check_health(self) -> dict[str, Any]:
        if not self.base_url:
            return {"connected": False, "base_url": None, "model": self.model}
        for path in ("/health", "/v1/models"):
            try:
                with self._opener.open(f"{self.base_url}{path}", timeout=2) as resp:
                    if resp.status == 200:
                        return {"connected": True, "base_url": self.base_url, "model": self.model}
            except Exception:
                continue
        return {"connected": False, "base_url": self.base_url, "model": self.model}

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("No local LLM server URL configured (set PAPER_QNA_LLM_URL).")
        url = f"{self.base_url}/v1/chat/completions"
        use_json = json_mode
        cur_msgs = [dict(m) for m in messages]
        for attempt in range(3):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": cur_msgs,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            if use_json:
                payload["response_format"] = {"type": "json_object"}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                txt = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            except urllib.error.HTTPError as he:
                body = he.read().decode("utf-8", errors="ignore")
                low = body.lower()
                if he.code == 400 and use_json and "response_format" in low:
                    use_json = False
                    continue
                if he.code in (400, 500) and ("context" in low or "exceed" in low or "max_tokens" in low):
                    max_tokens = max(384, int(max_tokens * 0.7))
                    for m in cur_msgs:
                        if m.get("role") == "user" and len(m.get("content", "")) > 2400:
                            m["content"] = m["content"][: int(len(m["content"]) * 0.65)]
                    continue
                if attempt == 2:
                    raise RuntimeError(f"LLM HTTP {he.code}: {body[:200]}") from he
        return ""

    def translate_ko_to_en(self, question: str) -> str:
        if not self.base_url:
            return question
        msgs = [
            {
                "role": "system",
                "content": "Translate this Korean scientific paper question into concise academic English search terms. Preserve all technical terms, model names, symbols, and numbers. Output ONLY the English translation.",
            },
            {"role": "user", "content": question.strip()},
        ]
        try:
            tr = self.chat(msgs, max_tokens=96, temperature=0.0, json_mode=False)
            return tr.splitlines()[0].strip() if tr else question
        except Exception:
            return question
