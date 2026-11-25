"""Unified Evidence Grounded Paper Q&A engine."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from paper_qna.llm_client import OpenAICompatibleLLMClient
from paper_qna.nli_gate import ConformalGate, ScientificNLIOnnx, ScientificNLIPyTorch
from paper_qna.pdf_parser import load_corpus30_papers, parse_pdf_bytes
from paper_qna.retrieval import HybridRetriever, tokenize
from paper_qna.verifier import (
    build_v2_grounded_messages,
    canonical_abstention_payload,
    verify_and_assemble_v2,
)

ROOT = Path(__file__).resolve().parents[1]
HANGUL_RE = re.compile(r"[가-힣]")


def detect_language(text: str) -> str:
    return "ko" if HANGUL_RE.search(str(text or "")) else "en"


def _extractive_fallback(
    question: str,
    evidence: list[dict[str, Any]],
    language: str,
    nli_res: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic grounded sentence extraction when no local LLM server is attached."""
    q_toks = set(tokenize(question))
    best_sents: list[tuple[float, str, dict[str, Any]]] = []
    for ev in evidence[:3]:
        for sent in re.split(r"(?<=[.!?])\s+", ev.get("text", "")):
            s = sent.strip()
            if len(s) < 35 or len(s) > 420:
                continue
            s_toks = set(tokenize(s))
            overlap = len(q_toks & s_toks) / max(len(q_toks), 1)
            score = overlap + 0.25 * float(ev.get("reranker_score", 0.0))
            best_sents.append((score, s, ev))
    best_sents.sort(key=lambda item: -item[0])

    if not best_sents or (best_sents[0][0] < 0.12 and language == "en"):
        reason = (
            "로컬 LLM 서버(Qwen3-4B)가 연결되지 않았고, 검색된 근거에서 질문과 직접 일치하는 문장을 찾지 못했습니다."
            if language == "ko"
            else "No local LLM server (Qwen3-4B) is connected and retrieved passages lacked direct lexical match."
        )
        return verify_and_assemble_v2("", evidence, language, pre_abstain_reason=reason)

    top_items = best_sents[:2]
    claims = []
    for _, sent, ev in top_items:
        claim_text = (
            f"[근거 원문 발췌 · p.{ev['page']}] {sent}"
            if language == "ko"
            else sent
        )
        claims.append(
            {
                "text": claim_text,
                "evidence_id": ev["evidence_id"],
                "verbatim_quote": sent[:180],
            }
        )
    raw_json = json.dumps(
        {
            "premise_check": "supported",
            "premise_reason": "Grounded via deterministic extractive fallback (connect local Qwen3-4B llama-server for full bilingual synthesis).",
            "claims": claims,
        },
        ensure_ascii=False,
    )
    out = verify_and_assemble_v2(raw_json, evidence, language)
    out["pipeline"]["generation_mode"] = "extractive_fallback_no_llm"
    return out


class PaperQnAEngine:
    """Thread safe local Paper Q&A engine supporting PDF uploads, Corpus30 presets, local Scientific NLI, and local Qwen3-4B."""

    def __init__(
        self,
        root: Path = ROOT,
        nli_dir: Path | None = None,
        gate_path: Path | None = None,
        llm_url: str | None = None,
        llm_model: str | None = None,
        use_neural_retrieval: bool = False,
        device: str = "cpu",
        nli_threads: int = 4,
    ):
        self.root = Path(root)
        self.nli_dir = Path(nli_dir) if nli_dir else (self.root / "models" / "scientific_nli_int8")
        self.gate_path = Path(gate_path) if gate_path else (self.root / "reports" / "calibrated_gate.json")
        self._lock = threading.Lock()

        self.papers: dict[str, dict[str, Any]] = load_corpus30_papers(self.root)
        self.presets: list[dict[str, Any]] = self._load_presets()
        self._preset_lookup: dict[tuple[str, str], dict[str, Any]] = {
            (p["paper_id"], p["question"].strip().casefold()): p for p in self.presets
        }

        self.retriever = HybridRetriever(use_neural=use_neural_retrieval, device=device)
        self.llm = OpenAICompatibleLLMClient(base_url=llm_url, model=llm_model)
        self.gate: ConformalGate | None = (
            ConformalGate.from_file(self.gate_path) if self.gate_path.is_file() else None
        )
        self.nli: ScientificNLIPyTorch | ScientificNLIOnnx | None = None
        self.nli_backend_status: str = "unavailable"
        self.nli_parity_gate: dict[str, Any] | None = None
        self.nli_error: str | None = None

        fp32_dir = self.root / "models" / "scientific_nli_mdeberta"
        if (fp32_dir / "config.json").is_file():
            try:
                self.nli = ScientificNLIPyTorch(fp32_dir, device=device)
                self.nli_backend_status = "fp32_pytorch_finetuned"
            except Exception as exc:
                self.nli_error = str(exc)
        elif (self.nli_dir / "model_quantized.onnx").is_file():
            try:
                cand = ScientificNLIOnnx(self.nli_dir, num_threads=nli_threads)
                self.nli_parity_gate = cand.check_padding_invariance()
                if self.nli_parity_gate["passed"]:
                    self.nli = cand
                    self.nli_backend_status = "onnx_int8_verified"
                else:
                    # Block degraded TorchScript INT8 DeBERTa-v2 graph from corrupting live gating
                    self.nli_backend_status = "onnx_int8_blocked_by_parity_gate"
            except Exception as exc:
                self.nli_error = str(exc)

    def _load_presets(self) -> list[dict[str, Any]]:
        path = self.root / "data" / "benchmarks" / "corpus30" / "demo_holdout_presets.json"
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        llm_health = self.llm.check_health()
        return {
            "version": "2.0.0",
            "papers_loaded": len(self.papers),
            "presets_loaded": len(self.presets),
            "retrieval_mode": self.retriever.mode,
            "nli_onnx": {
                "available": self.nli is not None,
                "backend": self.nli_backend_status,
                "model_dir": str(self.nli_dir),
                "size_mib": getattr(self.nli, "model_size_mib", None) or (
                    round((self.nli_dir / "model_quantized.onnx").stat().st_size / (1024 ** 2), 2)
                    if (self.nli_dir / "model_quantized.onnx").is_file()
                    else None
                ),
                "default_threshold": self.gate.nli_contra_threshold if self.gate else 0.95,
                "parity_gate": self.nli_parity_gate,
                "error": self.nli_error,
            },
            "conformal_gate": {
                "available": self.gate is not None,
                "oof_aurc": self.gate.oof_aurc if self.gate else None,
                "thresholds": self.gate.thresholds if self.gate else {},
            },
            "llm": llm_health,
        }

    def list_papers(self) -> list[dict[str, Any]]:
        out = []
        for pid, p in sorted(self.papers.items()):
            out.append(
                {
                    "paper_id": pid,
                    "title": p["title"],
                    "page_count": p["page_count"],
                    "chunk_count": p["chunk_count"],
                    "source": p.get("source", "corpus30"),
                    "split": p.get("split", "uploaded"),
                }
            )
        return out

    def register_pdf(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> dict[str, Any]:
        parsed = parse_pdf_bytes(pdf_bytes, filename=filename)
        with self._lock:
            self.papers[parsed["paper_id"]] = parsed
        return {
            "paper_id": parsed["paper_id"],
            "title": parsed["title"],
            "filename": parsed["filename"],
            "page_count": parsed["page_count"],
            "chunk_count": parsed["chunk_count"],
            "source": "uploaded",
        }

    def check_nli(self, question: str, evidence_texts: list[str], threshold: float = 0.95) -> dict[str, Any]:
        if self.nli is None:
            raise RuntimeError("Local Scientific NLI ONNX INT8 model is not loaded.")
        ev_objs = [
            {"evidence_id": f"ev-{i:02d}", "page": i, "text": txt}
            for i, txt in enumerate(evidence_texts, start=1)
            if str(txt).strip()
        ]
        res = self.nli.score_question_against_evidence(question, ev_objs)
        res["threshold"] = float(threshold)
        res["contradicted"] = bool(res["max_contra"] >= threshold and res["max_contra"] > res["max_entail"])
        return res

    def ask(
        self,
        paper_id: str,
        question: str,
        nli_threshold: float = 0.95,
        conformal_alpha: float = 0.05,
        prefer_live_llm: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if paper_id not in self.papers:
            raise KeyError(f"Unknown paper_id: {paper_id}")
        q_clean = str(question or "").strip()
        if not q_clean:
            raise ValueError("Question must not be empty.")
        lang = detect_language(q_clean)
        paper = self.papers[paper_id]
        preset = self._preset_lookup.get((paper_id, q_clean.casefold()))

        llm_live = prefer_live_llm and self.llm.check_health()["connected"]

        # For curated Holdout presets when no live LLM is attached, use verified retrieved evidence
        # and run local Scientific NLI + live Conformal Gate on top.
        if preset is not None and not llm_live:
            evidence = preset["retrieved_evidence"]
            top_score = float(preset["pipeline"].get("top_reranker_score", 0.0))
            margin_1_5 = float(preset["pipeline"].get("score_margin_1_5", 0.0))
            en_q = preset.get("en_query")
            retrieval_mode = "bge_m3_parent1800_verified"
        else:
            en_q = self.llm.translate_ko_to_en(q_clean) if (lang == "ko" and llm_live) else (preset.get("en_query") if preset else None)
            ret = self.retriever.retrieve(q_clean, paper_id, paper["chunks"], en_query=en_q)
            evidence = ret["evidence"]
            top_score = ret["top_reranker_score"]
            margin_1_5 = ret["score_margin_1_5"]
            retrieval_mode = ret["retrieval_mode"]

        # Stage 1: Local Scientific NLI Premise Gate
        nli_info: dict[str, Any]
        if self.nli is not None and evidence:
            nli_info = self.nli.score_question_against_evidence(q_clean, evidence)
            nli_info["engine"] = getattr(self.nli, "backend_name", "local_nli")
        elif preset is not None:
            nli_info = {
                "max_entail": float(preset["pipeline"].get("nli_max_entailment", 0.0)),
                "max_contra": float(preset["pipeline"].get("nli_max_contradiction", 0.0)),
                "window_count": len(evidence),
                "latency_ms": 0.0,
                "top_contra_evidence_id": evidence[0]["evidence_id"] if evidence else None,
                "top_contra_page": evidence[0]["page"] if evidence else None,
                "engine": "fp32_verified_preset",
            }
        else:
            nli_info = {
                "max_entail": 0.5,
                "max_contra": 0.0,
                "window_count": 0,
                "latency_ms": 0.0,
                "top_contra_evidence_id": None,
                "top_contra_page": None,
                "engine": "disabled",
            }

        nli_ent = float(nli_info["max_entail"])
        nli_con = float(nli_info["max_contra"])

        # Stage 2: Pre Gate False Premise and Low Relevance Check
        pre_reason: str | None = None
        abstention_src: str | None = None
        if top_score < -3.0:
            pre_reason = (
                "검색된 논문 근거의 관련도가 매우 낮아 답변할 수 없습니다."
                if lang == "ko"
                else "Retrieved paper evidence relevance is too low to support an answer."
            )
            abstention_src = "low_relevance_gate"
        elif nli_con >= float(nli_threshold) and nli_con > nli_ent:
            engine_lbl = "Scientific NLI"
            pre_reason = (
                f"질문의 전제가 논문 원문 근거와 모순됩니다 ({engine_lbl} 모순 확률={nli_con:.4f} >= {nli_threshold:.2f})."
                if lang == "ko"
                else f"The question premise contradicts the retrieved paper evidence ({engine_lbl} contradiction={nli_con:.4f} >= {nli_threshold:.2f})."
            )
            abstention_src = "scientific_nli_contradiction"

        # Stage 3: Generation + Deterministic Quote/Numeric Verification
        if pre_reason is not None:
            out = verify_and_assemble_v2("", evidence, lang, pre_abstain_reason=pre_reason)
            out["pipeline"]["abstention_source"] = abstention_src
            out["pipeline"]["generation_mode"] = "pre_gate_blocked"
        elif llm_live:
            msgs = build_v2_grounded_messages(q_clean, evidence, lang)
            raw = self.llm.chat(msgs, max_tokens=1024, temperature=0.0, json_mode=True)
            out = verify_and_assemble_v2(raw, evidence, lang)
            out["pipeline"]["generation_mode"] = f"live_llm ({self.llm.model})"
        elif preset is not None:
            out = {
                "response": dict(preset["response"]),
                "cited_evidence": list(preset["cited_evidence"]),
                "retrieved_evidence": list(evidence),
                "answer_alignment": list(preset["answer_alignment"]),
                "pipeline": dict(preset["pipeline"]),
            }
            out["pipeline"]["generation_mode"] = (
                f"verified_holdout_replay+{nli_info.get('engine', 'fp32_nli')}"
            )
        else:
            out = _extractive_fallback(q_clean, evidence, lang, nli_info)

        # Stage 4: Paper Grouped 5 Fold OOF Conformal Gate Telemetry
        conformal_info = None
        if self.gate is not None:
            pipe = out["pipeline"]
            conformal_info = self.gate.score(
                {
                    "top_reranker_score": top_score,
                    "score_margin_1_5": margin_1_5,
                    "is_ko": 1.0 if lang == "ko" else 0.0,
                    "nli_max_entailment": nli_ent,
                    "nli_max_contradiction": nli_con,
                    "quote_verified_ratio": float(pipe.get("quote_verified_ratio", 0.0)),
                    "numeric_verified_ratio": float(pipe.get("numeric_verified_ratio", 0.0)),
                },
                alpha=conformal_alpha,
            )

        wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        out["paper"] = {"paper_id": paper_id, "title": paper["title"], "page_count": paper["page_count"]}
        out["question"] = q_clean
        out["language"] = lang
        out["nli"] = {**nli_info, "threshold": float(nli_threshold), "triggered": abstention_src == "scientific_nli_contradiction"}
        out["conformal"] = conformal_info
        out["pipeline"].update(
            {
                "retrieval_mode": retrieval_mode,
                "top_reranker_score": round(top_score, 4),
                "score_margin_1_5": round(margin_1_5, 4),
                "en_query": en_q,
                "nli_max_entailment": nli_ent,
                "nli_max_contradiction": nli_con,
                "wall_ms": wall_ms,
            }
        )
        if preset is not None:
            out["holdout_meta"] = {
                "case_id": preset["id"],
                "category": preset["category"],
                "split": "holdout",
            }
        return out
