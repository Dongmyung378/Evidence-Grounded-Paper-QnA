"""Unit, contract, ONNX INT8 NLI, Conformal Gate, and FastAPI Web Demo tests for Paper Q&A v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paper_qna.nli_gate import ConformalGate, ScientificNLIOnnx, extract_premise_claims, sliding_windows
from paper_qna.pdf_parser import chunk_page_text, load_corpus30_papers
from paper_qna.pipeline import PaperQnAEngine
from paper_qna.retrieval import HybridRetriever, rrf_fuse
from paper_qna.verifier import (
    AnswerValidationError,
    extract_canonical_numbers,
    extract_json_obj,
    numeric_fidelity_ok,
    quote_in_source,
    repair_pdf_chunk_text,
    validate_answer_payload,
    verify_and_assemble_v2,
)
from paper_qna.web_demo import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_text_repair_and_chunking():
    repaired = repair_pdf_chunk_text("thisalgorithm\u00ad achieves 50–225 MHz.")
    assert "this algorithm" in repaired
    assert "\u00ad" not in repaired
    long_text = " ".join(f"Sentence number {i} reports 95.4% accuracy on the benchmark." for i in range(60))
    chunks = chunk_page_text(long_text, chunk_size=600, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 650 for c in chunks)


def test_quote_numeric_and_contract_verifier():
    ev = [
        {
            "evidence_id": "ev-01-p001-c001",
            "paper_id": "paper-001",
            "page": 1,
            "section": "Methods",
            "text": "The dual-band receiver operates across 50-225 MHz with 12.5 dB gain and does not eliminate the bias floor.",
        }
    ]
    raw_ok = json.dumps(
        {
            "premise_check": "supported",
            "premise_reason": "Directly stated.",
            "claims": [
                {
                    "text": "수신기는 50-225 MHz 대역에서 12.5 dB 이득으로 동작합니다.",
                    "evidence_id": "ev-01-p001-c001",
                    "verbatim_quote": "operates across 50-225 MHz with 12.5 dB gain",
                }
            ],
        },
        ensure_ascii=False,
    )
    out = verify_and_assemble_v2(raw_ok, ev, language="ko")
    assert out["response"]["sufficiency"] == "sufficient"
    assert out["pipeline"]["quote_verified"] is True
    assert out["pipeline"]["numeric_verified"] is True
    assert quote_in_source("operates across 50-225 MHz", ev[0]["text"])
    assert numeric_fidelity_ok("50-225 MHz and 12.5 dB", ev[0]["text"])
    assert not numeric_fidelity_ok("999 MHz", ev[0]["text"])
    assert "50" in extract_canonical_numbers("50–225 MHz")

    # False-premise contradiction abstention
    raw_fp = json.dumps(
        {
            "premise_check": "contradicted",
            "premise_reason": "The paper states it does NOT eliminate the bias floor.",
            "claims": [],
        }
    )
    out_fp = verify_and_assemble_v2(raw_fp, ev, language="en")
    assert out_fp["response"]["sufficiency"] == "insufficient"
    assert out_fp["cited_evidence"] == []
    assert out_fp["pipeline"]["abstention_source"] == "llm_contradicted"

    # Contract rejects Korean sufficient answer without Hangul
    with pytest.raises(AnswerValidationError):
        validate_answer_payload(
            {"answer": "English only text", "evidence_ids": ["ev-01-p001-c001"], "sufficiency": "sufficient", "abstention_reason": None},
            ev,
            language="ko",
        )


def test_extract_json_handles_think_and_fences():
    raw = '<think>reasoning</think>\n```json\n{"premise_check": "supported", "claims": []}\n```'
    assert extract_json_obj(raw)["premise_check"] == "supported"


def test_premise_claim_extraction_and_windows():
    en_hyps = extract_premise_claims("Since the paper establishes that the bias floor is eliminated, how was it measured?")
    assert len(en_hyps) == 2 and "bias floor is eliminated" in en_hyps[1]
    ko_hyps = extract_premise_claims("논문이 왜곡이 5임을 증명했다고 입증했으므로 그 근거는 무엇인가요?")
    assert len(ko_hyps) == 2
    wins = sliding_windows(" ".join(["word"] * 450), win_words=200, stride=120)
    assert len(wins) >= 3


def test_hybrid_retriever_and_parent_expansion():
    papers = load_corpus30_papers(ROOT)
    assert len(papers) == 30
    p = papers["paper-030"]
    ret = HybridRetriever(use_neural=False)
    res = ret.retrieve("three screening heuristics ranking baselines", "paper-030", p["chunks"])
    assert len(res["evidence"]) == 5
    assert all("evidence_id" in e and "page" in e for e in res["evidence"])
    fused = rrf_fuse([[{"chunk": p["chunks"][0]}], [{"chunk": p["chunks"][0]}]])
    assert fused[0]["chunk"]["chunk_id"] == p["chunks"][0]["chunk_id"]


def test_conformal_gate_and_frozen_artifacts():
    gate = ConformalGate.from_file(ROOT / "reports" / "calibrated_gate.json")
    assert gate.oof_aurc == pytest.approx(0.0669, abs=1e-4)
    assert gate.nli_contra_threshold == pytest.approx(0.95, abs=1e-4)
    high = gate.score(
        {
            "top_reranker_score": 0.95,
            "score_margin_1_5": 0.5,
            "is_ko": 0.0,
            "nli_max_entailment": 0.2,
            "nli_max_contradiction": 0.001,
            "quote_verified_ratio": 1.0,
            "numeric_verified_ratio": 1.0,
        },
        alpha=0.05,
    )
    low = gate.score(
        {
            "top_reranker_score": 0.1,
            "score_margin_1_5": 0.05,
            "is_ko": 1.0,
            "nli_max_entailment": 0.0,
            "nli_max_contradiction": 0.999,
            "quote_verified_ratio": 0.0,
            "numeric_verified_ratio": 0.0,
        },
        alpha=0.05,
    )
    assert high["probability"] > low["probability"]
    assert high["accepted"] is True
    assert low["accepted"] is False

    freeze = json.loads((ROOT / "data" / "corpus_v2_holdout" / "freeze_v2.json").read_text(encoding="utf-8"))
    for fname, expected in freeze["sha256"].items():
        actual = hashlib.sha256((ROOT / "data" / "corpus_v2_holdout" / fname).read_bytes()).hexdigest()
        assert actual == expected


def test_fastapi_web_demo_routes():
    engine = PaperQnAEngine(root=ROOT)
    client = TestClient(create_app(engine=engine))

    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/api/papers").json()["papers"]) == 30
    presets = client.get("/api/presets").json()["presets"]
    assert len(presets) >= 6

    ko_ans = next(p for p in presets if p["category"] == "ans" and p["language"] == "ko")
    r_ans = client.post("/api/ask", json={"paper_id": ko_ans["paper_id"], "question": ko_ans["question"]})
    assert r_ans.status_code == 200
    assert r_ans.json()["response"]["sufficiency"] == "sufficient"

    fp_en = next(p for p in presets if p["category"] == "fp" and p["language"] == "en")
    r_fp = client.post("/api/ask", json={"paper_id": fp_en["paper_id"], "question": fp_en["question"]})
    assert r_fp.status_code == 200
    assert r_fp.json()["response"]["sufficiency"] == "insufficient"

    # Reject non-PDF upload cleanly
    r_bad = client.post("/api/upload", files={"file": ("bad.txt", b"not a pdf", "text/plain")})
    assert r_bad.status_code == 400


def test_borderline_nli_threshold_and_conformal_gate_enforcement():
    engine = PaperQnAEngine(root=ROOT)
    client = TestClient(create_app(engine=engine))
    presets = client.get("/api/presets").json()["presets"]

    # Borderline NLI case: passes at 0.95, blocked when lowered to 0.78
    bnli = next(p for p in presets if p["id"] == "paper-017-security-ko")
    r_pass = client.post("/api/ask", json={"paper_id": bnli["paper_id"], "question": bnli["question"], "nli_threshold": 0.95}).json()
    r_block = client.post("/api/ask", json={"paper_id": bnli["paper_id"], "question": bnli["question"], "nli_threshold": 0.78}).json()
    assert r_pass["response"]["sufficiency"] == "sufficient"
    assert r_block["response"]["sufficiency"] == "insufficient"
    assert r_block["nli"]["triggered"] is True

    # Borderline Conformal case: accepted at alpha=0.05, held/blocked at alpha=0.02 when enforce_conformal=True
    bconf = next(p for p in presets if p["id"] == "paper-012-copeland-en")
    c_pass = client.post(
        "/api/ask",
        json={"paper_id": bconf["paper_id"], "question": bconf["question"], "conformal_alpha": 0.05, "enforce_conformal": True},
    ).json()
    c_hold = client.post(
        "/api/ask",
        json={"paper_id": bconf["paper_id"], "question": bconf["question"], "conformal_alpha": 0.02, "enforce_conformal": True},
    ).json()
    assert c_pass["conformal"]["accepted"] is True
    assert c_pass["response"]["sufficiency"] == "sufficient"
    assert c_hold["conformal"]["accepted"] is False
    assert c_hold["conformal"]["blocked"] is True
    assert c_hold["response"]["sufficiency"] == "insufficient"

    # Unrelated Korean and English questions in extractive fallback must abstain safely (never echo page 1)
    r_ko_irrel = client.post("/api/ask", json={"paper_id": "paper-030", "question": "박동명의 생일은?"}).json()
    assert r_ko_irrel["response"]["sufficiency"] == "insufficient"
    assert r_ko_irrel["cited_evidence"] == []

    r_en_irrel = client.post(
        "/api/ask",
        json={"paper_id": "paper-030", "question": "What is the clinical trial survival rate in oncology patients treated with chemotherapy?"},
    ).json()
    assert r_en_irrel["response"]["sufficiency"] == "insufficient"
    assert r_en_irrel["cited_evidence"] == []


def test_local_scientific_nli_onnx_parity_gate_if_present():
    onnx_dir = ROOT / "models" / "scientific_nli_int8"
    if not (onnx_dir / "model_quantized.onnx").is_file():
        pytest.skip("Local ONNX INT8 model binary not installed in CI")
    nli = ScientificNLIOnnx(onnx_dir, num_threads=4)
    gate = nli.check_padding_invariance()
    assert "abs_diff" in gate and "passed" in gate
    eng = PaperQnAEngine(root=ROOT)
    if not gate["passed"]:
        assert eng.nli is None
        assert eng.nli_backend_status == "onnx_int8_blocked_by_parity_gate"
    else:
        assert eng.nli is not None
