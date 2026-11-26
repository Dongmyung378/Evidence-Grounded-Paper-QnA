"""Deterministic response contract, verbatim quote, and numeric fidelity verifier."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

ABSTENTION_ANSWERS = {
    "en": "The supplied paper evidence is insufficient to answer this question.",
    "ko": "제공된 논문 근거만으로는 이 질문에 답하기에 충분하지 않습니다.",
}

NUM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-+−]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][-+−]?\d+)?"
)


class AnswerValidationError(ValueError):
    """Raised when a structured answer violates the grounded response contract."""


def repair_pdf_chunk_text(text: str) -> str:
    """Apply deterministic NFKC, soft-hyphen removal, and PDF word-glue repair."""
    t = unicodedata.normalize("NFKC", str(text or "")).replace("\u00ad", "").replace("\u200b", "")
    t = re.sub(
        r"\b(this|that|these|those|when|where|with|from|into|such|each)(?=[a-z]{6,}\b)",
        r"\1 ",
        t,
    )
    t = re.sub(r"(?<=\bWhen)([A-Z])(?=(?:and|or)\b)", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def canonical_abstention_payload(language: str, reason: str) -> dict[str, Any]:
    lang = "ko" if str(language).lower().startswith("ko") else "en"
    clean_reason = str(reason or "").strip()[:900] or ABSTENTION_ANSWERS[lang]
    return {
        "answer": ABSTENTION_ANSWERS[lang],
        "evidence_ids": [],
        "sufficiency": "insufficient",
        "abstention_reason": clean_reason,
    }


def validate_answer_payload(
    response: dict[str, Any],
    cited_evidence: list[dict[str, Any]],
    language: str,
) -> None:
    if not isinstance(response, dict):
        raise AnswerValidationError("Response must be a JSON object.")
    sufficiency = response.get("sufficiency")
    if sufficiency not in {"sufficient", "insufficient"}:
        raise AnswerValidationError(f"Invalid sufficiency value: {sufficiency!r}")
    answer = str(response.get("answer") or "").strip()
    if not answer or len(answer) > 4000:
        raise AnswerValidationError("Answer text must be 1..4000 characters.")
    evidence_ids = response.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        raise AnswerValidationError("evidence_ids must be a list.")

    if sufficiency == "insufficient":
        if evidence_ids or cited_evidence:
            raise AnswerValidationError("Insufficient responses must not cite evidence.")
        if not str(response.get("abstention_reason") or "").strip():
            raise AnswerValidationError("Insufficient responses require abstention_reason.")
        return

    if not evidence_ids or len(evidence_ids) > 5:
        raise AnswerValidationError("Sufficient responses must cite 1..5 evidence_ids.")
    if response.get("abstention_reason") is not None:
        raise AnswerValidationError("Sufficient responses must have abstention_reason=None.")
    cited_ids = [str(e.get("evidence_id", "")) for e in cited_evidence]
    if set(evidence_ids) != set(cited_ids):
        raise AnswerValidationError("evidence_ids must match cited_evidence.")
    lang = "ko" if str(language).lower().startswith("ko") else "en"
    if lang == "ko" and not re.search(r"[가-힣]", answer):
        raise AnswerValidationError("Korean sufficient answers must contain Hangul characters.")


def build_v2_grounded_messages(
    question: str,
    evidence: list[dict[str, Any]],
    language: str,
) -> list[dict[str, str]]:
    """Construct the v2 bilingual premise-aware Guided JSON prompt."""
    target_lang = "Korean" if language == "ko" else "English"
    ev_block = "\n".join(
        json.dumps(
            {
                "evidence_id": item["evidence_id"],
                "page": item["page"],
                "section": item["section"],
                "text": item["text"],
            },
            ensure_ascii=False,
        )
        for item in evidence
    )
    system_msg = f"""You are a strict evidence-grounded scientific paper Q&A assistant.

Rules:
1. Rely ONLY on the supplied Evidence JSON lines. NEVER use outside knowledge.
2. First check whether the question contains an explicit or implicit premise that is CONTRADICTED by the evidence (e.g., claiming a method eliminates a bias floor when the paper states it does NOT, or assuming models were trained identically when they were NOT).
   - If contradicted -> set "premise_check": "contradicted", "claims": [], and explain the exact contradiction in "premise_reason" ({target_lang}) citing the contradicting fact.
   - If the evidence does not supply the requested information -> set "premise_check": "unsupported", "claims": [], and explain briefly in "premise_reason" ({target_lang}).
   - Otherwise -> set "premise_check": "supported" and write complete, faithful claims in "claims".
3. Write each claim "text" in natural, accurate academic {target_lang}. Preserve ALL exact numbers, ranges (e.g., 50-225 MHz), units, percentages, dataset/model names, conditions, and qualifiers verbatim.
4. Cover ALL requested items/aspects in the question concisely (at most 3-4 claims for overview questions).
5. Every claim MUST include a valid "evidence_id" from the supplied list and a SHORT "verbatim_quote" (10 to 25 words) copied directly from that evidence text.
6. Output ONLY a single valid JSON object with keys: "premise_check", "premise_reason", "claims". No Markdown code fences."""

    user_msg = f"""Question ({target_lang}):
{question.strip()}

Evidence:
{ev_block}

Return ONLY JSON matching:
{{"premise_check": "supported|unsupported|contradicted", "premise_reason": "...", "claims": [{{"text": "...", "evidence_id": "ev-01-...", "verbatim_quote": "..."}}]}}"""
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def _recover_truncated_claims_json(cleaned: str) -> dict[str, Any] | None:
    """Recover complete claims when a local LLM hits max_tokens mid-JSON inside claims[]."""
    m_pc = re.search(r'"premise_check"\s*:\s*"([^"]+)"', cleaned)
    if not m_pc:
        return None
    verdict = m_pc.group(1)
    m_pr = re.search(r'"premise_reason"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    reason = json.loads(f'"{m_pr.group(1)}"') if m_pr else ""
    claims_pos = cleaned.find('"claims"')
    if claims_pos == -1:
        return {"premise_check": verdict, "premise_reason": reason, "claims": []}
    arr_start = cleaned.find("[", claims_pos)
    if arr_start == -1:
        return None
    recovered: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = arr_start + 1
    n = len(cleaned)
    while idx < n:
        while idx < n and cleaned[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or cleaned[idx] == "]":
            break
        if cleaned[idx] != "{":
            break
        try:
            item, end_idx = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict) and item.get("text") and item.get("evidence_id"):
            recovered.append(item)
        idx = end_idx
    if verdict in {"unsupported", "contradicted"} or recovered:
        return {"premise_check": verdict, "premise_reason": reason, "claims": recovered}
    return None


def extract_json_obj(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.S).strip()
    m = re.match(r"^```(?:json)?\s*(\{.*\})\s*```$", cleaned, flags=re.I | re.S)
    if m:
        cleaned = m.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        elif start != -1:
            cleaned = cleaned[start:]
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            obj = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            rec = _recover_truncated_claims_json(cleaned)
            if rec is None:
                raise
            obj = rec
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON must be an object.")
    return obj


def quote_in_source(quote: str, source_text: str) -> bool:
    if not quote or len(quote.strip()) < 8:
        return False
    q_norm = re.sub(r"\s+", " ", quote.casefold()).strip()
    s_norm = re.sub(r"\s+", " ", source_text.casefold()).strip()
    if q_norm in s_norm:
        return True
    q_tokens = re.findall(r"[a-z0-9]+", q_norm)
    if not q_tokens:
        return False
    s_tokens = set(re.findall(r"[a-z0-9]+", s_norm))
    return (sum(t in s_tokens for t in q_tokens) / len(q_tokens)) >= 0.72


def extract_canonical_numbers(text: str) -> set[str]:
    nums: set[str] = set()
    cleaned = str(text or "").replace("–", "-").replace("−", "-")
    for raw in NUM_PATTERN.findall(cleaned):
        val = raw.lstrip("+").replace(",", "")
        if val.startswith("."):
            val = "0" + val
        try:
            f = float(val)
            nums.add(f"{f:g}")
            nums.add(f"{abs(f):g}")
        except ValueError:
            nums.add(val)
    return nums


def numeric_fidelity_ok(claim_text: str, source_text: str) -> bool:
    c_nums = extract_canonical_numbers(claim_text)
    if not c_nums:
        return True
    s_nums = extract_canonical_numbers(source_text)
    if not s_nums:
        return False
    return c_nums.issubset(s_nums)


def verify_and_assemble_v2(
    raw_output: str,
    evidence: list[dict[str, Any]],
    language: str,
    pre_abstain_reason: str | None = None,
    strict_drop_unverified_numeric: bool = False,
) -> dict[str, Any]:
    ev_by_id = {e["evidence_id"]: e for e in evidence}
    if pre_abstain_reason:
        return {
            "response": canonical_abstention_payload(language, pre_abstain_reason),
            "cited_evidence": [],
            "retrieved_evidence": evidence,
            "answer_alignment": [],
            "pipeline": {
                "abstention_source": "pre_gate",
                "fallback_used": False,
                "quote_verified": True,
                "quote_verified_ratio": 1.0,
                "numeric_verified": True,
                "numeric_verified_ratio": 1.0,
                "raw_verdict": "pre_abstain",
            },
        }
    try:
        parsed = extract_json_obj(raw_output)
    except Exception as exc:
        reason = "구조화 JSON 응답 파싱 실패" if language == "ko" else f"JSON parse failed: {exc}"
        return {
            "response": canonical_abstention_payload(language, reason),
            "cited_evidence": [],
            "retrieved_evidence": evidence,
            "answer_alignment": [],
            "pipeline": {
                "abstention_source": "json_fallback",
                "fallback_used": True,
                "quote_verified": False,
                "quote_verified_ratio": 0.0,
                "numeric_verified": False,
                "numeric_verified_ratio": 0.0,
            },
        }

    verdict = str(parsed.get("premise_check", "unsupported")).strip().lower()
    reason = str(parsed.get("premise_reason") or "").strip()

    if verdict in {"unsupported", "contradicted"}:
        default_r = (
            "질문의 전제가 논문 근거와 모순되거나 근거가 부족합니다."
            if language == "ko"
            else "The question premise is contradicted or unsupported by the paper evidence."
        )
        return {
            "response": canonical_abstention_payload(language, (reason or default_r)[:900]),
            "cited_evidence": [],
            "retrieved_evidence": evidence,
            "answer_alignment": [],
            "pipeline": {
                "abstention_source": f"llm_{verdict}",
                "fallback_used": False,
                "quote_verified": True,
                "quote_verified_ratio": 1.0,
                "numeric_verified": True,
                "numeric_verified_ratio": 1.0,
                "raw_verdict": verdict,
            },
        }

    claims = parsed.get("claims") or []
    verified_texts: list[str] = []
    cited_ids: list[str] = []
    alignments: list[dict[str, Any]] = []
    quote_hits, num_hits = 0, 0
    all_ev_text = " ".join(e["text"] for e in evidence)

    for cl in claims:
        if not isinstance(cl, dict):
            continue
        text = str(cl.get("text", "")).strip()
        eid = str(cl.get("evidence_id", "")).strip()
        quote = str(cl.get("verbatim_quote", "")).strip()
        if not text or eid not in ev_by_id:
            continue
        src = ev_by_id[eid]
        q_ok = quote_in_source(quote, src["text"])
        n_ok = numeric_fidelity_ok(text, src["text"]) or numeric_fidelity_ok(text, all_ev_text)
        if strict_drop_unverified_numeric and not n_ok:
            continue
        quote_hits += int(q_ok)
        num_hits += int(n_ok)
        verified_texts.append(text)
        if eid not in cited_ids and len(cited_ids) < 5:
            cited_ids.append(eid)
        alignments.append(
            {
                "answer_text": text,
                "evidence_id": eid,
                "page": src["page"],
                "section": src["section"],
                "verbatim_quote": quote,
                "quote_verified": q_ok,
                "numeric_verified": n_ok,
            }
        )

    if not verified_texts or not cited_ids:
        fb = (
            "검증 가능한 인용 근거가 포함된 답변을 구성하지 못했습니다."
            if language == "ko"
            else "No verified cited claims were produced."
        )
        return {
            "response": canonical_abstention_payload(language, fb),
            "cited_evidence": [],
            "retrieved_evidence": evidence,
            "answer_alignment": [],
            "pipeline": {
                "abstention_source": "claim_verification_fallback",
                "fallback_used": True,
                "quote_verified": False,
                "quote_verified_ratio": 0.0,
                "numeric_verified": False,
                "numeric_verified_ratio": 0.0,
            },
        }

    answer_text = " ".join(verified_texts)[:3950]
    cited_evidence = [ev_by_id[eid] for eid in cited_ids]
    response = {
        "answer": answer_text,
        "evidence_ids": cited_ids,
        "sufficiency": "sufficient",
        "abstention_reason": None,
    }
    try:
        validate_answer_payload(response, cited_evidence, language)
    except AnswerValidationError as exc:
        fb = f"응답 계약 검증 실패: {exc}" if language == "ko" else f"Contract validation failed: {exc}"
        return {
            "response": canonical_abstention_payload(language, fb[:900]),
            "cited_evidence": [],
            "retrieved_evidence": evidence,
            "answer_alignment": [],
            "pipeline": {
                "abstention_source": "contract_fallback",
                "fallback_used": True,
                "quote_verified": False,
                "quote_verified_ratio": 0.0,
                "numeric_verified": False,
                "numeric_verified_ratio": 0.0,
            },
        }

    return {
        "response": response,
        "cited_evidence": cited_evidence,
        "retrieved_evidence": evidence,
        "answer_alignment": alignments,
        "pipeline": {
            "abstention_source": None,
            "fallback_used": False,
            "raw_verdict": verdict,
            "quote_verified": (quote_hits == len(alignments)),
            "quote_verified_ratio": round(quote_hits / max(len(alignments), 1), 3),
            "numeric_verified": (num_hits == len(alignments)),
            "numeric_verified_ratio": round(num_hits / max(len(alignments), 1), 3),
        },
    }
