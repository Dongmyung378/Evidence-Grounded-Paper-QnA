"""Generate clean current-system reports (JSON/CSV) and localized EN/KO publication figures."""

from __future__ import annotations

import copy
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_qna.verifier import canonical_abstention_payload, validate_answer_payload

TEAL = "#002147"
TEAL_MID = "#234E7D"
TEAL_SOFT = "#FCF5DC"
AMBER = "#E8B923"
AMBER_TEXT = "#9E7100"
AMBER_SOFT = "#FCF5DC"
CRIMSON = "#A62B3A"
SLATE = "#5C6E80"
INK = "#132234"
PAPER = "#FAF7F0"
GRID = "#E2DCCB"


def set_plot_font(lang: str) -> None:
    if lang == "ko":
        plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Noto Sans KR", "AppleGothic", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_eval(text: str) -> str:
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", str(text or ""))
    return re.sub(r"\s+", " ", text.casefold().replace("\u2013", "-").replace("\u2212", "-"))


def _contains_alias(text: str, alias: str) -> bool:
    t, a = _norm_eval(text), _norm_eval(alias)
    if re.fullmatch(r"[\d,.]+", a):
        return bool(re.search(r"(?<![\d.])" + re.escape(a) + r"(?![\d.])", t))
    return a in t


def evaluate_predictions(records: list[dict[str, Any]], questions: list[dict[str, Any]], split: str) -> dict[str, Any]:
    pool = [q for q in questions if q["split"] == split]
    lookup = {q["case_id"]: q for q in pool}
    evaluated = []
    for rec in records:
        cid = rec["case_id"]
        if cid not in lookup:
            continue
        case = lookup[cid]
        out = rec.get("output") or {}
        resp = out.get("response") or {}
        cited = out.get("cited_evidence") or []
        contract = True
        try:
            validate_answer_payload(resp, cited, case["language"])
        except Exception:
            contract = False
        answered = contract and resp.get("sufficiency") == "sufficient"
        refused = contract and resp.get("sufficiency") == "insufficient" and not cited
        facts = [
            any(_contains_alias(resp.get("answer", ""), alias) for alias in fact["any_of"])
            for fact in case.get("required_facts", [])
        ]
        gold_pages = set(case.get("gold_pages", []))
        cit_pages = {e["page"] for e in cited}
        pipe = out.get("pipeline") or {}
        evaluated.append(
            {
                "case_id": cid,
                "meaning_id": case["meaning_id"],
                "paper_id": case["paper_id"],
                "split": case["split"],
                "language": case["language"],
                "answerability": case["answerability"],
                "contract_valid": contract,
                "answered": bool(answered),
                "refused": bool(refused),
                "all_fact_aliases": bool(answered and facts and all(facts)),
                "acceptable_source_page_hit": bool(answered and (gold_pages & cit_pages)),
                "quote_verified": bool(pipe.get("quote_verified", True)),
                "numeric_verified": bool(pipe.get("numeric_verified", True)),
                "wall_seconds": float(rec.get("wall_seconds", 0.0)),
            }
        )

    def _sum(rows: list[dict[str, Any]], paired: bool) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(r["meaning_id"] if paired else r["case_id"], []).append(r)
        pos = [g for g in grouped.values() if g[0]["answerability"] == "answerable"]
        neg = [g for g in grouped.values() if g[0]["answerability"] != "answerable"]
        ans_rows = [r for r in rows if r["answered"]]
        return {
            "n_outputs": len(rows),
            "n_answerable": len(pos),
            "n_unsupported": sum(1 for g in neg if g[0]["answerability"] == "unsupported"),
            "n_false_premise": sum(1 for g in neg if g[0]["answerability"] == "false_premise"),
            "contract_valid": sum(r["contract_valid"] for r in rows),
            "answerable_retained": sum(all(r["answered"] for r in g) for g in pos),
            "all_fact_aliases": sum(all(r["all_fact_aliases"] for r in g) for g in pos),
            "acceptable_source_page_hit": sum(all(r["acceptable_source_page_hit"] for r in g) for g in pos),
            "unsupported_refused": sum(all(r["refused"] for r in g) for g in neg if g[0]["answerability"] == "unsupported"),
            "false_premise_refused": sum(all(r["refused"] for r in g) for g in neg if g[0]["answerability"] == "false_premise"),
            "quote_verified_rate": round(sum(r["quote_verified"] for r in ans_rows) / max(len(ans_rows), 1), 3),
            "numeric_verified_rate": round(sum(r["numeric_verified"] for r in ans_rows) / max(len(ans_rows), 1), 3),
            "mean_wall_seconds": round(statistics.mean(r["wall_seconds"] for r in rows), 3) if rows else 0.0,
        }

    return {
        "split": split,
        "paired": _sum(evaluated, True),
        "all_outputs": _sum(evaluated, False),
        "en": _sum([r for r in evaluated if r["language"] == "en"], False),
        "ko": _sum([r for r in evaluated if r["language"] == "ko"], False),
        "records": evaluated,
    }


def compute_paired_stat_tests(
    comparison_name: str,
    base_report: dict[str, Any],
    cand_report: dict[str, Any],
    n_boot: int = 2000,
    seed: int = 378,
) -> list[dict[str, Any]]:
    b_map = {r["case_id"]: r for r in base_report["records"]}
    c_map = {r["case_id"]: r for r in cand_report["records"]}
    common = sorted(set(b_map) & set(c_map))
    rng = np.random.default_rng(seed)

    specs = [
        ("answerable_retained", "answerable", "answered"),
        ("all_fact_aliases", "answerable", "all_fact_aliases"),
        ("acceptable_source_page_hit", "answerable", "acceptable_source_page_hit"),
        ("unsupported_refused", "unsupported", "refused"),
        ("false_premise_refused", "false_premise", "refused"),
    ]
    rows = []
    for label, atype, field in specs:
        sub = [cid for cid in common if b_map[cid]["answerability"] == atype]
        b_vals = np.array([int(b_map[cid][field]) for cid in sub])
        c_vals = np.array([int(c_map[cid][field]) for cid in sub])
        n01 = int(((b_vals == 0) & (c_vals == 1)).sum())
        n10 = int(((b_vals == 1) & (c_vals == 0)).sum())
        p_val = float(binomtest(n01, n01 + n10, 0.5).pvalue) if (n01 + n10) > 0 else 1.0

        by_p: dict[str, list[int]] = defaultdict(list)
        for idx, cid in enumerate(sub):
            by_p[b_map[cid]["paper_id"]].append(idx)
        pids = list(by_p.keys())
        deltas = []
        for _ in range(n_boot):
            sampled_pids = rng.choice(pids, size=len(pids), replace=True)
            idxs = [i for p in sampled_pids for i in by_p[p]]
            deltas.append(float(c_vals[idxs].mean() - b_vals[idxs].mean()))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append(
            {
                "comparison": comparison_name,
                "metric": label,
                "n_outputs": len(sub),
                "base_rate": round(float(b_vals.mean()), 4),
                "cand_rate": round(float(c_vals.mean()), 4),
                "delta": round(float(c_vals.mean() - b_vals.mean()), 4),
                "ci95_low": round(float(lo), 4),
                "ci95_high": round(float(hi), 4),
                "gain_n01": n01,
                "loss_n10": n10,
                "exact_mcnemar_p": float(f"{p_val:.6g}"),
            }
        )
    return rows


def write_hero_svgs(en_dir: Path, ko_dir: Path) -> None:
    ko_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">논문 질의응답과 근거 추적: BGE-M3 · Scientific NLI · Qwen3-4B</title>
<desc id="desc">미공개 corpus30 Holdout 240문항에서 응답 계약 100%, 거짓전제 거절 97.5%(영어 20/20, 한국어 19/20), 근거부족 거절 100%, 충실도 0.888, 문항당 1.77초를 달성한 이중언어 논문 질의응답 시스템 요약입니다.</desc>
<style>text{font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo','Segoe UI',sans-serif;font-variant-numeric:tabular-nums}</style>
<rect width="1200" height="760" fill="#FAF7F0"/>
<text x="48" y="72" fill="#002147" font-size="42" font-weight="760">논문 질의응답과 근거 추적</text>
<text x="48" y="114" fill="#4D5F73" font-size="20.5">BGE-M3 하이브리드 검색 · FP32 mDeBERTa-v3 Scientific NLI · Qwen3-4B Guided JSON · Conformal Gate</text>
<line x1="48" y1="142" x2="1152" y2="142" stroke="#E2DCCB" stroke-width="1.5"/>

<rect x="48" y="168" width="442" height="418" rx="14" fill="#FFFFFF" stroke="#E2DCCB" stroke-width="1.3"/>
<text x="74" y="208" fill="#002147" font-size="22.5" font-weight="750">근거 검증 파이프라인</text>
<text x="74" y="234" fill="#586778" font-size="14.5">영문 논문 PDF → 한·영 질문 응답 및 문장별 원문 인용</text>

<rect x="74" y="254" width="390" height="68" rx="10" fill="#F4F7FB" stroke="#A6BDD8" stroke-width="1.3"/>
<text x="92" y="280" fill="#002147" font-size="12.5" font-weight="750">01 하이브리드 검색 및 Parent 1,800자 확장</text>
<text x="92" y="306" fill="#132234" font-size="16" font-weight="700">BM25 + BGE-M3 + 이중 점수 리랭킹</text>

<path d="M269 322V344" stroke="#002147" stroke-width="2.6"/>
<path d="M262 337L269 345L276 337" fill="none" stroke="#002147" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="348" width="390" height="72" rx="10" fill="#FCF5DC" stroke="#E8B923" stroke-width="1.8"/>
<text x="92" y="374" fill="#8C6400" font-size="12.5" font-weight="750">02 Scientific NLI 사전 차단 게이트 (모순 임계 0.95)</text>
<text x="92" y="399" fill="#132234" font-size="15.8" font-weight="700">mDeBERTa-v3 FP32 파인튜닝 · 패리티 게이트 내장</text>

<path d="M269 420V442" stroke="#C28E0E" stroke-width="2.6"/>
<path d="M262 435L269 443L276 435" fill="none" stroke="#C28E0E" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="446" width="390" height="70" rx="10" fill="#F4F7FB" stroke="#002147" stroke-width="1.6"/>
<text x="92" y="472" fill="#002147" font-size="12.5" font-weight="750">03 Guided JSON 및 결정론적 인용·수치 검증</text>
<text x="92" y="497" fill="#132234" font-size="15.8" font-weight="700">Qwen3-4B-Instruct-2507 + Conformal Gate</text>

<rect x="74" y="530" width="390" height="38" rx="8" fill="#FCF5DC" stroke="#E8B923" stroke-width="1"/>
<text x="269" y="554" text-anchor="middle" fill="#6E4E00" font-size="13.8" font-weight="750">논문 그룹 5폴드 OOF AURC 0.0669 · 응답 계약 240/240</text>

<path d="M490 377H542V230H590" fill="none" stroke="#5C6E80" stroke-width="2.8" stroke-linejoin="round"/>
<path d="M490 377H590" fill="none" stroke="#E8B923" stroke-width="3.0"/>
<path d="M490 377H542V522H590" fill="none" stroke="#002147" stroke-width="3.4" stroke-linejoin="round"/>
<circle cx="542" cy="377" r="5" fill="#002147"/>
<path d="M581 223L591 230L581 237" fill="none" stroke="#5C6E80" stroke-width="2.8" stroke-linecap="round"/>
<path d="M581 370L591 377L581 384" fill="none" stroke="#E8B923" stroke-width="3.0" stroke-linecap="round"/>
<path d="M581 515L591 522L581 529" fill="none" stroke="#002147" stroke-width="3.4" stroke-linecap="round"/>

<rect x="598" y="168" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#E2DCCB" stroke-width="1.3"/>
<rect x="598" y="168" width="9" height="124" rx="4" fill="#5C6E80"/>
<text x="628" y="204" fill="#4D5F73" font-size="13.5" font-weight="750">미공개 HOLDOUT 정답 유지 및 근거 정확도 (240문항)</text>
<text x="628" y="241" fill="#002147" font-size="25" font-weight="780">정답 유지 88.8% · 페이지 인용 76.9%</text>
<text x="628" y="272" fill="#4D5F73" font-size="14.5">영어 핵심사실 54/80 (67.5%) · 한국어 41/80 (51.3%) · 수치 검증 95.1%</text>

<rect x="598" y="314" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#E8B923" stroke-width="1.8"/>
<rect x="598" y="314" width="9" height="124" rx="4" fill="#E8B923"/>
<text x="628" y="350" fill="#8C6400" font-size="13.5" font-weight="750">환각 및 거짓전제 방어 성능 (SCIENTIFIC NLI)</text>
<text x="628" y="387" fill="#002147" font-size="25" font-weight="780">거짓전제 거절 97.5% · 근거부족 100%</text>
<text x="628" y="418" fill="#4D5F73" font-size="14.5">영어 거짓전제 20/20 (100%) · 한국어 19/20 (95%) · DEV 분리도 0.001 대 0.999</text>

<rect x="598" y="460" width="554" height="126" rx="14" fill="#FFFFFF" stroke="#002147" stroke-width="2.2"/>
<rect x="598" y="460" width="10" height="126" rx="4" fill="#002147"/>
<rect x="1028" y="476" width="104" height="26" rx="13" fill="#FCF5DC" stroke="#E8B923" stroke-width="1"/>
<text x="1080" y="494" text-anchor="middle" fill="#6E4E00" font-size="12.5" font-weight="750">최종 채택 모델</text>
<text x="628" y="496" fill="#002147" font-size="13.5" font-weight="750">의미 평가 · 응답 속도 · 소거 검증 규율</text>
<text x="628" y="533" fill="#002147" font-size="24" font-weight="780">충실도 0.888 · 문항당 1.77초 추론</text>
<text x="628" y="564" fill="#4D5F73" font-size="14.2">EN 완전성 0.849 / KO 0.711 · Post NLI와 LoRA SFT 과적합 소거 후 최종 채택</text>

<rect x="48" y="610" width="1104" height="106" rx="16" fill="#002147"/>
<text x="80" y="654" fill="#FFFFFF" font-size="26.5" font-weight="760">미공개 Holdout 240문항 실측 및 RTX 3060 로컬 검증 완료</text>
<text x="80" y="690" fill="#D7E2F0" font-size="18.5">FastAPI 인터랙티브 데모 · FP32 1.048 GiB 패딩 오차 0.000000 · ONNX INT8 배포 게이트 자동 차단</text>
<text x="1120" y="660" text-anchor="end" fill="#F4CC47" font-size="38" font-weight="780">97.5% / 0.888</text>
<text x="1120" y="690" text-anchor="end" fill="#F6E5A8" font-size="16.5">거짓전제 방어 · 의미 충실도</text>
</svg>
"""

    en_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">Evidence-Grounded Paper Q&amp;A: BGE-M3 · Scientific NLI · Qwen3-4B</title>
<desc id="desc">Bilingual EN and KO scientific paper Q&amp;A achieving 100% contract validity, 97.5% false-premise refusal (EN 20/20, KO 19/20), 100% unsupported refusal, 0.888 semantic faithfulness, and 1.77s per query on 240 unseen corpus30 Holdout outputs.</desc>
<style>text{font-family:'Segoe UI','Noto Sans','Helvetica Neue',Arial,sans-serif;font-variant-numeric:tabular-nums}</style>
<rect width="1200" height="760" fill="#FAF7F0"/>
<text x="48" y="72" fill="#002147" font-size="40" font-weight="760">Evidence-Grounded Paper Q&amp;A</text>
<text x="48" y="114" fill="#4D5F73" font-size="20">BGE-M3 Hybrid Retrieval · FP32 mDeBERTa-v3 Scientific NLI · Qwen3-4B Guided JSON · Conformal Gate</text>
<line x1="48" y1="142" x2="1152" y2="142" stroke="#E2DCCB" stroke-width="1.5"/>

<rect x="48" y="168" width="442" height="418" rx="14" fill="#FFFFFF" stroke="#E2DCCB" stroke-width="1.3"/>
<text x="74" y="208" fill="#002147" font-size="22" font-weight="750">Grounded Verification Stack</text>
<text x="74" y="234" fill="#586778" font-size="14.2">Single English Paper → Bilingual EN/KO Claims &amp; Quotes</text>

<rect x="74" y="254" width="390" height="68" rx="10" fill="#F4F7FB" stroke="#A6BDD8" stroke-width="1.3"/>
<text x="92" y="280" fill="#002147" font-size="12.2" font-weight="750">01 HYBRID RETRIEVAL &amp; PARENT (1,800 CHARS)</text>
<text x="92" y="306" fill="#132234" font-size="16" font-weight="700">BM25 + BGE-M3 + Dual-Score Rerank</text>

<path d="M269 322V344" stroke="#002147" stroke-width="2.6"/>
<path d="M262 337L269 345L276 337" fill="none" stroke="#002147" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="348" width="390" height="72" rx="10" fill="#FCF5DC" stroke="#E8B923" stroke-width="1.8"/>
<text x="92" y="374" fill="#8C6400" font-size="12.2" font-weight="750">02 SCIENTIFIC NLI PRE-GATE (THRESHOLD 0.95)</text>
<text x="92" y="399" fill="#132234" font-size="15.8" font-weight="700">FP32 mDeBERTa-v3 · Built-In Parity Gate</text>

<path d="M269 420V442" stroke="#C28E0E" stroke-width="2.6"/>
<path d="M262 435L269 443L276 435" fill="none" stroke="#C28E0E" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="446" width="390" height="70" rx="10" fill="#F4F7FB" stroke="#002147" stroke-width="1.6"/>
<text x="92" y="472" fill="#002147" font-size="12.2" font-weight="750">03 GUIDED JSON &amp; QUOTE / NUMERIC VERIFIER</text>
<text x="92" y="497" fill="#132234" font-size="15.8" font-weight="700">Qwen3-4B-Instruct-2507 + Conformal Gate</text>

<rect x="74" y="530" width="390" height="38" rx="8" fill="#FCF5DC" stroke="#E8B923" stroke-width="1"/>
<text x="269" y="554" text-anchor="middle" fill="#6E4E00" font-size="13.5" font-weight="750">Paper-Grouped 5-Fold OOF AURC 0.0669 · 240/240 Valid</text>

<path d="M490 377H542V230H590" fill="none" stroke="#5C6E80" stroke-width="2.8" stroke-linejoin="round"/>
<path d="M490 377H590" fill="none" stroke="#E8B923" stroke-width="3.0"/>
<path d="M490 377H542V522H590" fill="none" stroke="#002147" stroke-width="3.4" stroke-linejoin="round"/>
<circle cx="542" cy="377" r="5" fill="#002147"/>
<path d="M581 223L591 230L581 237" fill="none" stroke="#5C6E80" stroke-width="2.8" stroke-linecap="round"/>
<path d="M581 370L591 377L581 384" fill="none" stroke="#E8B923" stroke-width="3.0" stroke-linecap="round"/>
<path d="M581 515L591 522L581 529" fill="none" stroke="#002147" stroke-width="3.4" stroke-linecap="round"/>

<rect x="598" y="168" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#E2DCCB" stroke-width="1.3"/>
<rect x="598" y="168" width="9" height="124" rx="4" fill="#5C6E80"/>
<text x="628" y="204" fill="#4D5F73" font-size="13.2" font-weight="750">ANSWER RETENTION &amp; GROUNDING · HOLDOUT (240 OUTPUTS)</text>
<text x="628" y="241" fill="#002147" font-size="25" font-weight="780">88.8% Retained · 76.9% Page Hit</text>
<text x="628" y="272" fill="#4D5F73" font-size="14.2">EN Fact Recall 54/80 (67.5%) · KO 41/80 (51.3%) · Numeric Verified 95.1%</text>

<rect x="598" y="314" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#E8B923" stroke-width="1.8"/>
<rect x="598" y="314" width="9" height="124" rx="4" fill="#E8B923"/>
<text x="628" y="350" fill="#8C6400" font-size="13.2" font-weight="750">HALLUCINATION &amp; FALSE-PREMISE DEFENSE (SCIENTIFIC NLI)</text>
<text x="628" y="387" fill="#002147" font-size="25" font-weight="780">97.5% False Premise · 100% Unsupported</text>
<text x="628" y="418" fill="#4D5F73" font-size="14.2">EN 20/20 (100%) · KO 19/20 (95%) · DEV NLI Split 0.001 vs 0.999</text>

<rect x="598" y="460" width="554" height="126" rx="14" fill="#FFFFFF" stroke="#002147" stroke-width="2.2"/>
<rect x="598" y="460" width="10" height="126" rx="4" fill="#002147"/>
<rect x="1024" y="476" width="108" height="26" rx="13" fill="#FCF5DC" stroke="#E8B923" stroke-width="1"/>
<text x="1078" y="494" text-anchor="middle" fill="#6E4E00" font-size="12.5" font-weight="750">FINAL MODEL</text>
<text x="628" y="496" fill="#002147" font-size="13.2" font-weight="750">SEMANTIC JUDGE · LATENCY · ABLATION DISCIPLINE</text>
<text x="628" y="533" fill="#002147" font-size="23.5" font-weight="780">Faithfulness 0.888 · 1.77s / Query</text>
<text x="628" y="564" fill="#4D5F73" font-size="14">EN Completeness 0.849 / KO 0.711 · Post-NLI &amp; LoRA SFT rejected on DEV</text>

<rect x="48" y="610" width="1104" height="106" rx="16" fill="#002147"/>
<text x="80" y="654" fill="#FFFFFF" font-size="25.5" font-weight="760">Unseen Holdout (240) &amp; Local RTX 3060 Verified</text>
<text x="80" y="690" fill="#D7E2F0" font-size="18.2">FastAPI Interactive Demo · FP32 1.048 GiB (0.000000 Pad Diff) · ONNX INT8 Auto-Blocked</text>
<text x="1120" y="660" text-anchor="end" fill="#F4CC47" font-size="38" font-weight="780">97.5% / 0.888</text>
<text x="1120" y="690" text-anchor="end" fill="#F6E5A8" font-size="16.5">FP Refusal · Faithfulness</text>
</svg>
"""
    (ko_dir / "paper-qna-thumbnail.svg").write_text(ko_svg, encoding="utf-8")
    (en_dir / "paper-qna-thumbnail.svg").write_text(en_svg, encoding="utf-8")


def plot_holdout_performance(out_path: Path, exp4_rep: dict[str, Any], sem_holdout: dict[str, Any], lang: str) -> None:
    set_plot_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=220, gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.patch.set_facecolor(PAPER)

    en, ko, pair = exp4_rep["en"], exp4_rep["ko"], exp4_rep["paired"]
    en_vals = [
        100 * en["answerable_retained"] / 80,
        100 * en["all_fact_aliases"] / 80,
        100 * en["acceptable_source_page_hit"] / 80,
        100 * en["unsupported_refused"] / 20,
        100 * en["false_premise_refused"] / 20,
    ]
    ko_vals = [
        100 * ko["answerable_retained"] / 80,
        100 * ko["all_fact_aliases"] / 80,
        100 * ko["acceptable_source_page_hit"] / 80,
        100 * ko["unsupported_refused"] / 20,
        100 * ko["false_premise_refused"] / 20,
    ]
    pair_vals = [
        100 * pair["answerable_retained"] / 80,
        100 * pair["all_fact_aliases"] / 80,
        100 * pair["acceptable_source_page_hit"] / 80,
        100 * pair["unsupported_refused"] / 20,
        100 * pair["false_premise_refused"] / 20,
    ]

    if lang == "ko":
        x_labels = [
            "정답가능 유지\n(80쌍)",
            "핵심사실 일치\n(80쌍)",
            "정답 페이지 인용\n(80쌍)",
            "근거부족 거절\n(20쌍)",
            "거짓전제 거절\n(20쌍)",
        ]
        leg = ["영어 (120문항)", "한국어 (120문항)", "한·영 동시 통과 (120쌍)"]
        y_lbl = "통과율 (%)"
        t1 = "A. 미공개 Holdout 이중언어 세부 성능 (20편, 240문항)"
        q_labels = [
            "응답 계약 준수 (240/240)",
            "수치 충실도 검증 (결정론적)",
            "평균 의미 충실도 (14B Judge)",
            "원문 인용 일치 (결정론적)",
            "영어 의미 완전성 (14B Judge)",
            "한국어 의미 완전성 (14B Judge)",
            "한·영 동시 완전성 (14B Judge)",
        ]
        x2_lbl = "검증 점수 (%)"
        t2 = "B. 결정론적 검증 및 14B 의미 평가 요약"
        foot = f"평균 응답 시간: {pair['mean_wall_seconds']:.2f}초/문항 (vLLM L4)"
    else:
        x_labels = [
            "Answerable\nRetained (80)",
            "All Fact\nAliases (80)",
            "Source Page\nHit (80)",
            "Unsupported\nRefused (20)",
            "False Premise\nRefused (20)",
        ]
        leg = ["English (120 outputs)", "Korean (120 outputs)", "Paired Strict (120 pairs)"]
        y_lbl = "Pass Rate (%)"
        t1 = "A. Unseen Holdout Bilingual Breakdown (20 Papers, 240 Outputs)"
        q_labels = [
            "Contract Valid (240/240)",
            "Numeric Fidelity (Verifier)",
            "Mean Faithfulness (14B Judge)",
            "Verbatim Quote Match (Verifier)",
            "EN Completeness (14B Judge)",
            "KO Completeness (14B Judge)",
            "Paired Completeness (14B Judge)",
        ]
        x2_lbl = "Score (%)"
        t2 = "B. Deterministic Verification & Semantic Judge"
        foot = f"Mean wall latency: {pair['mean_wall_seconds']:.2f}s/query (vLLM L4)"

    ax1.set_facecolor("#FFFFFF")
    x = np.arange(len(x_labels))
    w = 0.25
    b_en = ax1.bar(x - w, en_vals, w, label=leg[0], color=TEAL, edgecolor="#001633", linewidth=0.7)
    b_ko = ax1.bar(x, ko_vals, w, label=leg[1], color=AMBER, edgecolor="#B58400", linewidth=0.8)
    b_pr = ax1.bar(x + w, pair_vals, w, label=leg[2], color=SLATE, edgecolor="#3F4E5E", linewidth=0.7)

    for bars, col in ((b_en, TEAL), (b_ko, AMBER_TEXT), (b_pr, "#334155")):
        for rect in bars:
            val = rect.get_height()
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                val + 1.5,
                f"{val:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=col,
                fontweight="bold",
            )

    ax1.set_ylim(0, 118)
    ax1.set_ylabel(y_lbl, fontsize=10, fontweight="bold", color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8.8, color=INK)
    ax1.set_title(t1, fontsize=11.0, fontweight="bold", color=INK, pad=11)
    ax1.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    ax2.set_facecolor("#FFFFFF")
    q_vals = [
        100.0 * pair["contract_valid"] / 240,
        100.0 * pair["numeric_verified_rate"],
        100.0 * sem_holdout["mean_faithfulness"],
        100.0 * pair["quote_verified_rate"],
        100.0 * sem_holdout["en_completeness"],
        100.0 * sem_holdout["ko_completeness"],
        100.0 * sem_holdout["paired_completeness"],
    ]
    q_cols = [TEAL, TEAL, TEAL_MID, TEAL_MID, AMBER, AMBER, SLATE]
    y = np.arange(len(q_labels))[::-1]
    ax2.barh(y, q_vals, height=0.62, color=q_cols, edgecolor="#2B3D41", linewidth=0.5)
    for yi, val in zip(y, q_vals):
        ax2.text(val + 1.4, yi, f"{val:.1f}%", va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(q_labels, fontsize=8.5, color=INK)
    ax2.set_xlim(0, 116)
    ax2.set_ylim(-1.25, len(q_labels) - 0.35)
    ax2.set_xlabel(x2_lbl, fontsize=9.8, fontweight="bold", color=INK)
    ax2.set_title(t2, fontsize=11.0, fontweight="bold", color=INK, pad=11)
    ax2.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(
        0.50,
        0.035,
        foot,
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=INK,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=AMBER_SOFT, edgecolor="#DEC299", linewidth=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor=PAPER)
    plt.close(fig)


def plot_dev_ablation(out_path: Path, dev_rows: list[dict[str, Any]], lang: str) -> None:
    set_plot_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.1), dpi=220, gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor(PAPER)

    if lang == "ko":
        short_map = {
            "exp1c_bge_m3_qwen3_4b_2507_zeroshot": "Exp 1C\nBase 4B",
            "exp1c_teacher_qwen3_14b_4bit": "Exp 1C\n14B 교사",
            "exp2a_pre_sliding_nli": "Exp 2A\nNLI 0.70",
            "exp2b_pre_and_2stage_post_nli": "Exp 2B\nPost NLI\n(기각)",
            "exp3a_qwen3_4b_2507_comp_only_sft": "Exp 3A\nLoRA SFT\n(기각)",
            "exp2c_base4b_pre_nli_095": "Exp 2C\nNLI 0.95\n(최종)",
        }
        leg1 = [
            "한·영 정답가능 유지 (%)",
            "한·영 핵심사실 일치 (%)",
            "한·영 거짓전제 거절 (%)",
            "최종 채택 구성 (Exp 2C)",
        ]
        y1_lbl = "DEV 통과율 (%)"
        t1 = "A. DEV 120문항 단계별 소거 실험 및 기각 판정"
        leg2 = ["영어 핵심사실 (40문항)", "한국어 핵심사실 (40문항)"]
        x2_lbl = "필수 핵심사실 모두 일치 (%)"
        t2 = "B. 한·영 핵심사실 비교 (Post NLI·SFT 기각 근거)"
    else:
        short_map = {
            "exp1c_bge_m3_qwen3_4b_2507_zeroshot": "Exp 1C\nBase 4B",
            "exp1c_teacher_qwen3_14b_4bit": "Exp 1C\n14B AWQ",
            "exp2a_pre_sliding_nli": "Exp 2A\nNLI 0.70",
            "exp2b_pre_and_2stage_post_nli": "Exp 2B\nPost NLI\n(Rejected)",
            "exp3a_qwen3_4b_2507_comp_only_sft": "Exp 3A\nLoRA SFT\n(Rejected)",
            "exp2c_base4b_pre_nli_095": "Exp 2C\nNLI 0.95\n(Winner)",
        }
        leg1 = [
            "Paired Answerable Retained (%)",
            "Paired All Fact Aliases (%)",
            "Paired False Premise Refused (%)",
            "Selected Final Stack (Exp 2C)",
        ]
        y1_lbl = "DEV Rate (%)"
        t1 = "A. DEV 120 Output Ablation Progression & Rejections"
        leg2 = ["English Facts (40)", "Korean Facts (40)"]
        x2_lbl = "All Required Fact Aliases Matched (%)"
        t2 = "B. EN vs KO Fact Recall (Why Post NLI & SFT Failed)"

    names = [short_map[r["exp_id"]] for r in dev_rows]
    x = np.arange(len(names))
    ans_pct = [100 * r["pair_ans"] / 40 for r in dev_rows]
    facts_pct = [100 * r["pair_facts"] / 40 for r in dev_rows]
    fp_pct = [100 * r["fp_pair"] / 10 for r in dev_rows]

    ax1.set_facecolor("#FFFFFF")
    ax1.plot(x, ans_pct, marker="o", linewidth=2.3, color=TEAL, label=leg1[0])
    ax1.plot(x, facts_pct, marker="s", linewidth=2.4, color=AMBER, label=leg1[1])
    ax1.plot(x, fp_pct, marker="^", linewidth=2.2, linestyle="--", color=CRIMSON, label=leg1[2])
    ax1.axvspan(len(names) - 1.42, len(names) - 0.58, color=TEAL_SOFT, alpha=0.7, label=leg1[3])

    for i in range(len(names)):
        ax1.annotate(
            f"{ans_pct[i]:.0f}%",
            (x[i], ans_pct[i]),
            textcoords="offset points",
            xytext=(0, -15 if i in (0, 1) else -14),
            ha="center",
            fontsize=7.9,
            color=TEAL,
            fontweight="bold",
        )
        ax1.annotate(
            f"{facts_pct[i]:.0f}%",
            (x[i], facts_pct[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.0,
            color=AMBER_TEXT,
            fontweight="bold",
        )
        fp_dy = 7 if fp_pct[i] >= 95 else (-14 if i == 1 else 7)
        ax1.annotate(
            f"{fp_pct[i]:.0f}%",
            (x[i], fp_pct[i]),
            textcoords="offset points",
            xytext=(0, fp_dy),
            ha="center",
            fontsize=8.0,
            color=CRIMSON,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8.3, color=INK)
    ax1.set_ylim(14, 114)
    ax1.set_ylabel(y1_lbl, fontsize=10, fontweight="bold", color=INK)
    ax1.set_title(t1, fontsize=11.0, fontweight="bold", color=INK, pad=11)
    ax1.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower center", ncol=2, fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    ax2.set_facecolor("#FFFFFF")
    key_idxs = [0, 1, 3, 4, 5]  # Base4B, 14B, Post-NLI, SFT, Winner 2C
    k_names = [short_map[dev_rows[i]["exp_id"]].replace("\n", " ") for i in key_idxs]
    en_vals = [100 * dev_rows[i]["en_facts"] / 40 for i in key_idxs]
    ko_vals = [100 * dev_rows[i]["ko_facts"] / 40 for i in key_idxs]
    y = np.arange(len(k_names))[::-1]
    h = 0.35
    ax2.barh(y + h / 2, en_vals, h, label=leg2[0], color=TEAL)
    ax2.barh(y - h / 2, ko_vals, h, label=leg2[1], color=AMBER)
    for yi, ev, kv in zip(y, en_vals, ko_vals):
        ax2.text(ev + 1.2, yi + h / 2, f"{ev:.1f}%", va="center", fontsize=8.2, color=TEAL, fontweight="bold")
        ax2.text(kv + 1.2, yi - h / 2, f"{kv:.1f}%", va="center", fontsize=8.2, color=AMBER_TEXT, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(k_names, fontsize=8.2, color=INK)
    ax2.set_xlim(0, 108)
    ax2.set_ylim(-0.6, len(k_names) + 0.35)
    ax2.set_xlabel(x2_lbl, fontsize=9.6, fontweight="bold", color=INK)
    ax2.set_title(t2, fontsize=11.0, fontweight="bold", color=INK, pad=11)
    ax2.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", ncol=2, fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor=PAPER)
    plt.close(fig)


def plot_nli_and_conformal(
    out_path: Path,
    sweep_rows: list[dict[str, Any]],
    gate_payload: dict[str, Any],
    ans_contras: list[float],
    fp_contras: list[float],
    lang: str,
) -> None:
    set_plot_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=220)
    fig.patch.set_facecolor(PAPER)

    thrs = [r["nli_thr"] for r in sweep_rows if r["nli_thr"] <= 1.0]
    p_ans = [100 * r["pair_ans"] / 40 for r in sweep_rows if r["nli_thr"] <= 1.0]
    p_facts = [100 * r["pair_facts"] / 40 for r in sweep_rows if r["nli_thr"] <= 1.0]
    p_fp = [100 * r["fp_pair"] / 10 for r in sweep_rows if r["nli_thr"] <= 1.0]

    if lang == "ko":
        leg1 = [
            "거짓전제 거절 (0.70~0.99 전 구간 100%)",
            "한·영 정답가능 유지 (87.5% → 92.5%)",
            "한·영 핵심사실 일치 (57.5% → 60.0%)",
            "최종 채택 임계값 = 0.95",
        ]
        box_txt = (
            f"FP32 Scientific NLI 분포 분리 (DEV)\n"
            f"• 정답가능 중앙값 = {statistics.median(ans_contras):.3f}\n"
            f"• 거짓전제 최소값 = {min(fp_contras):.3f}\n"
            f"• 0.95 이상 차단: {sum(x >= 0.95 for x in fp_contras)}/{len(fp_contras)}건 (100%)"
        )
        x1_lbl = "Scientific NLI 모순 차단 임계값"
        y1_lbl = "DEV 성능 지표 (%)"
        t1 = "A. Scientific NLI 분포 분리 및 임계값 평탄 구간"
        leg2 = ["정답가능 커버리지 (%)", "부정 문항 누출률 (%)"]
        y2_lbl = "논문 그룹 5폴드 OOF 비율 (%)"
        t2 = f"B. Conformal 위험·커버리지 게이트 (OOF AURC = {gate_payload['oof_aurc']:.4f})"
        tick_fmt = lambda a, t: f"α = {a:.2f}\n(임계 {t:.3f})"
    else:
        leg1 = [
            "False Premise Refused (100% across 0.70 to 0.99)",
            "Paired Answerable Retained (87.5% -> 92.5%)",
            "Paired All Fact Aliases (57.5% -> 60.0%)",
            "Selected Gate Threshold = 0.95",
        ]
        box_txt = (
            f"FP32 Scientific NLI Separation (DEV)\n"
            f"• Answerable median = {statistics.median(ans_contras):.3f}\n"
            f"• False Premise min = {min(fp_contras):.3f}\n"
            f"• Caught >= 0.95: {sum(x >= 0.95 for x in fp_contras)}/{len(fp_contras)} (100%)"
        )
        x1_lbl = "Scientific NLI Contradiction Threshold"
        y1_lbl = "DEV Metric (%)"
        t1 = "A. Scientific NLI Separation & Threshold Plateau"
        leg2 = ["Answerable Coverage (%)", "Negative Leak Rate (%)"]
        y2_lbl = "5 Fold Paper Grouped OOF Rate (%)"
        t2 = f"B. Conformal Risk Coverage Gate (OOF AURC = {gate_payload['oof_aurc']:.4f})"
        tick_fmt = lambda a, t: f"α = {a:.2f}\n(thr {t:.3f})"

    ax1.set_facecolor("#FFFFFF")
    ax1.plot(thrs, p_fp, marker="^", linewidth=2.4, color=CRIMSON, label=leg1[0])
    ax1.plot(thrs, p_ans, marker="o", linewidth=2.4, color=TEAL, label=leg1[1])
    ax1.plot(thrs, p_facts, marker="s", linewidth=2.4, color=AMBER, label=leg1[2])
    ax1.axvline(0.95, color=INK, linestyle=":", linewidth=1.8, label=leg1[3])
    ax1.axvspan(0.90, 0.99, color=TEAL_SOFT, alpha=0.55)
    ax1.text(
        0.70,
        30,
        box_txt,
        fontsize=8.2,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor=TEAL, linewidth=1.0),
    )
    ax1.set_xlim(0.68, 1.01)
    ax1.set_ylim(0, 114)
    ax1.set_xlabel(x1_lbl, fontsize=9.8, fontweight="bold", color=INK)
    ax1.set_ylabel(y1_lbl, fontsize=9.8, fontweight="bold", color=INK)
    ax1.set_title(t1, fontsize=11.0, fontweight="bold", color=INK, pad=11)
    ax1.grid(color=GRID, linestyle="--", linewidth=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    ax2.set_facecolor("#FFFFFF")
    rc = gate_payload["oof_risk_coverage"]
    alphas = [r["alpha"] for r in rc]
    covs = [100 * r["answerable_coverage"] for r in rc]
    leaks = [100 * r["negative_leak_rate"] for r in rc]
    thrs_c = [r["oof_threshold"] for r in rc]
    x = np.arange(len(alphas))
    w = 0.34
    b_cov = ax2.bar(x - w / 2, covs, w, color=TEAL, label=leg2[0])
    b_lk = ax2.bar(x + w / 2, leaks, w, color=CRIMSON, label=leg2[1])
    for r, v in zip(b_cov, covs):
        ax2.text(r.get_x() + r.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", fontsize=8.8, color=TEAL, fontweight="bold")
    for r, v in zip(b_lk, leaks):
        ax2.text(r.get_x() + r.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", fontsize=8.8, color=CRIMSON, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([tick_fmt(a, t) for a, t in zip(alphas, thrs_c)], fontsize=8.8, color=INK)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel(y2_lbl, fontsize=9.8, fontweight="bold", color=INK)
    ax2.set_title(t2, fontsize=11.0, fontweight="bold", color=INK, pad=11)
    ax2.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.4, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor=PAPER)
    plt.close(fig)


def main() -> int:
    reports_dir = ROOT / "reports"
    en_fig_dir = ROOT / "docs" / "figures" / "en"
    ko_fig_dir = ROOT / "docs" / "figures" / "ko"
    ckpt_dir = ROOT / "outputs" / "v2_checkpoints"
    en_fig_dir.mkdir(parents=True, exist_ok=True)
    ko_fig_dir.mkdir(parents=True, exist_ok=True)

    questions = read_jsonl(ROOT / "data" / "benchmarks" / "corpus30" / "questions.jsonl")
    dev_qs = [q for q in questions if q["split"] == "dev"]
    raw_log = read_jsonl(ckpt_dir / "experiment_log_raw.jsonl")
    exp1b = json.loads((reports_dir / "exp1b_retrieval_ab.json").read_text(encoding="utf-8"))
    sem_dev = json.loads((reports_dir / "exp1c_qwen3_4b_2507_semantic_summary.json").read_text(encoding="utf-8"))
    sem_holdout = json.loads((reports_dir / "exp4_v2_final_holdout_semantic_summary.json").read_text(encoding="utf-8"))
    gate_payload = json.loads((reports_dir / "calibrated_gate.json").read_text(encoding="utf-8"))
    freeze_v2 = json.loads((ROOT / "data" / "corpus_v2_holdout" / "freeze_v2.json").read_text(encoding="utf-8"))

    # Keep clean deduplicated experiments (skip superseded BF16 NaN runs at raw indices 2 and 3)
    canonical_entries = [raw_log[0], raw_log[1], raw_log[4], raw_log[5], raw_log[6], raw_log[7], raw_log[8]]
    (reports_dir / "experiment_log.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in canonical_entries) + "\n",
        encoding="utf-8",
    )

    # Evaluate DEV & Holdout checkpoints directly
    c1_recs = read_jsonl(ckpt_dir / "exp1c_bge_m3_qwen3_4b_2507_zeroshot_dev.checkpoint.jsonl")
    c2a_recs = read_jsonl(ckpt_dir / "exp2a_pre_sliding_nli_dev.checkpoint.jsonl")
    c2b_recs = read_jsonl(ckpt_dir / "exp2b_pre_and_2stage_post_nli_dev.checkpoint.jsonl")
    c3a_recs = read_jsonl(ckpt_dir / "exp3a_qwen3_4b_2507_comp_only_sft_dev.checkpoint.jsonl")
    exp4_recs = read_jsonl(ckpt_dir / "exp4_v2_final_regression_holdout.checkpoint.jsonl")

    rep_1c = evaluate_predictions(c1_recs, questions, split="dev")
    rep_2a = evaluate_predictions(c2a_recs, questions, split="dev")
    rep_2b = evaluate_predictions(c2b_recs, questions, split="dev")
    rep_3a = evaluate_predictions(c3a_recs, questions, split="dev")
    exp4_report = evaluate_predictions(exp4_recs, questions, split="holdout")

    # Exact DEV NLI threshold sweep + reconstruct Exp 2C exact records (thr=0.95)
    c1_map = {r["case_id"]: r for r in c1_recs}
    c2_map = {r["case_id"]: r for r in c2a_recs}
    ans_contras = [float(c2_map[q["case_id"]]["output"]["pipeline"].get("nli_max_contradiction", 0.0)) for q in dev_qs if q["answerability"] == "answerable"]
    fp_contras = [float(c2_map[q["case_id"]]["output"]["pipeline"].get("nli_max_contradiction", 0.0)) for q in dev_qs if q["answerability"] == "false_premise"]

    sweep_rows = []
    rep_2c = None
    for thr in [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.01]:
        sim = []
        for q in dev_qs:
            cid = q["case_id"]
            rec = copy.deepcopy(c1_map[cid])
            p2 = c2_map[cid]["output"]["pipeline"]
            nc, ne = float(p2.get("nli_max_contradiction", 0.0)), float(p2.get("nli_max_entailment", 0.5))
            if nc >= thr and nc > ne:
                rec["output"]["response"] = canonical_abstention_payload(q["language"], f"NLI contra={nc:.2f}")
                rec["output"]["cited_evidence"] = []
            sim.append(rec)
        rep = evaluate_predictions(sim, questions, split="dev")
        if abs(thr - 0.95) < 1e-6:
            rep_2c = rep
        sweep_rows.append(
            {
                "nli_thr": thr,
                "pair_ans": rep["paired"]["answerable_retained"],
                "ko_ans": rep["ko"]["answerable_retained"],
                "pair_facts": rep["paired"]["all_fact_aliases"],
                "en_facts": rep["en"]["all_fact_aliases"],
                "ko_facts": rep["ko"]["all_fact_aliases"],
                "unsup_pair": rep["paired"]["unsupported_refused"],
                "fp_en": rep["en"]["false_premise_refused"],
                "fp_ko": rep["ko"]["false_premise_refused"],
                "fp_pair": rep["paired"]["false_premise_refused"],
            }
        )
    assert rep_2c is not None
    with (reports_dir / "nli_threshold_sweep_dev.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # Within-system DEV statistical tests (Exact McNemar & Paper Cluster Bootstrap)
    dev_stat_tests = (
        compute_paired_stat_tests("Exp1C_Base4B_vs_Exp2C_PreNLI095", rep_1c, rep_2c)
        + compute_paired_stat_tests("Exp2A_PreNLI_vs_Exp2B_PostNLI", rep_2a, rep_2b)
        + compute_paired_stat_tests("Exp2A_Base4B_vs_Exp3A_LoRASFT", rep_2a, rep_3a)
    )
    with (reports_dir / "dev_ablation_statistical_tests.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dev_stat_tests[0].keys()))
        w.writeheader()
        w.writerows(dev_stat_tests)

    # Clean DEV ablation summary CSV (current system only)
    decisions = [
        "Adopted Generator",
        "Teacher Ceiling",
        "Threshold Low",
        "Rejected",
        "Rejected",
        "DEV Winner",
    ]
    dev_table = []
    for entry, dec in zip(canonical_entries[:6], decisions):
        p, en, ko = entry["paired"], entry["en"], entry["ko"]
        dev_table.append(
            {
                "exp_id": entry["exp_name"],
                "decision": dec,
                "pair_ans": p["answerable_retained"],
                "ko_ans": ko["answerable_retained"],
                "pair_facts": p["all_fact_aliases"],
                "en_facts": en["all_fact_aliases"],
                "ko_facts": ko["all_fact_aliases"],
                "unsup_pair": p["unsupported_refused"],
                "fp_en": en["false_premise_refused"],
                "fp_ko": ko["false_premise_refused"],
                "fp_pair": p["false_premise_refused"],
                "quote_rate": p.get("quote_verified_rate"),
                "numeric_rate": p.get("numeric_verified_rate"),
                "sec_per_q": round(p["mean_wall_seconds"], 2),
            }
        )
    with (reports_dir / "ablation_dev_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dev_table[0].keys()))
        w.writeheader()
        w.writerows(dev_table)

    # Holdout bilingual breakdown CSV
    holdout_rows = []
    for slice_name, d, sem_comp in [
        ("en", exp4_report["en"], sem_holdout["en_completeness"]),
        ("ko", exp4_report["ko"], sem_holdout["ko_completeness"]),
        ("paired", exp4_report["paired"], sem_holdout["paired_completeness"]),
        ("all_outputs", exp4_report["all_outputs"], None),
    ]:
        holdout_rows.append(
            {
                "slice": slice_name,
                "n_outputs": d["n_outputs"],
                "contract_valid": d["contract_valid"],
                "answerable_retained": d["answerable_retained"],
                "n_answerable": d["n_answerable"],
                "all_fact_aliases": d["all_fact_aliases"],
                "acceptable_source_page_hit": d["acceptable_source_page_hit"],
                "unsupported_refused": d["unsupported_refused"],
                "n_unsupported": d["n_unsupported"],
                "false_premise_refused": d["false_premise_refused"],
                "n_false_premise": d["n_false_premise"],
                "quote_verified_rate": d["quote_verified_rate"],
                "numeric_verified_rate": d["numeric_verified_rate"],
                "semantic_completeness": sem_comp,
                "mean_faithfulness": sem_holdout["mean_faithfulness"] if slice_name == "paired" else None,
                "mean_wall_seconds": d["mean_wall_seconds"],
            }
        )
    with (reports_dir / "holdout_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(holdout_rows[0].keys()))
        w.writeheader()
        w.writerows(holdout_rows)

    local_ver_path = reports_dir / "local_verification.json"
    local_ver = json.loads(local_ver_path.read_text(encoding="utf-8")) if local_ver_path.is_file() else None

    summary = {
        "schema_version": 2,
        "project": "Evidence-Grounded-Paper-QnA",
        "winner_stack": {
            "retrieval": "BM25 + BAAI/bge-m3 + RRF(k=60) + Dual Score BAAI/bge-reranker-v2-m3 + Parent(1800 chars)",
            "scientific_nli": "Fine tuned FP32 MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (threshold=0.95, post NLI=None, runtime parity gate)",
            "generator": "Qwen/Qwen3-4B-Instruct-2507 (Guided JSON response_format, zero shot, no SFT adapter)",
            "conformal_gate": "Paper Grouped 5 Fold OOF Logistic Conformal Gate (OOF AURC=0.0669)",
        },
        "retrieval_ab_dev": exp1b,
        "nli_separation_dev": {
            "answerable_contra_median": round(statistics.median(ans_contras), 4),
            "answerable_contra_max": round(max(ans_contras), 4),
            "false_premise_contra_median": round(statistics.median(fp_contras), 4),
            "false_premise_contra_min": round(min(fp_contras), 4),
            "false_premise_caught_at_095": sum(x >= 0.95 for x in fp_contras),
            "false_premise_total": len(fp_contras),
        },
        "dev_ablation_120": dev_table,
        "dev_ablation_statistical_tests": dev_stat_tests,
        "holdout_240": {
            "paired": exp4_report["paired"],
            "all_outputs": exp4_report["all_outputs"],
            "en": exp4_report["en"],
            "ko": exp4_report["ko"],
            "semantic_judge_holdout": sem_holdout,
            "semantic_judge_dev_exp1c": sem_dev,
        },
        "conformal_gate": {
            "oof_aurc": gate_payload["oof_aurc"],
            "nli_contra_threshold": gate_payload["nli_contra_threshold"],
            "oof_risk_coverage": gate_payload["oof_risk_coverage"],
        },
        "corpus_v2_holdout_freeze": freeze_v2,
        "local_verification": local_ver,
    }
    (reports_dir / "final-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Generate strictly localized EN and KO figures
    write_hero_svgs(en_fig_dir, ko_fig_dir)
    for lang, out_dir in (("en", en_fig_dir), ("ko", ko_fig_dir)):
        plot_holdout_performance(out_dir / "holdout-performance.png", exp4_report, sem_holdout, lang)
        plot_dev_ablation(out_dir / "dev-ablation-progression.png", dev_table, lang)
        plot_nli_and_conformal(out_dir / "nli-conformal-calibration.png", sweep_rows, gate_payload, ans_contras, fp_contras, lang)

    print("[OK] Generated clean reports and strictly localized EN/KO figures!")
    for folder in (en_fig_dir, ko_fig_dir):
        print(f"Folder: {folder.name}/")
        for p in sorted(folder.iterdir()):
            print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KiB)")
    print("\nHoldout exact counts:")
    print("  EN    :", exp4_report["en"])
    print("  KO    :", exp4_report["ko"])
    print("  PAIRED:", exp4_report["paired"])
    print("  ALL   :", exp4_report["all_outputs"])
    print("\nWithin-system DEV statistical tests (non-trivial rows):")
    for row in dev_stat_tests:
        if row["delta"] != 0:
            print(" ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
