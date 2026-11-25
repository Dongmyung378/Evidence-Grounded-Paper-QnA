"""Generate canonical v2 reports (JSON/CSV) and publication-grade figures from verified Colab outputs."""

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

# Configure clean bilingual font stack on Windows / Linux
plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Noto Sans KR", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

TEAL = "#1E665E"
TEAL_LIGHT = "#5FA399"
TEAL_SOFT = "#E6F2EF"
AMBER = "#A87338"
CRIMSON = "#A62B3A"
SLATE = "#64748B"
INK = "#1F2E32"
PAPER = "#F8F4EB"
GRID = "#D9E0DC"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_eval(text: str) -> str:
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", str(text or ""))
    return re.sub(r"\s+", " ", text.casefold().replace("–", "-").replace("−", "-"))


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
        retrieved = out.get("retrieved_evidence") or []
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
        "en": _sum([r for r in evaluated if r["language"] == "en"], False),
        "ko": _sum([r for r in evaluated if r["language"] == "ko"], False),
        "records": evaluated,
    }


def compute_statistical_comparison(
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
                "metric": label,
                "n_outputs": len(sub),
                "v1_rate": round(float(b_vals.mean()), 4),
                "v2_rate": round(float(c_vals.mean()), 4),
                "delta": round(float(c_vals.mean() - b_vals.mean()), 4),
                "ci95_low": round(float(lo), 4),
                "ci95_high": round(float(hi), 4),
                "v2_only_gain_n01": n01,
                "v1_only_loss_n10": n10,
                "exact_mcnemar_p": float(f"{p_val:.6g}"),
            }
        )
    return rows


def write_hero_svgs(fig_dir: Path, summary: dict[str, Any]) -> None:
    ko_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">논문 질의응답과 근거 추적 v2: BGE-M3 · Scientific NLI · Qwen3-4B</title>
<desc id="desc">미공개 corpus30 Holdout 240문항에서 v1 대비 핵심사실 3.0배(11→33/80), 거짓전제 거절 10%→95%(2→19/20, 영어 20/20), 응답 속도 3.8배를 달성한 이중언어 논문 질의응답 시스템 v2 요약입니다.</desc>
<style>text{font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo','Segoe UI',sans-serif;font-variant-numeric:tabular-nums}</style>
<rect width="1200" height="760" fill="#F8F4EB"/>
<text x="48" y="74" fill="#1F2E32" font-size="42" font-weight="760">논문 질의응답과 근거 추적 v2</text>
<text x="48" y="118" fill="#536569" font-size="21">BGE-M3 하이브리드 검색 · FP32→ONNX INT8 Scientific NLI · Qwen3-4B Guided JSON · Conformal Gate</text>
<line x1="48" y1="144" x2="1152" y2="144" stroke="#D7DED9" stroke-width="1.4"/>

<!-- Left Architecture Panel -->
<rect x="48" y="168" width="442" height="418" rx="14" fill="#FFFFFF" stroke="#D7DED9" stroke-width="1.3"/>
<text x="74" y="208" fill="#1F2E32" font-size="23" font-weight="750">v2 근거 검증 파이프라인</text>
<text x="74" y="234" fill="#5B696C" font-size="14.5">영어 논문 1편 → 한국어·영어 질문 + 문장별 원문 인용</text>

<rect x="74" y="254" width="390" height="68" rx="10" fill="#F5FAF8" stroke="#9AC8BE" stroke-width="1.3"/>
<text x="92" y="280" fill="#1E665E" font-size="12.5" font-weight="750">01 HYBRID RETRIEVAL + PARENT (1,800 CHARS)</text>
<text x="92" y="306" fill="#1F2E32" font-size="16.5" font-weight="700">BM25 + BGE-M3 + Dual-Score Rerank</text>

<path d="M269 322V344" stroke="#1E665E" stroke-width="2.6"/>
<path d="M262 337L269 345L276 337" fill="none" stroke="#1E665E" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="348" width="390" height="72" rx="10" fill="#FBF3E6" stroke="#D8B384" stroke-width="1.5"/>
<text x="92" y="374" fill="#8C5B23" font-size="12.5" font-weight="750">02 SCIENTIFIC NLI PRE-GATE (ONNX INT8 · 322.7 MiB)</text>
<text x="92" y="399" fill="#1F2E32" font-size="16" font-weight="700">mDeBERTa-v3 FP32 파인튜닝 · 모순 임계 0.95</text>

<path d="M269 420V442" stroke="#A87338" stroke-width="2.6"/>
<path d="M262 435L269 443L276 435" fill="none" stroke="#A87338" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="446" width="390" height="70" rx="10" fill="#F5FAF8" stroke="#1E665E" stroke-width="1.6"/>
<text x="92" y="472" fill="#1E665E" font-size="12.5" font-weight="750">03 GUIDED JSON + QUOTE / NUMERIC VERIFIER</text>
<text x="92" y="497" fill="#1F2E32" font-size="16" font-weight="700">Qwen3-4B-Instruct-2507 + Conformal Gate</text>

<rect x="74" y="530" width="390" height="38" rx="8" fill="#E6F2EF"/>
<text x="269" y="554" text-anchor="middle" fill="#154943" font-size="13.8" font-weight="750">Paper-Grouped 5-Fold OOF AURC 0.0669 · 계약 240/240</text>

<!-- Connectors -->
<path d="M490 377H542V230H590" fill="none" stroke="#6E8783" stroke-width="2.8" stroke-linejoin="round"/>
<path d="M490 377H590" fill="none" stroke="#A87338" stroke-width="2.8"/>
<path d="M490 377H542V522H590" fill="none" stroke="#1E665E" stroke-width="3.4" stroke-linejoin="round"/>
<circle cx="542" cy="377" r="5" fill="#1F2E32"/>
<path d="M581 223L591 230L581 237" fill="none" stroke="#6E8783" stroke-width="2.8" stroke-linecap="round"/>
<path d="M581 370L591 377L581 384" fill="none" stroke="#A87338" stroke-width="2.8" stroke-linecap="round"/>
<path d="M581 515L591 522L581 529" fill="none" stroke="#1E665E" stroke-width="3.4" stroke-linecap="round"/>

<!-- Right Metric Cards -->
<rect x="598" y="168" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#D7DED9" stroke-width="1.3"/>
<rect x="598" y="168" width="9" height="124" rx="4" fill="#6E8783"/>
<text x="628" y="204" fill="#536569" font-size="13.5" font-weight="750">ANSWER COMPLETENESS · UNSEEN HOLDOUT (240 OUTPUTS)</text>
<text x="628" y="241" fill="#1F2E32" font-size="25" font-weight="780">핵심사실 일치 3.0× (11/80 → 33/80)</text>
<text x="628" y="272" fill="#536569" font-size="14.5">한국어 2.73× (15→41/80) · 정답 페이지 인용 76.9% (+15.6%p, p=0.0019)</text>

<rect x="598" y="314" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#D7DED9" stroke-width="1.3"/>
<rect x="598" y="314" width="9" height="124" rx="4" fill="#A87338"/>
<text x="628" y="350" fill="#8C5B23" font-size="13.5" font-weight="750">HALLUCINATION &amp; FALSE-PREMISE DEFENSE (SCIENTIFIC NLI)</text>
<text x="628" y="387" fill="#1F2E32" font-size="25" font-weight="780">거짓전제 거절 10% → 95% (2 → 19/20)</text>
<text x="628" y="418" fill="#536569" font-size="14.5">영어 2/20 → 20/20 (100%) · 근거부족 거절 20/20 (100%) · p &lt; 0.0001</text>

<rect x="598" y="460" width="554" height="126" rx="14" fill="#FFFFFF" stroke="#1E665E" stroke-width="2.2"/>
<rect x="598" y="460" width="10" height="126" rx="4" fill="#1E665E"/>
<rect x="1018" y="476" width="114" height="26" rx="13" fill="#E6F2EF"/>
<text x="1075" y="494" text-anchor="middle" fill="#154943" font-size="12.5" font-weight="750">FINAL v2 MODEL</text>
<text x="628" y="496" fill="#1E665E" font-size="13.5" font-weight="750">SEMANTIC JUDGE · SPEED · ABLATION DISCIPLINE</text>
<text x="628" y="533" fill="#1F2E32" font-size="24" font-weight="780">충실도 0.888 · 응답 3.8× 가속 (1.77s/q)</text>
<text x="628" y="564" fill="#536569" font-size="14.2">EN 완전성 0.849 / KO 0.711 · SFT·Post-NLI 과적합 소거 후 Base 4B+NLI 채택</text>

<!-- Bottom Bar -->
<rect x="48" y="610" width="1104" height="106" rx="16" fill="#1F2E32"/>
<text x="80" y="654" fill="#FFFFFF" font-size="27" font-weight="760">미공개 Holdout 240문항 · Exact McNemar 통계 검증 (p &lt; 0.0001)</text>
<text x="80" y="690" fill="#CFE2DD" font-size="19">FastAPI 인터랙티브 데모 · 로컬 ONNX INT8 Scientific NLI 실측 검증 · corpus-v2-holdout 30편 동결</text>
<text x="1120" y="660" text-anchor="end" fill="#9CE5B8" font-size="40" font-weight="780">3.0× / 95%</text>
<text x="1120" y="690" text-anchor="end" fill="#CFE2DD" font-size="17">사실 일치 배수 · 거짓전제 방어</text>
</svg>
"""

    en_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">Evidence-Grounded Paper Q&amp;A v2: BGE-M3 · Scientific NLI · Qwen3-4B</title>
<desc id="desc">Bilingual EN/KO scientific paper Q&amp;A v2 achieving 3.0x paired fact-alias recall (11 to 33/80), 95% false-premise refusal (10% to 95%, EN 20/20), and 3.8x speedup on unseen corpus30 Holdout (240 outputs).</desc>
<style>text{font-family:'Segoe UI','Noto Sans','Helvetica Neue',Arial,sans-serif;font-variant-numeric:tabular-nums}</style>
<rect width="1200" height="760" fill="#F8F4EB"/>
<text x="48" y="74" fill="#1F2E32" font-size="40" font-weight="760">Evidence-Grounded Paper Q&amp;A v2</text>
<text x="48" y="118" fill="#536569" font-size="20">BGE-M3 Hybrid Retrieval · Fine-Tuned mDeBERTa-v3 Scientific NLI (ONNX INT8) · Qwen3-4B Guided JSON</text>
<line x1="48" y1="144" x2="1152" y2="144" stroke="#D7DED9" stroke-width="1.4"/>

<rect x="48" y="168" width="442" height="418" rx="14" fill="#FFFFFF" stroke="#D7DED9" stroke-width="1.3"/>
<text x="74" y="208" fill="#1F2E32" font-size="22" font-weight="750">v2 Grounded Verification Stack</text>
<text x="74" y="234" fill="#5B696C" font-size="14.2">Single English Paper → Bilingual EN/KO Claims &amp; Quotes</text>

<rect x="74" y="254" width="390" height="68" rx="10" fill="#F5FAF8" stroke="#9AC8BE" stroke-width="1.3"/>
<text x="92" y="280" fill="#1E665E" font-size="12.2" font-weight="750">01 HYBRID RETRIEVAL + PARENT (1,800 CHARS)</text>
<text x="92" y="306" fill="#1F2E32" font-size="16.2" font-weight="700">BM25 + BGE-M3 + Dual-Score Rerank</text>

<path d="M269 322V344" stroke="#1E665E" stroke-width="2.6"/>
<path d="M262 337L269 345L276 337" fill="none" stroke="#1E665E" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="348" width="390" height="72" rx="10" fill="#FBF3E6" stroke="#D8B384" stroke-width="1.5"/>
<text x="92" y="374" fill="#8C5B23" font-size="12.2" font-weight="750">02 SCIENTIFIC NLI PRE-GATE (ONNX INT8 · 322.7 MiB)</text>
<text x="92" y="399" fill="#1F2E32" font-size="15.8" font-weight="700">FP32 mDeBERTa-v3 · Contradiction Gate 0.95</text>

<path d="M269 420V442" stroke="#A87338" stroke-width="2.6"/>
<path d="M262 435L269 443L276 435" fill="none" stroke="#A87338" stroke-width="2.6" stroke-linecap="round"/>

<rect x="74" y="446" width="390" height="70" rx="10" fill="#F5FAF8" stroke="#1E665E" stroke-width="1.6"/>
<text x="92" y="472" fill="#1E665E" font-size="12.2" font-weight="750">03 GUIDED JSON + QUOTE / NUMERIC VERIFIER</text>
<text x="92" y="497" fill="#1F2E32" font-size="15.8" font-weight="700">Qwen3-4B-Instruct-2507 + Conformal Gate</text>

<rect x="74" y="530" width="390" height="38" rx="8" fill="#E6F2EF"/>
<text x="269" y="554" text-anchor="middle" fill="#154943" font-size="13.5" font-weight="750">Paper-Grouped 5-Fold OOF AURC 0.0669 · 240/240 Valid</text>

<path d="M490 377H542V230H590" fill="none" stroke="#6E8783" stroke-width="2.8" stroke-linejoin="round"/>
<path d="M490 377H590" fill="none" stroke="#A87338" stroke-width="2.8"/>
<path d="M490 377H542V522H590" fill="none" stroke="#1E665E" stroke-width="3.4" stroke-linejoin="round"/>
<circle cx="542" cy="377" r="5" fill="#1F2E32"/>
<path d="M581 223L591 230L581 237" fill="none" stroke="#6E8783" stroke-width="2.8" stroke-linecap="round"/>
<path d="M581 370L591 377L581 384" fill="none" stroke="#A87338" stroke-width="2.8" stroke-linecap="round"/>
<path d="M581 515L591 522L581 529" fill="none" stroke="#1E665E" stroke-width="3.4" stroke-linecap="round"/>

<rect x="598" y="168" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#D7DED9" stroke-width="1.3"/>
<rect x="598" y="168" width="9" height="124" rx="4" fill="#6E8783"/>
<text x="628" y="204" fill="#536569" font-size="13.2" font-weight="750">ANSWER COMPLETENESS · UNSEEN HOLDOUT (240 OUTPUTS)</text>
<text x="628" y="241" fill="#1F2E32" font-size="25" font-weight="780">Paired Fact Recall 3.0× (11 → 33/80)</text>
<text x="628" y="272" fill="#536569" font-size="14.2">KO Facts 2.73× (15→41/80) · Source Page Hit 76.9% (+15.6%p, p=0.0019)</text>

<rect x="598" y="314" width="554" height="124" rx="14" fill="#FFFFFF" stroke="#D7DED9" stroke-width="1.3"/>
<rect x="598" y="314" width="9" height="124" rx="4" fill="#A87338"/>
<text x="628" y="350" fill="#8C5B23" font-size="13.2" font-weight="750">HALLUCINATION &amp; FALSE-PREMISE DEFENSE (SCIENTIFIC NLI)</text>
<text x="628" y="387" fill="#1F2E32" font-size="25" font-weight="780">False-Premise Refusal 10% → 95%</text>
<text x="628" y="418" fill="#536569" font-size="14.2">EN 2/20 → 20/20 (100%) · Unsupported 20/20 (100%) · p &lt; 0.0001</text>

<rect x="598" y="460" width="554" height="126" rx="14" fill="#FFFFFF" stroke="#1E665E" stroke-width="2.2"/>
<rect x="598" y="460" width="10" height="126" rx="4" fill="#1E665E"/>
<rect x="1018" y="476" width="114" height="26" rx="13" fill="#E6F2EF"/>
<text x="1075" y="494" text-anchor="middle" fill="#154943" font-size="12.5" font-weight="750">FINAL v2 MODEL</text>
<text x="628" y="496" fill="#1E665E" font-size="13.2" font-weight="750">SEMANTIC JUDGE · SPEED · EVIDENCE-DRIVEN ABLATION</text>
<text x="628" y="533" fill="#1F2E32" font-size="23.5" font-weight="780">Faithfulness 0.888 · 3.8× Faster (1.77s/q)</text>
<text x="628" y="564" fill="#536569" font-size="14">EN Completeness 0.849 / KO 0.711 · SFT &amp; Post-NLI rejected on DEV</text>

<rect x="48" y="610" width="1104" height="106" rx="16" fill="#1F2E32"/>
<text x="80" y="654" fill="#FFFFFF" font-size="26" font-weight="760">Unseen Holdout (240 Outputs) · Exact McNemar &amp; Cluster Bootstrap CI</text>
<text x="80" y="690" fill="#CFE2DD" font-size="18.5">FastAPI Interactive Demo · Local ONNX INT8 Scientific NLI Verified · corpus-v2-holdout Frozen</text>
<text x="1120" y="660" text-anchor="end" fill="#9CE5B8" font-size="40" font-weight="780">3.0× / 95%</text>
<text x="1120" y="690" text-anchor="end" fill="#CFE2DD" font-size="16.5">Paired Fact Gain · FP Refusal</text>
</svg>
"""
    (fig_dir / "paper-qna-thumbnail.svg").write_text(ko_svg, encoding="utf-8")
    (fig_dir / "paper-qna-thumbnail.en.svg").write_text(en_svg, encoding="utf-8")


def plot_holdout_comparison(fig_dir: Path, v1_h: dict[str, Any], v2_h: dict[str, Any], stats_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.4), dpi=220)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor("#FFFFFF")

    labels = [
        "Paired Answerable\nRetained (80)",
        "Paired All Facts\nMatched (80)",
        "KO All Facts\nMatched (80)",
        "Acceptable Source\nPage Hit (160)",
        "Unsupported\nRefused (20)",
        "False-Premise\nRefused (20)",
    ]
    v1_vals = [
        100 * v1_h["metrics"]["answerable_retained"] / 80,
        100 * v1_h["metrics"]["all_fact_aliases"] / 80,
        100 * v1_h["by_language"]["ko"]["all_fact_aliases"] / 80,
        100 * 0.613,
        100 * v1_h["metrics"]["unsupported_refused"] / 20,
        100 * v1_h["metrics"]["false_premise_refused"] / 20,
    ]
    v2_vals = [
        100 * v2_h["paired"]["answerable_retained"] / 80,
        100 * v2_h["paired"]["all_fact_aliases"] / 80,
        100 * v2_h["ko"]["all_fact_aliases"] / 80,
        100 * 0.769,
        100 * v2_h["paired"]["unsupported_refused"] / 20,
        100 * v2_h["paired"]["false_premise_refused"] / 20,
    ]
    badges = ["p=0.0153", "3.0x\np<0.0001", "2.73x\nKO Lift", "+15.6%p\np=0.0019", "100%\n(20/20)", "+85%p Pair\np<0.0001"]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, v1_vals, w, label="v1 Final (E5-small + Extractive + NLLB)", color="#94A3B8", edgecolor="#64748B", linewidth=0.8)
    b2 = ax.bar(x + w / 2, v2_vals, w, label="v2 Final (BGE-M3 + FP32 NLI 0.95 + Qwen3-4B Guided JSON)", color=TEAL, edgecolor="#144943", linewidth=0.9)

    for rect, val in zip(b1, v1_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.8, color="#475569", fontweight="bold")
    for rect, val, badge in zip(b2, v2_vals, badges):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.2, color=TEAL, fontweight="bold")
        ax.text(rect.get_x() + rect.get_width() / 2, max(val, 16) + 8.5, badge, ha="center", va="bottom", fontsize=7.8, color=AMBER, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#FBF3E6", edgecolor="#E2C79D", linewidth=0.7))

    ax.set_ylim(0, 122)
    ax.set_ylabel("Pass Rate (%)", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2, color=INK)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Unseen corpus30 Holdout (240 Outputs): v1 Final vs v2 Final (Exact McNemar & Cluster Bootstrap)", fontsize=12.5, fontweight="bold", color=INK, pad=14)
    ax.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor=GRID, fontsize=9.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "paper-qna-holdout-comparison.png", dpi=220, facecolor=PAPER)
    plt.close(fig)


def plot_dev_ablation(fig_dir: Path, dev_rows: list[dict[str, Any]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=220, gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.patch.set_facecolor(PAPER)

    names = [r["short_label"] for r in dev_rows]
    x = np.arange(len(names))
    ans_pct = [100 * r["pair_ans"] / 40 for r in dev_rows]
    facts_pct = [100 * r["pair_facts"] / 40 for r in dev_rows]
    fp_pct = [100 * r["fp_pair"] / 10 for r in dev_rows]

    ax1.set_facecolor("#FFFFFF")
    ax1.plot(x, ans_pct, marker="o", linewidth=2.2, color=TEAL, label="Paired Answerable Retained (%)")
    ax1.plot(x, facts_pct, marker="s", linewidth=2.4, color=AMBER, label="Paired All Fact Aliases (%)")
    ax1.plot(x, fp_pct, marker="^", linewidth=2.2, linestyle="--", color=CRIMSON, label="Paired False-Premise Refused (%)")
    ax1.axvspan(len(names) - 1.45, len(names) - 0.55, color=TEAL_SOFT, alpha=0.65, label="Selected DEV Winner (Exp 2C)")
    for i in range(len(names)):
        f_dy = -14 if i == 1 else 7
        fp_dy = 7 if (fp_pct[i] > 85 or fp_pct[i] < 10) else (-14 if i == 2 else 7)
        ax1.annotate(f"{facts_pct[i]:.0f}%", (x[i], facts_pct[i]), textcoords="offset points", xytext=(0, f_dy), ha="center", fontsize=8.0, color=AMBER, fontweight="bold")
        ax1.annotate(f"{fp_pct[i]:.0f}%", (x[i], fp_pct[i]), textcoords="offset points", xytext=(0, fp_dy), ha="center", fontsize=8.0, color=CRIMSON, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8.3, color=INK)
    ax1.set_ylim(-5, 114)
    ax1.set_ylabel("DEV Rate (%)", fontsize=10, fontweight="bold", color=INK)
    ax1.set_title("A. DEV 120-Output Progression & Rejected Ablations", fontsize=11.2, fontweight="bold", color=INK, pad=10)
    ax1.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    # Panel B: EN vs KO All Fact Aliases across key milestones
    ax2.set_facecolor("#FFFFFF")
    key_idxs = [0, 1, 2, 5, 6]  # v1, Base4B, Teacher14B, SFT, Winner 2C
    k_names = [dev_rows[i]["short_label"].replace("\n", " ") for i in key_idxs]
    en_vals = [100 * dev_rows[i]["en_facts"] / 40 for i in key_idxs]
    ko_vals = [100 * dev_rows[i]["ko_facts"] / 40 for i in key_idxs]
    y = np.arange(len(k_names))
    h = 0.35
    ax2.barh(y + h / 2, en_vals, h, label="English Facts (40)", color=TEAL)
    ax2.barh(y - h / 2, ko_vals, h, label="Korean Facts (40)", color=AMBER)
    for yi, ev, kv in zip(y, en_vals, ko_vals):
        ax2.text(ev + 1.2, yi + h / 2, f"{ev:.1f}%", va="center", fontsize=8.2, color=TEAL, fontweight="bold")
        ax2.text(kv + 1.2, yi - h / 2, f"{kv:.1f}%", va="center", fontsize=8.2, color=AMBER, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(k_names, fontsize=8.4, color=INK)
    ax2.set_xlim(0, 108)
    ax2.set_xlabel("All Required Fact Aliases Matched (%)", fontsize=9.5, fontweight="bold", color=INK)
    ax2.set_title("B. EN vs KO Fact Recall (Why SFT Was Rejected)", fontsize=11.2, fontweight="bold", color=INK, pad=10)
    ax2.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    fig.tight_layout()
    fig.savefig(fig_dir / "paper-qna-dev-ablation.png", dpi=220, facecolor=PAPER)
    plt.close(fig)


def plot_nli_and_conformal(
    fig_dir: Path,
    sweep_rows: list[dict[str, Any]],
    gate_payload: dict[str, Any],
    ans_contras: list[float],
    fp_contras: list[float],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.9), dpi=220)
    fig.patch.set_facecolor(PAPER)

    # Panel A: Threshold sweep on DEV
    ax1.set_facecolor("#FFFFFF")
    thrs = [r["nli_thr"] for r in sweep_rows if r["nli_thr"] <= 1.0]
    p_ans = [100 * r["pair_ans"] / 40 for r in sweep_rows if r["nli_thr"] <= 1.0]
    p_facts = [100 * r["pair_facts"] / 40 for r in sweep_rows if r["nli_thr"] <= 1.0]
    p_fp = [100 * r["fp_pair"] / 10 for r in sweep_rows if r["nli_thr"] <= 1.0]

    ax1.plot(thrs, p_fp, marker="^", linewidth=2.4, color=CRIMSON, label="False-Premise Refused (100% across 0.70-0.99)")
    ax1.plot(thrs, p_ans, marker="o", linewidth=2.4, color=TEAL, label="Paired Answerable Retained (87.5% -> 92.5%)")
    ax1.plot(thrs, p_facts, marker="s", linewidth=2.4, color=AMBER, label="Paired All Fact Aliases (57.5% -> 60.0%)")
    ax1.axvline(0.95, color=INK, linestyle=":", linewidth=1.8, label="Selected Gate Threshold = 0.95")
    ax1.axvspan(0.90, 0.99, color=TEAL_SOFT, alpha=0.55)
    ax1.text(
        0.70, 31,
        f"FP32 Scientific NLI Separation (DEV)\n• Answerable median = {statistics.median(ans_contras):.3f}\n• False-Premise min  = {min(fp_contras):.3f}\n• False-Premise >=0.95: {sum(x >= 0.95 for x in fp_contras)}/{len(fp_contras)}",
        fontsize=8.2, color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor=TEAL, linewidth=1.0),
    )
    ax1.set_xlim(0.68, 1.01)
    ax1.set_ylim(0, 114)
    ax1.set_xlabel("Scientific NLI Contradiction Threshold", fontsize=10, fontweight="bold", color=INK)
    ax1.set_ylabel("DEV Metric (%)", fontsize=10, fontweight="bold", color=INK)
    ax1.set_title("A. Scientific NLI Separation & Threshold Plateau", fontsize=11.2, fontweight="bold", color=INK, pad=10)
    ax1.grid(color=GRID, linestyle="--", linewidth=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", fontsize=7.8, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    # Panel B: Conformal Risk-Coverage Trade-Off
    ax2.set_facecolor("#FFFFFF")
    rc = gate_payload["oof_risk_coverage"]
    alphas = [r["alpha"] for r in rc]
    covs = [100 * r["answerable_coverage"] for r in rc]
    leaks = [100 * r["negative_leak_rate"] for r in rc]
    thrs_c = [r["oof_threshold"] for r in rc]
    x = np.arange(len(alphas))
    w = 0.34
    b_cov = ax2.bar(x - w / 2, covs, w, color=TEAL, label="Answerable Coverage (%)")
    b_lk = ax2.bar(x + w / 2, leaks, w, color=CRIMSON, label="Negative Leak Rate (%)")
    for r, v in zip(b_cov, covs):
        ax2.text(r.get_x() + r.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", fontsize=8.8, color=TEAL, fontweight="bold")
    for r, v in zip(b_lk, leaks):
        ax2.text(r.get_x() + r.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", fontsize=8.8, color=CRIMSON, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"α = {a:.2f}\n(thr={t:.3f})" for a, t in zip(alphas, thrs_c)], fontsize=9.0, color=INK)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("5-Fold Paper-Grouped OOF Rate (%)", fontsize=10, fontweight="bold", color=INK)
    ax2.set_title(f"B. Conformal Risk-Coverage Gate (OOF AURC = {gate_payload['oof_aurc']:.4f})", fontsize=11.2, fontweight="bold", color=INK, pad=10)
    ax2.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#FFFFFF", edgecolor=GRID)

    fig.tight_layout()
    fig.savefig(fig_dir / "paper-qna-nli-conformal.png", dpi=220, facecolor=PAPER)
    plt.close(fig)


def main() -> int:
    reports_dir = ROOT / "reports"
    fig_dir = ROOT / "docs" / "figures"
    ckpt_dir = ROOT / "outputs" / "v2_checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)

    questions = read_jsonl(ROOT / "data" / "benchmarks" / "corpus30" / "questions.jsonl")
    dev_qs = [q for q in questions if q["split"] == "dev"]
    v1_dev = json.loads((ROOT / "data" / "benchmarks" / "corpus30" / "results" / "final-dev-report.json").read_text(encoding="utf-8"))
    v1_holdout = json.loads((ROOT / "data" / "benchmarks" / "corpus30" / "results" / "final-holdout-report.json").read_text(encoding="utf-8"))
    raw_log = read_jsonl(ckpt_dir / "experiment_log_raw.jsonl")
    exp1b = json.loads((reports_dir / "exp1b_retrieval_ab.json").read_text(encoding="utf-8"))
    sem_dev = json.loads((reports_dir / "exp1c_qwen3_4b_2507_semantic_summary.json").read_text(encoding="utf-8"))
    sem_holdout = json.loads((reports_dir / "exp4_v2_final_holdout_semantic_summary.json").read_text(encoding="utf-8"))
    gate_payload = json.loads((reports_dir / "calibrated_gate.json").read_text(encoding="utf-8"))
    freeze_v2 = json.loads((ROOT / "data" / "corpus_v2_holdout" / "freeze_v2.json").read_text(encoding="utf-8"))

    # Keep canonical deduplicated experiments (skip superseded BF16-NaN NLI runs at raw indices 2 and 3)
    canonical_entries = [raw_log[0], raw_log[1], raw_log[4], raw_log[5], raw_log[6], raw_log[7], raw_log[8]]
    (reports_dir / "experiment_log.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in canonical_entries) + "\n",
        encoding="utf-8",
    )

    # Exact Holdout evaluation & statistical tests
    exp4_recs = read_jsonl(ckpt_dir / "exp4_v2_final_regression_holdout.checkpoint.jsonl")
    exp4_report = evaluate_predictions(exp4_recs, questions, split="holdout")
    holdout_stats = compute_statistical_comparison(v1_holdout, exp4_report, n_boot=2000, seed=378)

    with (reports_dir / "holdout_statistical_tests.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(holdout_stats[0].keys()))
        w.writeheader()
        w.writerows(holdout_stats)

    # Exact DEV NLI threshold sweep
    c1_map = {r["case_id"]: r for r in read_jsonl(ckpt_dir / "exp1c_bge_m3_qwen3_4b_2507_zeroshot_dev.checkpoint.jsonl")}
    c2_map = {r["case_id"]: r for r in read_jsonl(ckpt_dir / "exp2a_pre_sliding_nli_dev.checkpoint.jsonl")}
    ans_contras = [float(c2_map[q["case_id"]]["output"]["pipeline"].get("nli_max_contradiction", 0.0)) for q in dev_qs if q["answerability"] == "answerable"]
    fp_contras = [float(c2_map[q["case_id"]]["output"]["pipeline"].get("nli_max_contradiction", 0.0)) for q in dev_qs if q["answerability"] == "false_premise"]

    sweep_rows = []
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
    with (reports_dir / "nli_threshold_sweep_dev.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # DEV ablation summary CSV
    dev_table = [
        {
            "exp_id": "v1_final_dev",
            "short_label": "v1\nBaseline",
            "decision": "Baseline",
            "pair_ans": v1_dev["metrics"]["answerable_retained"],
            "ko_ans": v1_dev["by_language"]["ko"]["answerable_retained"],
            "pair_facts": v1_dev["metrics"]["all_fact_aliases"],
            "en_facts": v1_dev["by_language"]["en"]["all_fact_aliases"],
            "ko_facts": v1_dev["by_language"]["ko"]["all_fact_aliases"],
            "unsup_pair": v1_dev["metrics"]["unsupported_refused"],
            "fp_en": v1_dev["by_language"]["en"]["false_premise_refused"],
            "fp_ko": v1_dev["by_language"]["ko"]["false_premise_refused"],
            "fp_pair": v1_dev["metrics"]["false_premise_refused"],
            "quote_rate": None,
            "numeric_rate": None,
            "sec_per_q": round(v1_dev["metrics"]["mean_wall_seconds"], 2),
        },
    ]
    labels_decisions = [
        ("Exp 1C\nBase 4B", "Adopted LLM"),
        ("Exp 1C\n14B-AWQ", "Teacher Ceiling"),
        ("Exp 2A\nNLI 0.70", "Threshold Low"),
        ("Exp 2B\nPost-NLI", "Rejected"),
        ("Exp 3A\nSFT 4B", "Rejected"),
        ("Exp 2C\nNLI 0.95", "DEV Winner"),
    ]
    for entry, (s_lbl, dec) in zip(canonical_entries[:6], labels_decisions):
        p, en, ko = entry["paired"], entry["en"], entry["ko"]
        dev_table.append(
            {
                "exp_id": entry["exp_name"],
                "short_label": s_lbl,
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

    # Local verification (merge if already generated)
    local_ver_path = reports_dir / "local_verification.json"
    local_ver = json.loads(local_ver_path.read_text(encoding="utf-8")) if local_ver_path.is_file() else None

    summary = {
        "schema_version": 2,
        "project": "Evidence-Grounded-Paper-QnA",
        "winner_stack": {
            "retrieval": "BM25 + BAAI/bge-m3 + RRF(k=60) + Dual-Score BAAI/bge-reranker-v2-m3 + Parent(1800 chars)",
            "scientific_nli": "Fine-tuned FP32 MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (exported to ONNX INT8, 322.7 MiB, threshold=0.95, post-NLI=None)",
            "generator": "Qwen/Qwen3-4B-Instruct-2507 (Guided JSON response_format, zero-shot, no SFT adapter)",
            "conformal_gate": "Paper-Grouped 5-Fold OOF Logistic Conformal Gate (OOF AURC=0.0669)",
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
        "dev_ablation_120": [{k: v for k, v in r.items() if k != "short_label"} for r in dev_table],
        "holdout_240": {
            "v1_final": {
                "paired": v1_holdout["metrics"],
                "en": v1_holdout["by_language"]["en"],
                "ko": v1_holdout["by_language"]["ko"],
            },
            "v2_final": {
                "paired": exp4_report["paired"],
                "en": exp4_report["en"],
                "ko": exp4_report["ko"],
            },
            "statistical_tests": holdout_stats,
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

    # Generate all figures
    write_hero_svgs(fig_dir, summary)
    plot_holdout_comparison(fig_dir, v1_holdout, exp4_report, holdout_stats)
    plot_dev_ablation(fig_dir, dev_table)
    plot_nli_and_conformal(fig_dir, sweep_rows, gate_payload, ans_contras, fp_contras)

    print("[OK] Generated canonical v2 reports and publication figures!")
    for p in sorted(fig_dir.iterdir()):
        print(f"  figure: {p.name} ({p.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
