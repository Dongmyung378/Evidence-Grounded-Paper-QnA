"""Create the manually reviewed Day 19 retrieval failure analysis."""

import argparse
from collections import Counter
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


DEFAULT_COMPARISON = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.json"
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
DEFAULT_GOLD = PROJECT_ROOT / "data" / "evaluation" / "gold_evidence.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "failure_analysis.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "data" / "evaluation" / "failure_analysis.md"
RECALL_K = 5


# These labels and rationales are the result of a manual question/evidence/ranking review.
REVIEWED_CASES = {
    "q-049-ko": {
        "category": "chunking",
        "rationale": "정답이 있는 결론 페이지가 표·그림 캡션과 함께 길고 잡음 많은 청크로 추출되어 핵심 결론 표현이 희석됐다.",
        "recommended_action": "섹션 경계와 문단 경계를 우선하는 청킹을 검토한다.",
    },
    "q-005-en": {
        "category": "chunking",
        "rationale": "초록의 방법 요약이 여러 청크로 나뉘고 같은 방법명이 본문 여러 페이지에 반복되어 BM25의 정답 페이지가 6위로 밀렸다.",
        "recommended_action": "초록·결론은 섹션 단위 보존 또는 인접 청크 확장을 검토한다.",
    },
    "q-005-ko": {
        "category": "terminology",
        "rationale": "한국어의 일반적인 프레임워크 표현과 영어 논문의 MoE·MLA·Muon·μP 고유 용어가 직접 대응하지 않아 모든 검색기가 정답 페이지를 상위 5개에 올리지 못했다.",
        "recommended_action": "다국어 질의 확장과 모델·약어 사전을 검토한다.",
    },
    "q-001-ko": {
        "category": "terminology",
        "rationale": "'주요 문제'라는 한국어 일반 표현이 operational non-identifiability, RFI, subspace overlap이라는 논문의 구체 용어를 충분히 제공하지 않아 dense 순위가 8위였다.",
        "recommended_action": "질문 의도를 논문 핵심 용어로 확장하는 방식을 검토한다.",
    },
    "q-017-ko": {
        "category": "terminology",
        "rationale": "한국어의 표현 학습·과정 추론 표현과 영어 초록의 hierarchy·sensing·support·uncertainty 표현 사이 의미 대응이 약해 dense가 10위 안에 들지 못했다.",
        "recommended_action": "한국어-영어 과학 용어 동의어 확장을 검토한다.",
    },
    "q-038-ko": {
        "category": "numbers",
        "rationale": "정답 근거가 INT8, FP32, CPU 명칭, 모델명과 같은 숫자·식별자 중심이며 이 항목들이 여러 페이지에 반복되어 BM25 정답 페이지가 6위였다.",
        "recommended_action": "숫자·모델명·하드웨어 식별자에 대한 필드 가중치를 검토한다.",
    },
    "q-042-en": {
        "category": "numbers",
        "rationale": "평가 근거가 관측 섹터, 표본 수, 임계값 등 수치 중심이라 의미 임베딩이 일반 평가 설명보다 정답 페이지를 낮게 평가해 dense 순위가 9위였다.",
        "recommended_action": "숫자 토큰 보존과 lexical 점수 가중을 검토한다.",
    },
    "q-045-ko": {
        "category": "numbers",
        "rationale": "모델 폭, 학습률, 스케일과 μP 표기가 핵심 단서인데 dense가 정량적 식별자를 충분히 우선하지 않아 7위였다.",
        "recommended_action": "정량 질의에는 BM25 비중 또는 숫자 일치 보너스를 검토한다.",
    },
    "q-043-ko": {
        "category": "multiple_evidence",
        "rationale": "합성 데이터와 ABIDE 실험, 비교 기준, 평가 지표가 여러 문단·페이지에 분산되어 단일 gold 페이지와 단일 청크 중심 평가에서 hybrid 순위가 8위였다.",
        "recommended_action": "인접·보완 근거를 묶는 multi-evidence retrieval을 검토한다.",
    },
    "q-015-en": {
        "category": "multiple_evidence",
        "rationale": "개선 근거가 실험 결과와 학습 스케줄 설명, 결론에 분산되어 dense가 개별 결과 페이지를 우선하고 gold 결론 페이지는 6위에 머물렀다.",
        "recommended_action": "결과와 결론을 연결하는 다중 청크 집계를 검토한다.",
    },
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def clean_excerpt(text, limit=320):
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit].rstrip() + "..."


def rank_text(rank):
    return "-" if rank is None else str(rank)


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    questions = {row["question_id"]: row for row in load_jsonl(args.questions)}
    gold = {row["question_id"]: row for row in load_jsonl(args.gold)}
    result_maps = {
        method: {row["question_id"]: row for row in rows}
        for method, rows in comparison["results"].items()
    }

    cases = []
    for question_id, review in REVIEWED_CASES.items():
        question = questions[question_id]
        evidence = gold[question_id]
        method_results = {}
        missed_by = []
        for method in ("bm25", "dense", "hybrid_rrf"):
            result = result_maps[method][question_id]
            rank = result["gold_rank"]
            missed = rank is None or rank > RECALL_K
            if missed:
                missed_by.append(method)
            method_results[method] = {
                "gold_rank": rank,
                "hit_at_5": not missed,
                "predicted_pages_at_5": result["predicted_pages"][:RECALL_K],
            }
        if not missed_by:
            raise ValueError(f"{question_id} is not a Recall@5 failure for any retriever")
        cases.append({
            "question_id": question_id,
            "paper_id": question["paper_id"],
            "language": question["question_language"],
            "question": question["question"],
            "gold_page": evidence["gold_page"],
            "gold_section": evidence["gold_section"],
            "gold_evidence_excerpt": clean_excerpt(evidence["gold_evidence_text"]),
            "missed_by": missed_by,
            "method_results": method_results,
            **review,
        })

    category_counts = Counter(case["category"] for case in cases)
    artifact = {
        "schema_version": 1,
        "roadmap_day": 19,
        "review_type": "manual failure analysis grounded in question, gold evidence, and retrieval ranks",
        "failure_definition": "gold evidence page is outside a retriever's top 5 unique pages",
        "evaluation_set": "verified-40",
        "reviewed_cases": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "categories": {
            "chunking": "청크 경계·길이·추출 잡음으로 핵심 표현이 희석된 사례",
            "terminology": "질문과 논문 사이 언어·전문용어·약어 불일치 사례",
            "numbers": "수치·모델명·하드웨어 식별자 중심 근거 사례",
            "multiple_evidence": "답변 근거가 여러 청크나 페이지에 분산된 사례",
        },
        "cases": cases,
        "limitations": [
            "40문항·10편의 소규모 평가 세트에 대한 진단이며 일반화 성능 추정은 아니다.",
            "gold relevance가 단일 페이지 기준이므로 보완 근거 페이지는 오답처럼 집계될 수 있다.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Day 19 검색 실패 분석",
        "",
        "고정된 verified 40문항에서 한 검색기라도 gold 근거 페이지를 Top-5에 포함하지 못한 사례 중 10건을 수동 검토했다. 분류는 질문, gold 근거, 세 검색기의 페이지 순위를 함께 확인해 정했다.",
        "",
        "| 질문 ID | 언어 | 원인 | 실패 검색기 | Gold 순위 (BM25/Dense/Hybrid) | 판단 근거 |",
        "|---|---|---|---|---|---|",
    ]
    for case in cases:
        ranks = "/".join(rank_text(case["method_results"][method]["gold_rank"]) for method in ("bm25", "dense", "hybrid_rrf"))
        reason = case["rationale"].replace("|", "\\|")
        lines.append(
            f"| {case['question_id']} | {case['language']} | {case['category']} | "
            f"{', '.join(case['missed_by'])} | {ranks} | {reason} |"
        )
    lines.extend([
        "",
        "## 원인별 건수",
        "",
        *[f"- {category}: {count}건" for category, count in sorted(category_counts.items())],
        "",
        "## 해석 범위",
        "",
        "이 분류는 오류 원인의 가설을 기록한 진단 자료다. 현재 평가는 단일 gold 페이지 일치 방식이므로, 검색된 페이지가 답변에 유용하더라도 gold 페이지가 아니면 실패로 집계될 수 있다.",
        "",
    ])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"reviewed_cases": len(cases), "category_counts": artifact["category_counts"]}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")
    print(f"Saved: {args.markdown}")


if __name__ == "__main__":
    main()
