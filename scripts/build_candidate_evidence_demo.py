"""Build reproducible Day 23-24 candidate and evidence examples."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from candidate_evidence_pipeline import CandidateEvidencePipeline
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_evaluation_data


DEFAULT_JSON = PROJECT_ROOT / "data" / "evaluation" / "candidate_evidence_demo.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "data" / "evaluation" / "candidate_evidence_demo.md"
DEMO_CASES = [
    ("q-001-ko", "multilingual_success", "한국어 질문에서 영어 근거를 찾는 대표 성공 사례"),
    ("q-043-ko", "reranker_recovery", "Day 22 reranker가 Top-5로 복구한 사례"),
    ("q-005-ko", "known_limitation", "후보 생성 또는 용어 불일치를 사람이 판별하는 사례"),
]


def markdown(payload):
    lines = [
        "# 23일차부터 24일차 후보와 근거 사례",
        "",
        "## 고정 정책",
        "",
        "- 후보군: BM25 Top-20 + Dense Top-20, 동일 가중치 RRF, 고유 청크 20개",
        "- 재정렬: 후보 20개 전체에 다국어 Cross-Encoder 적용",
        "- 최종 근거: 중복이 아닌 상위 5개, PDF 페이지마다 최대 2개",
        "- 유사 중복: token 3-shingle Jaccard 유사도 0.85 이상",
        "- 범위: 텍스트 근거만 사용하며 답변 생성과 자동 거절 판정은 제외",
        "",
        "## 사람이 확인하는 답변 가능성 점검표",
        "",
        "각 사례의 근거 5개를 읽고 요청한 내용을 직접 뒷받침하는지, 중요한 한정 표현이",
        "있는지, 페이지와 절 위치만으로 확인 가능한지 살핀다. 낮은 점수만으로 답변을",
        "거절하지 않으며 자동 충분성 판정은 이후 로드맵 단계에서 다룬다.",
        "",
    ]
    for case in payload["examples"]:
        lines.extend([
            f"## {case['question_id']} - {case['case_type']}",
            "",
            f"- 질문: {case['question']}",
            f"- 논문: `{case['paper_id']}`",
            f"- Gold 페이지(평가 전용): {case['gold_page']}",
            f"- 후보 수: {case['candidate_count']}",
            f"- 근거 수: {case['evidence_count']}",
            f"- Top-5 근거에 Gold 페이지 포함: {'예' if case['gold_page_in_evidence'] else '아니요'}",
            f"- 목적: {case['summary']}",
            "",
        ])
        for evidence in case["evidence"]:
            excerpt = " ".join(evidence["text"].split())[:500].rstrip()
            lines.extend([
                f"### 근거 {evidence['rank']}: p. {evidence['page']} - {evidence['section']}",
                "",
                f"- chunk_id: `{evidence['chunk_id']}`",
                f"- 재정렬 점수: {evidence['scores']['reranker']:.6f}",
                f"- Hybrid 순위: {evidence['scores']['pre_rerank_rank']}",
                f"- 원문: {excerpt}",
                "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="기존 JSON을 사용해 Markdown 설명만 다시 만듭니다.",
    )
    args = parser.parse_args()

    if args.render_only:
        payload = json.loads(args.json_output.read_text(encoding="utf-8"))
        args.markdown_output.write_text(markdown(payload), encoding="utf-8")
        print(f"Saved: {args.markdown_output}")
        return

    questions, gold_by_question = load_evaluation_data()
    question_by_id = {row["question_id"]: row for row in questions}
    started = perf_counter()
    pipeline = CandidateEvidencePipeline()
    examples = []
    for question_id, case_type, summary in DEMO_CASES:
        question = question_by_id[question_id]
        result = pipeline.run(question["question"], question["paper_id"])
        gold_page = int(gold_by_question[question_id]["gold_page"])
        evidence_pages = [item["page"] for item in result["evidence"]]
        candidate_gold_ranks = [
            item["candidate_rank"]
            for item in result["candidates"]
            if item["page"] == gold_page
        ]
        examples.append({
            "question_id": question_id,
            "case_type": case_type,
            "summary": summary,
            "paper_id": question["paper_id"],
            "question": question["question"],
            "question_language": question["question_language"],
            "gold_page": gold_page,
            "gold_candidate_rank": min(candidate_gold_ranks) if candidate_gold_ranks else None,
            "gold_page_in_evidence": gold_page in evidence_pages,
            "candidate_count": len(result["candidates"]),
            "evidence_count": len(result["evidence"]),
            "candidate_policy": result["candidate_policy"],
            "evidence_selection": result["evidence_selection"],
            "candidates": result["candidates"],
            "evidence": result["evidence"],
        })

    payload = {
        "schema_version": 1,
        "roadmap_days": [23, 24],
        "purpose": "사람이 읽는 답변 가능성 검토와 UI 근거 계약 검증",
        "evaluation_note": "Gold 페이지는 이 사례 검증에만 포함하며 파이프라인 입력으로 사용하지 않는다.",
        "corpus": {"papers": 10, "chunks": len(pipeline.chunks)},
        "embedding_cache": "hit" if pipeline.embedding_cache_hit else "created",
        "examples": examples,
        "runtime_seconds": round(perf_counter() - started, 3),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(payload), encoding="utf-8")
    print(f"Saved: {args.json_output}")
    print(f"Saved: {args.markdown_output}")
    for case in examples:
        print(
            f"{case['question_id']}: candidates={case['candidate_count']} "
            f"evidence={case['evidence_count']} "
            f"gold_page_in_evidence={case['gold_page_in_evidence']}"
        )
    print(f"Runtime: {payload['runtime_seconds']:.3f}s")


if __name__ == "__main__":
    main()
