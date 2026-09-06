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
        "# Day 23-24 Candidate and Evidence Demo",
        "",
        "## Fixed policy",
        "",
        "- Candidate pool: BM25 Top-20 + dense Top-20, equal-weight RRF, 20 unique chunks",
        "- Reranking: multilingual Cross-Encoder over all 20 candidates",
        "- Final evidence: first 5 non-duplicate chunks, maximum 2 chunks per PDF page",
        "- Near duplicate: token 3-shingle Jaccard similarity >= 0.85",
        "- Scope: text evidence only; no answer generation or automatic abstention decision",
        "",
        "## Human answerability checklist",
        "",
        "For each case, read the five evidence texts and check whether they directly support",
        "the requested claim, whether critical qualifiers are present, and whether the page",
        "and section locators are sufficient for verification. A low score alone is not an",
        "abstention rule; automatic sufficiency is intentionally deferred to the later roadmap.",
        "",
    ]
    for case in payload["examples"]:
        lines.extend([
            f"## {case['question_id']} — {case['case_type']}",
            "",
            f"- Question: {case['question']}",
            f"- Paper: `{case['paper_id']}`",
            f"- Gold page (evaluation only): {case['gold_page']}",
            f"- Candidate count: {case['candidate_count']}",
            f"- Evidence count: {case['evidence_count']}",
            f"- Gold page in Top-5 evidence: {'yes' if case['gold_page_in_evidence'] else 'no'}",
            f"- Purpose: {case['summary']}",
            "",
        ])
        for evidence in case["evidence"]:
            excerpt = " ".join(evidence["text"].split())[:500]
            lines.extend([
                f"### Evidence {evidence['rank']}: p. {evidence['page']} · {evidence['section']}",
                "",
                f"- chunk_id: `{evidence['chunk_id']}`",
                f"- reranker score: {evidence['scores']['reranker']:.6f}",
                f"- Hybrid rank: {evidence['scores']['pre_rerank_rank']}",
                f"- Text: {excerpt}",
                "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

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
        "purpose": "human-readable answerability review and UI evidence contract verification",
        "evaluation_note": "Gold pages are included only to verify this demo; they are not pipeline inputs.",
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
